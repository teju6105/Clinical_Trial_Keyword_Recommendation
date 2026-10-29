
from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "clinical_trials_cleaned.csv"
RESULTS_DIR = ROOT / "results"
DETAIL_PATH = RESULTS_DIR / "improved_bm25_evaluation.csv"
SUMMARY_PATH = RESULTS_DIR / "improved_bm25_evaluation_report.txt"

SEED = 42
TEST_FRACTION = 0.20
HIDE_FRACTION = 0.30
K = 5
TOP_NEIGHBORS = 10

TEXT_COLUMNS = [
    "title",
    "official_title",
    "conditions",
    "brief_summary",
    "interventions",
]


def parse_keywords(value):
    if pd.isna(value) or not str(value).strip():
        return set()
    return {
        " ".join(part.lower().split())
        for part in str(value).split(";")
        if part.strip()
    }


def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def precision_at_k(recommendations, relevant):
    return len(set(recommendations[:K]) & relevant) / K


def recall_at_k(recommendations, relevant):
    if not relevant:
        return float("nan")
    return len(set(recommendations[:K]) & relevant) / len(relevant)


def ndcg_at_k(recommendations, relevant):
    dcg = sum(
        1 / math.log2(rank + 2)
        for rank, keyword in enumerate(recommendations[:K])
        if keyword in relevant
    )
    ideal_count = min(len(relevant), K)
    idcg = sum(
        1 / math.log2(rank + 2)
        for rank in range(ideal_count)
    )
    return dcg / idcg if idcg else float("nan")


def main():
    df = pd.read_csv(DATA_PATH).fillna("")

    required = ["nct_id", "original_keywords"] + TEXT_COLUMNS
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["keyword_set"] = df["original_keywords"].apply(parse_keywords)
    eligible = df[df["keyword_set"].map(len) >= 2].copy()

    # Reproduce the original BM25 evaluation split exactly.
    rng = np.random.default_rng(SEED)
    order = rng.permutation(eligible.index.to_numpy())
    test_count = max(1, round(len(order) * TEST_FRACTION))
    test_indices = set(order[:test_count])

    train = eligible.loc[~eligible.index.isin(test_indices)].copy()
    test = eligible.loc[eligible.index.isin(test_indices)].copy()

    test_cases = []
    for index, row in test.iterrows():
        all_keywords = sorted(row["keyword_set"])
        hidden_count = max(1, round(len(all_keywords) * HIDE_FRACTION))
        hidden_count = min(hidden_count, len(all_keywords) - 1)

        hidden = set(
            rng.choice(all_keywords, size=hidden_count, replace=False).tolist()
        )

        test_cases.append({
            "index": index,
            "row": row,
            "hidden": hidden,
            "visible": row["keyword_set"] - hidden,
        })

    train_text = (
        train[TEXT_COLUMNS].astype(str).agg(" ".join, axis=1)
        .str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    test_text = (
        test[TEXT_COLUMNS].astype(str).agg(" ".join, axis=1)
        .str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )

    train_tokens = [tokenize(text) or ["emptydocument"] for text in train_text]
    test_tokens = [tokenize(text) or ["emptydocument"] for text in test_text]

    bm25 = BM25Okapi(train_tokens)
    test_position = {index: pos for pos, index in enumerate(test.index)}
    train_keyword_sets = train["keyword_set"].tolist()

    # Compute keyword rarity using training trials only.
    keyword_document_frequency = {}
    for keywords in train_keyword_sets:
        for keyword in keywords:
            keyword_document_frequency[keyword] = (
                keyword_document_frequency.get(keyword, 0) + 1
            )

    total_train = len(train)
    keyword_idf = {
        keyword: math.log(
            1 + (total_train - frequency + 0.5) / (frequency + 0.5)
        )
        for keyword, frequency in keyword_document_frequency.items()
    }

    rows = []

    for case in test_cases:
        row = case["row"]
        hidden = case["hidden"]
        visible = case["visible"]
        pos = test_position[case["index"]]

        scores = bm25.get_scores(test_tokens[pos])

        neighbor_positions = sorted(
            range(len(scores)),
            key=lambda j: scores[j],
            reverse=True,
        )[:TOP_NEIGHBORS]

        candidate_scores = {}

        for neighbor_pos in neighbor_positions:
            similarity = float(scores[neighbor_pos])
            if similarity <= 0:
                continue

            for keyword in train_keyword_sets[neighbor_pos]:
                if keyword in visible:
                    continue

                rarity = keyword_idf.get(keyword, 0.0)
                combined_score = similarity * rarity

                candidate_scores[keyword] = (
                    candidate_scores.get(keyword, 0.0) + combined_score
                )

        recommendations = [
            keyword
            for keyword, _ in sorted(
                candidate_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )[:K]
        ]

        rows.append({
            "nct_id": row["nct_id"],
            "model": "improved_bm25",
            "hidden_keyword_count": len(hidden),
            "hidden_keywords": "; ".join(sorted(hidden)),
            "recommended_keywords": "; ".join(recommendations),
            "precision_at_5": precision_at_k(recommendations, hidden),
            "recall_at_5": recall_at_k(recommendations, hidden),
            "ndcg_at_5": ndcg_at_k(recommendations, hidden),
        })

    detail = pd.DataFrame(rows)
    summary = detail[
        ["precision_at_5", "recall_at_5", "ndcg_at_5"]
    ].mean()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)

    report_lines = [
        "IMPROVED BM25 KEYWORD RECOVERY EVALUATION",
        "=" * 44,
        f"Eligible trials: {len(eligible)}",
        f"Training trials: {len(train)}",
        f"Test trials: {len(test)}",
        f"Random seed: {SEED}",
        f"Hidden keyword fraction: {HIDE_FRACTION:.0%}",
        f"Evaluation cutoff: {K}",
        "Ranking: BM25 similarity multiplied by training-only keyword IDF.",
        "",
        summary.to_string(float_format=lambda value: f"{value:.4f}"),
        "",
        "Caveats:",
        "- BM25 and keyword IDF use training trials only.",
        "- Candidate keywords come only from training trials.",
        "- Test keywords are hidden from recommendation generation.",
        "- Metrics measure recovery of recorded keywords, not clinical validity.",
    ]

    SUMMARY_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nImproved BM25 evaluation completed!")
    print(f"Eligible trials: {len(eligible)}")
    print(f"Training trials: {len(train)}")
    print(f"Test trials: {len(test)}")
    print("\nImproved BM25 mean metrics:")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nDetailed results: {DETAIL_PATH}")
    print(f"Summary report: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
