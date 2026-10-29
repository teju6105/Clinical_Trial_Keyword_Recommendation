from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "clinical_trials_cleaned.csv"
RESULTS_DIR = ROOT / "results"
DETAIL_PATH = RESULTS_DIR / "bm25_evaluation.csv"
SUMMARY_PATH = RESULTS_DIR / "bm25_evaluation_report.txt"

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

    # Match the trial-level split and hidden-label procedure used by
    # evaluate_content_based.py.
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

    # Fit BM25 on training descriptions only.
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
            score = float(scores[neighbor_pos])
            if score <= 0:
                continue

            neighbor_keywords = train.iloc[neighbor_pos]["keyword_set"]
            for keyword in neighbor_keywords:
                if keyword not in visible:
                    candidate_scores[keyword] = (
                        candidate_scores.get(keyword, 0.0) + score
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
            "model": "bm25",
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
    ].mean().to_frame().T
    summary.index = ["bm25"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)

    report_lines = [
        "BM25 KEYWORD RECOVERY EVALUATION",
        "=" * 38,
        f"Eligible trials: {len(eligible)}",
        f"Training trials: {len(train)}",
        f"Test trials: {len(test)}",
        f"Random seed: {SEED}",
        f"Hidden keyword fraction: {HIDE_FRACTION:.0%}",
        f"Evaluation cutoff: {K}",
        "",
        summary.to_string(float_format=lambda value: f"{value:.4f}"),
        "",
        "Caveats:",
        "- BM25 is fitted on training descriptions only.",
        "- Candidate keywords come only from training trials.",
        "- Test keywords are hidden from recommendation generation.",
        "- Missing recorded keywords are not treated as negative labels.",
        "- Metrics measure recovery of recorded keywords, not clinical validity.",
        "- Compare with the content-based evaluation using the same split.",
    ]
    SUMMARY_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nBM25 evaluation completed!")
    print(f"Eligible trials: {len(eligible)}")
    print(f"Training trials: {len(train)}")
    print(f"Test trials: {len(test)}")
    print("\nBM25 mean metrics:")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nDetailed results: {DETAIL_PATH}")
    print(f"Summary report: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
