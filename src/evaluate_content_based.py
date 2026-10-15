from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "clinical_trials_cleaned.csv"
RESULTS_DIR = ROOT / "results"
DETAIL_PATH = RESULTS_DIR / "content_based_evaluation.csv"
SUMMARY_PATH = RESULTS_DIR / "content_based_evaluation_report.txt"

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


def precision_at_k(recommendations, relevant):
    return len(set(recommendations[:K]) & relevant) / K


def recall_at_k(recommendations, relevant):
    return (
        len(set(recommendations[:K]) & relevant) / len(relevant)
        if relevant else float("nan")
    )


def ndcg_at_k(recommendations, relevant):
    dcg = sum(
        1 / math.log2(rank + 2)
        for rank, keyword in enumerate(recommendations[:K])
        if keyword in relevant
    )
    ideal_count = min(len(relevant), K)
    idcg = sum(1 / math.log2(rank + 2) for rank in range(ideal_count))
    return dcg / idcg if idcg else float("nan")


def main():
    df = pd.read_csv(DATA_PATH).fillna("")
    required = ["nct_id", "original_keywords"] + TEXT_COLUMNS
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["keyword_set"] = df["original_keywords"].apply(parse_keywords)
    eligible = df[df["keyword_set"].map(len) >= 2].copy()

    # One reproducible trial-level split.
    rng = np.random.default_rng(SEED)
    order = rng.permutation(eligible.index.to_numpy())
    test_count = max(1, round(len(order) * TEST_FRACTION))
    test_indices = set(order[:test_count])

    train = eligible.loc[~eligible.index.isin(test_indices)].copy()
    test = eligible.loc[eligible.index.isin(test_indices)].copy()

    # Create hidden labels once; both models use exactly the same labels.
    test_cases = []
    for index, row in test.iterrows():
        all_keywords = sorted(row["keyword_set"])
        hidden_count = max(1, round(len(all_keywords) * HIDE_FRACTION))
        hidden_count = min(hidden_count, len(all_keywords) - 1)
        hidden = set(
            rng.choice(all_keywords, size=hidden_count, replace=False).tolist()
        )
        visible = row["keyword_set"] - hidden
        test_cases.append({
            "index": index,
            "row": row,
            "hidden": hidden,
            "visible": visible,
        })

    # Frequency baseline: learn frequencies from training trials only.
    frequency = {}
    for keywords in train["keyword_set"]:
        for keyword in keywords:
            frequency[keyword] = frequency.get(keyword, 0) + 1

    global_rank = sorted(
        frequency,
        key=lambda keyword: (-frequency[keyword], keyword)
    )

    # Fit text representation on training descriptions only.
    train_text = (
        train[TEXT_COLUMNS].astype(str).agg(" ".join, axis=1)
        .str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    test_text = (
        test[TEXT_COLUMNS].astype(str).agg(" ".join, axis=1)
        .str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=30000,
    )
    train_matrix = vectorizer.fit_transform(train_text)
    test_matrix = vectorizer.transform(test_text)
    similarities = cosine_similarity(test_matrix, train_matrix)

    rows = []
    test_position = {index: pos for pos, index in enumerate(test.index)}

    for case in test_cases:
        row = case["row"]
        hidden = case["hidden"]
        visible = case["visible"]
        pos = test_position[case["index"]]

        # Model 1: frequency baseline, excluding already visible labels.
        baseline_recs = [
            keyword for keyword in global_rank if keyword not in visible
        ][:K]

        # Model 2: similarity-weighted keywords from training trials.
        sim_scores = similarities[pos]
        neighbor_positions = np.argsort(sim_scores)[::-1][:TOP_NEIGHBORS]
        candidate_scores = {}

        for neighbor_pos in neighbor_positions:
            score = float(sim_scores[neighbor_pos])
            if score <= 0:
                continue
            neighbor_keywords = train.iloc[neighbor_pos]["keyword_set"]
            for keyword in neighbor_keywords:
                if keyword not in visible:
                    candidate_scores[keyword] = (
                        candidate_scores.get(keyword, 0.0) + score
                    )

        content_recs = [
            keyword for keyword, score in sorted(
                candidate_scores.items(),
                key=lambda item: (-item[1], item[0])
            )
        ][:K]

        for model_name, recommendations in [
            ("frequency_baseline", baseline_recs),
            ("content_based", content_recs),
        ]:
            rows.append({
                "nct_id": row["nct_id"],
                "model": model_name,
                "hidden_keyword_count": len(hidden),
                "hidden_keywords": "; ".join(sorted(hidden)),
                "recommended_keywords": "; ".join(recommendations),
                "precision_at_5": precision_at_k(recommendations, hidden),
                "recall_at_5": recall_at_k(recommendations, hidden),
                "ndcg_at_5": ndcg_at_k(recommendations, hidden),
            })

    detail = pd.DataFrame(rows)
    summary = detail.groupby("model")[
        ["precision_at_5", "recall_at_5", "ndcg_at_5"]
    ].mean()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)

    report_lines = [
        "COMPARABLE KEYWORD RECOVERY EVALUATION",
        "=" * 45,
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
        "- Both models use the same test trials and hidden keywords.",
        "- Text vectorizer is fitted on training descriptions only.",
        "- Missing recorded keywords are not treated as negative labels.",
        "- Metrics measure recovery of recorded keywords, not clinical validity.",
        "- Results depend on this initial dataset and random split.",
    ]
    SUMMARY_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nComparable evaluation completed!")
    print(f"Eligible trials: {len(eligible)}")
    print(f"Training trials: {len(train)}")
    print(f"Test trials: {len(test)}")
    print("\nMean metrics:")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nDetailed results: {DETAIL_PATH}")
    print(f"Summary report: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
