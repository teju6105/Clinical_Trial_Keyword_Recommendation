from pathlib import Path
import math
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clinical_trials_cleaned.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "baseline_evaluation.csv"
REPORT_PATH = RESULTS_DIR / "baseline_evaluation_report.txt"

RANDOM_SEED = 42
TEST_FRACTION = 0.20
HIDE_FRACTION = 0.30
K = 5


def parse_keywords(value):
    if pd.isna(value) or not str(value).strip():
        return set()
    return {
        " ".join(part.lower().split())
        for part in str(value).split(";")
        if part.strip()
    }


def precision_at_k(recommended, relevant, k):
    if k == 0:
        return 0.0
    return len(set(recommended[:k]) & relevant) / k


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return float("nan")
    return len(set(recommended[:k]) & relevant) / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, keyword in enumerate(recommended[:k])
        if keyword in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(
        1.0 / math.log2(rank + 2)
        for rank in range(ideal_hits)
    )
    return dcg / idcg if idcg else float("nan")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH).fillna("")
    if "original_keywords" not in df.columns:
        raise ValueError("The original_keywords column is missing.")

    df["keyword_set"] = df["original_keywords"].apply(parse_keywords)
    eligible = df[df["keyword_set"].map(len) >= 2].copy()

    if len(eligible) < 10:
        raise ValueError(
            "Not enough trials with at least two keywords for evaluation."
        )

    # Reproducible trial-level holdout split.
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled_indices = rng.permutation(eligible.index.to_numpy())
    test_size = max(1, int(round(len(shuffled_indices) * TEST_FRACTION)))
    test_indices = set(shuffled_indices[:test_size])
    train = eligible.loc[~eligible.index.isin(test_indices)].copy()
    test = eligible.loc[eligible.index.isin(test_indices)].copy()

    # Fit the frequency baseline on training trials only.
    frequency = {}
    for keyword_set in train["keyword_set"]:
        for keyword in keyword_set:
            frequency[keyword] = frequency.get(keyword, 0) + 1

    ranked_keywords = sorted(
        frequency,
        key=lambda keyword: (-frequency[keyword], keyword)
    )

    rows = []
    for _, row in test.iterrows():
        known_keywords = row["keyword_set"]
        hidden_count = max(1, int(round(len(known_keywords) * HIDE_FRACTION)))
        hidden_count = min(hidden_count, len(known_keywords) - 1)

        hidden = set(rng.choice(sorted(known_keywords), size=hidden_count, replace=False))
        visible = known_keywords - hidden

        # Do not recommend keywords already visible for this trial.
        recommendations = [
            keyword for keyword in ranked_keywords
            if keyword not in visible
        ][:K]

        # This is a held-out keyword recovery test. Some hidden keywords may
        # be absent from training; the frequency baseline cannot recommend them.
        rows.append({
            "trial_id": row.get("nct_id", row.name),
            "visible_keyword_count": len(visible),
            "hidden_keyword_count": len(hidden),
            "precision_at_5": precision_at_k(recommendations, hidden, K),
            "recall_at_5": recall_at_k(recommendations, hidden, K),
            "ndcg_at_5": ndcg_at_k(recommendations, hidden, K),
            "hidden_keywords": "; ".join(sorted(hidden)),
            "recommended_keywords": "; ".join(recommendations),
        })

    result = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    # Report metrics overall and separately for trials with at least one
    # hidden keyword known to the training vocabulary.
    metrics = ["precision_at_5", "recall_at_5", "ndcg_at_5"]
    overall = result[metrics].mean()

    report = [
        "BASELINE KEYWORD RECOVERY EVALUATION",
        "=" * 45,
        f"Eligible trials (at least 2 recorded keywords): {len(eligible)}",
        f"Training trials: {len(train)}",
        f"Test trials: {len(test)}",
        f"Random seed: {RANDOM_SEED}",
        f"Test fraction: {TEST_FRACTION:.0%}",
        f"Hidden keyword fraction: {HIDE_FRACTION:.0%}",
        f"Evaluation cutoff K: {K}",
        "",
        "Mean test metrics:",
        f"Precision@{K}: {overall['precision_at_5']:.4f}",
        f"Recall@{K}: {overall['recall_at_5']:.4f}",
        f"NDCG@{K}: {overall['ndcg_at_5']:.4f}",
        "",
        "Important limitations:",
        "- The baseline ranks keywords by frequency in training trials.",
        "- Missing recorded keywords are not treated as negative labels.",
        "- This test measures recovery of hidden recorded keywords, not",
        "  clinical correctness or real-world recommendation quality.",
        "- This is a baseline, not evidence that a model is clinically useful.",
    ]

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print("\nBaseline evaluation completed!")
    print(f"Eligible trials: {len(eligible)}")
    print(f"Training trials: {len(train)}")
    print(f"Test trials: {len(test)}")
    print(f"Precision@{K}: {overall['precision_at_5']:.4f}")
    print(f"Recall@{K}: {overall['recall_at_5']:.4f}")
    print(f"NDCG@{K}: {overall['ndcg_at_5']:.4f}")
    print(f"\nDetailed results: {OUTPUT_PATH}")
    print(f"Evaluation report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
