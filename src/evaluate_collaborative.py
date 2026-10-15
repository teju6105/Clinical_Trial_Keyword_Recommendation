from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "clinical_trials_cleaned.csv"
RESULTS_DIR = ROOT / "results"
DETAIL_PATH = RESULTS_DIR / "collaborative_evaluation.csv"
REPORT_PATH = RESULTS_DIR / "collaborative_evaluation_report.txt"

SEED = 42
TEST_FRACTION = 0.20
HIDE_FRACTION = 0.30
K = 5
TOP_NEIGHBORS = 10


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
    required = ["nct_id", "original_keywords"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["keyword_set"] = df["original_keywords"].apply(parse_keywords)

    # Use the same eligible set and deterministic split as the earlier
    # evaluation scripts: trials with at least two recorded keywords.
    eligible = df[df["keyword_set"].map(len) >= 2].copy()
    eligible = eligible.reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(eligible))
    test_count = max(1, round(len(order) * TEST_FRACTION))
    test_positions = sorted(order[:test_count].tolist())
    train_positions = sorted(order[test_count:].tolist())

    train = eligible.iloc[train_positions].reset_index(drop=True)
    test = eligible.iloc[test_positions].reset_index(drop=True)

    # Build test hidden labels once using the same seed and procedure as
    # evaluate_content_based.py.
    test_cases = []
    for _, row in test.iterrows():
        all_keywords = sorted(row["keyword_set"])
        hidden_count = max(1, round(len(all_keywords) * HIDE_FRACTION))
        hidden_count = min(hidden_count, len(all_keywords) - 1)

        hidden = set(
            rng.choice(all_keywords, size=hidden_count, replace=False).tolist()
        )
        visible = row["keyword_set"] - hidden

        test_cases.append({
            "row": row,
            "hidden": hidden,
            "visible": visible,
        })

    # Training-only keyword vocabulary and interaction matrix.
    vocabulary = sorted({
        keyword
        for keyword_set in train["keyword_set"]
        for keyword in keyword_set
    })
    keyword_index = {keyword: index for index, keyword in enumerate(vocabulary)}

    interactions = np.zeros((len(train), len(vocabulary)), dtype=np.float32)
    for i, keyword_set in enumerate(train["keyword_set"]):
        for keyword in keyword_set:
            interactions[i, keyword_index[keyword]] = 1.0

    rows = []

    for case in test_cases:
        row = case["row"]
        hidden = case["hidden"]
        visible = case["visible"]

        target_vector = np.zeros((1, len(vocabulary)), dtype=np.float32)
        for keyword in visible:
            if keyword in keyword_index:
                target_vector[0, keyword_index[keyword]] = 1.0

        similarities = cosine_similarity(target_vector, interactions).ravel()

        neighbor_indices = [
            index for index in np.argsort(similarities)[::-1]
            if similarities[index] > 0
        ][:TOP_NEIGHBORS]

        scores = {}
        for neighbor_index in neighbor_indices:
            similarity = float(similarities[neighbor_index])
            for keyword in train.iloc[neighbor_index]["keyword_set"]:
                if keyword in visible:
                    continue
                scores[keyword] = scores.get(keyword, 0.0) + similarity

        recommendations = [
            keyword for keyword, _ in sorted(
                scores.items(), key=lambda item: (-item[1], item[0])
            )[:K]
        ]

        rows.append({
            "nct_id": row["nct_id"],
            "hidden_keyword_count": len(hidden),
            "hidden_keywords": "; ".join(sorted(hidden)),
            "neighbor_count": len(neighbor_indices),
            "recommended_keywords": "; ".join(recommendations),
            "precision_at_5": precision_at_k(recommendations, hidden),
            "recall_at_5": recall_at_k(recommendations, hidden),
            "ndcg_at_5": ndcg_at_k(recommendations, hidden),
        })

    detail = pd.DataFrame(rows)
    metrics = ["precision_at_5", "recall_at_5", "ndcg_at_5"]
    summary = detail[metrics].mean()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)

    no_neighbor_count = int((detail["neighbor_count"] == 0).sum())
    report = [
        "COLLABORATIVE FILTERING EVALUATION",
        "=" * 42,
        f"Eligible trials: {len(eligible)}",
        f"Training trials: {len(train)}",
        f"Test trials: {len(test)}",
        f"Training vocabulary: {len(vocabulary)}",
        f"Training interaction matrix: {interactions.shape}",
        f"Test trials with no positive-similarity neighbors: {no_neighbor_count}",
        f"Neighborhood size: {TOP_NEIGHBORS}",
        f"Evaluation cutoff: {K}",
        "",
        "Mean metrics:",
        f"Precision@{K}: {summary['precision_at_5']:.4f}",
        f"Recall@{K}: {summary['recall_at_5']:.4f}",
        f"NDCG@{K}: {summary['ndcg_at_5']:.4f}",
        "",
        "Limitations:",
        "- Metrics measure recovery of hidden recorded keywords.",
        "- Missing recorded keywords are not treated as negative labels.",
        "- Sparse keyword interactions can leave trials without neighbors.",
        "- Results apply to this dataset and split, not clinical validity.",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print("\nCollaborative-filtering evaluation completed!")
    print(f"Eligible trials: {len(eligible)}")
    print(f"Training trials: {len(train)}")
    print(f"Test trials: {len(test)}")
    print(f"Training vocabulary: {len(vocabulary)}")
    print(f"Test trials with no neighbors: {no_neighbor_count}")
    print(f"Precision@{K}: {summary['precision_at_5']:.4f}")
    print(f"Recall@{K}: {summary['recall_at_5']:.4f}")
    print(f"NDCG@{K}: {summary['ndcg_at_5']:.4f}")
    print(f"\nDetailed results: {DETAIL_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
