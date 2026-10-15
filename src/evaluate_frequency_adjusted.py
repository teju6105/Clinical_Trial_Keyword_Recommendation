from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "clinical_trials_cleaned.csv"
RESULTS_DIR = ROOT / "results"

DETAIL_PATH = RESULTS_DIR / "frequency_adjusted_evaluation.csv"
REPORT_PATH = RESULTS_DIR / "frequency_adjusted_evaluation_report.txt"

SEED = 42
TEST_FRACTION = 0.20
HIDE_FRACTION = 0.30
K = 5
TOP_NEIGHBORS = 10
MIN_SIMILARITY = 0.05

TITLE_WEIGHT = 0.40
CONDITIONS_WEIGHT = 0.35
SUMMARY_WEIGHT = 0.25


def parse_keywords(value):
    if pd.isna(value) or not str(value).strip():
        return set()
    return {
        " ".join(part.lower().split())
        for part in str(value).split(";")
        if part.strip()
    }


def normalize_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def make_matrix(train_text, test_text):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=20000,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(train_text)
    test_matrix = vectorizer.transform(test_text)
    return train_matrix, test_matrix


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


def rank_recommendations(scores):
    return [
        keyword
        for keyword, _ in sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:K]
    ]


def main():
    df = pd.read_csv(DATA_PATH).fillna("")

    required = [
        "nct_id", "original_keywords", "title",
        "official_title", "conditions", "brief_summary",
        "interventions",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["keyword_set"] = df["original_keywords"].apply(parse_keywords)

    # Reproduce the eligible records and split used by the previous evaluation.
    eligible = df[df["keyword_set"].map(len) >= 2].copy()
    eligible = eligible.reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(eligible))
    test_count = max(1, round(len(order) * TEST_FRACTION))

    test_positions = sorted(order[:test_count].tolist())
    train_positions = sorted(order[test_count:].tolist())

    train = eligible.iloc[train_positions].reset_index(drop=True)
    test = eligible.iloc[test_positions].reset_index(drop=True)

    # Hide the same keywords using the same random procedure as the
    # existing evaluation scripts.
    cases = []

    for _, row in test.iterrows():
        all_keywords = sorted(row["keyword_set"])
        hidden_count = max(1, round(len(all_keywords) * HIDE_FRACTION))
        hidden_count = min(hidden_count, len(all_keywords) - 1)

        hidden = set(
            rng.choice(
                all_keywords,
                size=hidden_count,
                replace=False,
            ).tolist()
        )

        cases.append({
            "row": row,
            "hidden": hidden,
            "visible": row["keyword_set"] - hidden,
        })

    # Fit all three field vectorizers on training trials only.
    title_train, title_test = make_matrix(
        normalize_text(train["title"]),
        normalize_text(test["title"]),
    )

    conditions_train, conditions_test = make_matrix(
        normalize_text(train["conditions"]),
        normalize_text(test["conditions"]),
    )

    summary_train, summary_test = make_matrix(
        normalize_text(train["brief_summary"]),
        normalize_text(test["brief_summary"]),
    )

    title_sim = cosine_similarity(title_test, title_train)
    conditions_sim = cosine_similarity(conditions_test, conditions_train)
    summary_sim = cosine_similarity(summary_test, summary_train)

    similarities = (
        TITLE_WEIGHT * title_sim
        + CONDITIONS_WEIGHT * conditions_sim
        + SUMMARY_WEIGHT * summary_sim
    )

    # Calculate keyword document frequency from TRAINING trials only.
    keyword_frequency = {}

    for keyword_set in train["keyword_set"]:
        for keyword in keyword_set:
            keyword_frequency[keyword] = (
                keyword_frequency.get(keyword, 0) + 1
            )

    total_train = len(train)

    keyword_idf = {
        keyword: math.log(
            (total_train + 1) / (frequency + 1)
        ) + 1
        for keyword, frequency in keyword_frequency.items()
    }

    results = []

    for test_position, case in enumerate(cases):
        row = case["row"]
        hidden = case["hidden"]
        visible = case["visible"]

        neighbor_order = np.argsort(
            similarities[test_position]
        )[::-1]

        neighbor_indices = [
            int(j)
            for j in neighbor_order
            if similarities[test_position, j] >= MIN_SIMILARITY
        ][:TOP_NEIGHBORS]

        scores = {}

        for neighbor_index in neighbor_indices:
            similarity = float(
                similarities[test_position, neighbor_index]
            )

            if similarity <= 0:
                continue

            for keyword in train.iloc[neighbor_index]["keyword_set"]:
                if keyword in visible:
                    continue

                # Only keywords observed in training receive scores.
                if keyword not in keyword_idf:
                    continue

                adjusted_score = similarity * keyword_idf[keyword]
                scores[keyword] = scores.get(keyword, 0.0) + adjusted_score

        recommendations = rank_recommendations(scores)

        results.append({
            "nct_id": row["nct_id"],
            "hidden_keywords": "; ".join(sorted(hidden)),
            "recommended_keywords": "; ".join(recommendations),
            "neighbor_count": len(neighbor_indices),
            "precision_at_5": precision_at_k(recommendations, hidden),
            "recall_at_5": recall_at_k(recommendations, hidden),
            "ndcg_at_5": ndcg_at_k(recommendations, hidden),
        })

    detail = pd.DataFrame(results)
    metrics = ["precision_at_5", "recall_at_5", "ndcg_at_5"]
    summary = detail[metrics].mean()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)

    report = [
        "FREQUENCY-ADJUSTED CONTENT-BASED EVALUATION",
        "=" * 47,
        f"Eligible trials: {len(eligible)}",
        f"Training trials: {len(train)}",
        f"Test trials: {len(test)}",
        f"Random seed: {SEED}",
        f"Training keyword vocabulary: {len(keyword_idf)}",
        f"Title weight: {TITLE_WEIGHT}",
        f"Conditions weight: {CONDITIONS_WEIGHT}",
        f"Summary weight: {SUMMARY_WEIGHT}",
        f"Minimum similarity: {MIN_SIMILARITY}",
        f"Maximum neighbors: {TOP_NEIGHBORS}",
        "",
        summary.to_string(float_format=lambda value: f"{value:.4f}"),
        "",
        "Limitations:",
        "- Text vectorizers are fitted on training trials only.",
        "- Keyword-frequency weights use training labels only.",
        "- Metrics evaluate recovery of recorded keywords, not clinical validity.",
        "- Field weights and similarity threshold remain exploratory.",
    ]

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print("\nFrequency-adjusted evaluation completed!")
    print(f"Eligible trials: {len(eligible)}")
    print(f"Training trials: {len(train)}")
    print(f"Test trials: {len(test)}")
    print("\nMean metrics:")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nDetailed results: {DETAIL_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
