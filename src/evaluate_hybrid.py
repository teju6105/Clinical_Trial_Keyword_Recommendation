from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "clinical_trials_cleaned.csv"
RESULTS_DIR = ROOT / "results"
DETAIL_PATH = RESULTS_DIR / "hybrid_evaluation.csv"
SUMMARY_PATH = RESULTS_DIR / "hybrid_evaluation_report.txt"

SEED = 42
TEST_FRACTION = 0.20
HIDE_FRACTION = 0.30
K = 5
TOP_NEIGHBORS = 10
CONTENT_WEIGHT = 0.7
COLLABORATIVE_WEIGHT = 0.3

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


def normalize(scores):
    if not scores:
        return {}
    maximum = max(scores.values())
    if maximum <= 0:
        return {}
    return {key: value / maximum for key, value in scores.items()}


def main():
    df = pd.read_csv(DATA_PATH).fillna("")
    required = ["nct_id", "original_keywords"] + TEXT_COLUMNS
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["keyword_set"] = df["original_keywords"].apply(parse_keywords)
    eligible = df[df["keyword_set"].map(len) >= 2].copy()
    eligible = eligible.reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(eligible))
    test_count = max(1, round(len(order) * TEST_FRACTION))
    test_positions = sorted(order[:test_count].tolist())
    train_positions = sorted(order[test_count:].tolist())

    train = eligible.iloc[train_positions].reset_index(drop=True)
    test = eligible.iloc[test_positions].reset_index(drop=True)

    # Hide labels once; every model receives identical visible/hidden labels.
    cases = []
    for _, row in test.iterrows():
        all_keywords = sorted(row["keyword_set"])
        hidden_count = max(1, round(len(all_keywords) * HIDE_FRACTION))
        hidden_count = min(hidden_count, len(all_keywords) - 1)
        hidden = set(
            rng.choice(all_keywords, size=hidden_count, replace=False).tolist()
        )
        cases.append({
            "row": row,
            "hidden": hidden,
            "visible": row["keyword_set"] - hidden,
        })

    # Training-only vocabulary and binary interaction matrix.
    vocabulary = sorted({
        keyword
        for keyword_set in train["keyword_set"]
        for keyword in keyword_set
    })
    keyword_index = {keyword: i for i, keyword in enumerate(vocabulary)}
    interactions = np.zeros((len(train), len(vocabulary)), dtype=np.float32)

    for i, keyword_set in enumerate(train["keyword_set"]):
        for keyword in keyword_set:
            interactions[i, keyword_index[keyword]] = 1.0

    # Frequency baseline, trained on training labels only.
    frequency = {}
    for keyword_set in train["keyword_set"]:
        for keyword in keyword_set:
            frequency[keyword] = frequency.get(keyword, 0) + 1
    global_rank = sorted(
        frequency, key=lambda keyword: (-frequency[keyword], keyword)
    )

    # Content model fitted on training descriptions only.
    def text_for(frame):
        return (
            frame[TEXT_COLUMNS].astype(str).agg(" ".join, axis=1)
            .str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=30000,
    )
    train_text_matrix = vectorizer.fit_transform(text_for(train))
    test_text_matrix = vectorizer.transform(text_for(test))
    content_similarity = cosine_similarity(test_text_matrix, train_text_matrix)

    result_rows = []

    for test_position, case in enumerate(cases):
        row = case["row"]
        hidden = case["hidden"]
        visible = case["visible"]

        # Frequency baseline.
        baseline_recs = [
            keyword for keyword in global_rank if keyword not in visible
        ][:K]

        # Content-based scores from similar training descriptions.
        content_scores = {}
        content_neighbors = np.argsort(
            content_similarity[test_position]
        )[::-1][:TOP_NEIGHBORS]

        for neighbor_index in content_neighbors:
            sim = float(content_similarity[test_position, neighbor_index])
            if sim <= 0:
                continue
            for keyword in train.iloc[neighbor_index]["keyword_set"]:
                if keyword not in visible:
                    content_scores[keyword] = (
                        content_scores.get(keyword, 0.0) + sim
                    )

        # Collaborative scores from visible keywords.
        target_vector = np.zeros((1, len(vocabulary)), dtype=np.float32)
        for keyword in visible:
            if keyword in keyword_index:
                target_vector[0, keyword_index[keyword]] = 1.0

        collaborative_scores = {}
        if target_vector.sum() > 0:
            interaction_similarity = cosine_similarity(
                target_vector, interactions
            ).ravel()
            neighbor_indices = [
                index for index in np.argsort(interaction_similarity)[::-1]
                if interaction_similarity[index] > 0
            ][:TOP_NEIGHBORS]

            for neighbor_index in neighbor_indices:
                sim = float(interaction_similarity[neighbor_index])
                for keyword in train.iloc[neighbor_index]["keyword_set"]:
                    if keyword not in visible:
                        collaborative_scores[keyword] = (
                            collaborative_scores.get(keyword, 0.0) + sim
                        )

        # Normalize component scores before combining.
        content_scores = normalize(content_scores)
        collaborative_scores = normalize(collaborative_scores)

        candidates = set(content_scores) | set(collaborative_scores)
        hybrid_scores = {
            keyword: (
                CONTENT_WEIGHT * content_scores.get(keyword, 0.0)
                + COLLABORATIVE_WEIGHT
                * collaborative_scores.get(keyword, 0.0)
            )
            for keyword in candidates
        }

        hybrid_recs = [
            keyword for keyword, _ in sorted(
                hybrid_scores.items(),
                key=lambda item: (-item[1], item[0])
            )[:K]
        ]

        for model_name, recommendations in [
            ("frequency_baseline", baseline_recs),
            ("content_based", [
                keyword for keyword, _ in sorted(
                    content_scores.items(),
                    key=lambda item: (-item[1], item[0])
                )[:K]
            ]),
            ("collaborative_filtering", [
                keyword for keyword, _ in sorted(
                    collaborative_scores.items(),
                    key=lambda item: (-item[1], item[0])
                )[:K]
            ]),
            ("hybrid", hybrid_recs),
        ]:
            result_rows.append({
                "nct_id": row["nct_id"],
                "model": model_name,
                "hidden_keywords": "; ".join(sorted(hidden)),
                "recommended_keywords": "; ".join(recommendations),
                "precision_at_5": precision_at_k(recommendations, hidden),
                "recall_at_5": recall_at_k(recommendations, hidden),
                "ndcg_at_5": ndcg_at_k(recommendations, hidden),
            })

    detail = pd.DataFrame(result_rows)
    metrics = ["precision_at_5", "recall_at_5", "ndcg_at_5"]
    summary = detail.groupby("model")[metrics].mean()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)

    report_lines = [
        "FOUR-MODEL COMPARISON",
        "=" * 35,
        f"Eligible trials: {len(eligible)}",
        f"Training trials: {len(train)}",
        f"Test trials: {len(test)}",
        f"Training keyword vocabulary: {len(vocabulary)}",
        f"TF-IDF vocabulary: {len(vectorizer.vocabulary_)}",
        f"Seed: {SEED}",
        f"Hybrid weights: content={CONTENT_WEIGHT}, collaborative={COLLABORATIVE_WEIGHT}",
        "",
        summary.to_string(float_format=lambda value: f"{value:.4f}"),
        "",
        "Interpretation cautions:",
        "- All models use the same test cases and hidden keywords in this run.",
        "- All models are fitted from training data only.",
        "- Missing recorded keywords are not assumed to be negative labels.",
        "- Metrics measure recovery of recorded keywords, not clinical validity.",
        "- Hybrid weights are fixed exploratory values, not validation-tuned.",
    ]
    SUMMARY_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nFour-model evaluation completed!")
    print(f"Eligible trials: {len(eligible)}")
    print(f"Training trials: {len(train)}")
    print(f"Test trials: {len(test)}")
    print("\nMean metrics:")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nDetailed results: {DETAIL_PATH}")
    print(f"Report: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
