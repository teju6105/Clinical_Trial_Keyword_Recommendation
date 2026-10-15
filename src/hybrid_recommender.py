from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "clinical_trials_cleaned.csv"
OUTPUT_PATH = ROOT / "results" / "hybrid_recommendations.csv"
REPORT_PATH = ROOT / "results" / "hybrid_report.txt"

TOP_NEIGHBORS = 10
TOP_RECOMMENDATIONS = 10
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


def prepare_text(df):
    return (
        df[TEXT_COLUMNS]
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH).fillna("")
    required = ["nct_id", "title", "original_keywords"] + TEXT_COLUMNS
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["keyword_set"] = df["original_keywords"].apply(parse_keywords)
    eligible = df[df["keyword_set"].map(len) >= 1].copy()
    eligible = eligible.reset_index(drop=True)

    # Exploratory split. The evaluation script will later handle a fixed,
    # shared train/test split for all models.
    rng = np.random.default_rng(42)
    order = rng.permutation(len(eligible))
    test_count = max(1, round(len(eligible) * 0.20))
    test_positions = sorted(order[:test_count].tolist())
    train_positions = sorted(order[test_count:].tolist())

    train = eligible.iloc[train_positions].reset_index(drop=True)
    test = eligible.iloc[test_positions].reset_index(drop=True)

    # Fit TF-IDF on training trial descriptions only.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=30000,
    )
    train_text = prepare_text(train)
    test_text = prepare_text(test)

    train_text_matrix = vectorizer.fit_transform(train_text)
    test_text_matrix = vectorizer.transform(test_text)
    content_similarity = cosine_similarity(
        test_text_matrix, train_text_matrix
    )

    # Build a binary interaction matrix using training labels only.
    vocabulary = sorted({
        keyword
        for keyword_set in train["keyword_set"]
        for keyword in keyword_set
    })
    keyword_index = {
        keyword: index for index, keyword in enumerate(vocabulary)
    }

    interactions = np.zeros(
        (len(train), len(vocabulary)),
        dtype=np.float32,
    )
    for i, keyword_set in enumerate(train["keyword_set"]):
        for keyword in keyword_set:
            interactions[i, keyword_index[keyword]] = 1.0

    rows = []

    for test_position, (_, row) in enumerate(test.iterrows()):
        target_keywords = row["keyword_set"]

        # Content-based candidate scores from the most similar training trials.
        content_scores = {}
        content_order = np.argsort(
            content_similarity[test_position]
        )[::-1][:TOP_NEIGHBORS]

        for neighbor_index in content_order:
            similarity = float(
                content_similarity[test_position, neighbor_index]
            )
            if similarity <= 0:
                continue

            for keyword in train.iloc[neighbor_index]["keyword_set"]:
                if keyword not in target_keywords:
                    content_scores[keyword] = (
                        content_scores.get(keyword, 0.0) + similarity
                    )

        # Collaborative scores use the visible interaction profile where
        # training-vocabulary keywords overlap with the target's recorded set.
        target_vector = np.zeros(
            (1, len(vocabulary)), dtype=np.float32
        )
        for keyword in target_keywords:
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
                similarity = float(interaction_similarity[neighbor_index])
                for keyword in train.iloc[neighbor_index]["keyword_set"]:
                    if keyword not in target_keywords:
                        collaborative_scores[keyword] = (
                            collaborative_scores.get(keyword, 0.0) + similarity
                        )

        # Normalize each signal so that its raw scale does not dominate.
        def normalize(scores):
            if not scores:
                return {}
            maximum = max(scores.values())
            if maximum <= 0:
                return {}
            return {key: value / maximum for key, value in scores.items()}

        content_scores = normalize(content_scores)
        collaborative_scores = normalize(collaborative_scores)

        candidate_keywords = set(content_scores) | set(collaborative_scores)
        hybrid_scores = {}

        for keyword in candidate_keywords:
            content_score = content_scores.get(keyword, 0.0)
            collaborative_score = collaborative_scores.get(keyword, 0.0)
            hybrid_scores[keyword] = (
                CONTENT_WEIGHT * content_score
                + COLLABORATIVE_WEIGHT * collaborative_score
            )

        recommendations = sorted(
            hybrid_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:TOP_RECOMMENDATIONS]

        rows.append({
            "nct_id": row["nct_id"],
            "title": row["title"],
            "content_candidate_count": len(content_scores),
            "collaborative_candidate_count": len(collaborative_scores),
            "recommended_keywords": "; ".join(
                keyword for keyword, _ in recommendations
            ),
            "recommendation_scores": "; ".join(
                f"{score:.4f}" for _, score in recommendations
            ),
        })

    output = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    report = [
        "HYBRID KEYWORD RECOMMENDATION REPORT",
        "=" * 43,
        f"Trials with recorded keywords: {len(eligible)}",
        f"Training trials: {len(train)}",
        f"Test trials: {len(test)}",
        f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}",
        f"Training keyword vocabulary size: {len(vocabulary)}",
        f"Content weight: {CONTENT_WEIGHT}",
        f"Collaborative weight: {COLLABORATIVE_WEIGHT}",
        f"Neighborhood size: {TOP_NEIGHBORS}",
        "",
        "Method:",
        "1. Find similar trials using TF-IDF text similarity.",
        "2. Find neighbors using binary keyword interactions.",
        "3. Normalize each candidate-score signal.",
        "4. Combine signals using fixed weights.",
        "",
        "Important limitations:",
        "This is an exploratory run, not a final evaluation.",
        "Weights must be selected using training/validation data.",
        "A fixed random split is not a substitute for shared hidden labels.",
        "Recommendation quality must be evaluated on a consistent test set.",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print("\nHybrid recommendation generation completed!")
    print(f"Eligible trials: {len(eligible)}")
    print(f"Training trials: {len(train)}")
    print(f"Test trials: {len(test)}")
    print(f"TF-IDF vocabulary: {len(vectorizer.vocabulary_)}")
    print(f"Training keyword vocabulary: {len(vocabulary)}")
    print(f"\nRecommendations saved to: {OUTPUT_PATH}")
    print(f"Report saved to: {REPORT_PATH}")

    print("\nFirst 5 hybrid recommendations:")
    for _, row in output.head(5).iterrows():
        print(f"\nTrial: {row['nct_id']} - {row['title']}")
        print(
            "Keywords:",
            row["recommended_keywords"] or "(no recommendations)",
        )


if __name__ == "__main__":
    main()
