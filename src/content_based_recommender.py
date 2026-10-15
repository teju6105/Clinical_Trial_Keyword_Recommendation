from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clinical_trials_cleaned.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "content_based_recommendations.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "content_based_report.txt"

TOP_SIMILAR_TRIALS = 10
TOP_RECOMMENDATIONS = 10


def parse_keywords(value):
    if pd.isna(value) or not str(value).strip():
        return set()

    return {
        " ".join(part.lower().split())
        for part in str(value).split(";")
        if part.strip()
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH).fillna("")

    required_columns = [
        "nct_id",
        "title",
        "official_title",
        "conditions",
        "brief_summary",
        "interventions",
        "original_keywords",
    ]

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Build content using descriptive fields only.
    text_columns = [
        "title",
        "official_title",
        "conditions",
        "brief_summary",
        "interventions",
    ]

    df["content_text"] = (
        df[text_columns]
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df["keyword_set"] = df["original_keywords"].apply(parse_keywords)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=30000,
    )

    content_matrix = vectorizer.fit_transform(df["content_text"])
    similarity_matrix = cosine_similarity(content_matrix, dense_output=True)

    rows = []

    for i, row in df.iterrows():
        similarities = similarity_matrix[i].copy()
        similarities[i] = -1

        similar_indices = similarities.argsort()[::-1][:TOP_SIMILAR_TRIALS]

        candidate_scores = {}

        for j in similar_indices:
            similarity = float(similarities[j])

            # Only trials with recorded keywords can contribute candidates.
            for keyword in df.iloc[j]["keyword_set"]:
                candidate_scores[keyword] = (
                    candidate_scores.get(keyword, 0.0) + similarity
                )

        # Do not recommend a keyword already recorded for the target trial.
        existing_keywords = row["keyword_set"]

        ranked_candidates = sorted(
            (
                (keyword, score)
                for keyword, score in candidate_scores.items()
                if keyword not in existing_keywords
            ),
            key=lambda item: (-item[1], item[0]),
        )

        recommendations = ranked_candidates[:TOP_RECOMMENDATIONS]

        rows.append({
            "nct_id": row["nct_id"],
            "title": row["title"],
            "similar_trial_ids": "; ".join(
                str(df.iloc[j]["nct_id"]) for j in similar_indices
            ),
            "similarity_scores": "; ".join(
                f"{similarities[j]:.4f}" for j in similar_indices
            ),
            "recommended_keywords": "; ".join(
                keyword for keyword, _ in recommendations
            ),
            "recommendation_scores": "; ".join(
                f"{score:.4f}" for _, score in recommendations
            ),
            "recorded_keyword_count": len(existing_keywords),
        })

    output = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    report_lines = [
        "CONTENT-BASED KEYWORD RECOMMENDATION REPORT",
        "=" * 48,
        f"Trials processed: {len(df)}",
        f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}",
        f"Similar trials considered per target: {TOP_SIMILAR_TRIALS}",
        f"Maximum recommendations per trial: {TOP_RECOMMENDATIONS}",
        "",
        "Method:",
        "1. Represent trial descriptions with TF-IDF.",
        "2. Calculate cosine similarity between trial descriptions.",
        "3. Collect keywords from the most similar trials.",
        "4. Rank candidate keywords by summed similarity.",
        "5. Exclude keywords already recorded for the target trial.",
        "",
        "Important limitation:",
        "This is an exploratory recommendation output, not an evaluation.",
        "For valid evaluation, keyword labels from held-out trials must",
        "not contribute to recommendations, and preprocessing/model fitting",
        "must be performed using training data only.",
    ]

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nContent-based recommendation generation completed!")
    print(f"Trials processed: {len(df)}")
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Similar trials per target: {TOP_SIMILAR_TRIALS}")
    print(f"\nRecommendations saved to: {OUTPUT_PATH}")
    print(f"Report saved to: {REPORT_PATH}")

    print("\nExample recommendations:")
    for _, example in output.head(5).iterrows():
        print(f"\nTrial: {example['nct_id']} - {example['title']}")
        print(
            "Keywords:",
            example["recommended_keywords"]
            if example["recommended_keywords"]
            else "(no candidates found)",
        )


if __name__ == "__main__":
    main()
