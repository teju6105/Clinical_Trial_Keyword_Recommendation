from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "clinical_trials_cleaned.csv"
OUTPUT_PATH = ROOT / "results" / "improved_content_recommendations.csv"
REPORT_PATH = ROOT / "results" / "improved_content_report.txt"

TOP_NEIGHBORS = 10
TOP_RECOMMENDATIONS = 10

# Similarity weights for different text fields.
TITLE_WEIGHT = 0.40
CONDITIONS_WEIGHT = 0.35
SUMMARY_WEIGHT = 0.25

# Exploratory threshold; must be tuned using validation data.
MIN_SIMILARITY = 0.05

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

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH).fillna("")

    required = [
        "nct_id", "title", "conditions",
        "brief_summary", "original_keywords"
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["keyword_set"] = df["original_keywords"].apply(parse_keywords)

    # Create separate text representations for different fields.
    title_text = normalize_text(df["title"])
    conditions_text = normalize_text(df["conditions"])
    summary_text = normalize_text(df["brief_summary"])

    # Fit each vectorizer on the dataset for this exploratory run.
    # Use the separate evaluation script for training-only evaluation.
    def make_matrix(text):
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=20000,
            sublinear_tf=True,
        )
        return vectorizer.fit_transform(text)

    title_matrix = make_matrix(title_text)
    conditions_matrix = make_matrix(conditions_text)
    summary_matrix = make_matrix(summary_text)

    title_sim = cosine_similarity(title_matrix)
    conditions_sim = cosine_similarity(conditions_matrix)
    summary_sim = cosine_similarity(summary_matrix)

    combined_sim = (
        TITLE_WEIGHT * title_sim
        + CONDITIONS_WEIGHT * conditions_sim
        + SUMMARY_WEIGHT * summary_sim
    )

    rows = []

    for i, row in df.iterrows():
        similarities = combined_sim[i].copy()
        similarities[i] = -1.0

        # Consider only neighbors that pass the exploratory threshold.
        candidate_indices = [
            j for j in np.argsort(similarities)[::-1]
            if similarities[j] >= MIN_SIMILARITY
        ][:TOP_NEIGHBORS]

        candidate_scores = {}

        for j in candidate_indices:
            similarity = float(similarities[j])

            for keyword in df.iloc[j]["keyword_set"]:
                if keyword in row["keyword_set"]:
                    continue

                candidate_scores[keyword] = (
                    candidate_scores.get(keyword, 0.0) + similarity
                )

        recommendations = sorted(
            candidate_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:TOP_RECOMMENDATIONS]

        rows.append({
            "nct_id": row["nct_id"],
            "title": row["title"],
            "similar_trial_ids": "; ".join(
                str(df.iloc[j]["nct_id"]) for j in candidate_indices
            ),
            "similarity_scores": "; ".join(
                f"{similarities[j]:.4f}" for j in candidate_indices
            ),
            "similarity_neighbor_count": len(candidate_indices),
            "recommended_keywords": "; ".join(
                keyword for keyword, _ in recommendations
            ),
            "recommendation_scores": "; ".join(
                f"{score:.4f}" for _, score in recommendations
            ),
            "recorded_keyword_count": len(row["keyword_set"]),
        })

    output = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    report = [
        "IMPROVED CONTENT-BASED RECOMMENDATION REPORT",
        "=" * 48,
        f"Trials processed: {len(df)}",
        f"Title weight: {TITLE_WEIGHT}",
        f"Conditions weight: {CONDITIONS_WEIGHT}",
        f"Summary weight: {SUMMARY_WEIGHT}",
        f"Minimum similarity threshold: {MIN_SIMILARITY}",
        f"Maximum neighbors: {TOP_NEIGHBORS}",
        f"Maximum recommendations: {TOP_RECOMMENDATIONS}",
        "",
        "Method:",
        "1. Represent title, conditions, and summary separately using TF-IDF.",
        "2. Calculate cosine similarity for each field.",
        "3. Combine similarities using fixed exploratory weights.",
        "4. Exclude neighbors below the exploratory threshold.",
        "5. Rank unseen keywords by summed neighbor similarity.",
        "",
        "Limitations:",
        "This output is exploratory, not a validated recommendation model.",
        "The threshold and weights have not been tuned on validation data.",
        "Lexical similarity does not guarantee clinical relevance.",
        "This script fits TF-IDF on the full dataset; do not use its output",
        "as leakage-free evaluation. Evaluate using training-only fitting.",
    ]

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print("\nImproved content-based recommendations generated!")
    print(f"Trials processed: {len(df)}")
    print(f"Minimum similarity threshold: {MIN_SIMILARITY}")
    print(f"Recommendations saved to: {OUTPUT_PATH}")
    print(f"Report saved to: {REPORT_PATH}")

    target_id = "NCT07806708"
    target = output[output["nct_id"] == target_id]

    if not target.empty:
        print(f"\nInspection: {target_id}")
        print("Trial:", target.iloc[0]["title"])
        print("Neighbors:", target.iloc[0]["similar_trial_ids"] or "(none)")
        print(
            "Recommended keywords:",
            target.iloc[0]["recommended_keywords"] or "(none)",
        )

if __name__ == "__main__":
    main()
