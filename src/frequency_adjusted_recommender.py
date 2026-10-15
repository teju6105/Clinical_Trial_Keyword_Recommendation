from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "clinical_trials_cleaned.csv"
OUTPUT_PATH = ROOT / "results" / "frequency_adjusted_recommendations.csv"
REPORT_PATH = ROOT / "results" / "frequency_adjusted_report.txt"

TOP_NEIGHBORS = 10
TOP_RECOMMENDATIONS = 10
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


def make_matrix(text):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=20000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(text)
    return matrix


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH).fillna("")

    required = [
        "nct_id",
        "title",
        "conditions",
        "brief_summary",
        "original_keywords",
    ]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["keyword_set"] = df["original_keywords"].apply(parse_keywords)

    # Only trials with recorded keywords can supply recommendation evidence.
    labelled = df[df["keyword_set"].map(len) > 0].copy()
    labelled = labelled.reset_index(drop=True)

    if len(labelled) < 2:
        raise ValueError("Not enough labelled trials to recommend keywords.")

    # Text representations.
    title_matrix = make_matrix(normalize_text(labelled["title"]))
    conditions_matrix = make_matrix(normalize_text(labelled["conditions"]))
    summary_matrix = make_matrix(normalize_text(labelled["brief_summary"]))

    # Combine similarities from separate text fields.
    title_similarity = cosine_similarity(title_matrix)
    conditions_similarity = cosine_similarity(conditions_matrix)
    summary_similarity = cosine_similarity(summary_matrix)

    combined_similarity = (
        TITLE_WEIGHT * title_similarity
        + CONDITIONS_WEIGHT * conditions_similarity
        + SUMMARY_WEIGHT * summary_similarity
    )

    # Count the number of labelled trials containing each keyword.
    keyword_document_frequency = {}

    for keyword_set in labelled["keyword_set"]:
        for keyword in keyword_set:
            keyword_document_frequency[keyword] = (
                keyword_document_frequency.get(keyword, 0) + 1
            )

    total_labelled_trials = len(labelled)

    # Smoothed inverse document frequency.
    keyword_idf = {
        keyword: math.log(
            (total_labelled_trials + 1) / (frequency + 1)
        ) + 1
        for keyword, frequency in keyword_document_frequency.items()
    }

    rows = []

    for i, row in labelled.iterrows():
        similarities = combined_similarity[i].copy()

        # A trial cannot be its own neighbor.
        similarities[i] = -1.0

        candidate_indices = [
            int(j)
            for j in np.argsort(similarities)[::-1]
            if similarities[j] >= MIN_SIMILARITY
        ][:TOP_NEIGHBORS]

        candidate_scores = {}

        for j in candidate_indices:
            similarity = float(similarities[j])

            if similarity <= 0:
                continue

            for keyword in labelled.iloc[j]["keyword_set"]:
                # Do not recommend keywords already recorded for this trial.
                if keyword in row["keyword_set"]:
                    continue

                # Give less weight to keywords found in many trials.
                adjusted_score = similarity * keyword_idf[keyword]

                candidate_scores[keyword] = (
                    candidate_scores.get(keyword, 0.0)
                    + adjusted_score
                )

        recommendations = sorted(
            candidate_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:TOP_RECOMMENDATIONS]

        rows.append({
            "nct_id": row["nct_id"],
            "title": row["title"],
            "similar_trial_ids": "; ".join(
                str(labelled.iloc[j]["nct_id"])
                for j in candidate_indices
            ),
            "similarity_scores": "; ".join(
                f"{similarities[j]:.4f}"
                for j in candidate_indices
            ),
            "neighbor_count": len(candidate_indices),
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
        "FREQUENCY-ADJUSTED CONTENT-BASED RECOMMENDER",
        "=" * 48,
        f"Labelled trials: {len(labelled)}",
        f"Unique recorded keywords: {len(keyword_idf)}",
        f"Title weight: {TITLE_WEIGHT}",
        f"Conditions weight: {CONDITIONS_WEIGHT}",
        f"Summary weight: {SUMMARY_WEIGHT}",
        f"Minimum similarity: {MIN_SIMILARITY}",
        f"Maximum neighbors: {TOP_NEIGHBORS}",
        f"Maximum recommendations: {TOP_RECOMMENDATIONS}",
        "",
        "Method:",
        "1. Calculate text similarity separately for three fields.",
        "2. Combine field similarities using fixed weights.",
        "3. Select similar labelled trials above the threshold.",
        "4. Score unseen keywords using similarity multiplied by smoothed IDF.",
        "",
        "Limitations:",
        "This is an exploratory recommendation run, not a validated model.",
        "Text representations and keyword frequencies use the full labelled dataset.",
        "Do not use these outputs as leakage-free evaluation results.",
        "Rare keywords are not necessarily more relevant or clinically valid.",
    ]

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print("\nFrequency-adjusted recommendations generated!")
    print(f"Labelled trials: {len(labelled)}")
    print(f"Unique keywords: {len(keyword_idf)}")
    print(f"Recommendations saved to: {OUTPUT_PATH}")
    print(f"Report saved to: {REPORT_PATH}")

    target_id = "NCT07806708"
    target = output[output["nct_id"] == target_id]

    if not target.empty:
        print(f"\nInspection: {target_id}")
        print("Trial:", target.iloc[0]["title"])
        print("Neighbors:", target.iloc[0]["similar_trial_ids"] or "(none)")
        print(
            "Recommendations:",
            target.iloc[0]["recommended_keywords"] or "(none)",
        )


if __name__ == "__main__":
    main()
