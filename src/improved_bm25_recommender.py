import pandas as pd
import numpy as np
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "clinical_trials_cleaned.csv"
OUTPUT_PATH = ROOT / "results" / "improved_bm25_recommendations.csv"
REPORT_PATH = ROOT / "results" / "improved_bm25_report.txt"

TEXT_COLUMNS = [
    "title",
    "official_title",
    "conditions",
    "brief_summary",
    "interventions",
]


def tokenize(text):
    if pd.isna(text):
        return []
    return re.findall(r"\b[a-zA-Z0-9]+\b", str(text).lower())


def split_keywords(value):
    if pd.isna(value):
        return []
    return [
        keyword.strip()
        for keyword in str(value).split(";")
        if keyword.strip()
    ]


def main():
    df = pd.read_csv(DATA_PATH).fillna("")

    required = {"nct_id", "keywords"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    text_columns = [c for c in TEXT_COLUMNS if c in df.columns]

    df["combined_text"] = df[text_columns].astype(str).agg(" ".join, axis=1)
    corpus = [tokenize(text) for text in df["combined_text"]]

    bm25 = BM25Okapi(corpus)

    keyword_trials = {}
    for _, row in df.iterrows():
        for keyword in split_keywords(row["keywords"]):
            keyword_trials.setdefault(keyword, set()).add(row["nct_id"])

    total_trials = len(df)
    keyword_scores = {}

    for keyword, trial_ids in keyword_trials.items():
        frequency = len(trial_ids)

        # Smoothed inverse document frequency:
        # gives less weight to keywords appearing in many trials.
        idf = np.log(1 + (total_trials - frequency + 0.5) /
                     (frequency + 0.5))

        keyword_scores[keyword] = idf

    results = []

    for index, row in df.iterrows():
        scores = bm25.get_scores(corpus[index])
        scores[index] = -np.inf

        top_indices = np.argsort(scores)[::-1][:10]
        existing = set(split_keywords(row["keywords"]))
        recommendations = {}

        for neighbor_index in top_indices:
            if not np.isfinite(scores[neighbor_index]):
                continue

            neighbor_keywords = split_keywords(
                df.iloc[neighbor_index]["keywords"]
            )

            for keyword in neighbor_keywords:
                if keyword in existing:
                    continue

                # Combine BM25 similarity with keyword rarity.
                score = max(float(scores[neighbor_index]), 0.0)
                score *= keyword_scores.get(keyword, 0.0)

                recommendations[keyword] = (
                    recommendations.get(keyword, 0.0) + score
                )

        ranked = sorted(
            recommendations.items(),
            key=lambda item: item[1],
            reverse=True
        )[:5]

        results.append({
            "nct_id": row["nct_id"],
            "title": row.get("title", ""),
            "existing_keywords": "; ".join(sorted(existing)),
            "recommended_keywords": "; ".join(
                keyword for keyword, _ in ranked
            ),
            "recommendation_scores": "; ".join(
                f"{score:.4f}" for _, score in ranked
            ),
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    report = [
        "Improved BM25 Keyword Recommendation Report",
        "=" * 45,
        f"Trials processed: {len(df)}",
        f"Unique keywords: {len(keyword_trials)}",
        "Method: BM25 similarity combined with smoothed inverse keyword frequency.",
        "Note: These are exploratory recommendations, not clinical advice.",
        f"Output: {OUTPUT_PATH}",
    ]

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print("\nSample recommendations:")
    print(pd.DataFrame(results).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
