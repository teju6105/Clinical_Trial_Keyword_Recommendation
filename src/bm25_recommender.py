from pathlib import Path
import re

import pandas as pd
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clinical_trials_cleaned.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "bm25_recommendations.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "bm25_report.txt"

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


def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH).fillna("")

    text_columns = [
        "title",
        "official_title",
        "conditions",
        "brief_summary",
        "interventions",
    ]

    required_columns = ["nct_id", "original_keywords"] + text_columns
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["keyword_set"] = df["original_keywords"].apply(parse_keywords)

    df["content_text"] = (
        df[text_columns]
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    tokenized_documents = [
        tokenize(text) or ["emptydocument"]
        for text in df["content_text"]
    ]

    bm25 = BM25Okapi(tokenized_documents)
    rows = []

    for i, row in df.iterrows():
        scores = bm25.get_scores(tokenized_documents[i])
        scores[i] = float("-inf")

        similar_indices = sorted(
            range(len(scores)),
            key=lambda j: scores[j],
            reverse=True,
        )[:TOP_SIMILAR_TRIALS]

        candidate_scores = {}

        for j in similar_indices:
            score = float(scores[j])

            if score <= 0:
                continue

            for keyword in df.iloc[j]["keyword_set"]:
                if keyword not in row["keyword_set"]:
                    candidate_scores[keyword] = (
                        candidate_scores.get(keyword, 0.0) + score
                    )

        ranked_candidates = sorted(
            candidate_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:TOP_RECOMMENDATIONS]

        rows.append({
            "nct_id": row["nct_id"],
            "title": row["title"],
            "similar_trial_ids": "; ".join(
                str(df.iloc[j]["nct_id"]) for j in similar_indices
            ),
            "recommended_keywords": "; ".join(
                keyword for keyword, _ in ranked_candidates
            ),
            "recommendation_scores": "; ".join(
                f"{score:.4f}" for _, score in ranked_candidates
            ),
            "recorded_keyword_count": len(row["keyword_set"]),
        })

    output = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    report = [
        "BM25-BASED CLINICAL TRIAL KEYWORD RECOMMENDATION",
        "=" * 52,
        f"Trials processed: {len(df)}",
        f"BM25 corpus size: {len(tokenized_documents)}",
        f"Similar trials considered per target: {TOP_SIMILAR_TRIALS}",
        f"Maximum recommendations per trial: {TOP_RECOMMENDATIONS}",
        "",
        "Method:",
        "1. Tokenize clinical trial descriptions.",
        "2. Rank other trials using BM25 relevance scores.",
        "3. Collect recorded keywords from the highest-ranked trials.",
        "4. Rank candidate keywords by summed BM25 scores.",
        "5. Exclude keywords already recorded for the target trial.",
        "",
        "Limitations:",
        "- This is an exploratory recommendation output, not an evaluation.",
        "- The corpus includes all supplied trial descriptions.",
        "- Recorded keywords are incomplete for some trials.",
        "- Recommendations do not establish clinical validity.",
    ]

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print("\nBM25 recommendation generation completed!")
    print(f"Trials processed: {len(df)}")
    print(f"Recommendations saved to: {OUTPUT_PATH}")
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
