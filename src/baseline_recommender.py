from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clinical_trials_cleaned.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "baseline_recommendations.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "baseline_report.txt"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH).fillna("")
    keyword_column = "original_keywords"

    if keyword_column not in df.columns:
        raise ValueError(f"Missing column: {keyword_column}")

    keyword_counts = {}
    trials_with_keywords = 0

    for keyword_text in df[keyword_column]:
        if not str(keyword_text).strip():
            continue

        trials_with_keywords += 1

        # Keywords in this dataset are separated by semicolons.
        individual_keywords = str(keyword_text).split(";")

        # Count each keyword at most once per trial.
        keywords_in_trial = {
            " ".join(keyword.lower().split())
            for keyword in individual_keywords
            if keyword.strip()
        }

        for keyword in keywords_in_trial:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1

    if not keyword_counts:
        raise ValueError("No individual keywords found.")

    ranked_keywords = sorted(
        keyword_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    recommendations = pd.DataFrame(
        [
            {
                "rank": rank,
                "keyword": keyword,
                "trial_frequency": frequency,
            }
            for rank, (keyword, frequency) in enumerate(ranked_keywords, 1)
        ]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(OUTPUT_PATH, index=False)

    report_lines = [
        "BASELINE KEYWORD RECOMMENDATION REPORT",
        "=" * 45,
        f"Total trials in dataset: {len(df)}",
        f"Trials with original keywords: {trials_with_keywords}",
        f"Unique individual keywords: {len(keyword_counts)}",
        "",
        "Method: Rank individual keywords by trial frequency.",
        "Each keyword is counted at most once per trial.",
        "This is a simple frequency-based baseline, not a personalized model.",
        "",
        "Top 20 keywords:",
    ]

    for rank, (keyword, frequency) in enumerate(ranked_keywords[:20], 1):
        report_lines.append(
            f"{rank:2}. {keyword} - appears in {frequency} trials"
        )

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nBaseline model completed successfully!")
    print(f"Total trials: {len(df)}")
    print(f"Trials with keywords: {trials_with_keywords}")
    print(f"Unique individual keywords: {len(keyword_counts)}")
    print("\nTop 10 individual keywords:")

    for rank, (keyword, frequency) in enumerate(ranked_keywords[:10], 1):
        print(f"{rank:2}. {keyword} ({frequency} trials)")

    print(f"\nRecommendations saved to: {OUTPUT_PATH}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
