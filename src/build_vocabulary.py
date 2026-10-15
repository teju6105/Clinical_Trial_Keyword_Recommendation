import pandas as pd
from collections import Counter
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "clinical_trials_cleaned.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
VOCAB_FILE = OUTPUT_DIR / "keyword_vocabulary.csv"
SUMMARY_FILE = PROJECT_ROOT / "results" / "keyword_vocabulary_summary.txt"


def normalize_keyword(keyword):
    """Normalize keyword casing and whitespace."""
    return " ".join(str(keyword).strip().lower().split())


def build_vocabulary():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    keyword_counts = Counter()
    trials_with_keywords = 0

    for value in df["original_keywords"].fillna(""):
        keywords = {
            normalize_keyword(item)
            for item in str(value).split(";")
            if normalize_keyword(item)
        }

        if keywords:
            trials_with_keywords += 1
            keyword_counts.update(keywords)

    if not keyword_counts:
        raise ValueError("No keywords were found in the dataset.")

    vocabulary = pd.DataFrame(
        [
            {
                "keyword": keyword,
                "trial_frequency": frequency,
            }
            for keyword, frequency in keyword_counts.items()
        ]
    )

    vocabulary = vocabulary.sort_values(
        ["trial_frequency", "keyword"],
        ascending=[False, True],
    ).reset_index(drop=True)

    vocabulary.to_csv(
        VOCAB_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    unique_keywords = len(vocabulary)
    total_keyword_assignments = int(vocabulary["trial_frequency"].sum())

    summary = [
        "CLINICAL TRIAL KEYWORD VOCABULARY SUMMARY",
        "=" * 50,
        f"Total clinical trials: {len(df)}",
        f"Trials with original keywords: {trials_with_keywords}",
        f"Trials without original keywords: {len(df) - trials_with_keywords}",
        f"Unique normalized keywords: {unique_keywords}",
        f"Total keyword assignments: {total_keyword_assignments}",
        "",
        "Top 20 keywords by trial frequency:",
        vocabulary.head(20).to_string(index=False),
        "",
        f"Vocabulary saved to: {VOCAB_FILE}",
    ]

    report_text = "\n".join(summary)
    SUMMARY_FILE.write_text(report_text, encoding="utf-8")

    print("=" * 60)
    print("KEYWORD VOCABULARY BUILD COMPLETED")
    print("=" * 60)
    print(f"Total trials: {len(df)}")
    print(f"Trials with keywords: {trials_with_keywords}")
    print(f"Unique keywords: {unique_keywords}")
    print(f"Total keyword assignments: {total_keyword_assignments}")

    print("\nTop 20 keywords:")
    print(vocabulary.head(20).to_string(index=False))

    print(f"\nVocabulary file: {VOCAB_FILE}")
    print(f"Summary report: {SUMMARY_FILE}")


if __name__ == "__main__":
    build_vocabulary()
