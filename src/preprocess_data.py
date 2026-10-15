import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "clinical_trials.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "clinical_trials_cleaned.csv"

TEXT_COLUMNS = [
    "nct_id",
    "title",
    "official_title",
    "status",
    "study_type",
    "phase",
    "conditions",
    "keywords",
    "brief_summary",
    "interventions",
]


def clean_text(value):
    """Normalize whitespace and handle empty text values."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def preprocess_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw clinical trial dataset...")
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    initial_rows = len(df)

    print(f"Initial records: {initial_rows}")

    # Ensure expected columns are available
    missing_columns = [
        column for column in TEXT_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Expected columns are missing: {missing_columns}"
        )

    # Preserve the original keywords before cleaning
    df["original_keywords"] = df["keywords"].fillna("").astype(str)

    # Normalize whitespace in all text columns
    for column in TEXT_COLUMNS:
        df[column] = df[column].map(clean_text)

    # Remove records without a usable trial identifier
    missing_id_count = df["nct_id"].eq("").sum()

    if missing_id_count:
        print(
            f"Removing {missing_id_count} records "
            "without a trial ID."
        )
        df = df[df["nct_id"].ne("")].copy()

    # Keep one record per trial ID
    duplicate_id_count = df["nct_id"].duplicated().sum()

    if duplicate_id_count:
        print(
            f"Removing {duplicate_id_count} duplicate trial IDs."
        )
        df = df.drop_duplicates(
            subset=["nct_id"],
            keep="first"
        ).copy()

    # Build text for downstream keyword extraction.
    # Do not include existing keywords in this field,
    # so they can be evaluated separately later.
    df["text_for_keyword_extraction"] = (
        df["title"] + ". "
        + df["official_title"] + ". "
        + df["conditions"] + ". "
        + df["brief_summary"] + ". "
        + df["interventions"]
    ).str.strip()

    # Calculate quality indicators
    df["has_original_keywords"] = (
        df["original_keywords"].str.strip().ne("")
    )

    df["has_conditions"] = df["conditions"].str.strip().ne("")

    # Save cleaned data
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 55)
    print("PREPROCESSING COMPLETED")
    print("=" * 55)
    print(f"Records before cleaning: {initial_rows}")
    print(f"Records after cleaning: {len(df)}")
    print(f"Records removed: {initial_rows - len(df)}")
    print(
        "Records with original keywords: "
        f"{df['has_original_keywords'].sum()}"
    )
    print(
        "Records without original keywords: "
        f"{(~df['has_original_keywords']).sum()}"
    )
    print(
        "Records with conditions: "
        f"{df['has_conditions'].sum()}"
    )
    print(f"Output file: {OUTPUT_FILE}")

    print("\nPreview of cleaned data:")
    print(
        df[
            [
                "nct_id",
                "title",
                "original_keywords",
                "has_original_keywords",
            ]
        ].head().to_string(index=False)
    )

    print("\nMissing or empty values after cleaning:")
    empty_counts = df[TEXT_COLUMNS].apply(
        lambda column: column.str.strip().eq("").sum()
    )
    print(empty_counts.to_string())


if __name__ == "__main__":
    preprocess_data()
