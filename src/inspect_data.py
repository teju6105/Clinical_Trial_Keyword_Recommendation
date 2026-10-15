import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "clinical_trials.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CLINICAL TRIAL DATA QUALITY INSPECTION")
print("=" * 60)

# Check that the dataset exists
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

# Load dataset
df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

# 1. Dataset dimensions
print("\n1. DATASET DIMENSIONS")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# 2. Column names and data types
print("\n2. COLUMN NAMES AND DATA TYPES")
print(df.dtypes.to_string())

# 3. Duplicate records
print("\n3. DUPLICATE CHECK")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Duplicate trial IDs: {df['nct_id'].duplicated().sum()}")

# 4. Empty fields
print("\n4. EMPTY FIELD COUNTS")
empty_counts = df.fillna("").astype(str).apply(
    lambda column: column.str.strip().eq("").sum()
)
print(empty_counts.to_string())

# 5. Unique values
print("\n5. UNIQUE VALUES PER COLUMN")
print(df.nunique(dropna=False).to_string())

# 6. Keyword coverage
keyword_count = df["keywords"].fillna("").astype(str).str.strip().ne("").sum()
print("\n6. KEYWORD COVERAGE")
print(f"Trials with keywords: {keyword_count}")
print(f"Trials without keywords: {len(df) - keyword_count}")
print(f"Keyword coverage: {keyword_count / len(df) * 100:.2f}%")

# 7. Condition coverage
condition_count = df["conditions"].fillna("").astype(str).str.strip().ne("").sum()
print("\n7. CONDITION COVERAGE")
print(f"Trials with conditions: {condition_count}")
print(f"Trials without conditions: {len(df) - condition_count}")

# 8. Most frequent study statuses
print("\n8. STUDY STATUS DISTRIBUTION")
print(df["status"].fillna("").replace("", "Unknown").value_counts().head(15).to_string())

# 9. Most frequent study types
print("\n9. STUDY TYPE DISTRIBUTION")
print(df["study_type"].fillna("").replace("", "Unknown").value_counts().to_string())

# 10. Most frequent conditions
print("\n10. MOST FREQUENT CONDITIONS")
all_conditions = (
    df["conditions"]
    .fillna("")
    .astype(str)
    .str.split(";")
    .explode()
    .str.strip()
)
all_conditions = all_conditions[all_conditions.ne("")]
print(all_conditions.value_counts().head(20).to_string())

# 11. Summary statistics
print("\n11. TEXT LENGTH SUMMARY")
for column in ["title", "brief_summary", "conditions", "keywords"]:
    lengths = df[column].fillna("").astype(str).str.len()
    print(
        f"{column}: mean={lengths.mean():.1f}, "
        f"median={lengths.median():.1f}, max={lengths.max()}"
    )

# Save inspection report
report_file = RESULTS_DIR / "data_quality_report.txt"

with report_file.open("w", encoding="utf-8") as file:
    file.write("CLINICAL TRIAL DATA QUALITY REPORT\n")
    file.write("=" * 50 + "\n")
    file.write(f"Dataset: {DATA_FILE}\n")
    file.write(f"Rows: {df.shape[0]}\n")
    file.write(f"Columns: {df.shape[1]}\n\n")
    file.write("Data types:\n")
    file.write(df.dtypes.to_string())
    file.write("\n\nDuplicate rows:\n")
    file.write(str(df.duplicated().sum()))
    file.write("\n\nDuplicate trial IDs:\n")
    file.write(str(df["nct_id"].duplicated().sum()))
    file.write("\n\nEmpty field counts:\n")
    file.write(empty_counts.to_string())
    file.write("\n\nUnique values per column:\n")
    file.write(df.nunique(dropna=False).to_string())
    file.write("\n\nStudy status distribution:\n")
    file.write(df["status"].fillna("").replace("", "Unknown").value_counts().to_string())
    file.write("\n\nStudy type distribution:\n")
    file.write(df["study_type"].fillna("").replace("", "Unknown").value_counts().to_string())
    file.write("\n\nMost frequent conditions:\n")
    file.write(all_conditions.value_counts().head(20).to_string())

print("\n" + "=" * 60)
print("INSPECTION COMPLETED")
print(f"Report saved to: {report_file}")
print("=" * 60)
