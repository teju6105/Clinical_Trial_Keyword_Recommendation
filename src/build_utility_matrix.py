import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, save_npz

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "clinical_trials_cleaned.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

MATRIX_FILE = OUTPUT_DIR / "trial_keyword_utility.npz"
TRIALS_FILE = OUTPUT_DIR / "matrix_trial_ids.csv"
VOCAB_FILE = OUTPUT_DIR / "matrix_keyword_vocabulary.csv"


def build_utility_matrix():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    print("Loading cleaned clinical trial data...")
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig").fillna("")

    # Use only trials with original registry-provided keywords.
    # Missing keywords are unknown, not negative examples.
    df["original_keywords"] = df["original_keywords"].astype(str)

    df = df[
        df["original_keywords"].str.strip().ne("")
    ].copy()

    # Ensure one row per trial ID.
    df = df.drop_duplicates(subset=["nct_id"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No trials with original keywords were found.")

    print(f"Trials included in matrix: {len(df)}")

    # Build a normalized list of unique keywords for each trial.
    trial_keyword_lists = []

    for value in df["original_keywords"]:
        keywords = {
            " ".join(part.strip().lower().split())
            for part in str(value).split(";")
            if part.strip()
        }
        trial_keyword_lists.append(sorted(keywords))

    # Create a sorted global vocabulary.
    vocabulary = sorted({
        keyword
        for keywords in trial_keyword_lists
        for keyword in keywords
    })

    keyword_to_index = {
        keyword: index
        for index, keyword in enumerate(vocabulary)
    }

    # Each original keyword is represented as a binary occurrence.
    # Because the registry field lists assigned keywords, this is
    # occurrence in a trial, not term frequency in a document.
    row_indices = []
    col_indices = []

    for row_index, keywords in enumerate(trial_keyword_lists):
        for keyword in keywords:
            row_indices.append(row_index)
            col_indices.append(keyword_to_index[keyword])

    n_trials = len(df)
    n_keywords = len(vocabulary)

    binary_matrix = csr_matrix(
        (
            np.ones(len(row_indices), dtype=np.float64),
            (row_indices, col_indices),
        ),
        shape=(n_trials, n_keywords),
    )

    # Number of trials containing each keyword.
    document_frequency = np.asarray(
        (binary_matrix > 0).sum(axis=0)
    ).ravel()

    # Smoothed-free IDF, following the provisional formula.
    # A keyword occurring in every trial receives zero IDF.
    idf = np.log10(n_trials / document_frequency)

    # Each trial-keyword occurrence has c_ij = 1,
    # so log10(1 + c_ij) = log10(2).
    coo = binary_matrix.tocoo()

    weighted_values = (
        np.log10(2.0) * idf[coo.col]
    )

    utility_matrix = csr_matrix(
        (weighted_values, (coo.row, coo.col)),
        shape=binary_matrix.shape,
    )

    save_npz(MATRIX_FILE, utility_matrix)

    pd.DataFrame({
        "nct_id": df["nct_id"].tolist()
    }).to_csv(TRIALS_FILE, index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "keyword": vocabulary,
        "trial_frequency": document_frequency,
        "idf": idf,
    }).to_csv(VOCAB_FILE, index=False, encoding="utf-8-sig")

    total_cells = n_trials * n_keywords
    nonzero_cells = utility_matrix.nnz
    sparsity = 1 - (nonzero_cells / total_cells)

    print("\n" + "=" * 60)
    print("TRIAL-KEYWORD UTILITY MATRIX COMPLETED")
    print("=" * 60)
    print(f"Number of trials: {n_trials}")
    print(f"Number of unique keywords: {n_keywords}")
    print(f"Matrix shape: {utility_matrix.shape}")
    print(f"Nonzero entries: {nonzero_cells}")
    print(f"Sparsity: {sparsity:.4%}")
    print(f"Matrix saved to: {MATRIX_FILE}")
    print(f"Trial IDs saved to: {TRIALS_FILE}")
    print(f"Matrix vocabulary saved to: {VOCAB_FILE}")

    print("\nFirst five keywords by trial frequency:")
    preview = pd.DataFrame({
        "keyword": vocabulary,
        "trial_frequency": document_frequency,
        "idf": idf,
    }).sort_values(
        ["trial_frequency", "keyword"],
        ascending=[False, True],
    ).head(5)

    print(preview.to_string(index=False))


if __name__ == "__main__":
    build_utility_matrix()
