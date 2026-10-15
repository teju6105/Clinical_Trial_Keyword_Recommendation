import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

MATRIX_FILE = DATA_DIR / "trial_keyword_utility.npz"
VOCAB_FILE = DATA_DIR / "matrix_keyword_vocabulary.csv"
REPORT_FILE = RESULTS_DIR / "feature_analysis_report.txt"


def analyze_features():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not MATRIX_FILE.exists() or not VOCAB_FILE.exists():
        raise FileNotFoundError(
            "Utility matrix or matrix vocabulary is missing."
        )

    matrix = load_npz(MATRIX_FILE).tocsr()
    vocabulary = pd.read_csv(VOCAB_FILE, encoding="utf-8-sig")

    # Number of keywords assigned to each trial.
    keywords_per_trial = np.diff(matrix.indptr)

    # Keyword frequency across trials.
    keyword_frequency = vocabulary["trial_frequency"]

    # Each unique pair is counted once.
    similarity = cosine_similarity(matrix, dense_output=True)
    np.fill_diagonal(similarity, 0.0)

    upper_triangle = np.triu_indices(matrix.shape[0], k=1)
    pair_similarities = similarity[upper_triangle]

    positive_pairs = pair_similarities[pair_similarities > 0]

    # Frequency groups.
    rare_keywords = int((keyword_frequency == 1).sum())
    low_frequency_keywords = int((keyword_frequency <= 3).sum())
    common_keywords = int((keyword_frequency >= 5).sum())

    # Trials connected to at least one other trial.
    connected_trials = int((similarity > 0).any(axis=1).sum())
    isolated_trials = int(matrix.shape[0] - connected_trials)

    # Estimate the effect of removing extremely rare keywords.
    # This is an analysis only; the original matrix remains unchanged.
    keep_columns = keyword_frequency.to_numpy() >= 2
    reduced_matrix = matrix[:, keep_columns]

    reduced_similarity = cosine_similarity(
        reduced_matrix,
        dense_output=True,
    )
    np.fill_diagonal(reduced_similarity, 0.0)

    reduced_pairs = reduced_similarity[upper_triangle]
    reduced_positive_pairs = int((reduced_pairs > 0).sum())

    lines = [
        "CLINICAL TRIAL FEATURE ANALYSIS",
        "=" * 55,
        f"Trials: {matrix.shape[0]}",
        f"Keywords: {matrix.shape[1]}",
        f"Mean keywords per trial: {keywords_per_trial.mean():.2f}",
        f"Median keywords per trial: {np.median(keywords_per_trial):.2f}",
        "",
        "KEYWORD FREQUENCY",
        f"Keywords occurring in one trial: {rare_keywords}",
        f"Keywords occurring in at most three trials: {low_frequency_keywords}",
        f"Keywords occurring in at least five trials: {common_keywords}",
        "",
        "TRIAL OVERLAP",
        f"Trial pairs with positive similarity: {len(positive_pairs)}",
        f"Trials connected to another trial: {connected_trials}",
        f"Trials with no positive-similarity neighbor: {isolated_trials}",
        (
            "Mean positive cosine similarity: "
            f"{positive_pairs.mean():.6f}"
            if len(positive_pairs)
            else "Mean positive cosine similarity: undefined"
        ),
        "",
        "RARE KEYWORD SENSITIVITY ANALYSIS",
        f"Keywords retained when frequency >= 2: {int(keep_columns.sum())}",
        f"Positive-similarity pairs after filtering: {reduced_positive_pairs}",
        "",
        "TOP 20 KEYWORDS",
        vocabulary.sort_values(
            ["trial_frequency", "keyword"],
            ascending=[False, True],
        ).head(20).to_string(index=False),
    ]

    report = "\n".join(lines)
    REPORT_FILE.write_text(report, encoding="utf-8")

    print("=" * 60)
    print("FEATURE ANALYSIS COMPLETED")
    print("=" * 60)
    print(f"Trials: {matrix.shape[0]}")
    print(f"Keywords: {matrix.shape[1]}")
    print(f"Mean keywords per trial: {keywords_per_trial.mean():.2f}")
    print(f"Median keywords per trial: {np.median(keywords_per_trial):.2f}")
    print(f"Keywords occurring in one trial: {rare_keywords}")
    print(f"Keywords occurring in at most three trials: {low_frequency_keywords}")
    print(f"Keywords occurring in at least five trials: {common_keywords}")
    print(f"Trials connected to another trial: {connected_trials}")
    print(f"Trials with no positive-similarity neighbor: {isolated_trials}")
    print(f"Positive-similarity pairs after filtering: {reduced_positive_pairs}")
    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    analyze_features()
