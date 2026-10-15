import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

MATRIX_FILE = DATA_DIR / "trial_keyword_utility.npz"
TRIALS_FILE = DATA_DIR / "matrix_trial_ids.csv"
REPORT_FILE = RESULTS_DIR / "keyword_overlap_report.txt"


def analyze_overlap():
    if not MATRIX_FILE.exists():
        raise FileNotFoundError(f"Matrix not found: {MATRIX_FILE}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading trial-keyword utility matrix...")
    matrix = load_npz(MATRIX_FILE).tocsr()
    trials = pd.read_csv(TRIALS_FILE, encoding="utf-8-sig")

    if len(trials) != matrix.shape[0]:
        raise ValueError("Trial ID count does not match matrix rows.")

    n_trials, n_keywords = matrix.shape

    # Count keywords assigned to each trial.
    keyword_counts = np.diff(matrix.indptr)

    # Calculate pairwise cosine similarity.
    # The matrix is small enough for a dense similarity matrix.
    similarity = cosine_similarity(matrix, dense_output=True)

    # Exclude each trial's similarity with itself.
    np.fill_diagonal(similarity, 0.0)

    # Count other trials with positive similarity.
    similar_trial_counts = (similarity > 0).sum(axis=1)

    # Calculate statistics using unique trial pairs only.
    upper_triangle = np.triu_indices(n_trials, k=1)
    pair_similarities = similarity[upper_triangle]

    total_pairs = len(pair_similarities)
    shared_pairs = int((pair_similarities > 0).sum())
    zero_similarity_pairs = total_pairs - shared_pairs

    mean_similarity = (
        float(pair_similarities.mean())
        if total_pairs else 0.0
    )

    positive_similarities = pair_similarities[
        pair_similarities > 0
    ]

    mean_positive_similarity = (
        float(positive_similarities.mean())
        if len(positive_similarities) else 0.0
    )

    # Top 10 trials by number of other trials sharing keywords.
    top_indices = np.argsort(-similar_trial_counts)[:10]

    top_trials = pd.DataFrame({
        "nct_id": trials.iloc[top_indices]["nct_id"].values,
        "keyword_count": keyword_counts[top_indices],
        "similar_trial_count": similar_trial_counts[top_indices],
        "mean_positive_similarity": [
            (
                similarity[i, similarity[i] > 0].mean()
                if np.any(similarity[i] > 0)
                else 0.0
            )
            for i in top_indices
        ],
    })

    # Keyword-count distribution.
    count_distribution = (
        pd.Series(keyword_counts)
        .value_counts()
        .sort_index()
    )

    report_lines = [
        "CLINICAL TRIAL KEYWORD OVERLAP REPORT",
        "=" * 55,
        f"Number of trials: {n_trials}",
        f"Number of keywords: {n_keywords}",
        f"Total unique trial pairs: {total_pairs}",
        f"Pairs with positive cosine similarity: {shared_pairs}",
        f"Pairs with zero cosine similarity: {zero_similarity_pairs}",
        (
            "Percentage of pairs with shared keywords: "
            f"{shared_pairs / total_pairs * 100:.2f}%"
            if total_pairs else "No trial pairs available."
        ),
        f"Mean cosine similarity across all pairs: {mean_similarity:.6f}",
        (
            "Mean cosine similarity among positive pairs: "
            f"{mean_positive_similarity:.6f}"
        ),
        f"Mean keywords per trial: {keyword_counts.mean():.2f}",
        f"Median keywords per trial: {np.median(keyword_counts):.2f}",
        f"Minimum keywords per trial: {keyword_counts.min()}",
        f"Maximum keywords per trial: {keyword_counts.max()}",
        "",
        "KEYWORD COUNT DISTRIBUTION",
        count_distribution.to_string(),
        "",
        "TOP 10 TRIALS BY NUMBER OF SIMILAR TRIALS",
        top_trials.to_string(index=False),
    ]

    report_text = "\n".join(report_lines)
    REPORT_FILE.write_text(report_text, encoding="utf-8")

    print("\n" + "=" * 60)
    print("KEYWORD OVERLAP ANALYSIS COMPLETED")
    print("=" * 60)
    print(f"Trials: {n_trials}")
    print(f"Keywords: {n_keywords}")
    print(f"Unique trial pairs: {total_pairs}")
    print(f"Pairs with shared keywords: {shared_pairs}")
    print(f"Pairs with zero similarity: {zero_similarity_pairs}")

    if total_pairs:
        print(
            "Pairs with shared keywords: "
            f"{shared_pairs / total_pairs * 100:.2f}%"
        )

    print(f"Mean cosine similarity: {mean_similarity:.6f}")
    print(f"Mean positive similarity: {mean_positive_similarity:.6f}")
    print(f"Mean keywords per trial: {keyword_counts.mean():.2f}")
    print(f"Median keywords per trial: {np.median(keyword_counts):.2f}")
    print(f"Minimum keywords per trial: {keyword_counts.min()}")
    print(f"Maximum keywords per trial: {keyword_counts.max()}")

    print("\nKeyword count distribution:")
    print(count_distribution.to_string())

    print("\nTop 10 trials by number of similar trials:")
    print(top_trials.to_string(index=False))

    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    analyze_overlap()
