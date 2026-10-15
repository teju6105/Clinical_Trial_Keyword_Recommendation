from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "results" / "hybrid_evaluation.csv"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load evaluation results
df = pd.read_csv(INPUT_FILE)

required_columns = {
    "model",
    "precision_at_5",
    "recall_at_5",
    "ndcg_at_5",
}

missing = required_columns - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# Average metrics across the evaluated test trials
summary = (
    df.groupby("model")[
        ["precision_at_5", "recall_at_5", "ndcg_at_5"]
    ]
    .mean()
)

# Display models in a consistent order
model_order = [
    "frequency_baseline",
    "collaborative_filtering",
    "content_based",
    "hybrid",
]

summary = summary.reindex(
    [model for model in model_order if model in summary.index]
)

display_names = {
    "frequency_baseline": "Frequency Baseline",
    "collaborative_filtering": "Collaborative Filtering",
    "content_based": "Content-Based Filtering",
    "hybrid": "Hybrid Model",
}

# Save the aggregated metrics
summary.rename(index=display_names).to_csv(
    OUTPUT_DIR / "model_comparison_summary.csv"
)

# Create a separate graph for each metric
metrics = [
    ("precision_at_5", "Precision@5", "precision_at_5_comparison.png"),
    ("recall_at_5", "Recall@5", "recall_at_5_comparison.png"),
    ("ndcg_at_5", "NDCG@5", "ndcg_at_5_comparison.png"),
]

for column, title, filename in metrics:
    values = summary[column]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        [display_names[model] for model in values.index],
        values.values,
    )

    plt.title(f"Model Comparison: {title}")
    plt.xlabel("Recommendation Model")
    plt.ylabel(title)
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, values.values):
        plt.annotate(
            f"{value:.4f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

print("\nComparison graphs created successfully!")
print("\nAverage evaluation metrics:")
print(summary.rename(index=display_names).round(4))
print("\nFiles saved in:", OUTPUT_DIR)
print("- precision_at_5_comparison.png")
print("- recall_at_5_comparison.png")
print("- ndcg_at_5_comparison.png")
print("- model_comparison_summary.csv")
