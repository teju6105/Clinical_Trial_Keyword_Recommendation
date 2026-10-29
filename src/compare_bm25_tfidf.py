from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent.parent
results = root / "results"

tfidf_path = results / "content_based_evaluation.csv"
bm25_path = results / "bm25_evaluation.csv"
output_path = results / "bm25_vs_tfidf_comparison.csv"

tfidf = pd.read_csv(tfidf_path)
bm25 = pd.read_csv(bm25_path)

metrics = ["precision_at_5", "recall_at_5", "ndcg_at_5"]

tfidf_summary = tfidf.groupby("model")[metrics].mean()
bm25_summary = bm25.groupby("model")[metrics].mean()

comparison = pd.concat([tfidf_summary, bm25_summary])
comparison.index.name = "model"
comparison.to_csv(output_path)

print("\nBM25 vs TF-IDF comparison")
print(comparison.to_string(float_format=lambda x: f"{x:.4f}"))
print(f"\nSaved comparison to: {output_path}")
