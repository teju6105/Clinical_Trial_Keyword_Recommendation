from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "clinical_trials_cleaned.csv"
RESULTS_DIR = ROOT / "results"

DETAIL_PATH = RESULTS_DIR / "improved_content_evaluation.csv"
REPORT_PATH = RESULTS_DIR / "improved_content_evaluation_report.txt"

SEED = 42
TEST_FRACTION = 0.20
HIDE_FRACTION = 0.30
K = 5
TOP_NEIGHBORS = 10
MIN_SIMILARITY = 0.0

TITLE_WEIGHT = 0.40
CONDITIONS_WEIGHT = 0.35
SUMMARY_WEIGHT = 0.25

def parse_keywords(value):
    if pd.isna(value) or not str(value).strip():
        return set()
    return {
        " ".join(part.lower().split())
        for part in str(value).split(";")
        if part.strip()
    }

def normalize_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

def precision_at_k(recommendations, relevant):
    return len(set(recommendations[:K]) & relevant) / K

def recall_at_k(recommendations, relevant):
    if not relevant:
        return float("nan")
    return len(set(recommendations[:K]) & relevant) / len(relevant)

def ndcg_at_k(recommendations, relevant):
    dcg = sum(
        1 / math.log2(rank + 2)
        for rank, keyword in enumerate(recommendations[:K])
        if keyword in relevant
    )
    ideal_count = min(len(relevant), K)
    idcg = sum(
        1 / math.log2(rank + 2)
        for rank in range(ideal_count)
    )
    return dcg / idcg if idcg else float("nan")

def make_matrix(train_text, test_text):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=20000,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(train_text)
    test_matrix = vectorizer.transform(test_text)
    return train_matrix, test_matrix, len(vectorizer.vocabulary_)

def main():
    df = pd.read_csv(DATA_PATH).fillna("")

    required = [
        "nct_id", "original_keywords", "title",
        "conditions", "brief_summary"
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["keyword_set"] = df["original_keywords"].apply(parse_keywords)

    # Match the original evaluation's eligibility and split.
    eligible = df[df["keyword_set"].map(len) >= 2].copy()
    eligible = eligible.reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(eligible))
    test_count = max(1, round(len(order) * TEST_FRACTION))

    test_positions = sorted(order[:test_count].tolist())
    train_positions = sorted(order[test_count:].tolist())

    train = eligible.iloc[train_positions].reset_index(drop=True)
    test = eligible.iloc[test_positions].reset_index(drop=True)

    # Generate hidden labels once using the same procedure as the original.
    cases = []
    for _, row in test.iterrows():
        all_keywords = sorted(row["keyword_set"])
        hidden_count = max(1, round(len(all_keywords) * HIDE_FRACTION))
        hidden_count = min(hidden_count, len(all_keywords) - 1)

        hidden = set(
            rng.choice(
                all_keywords,
                size=hidden_count,
                replace=False
            ).tolist()
        )

        cases.append({
            "row": row,
            "hidden": hidden,
            "visible": row["keyword_set"] - hidden,
        })

    # Original content-based representation: all five text fields.
    text_columns = [
        "title", "official_title", "conditions",
        "brief_summary", "interventions"
    ]
    missing_text = [c for c in text_columns if c not in df.columns]
    if missing_text:
        raise ValueError(f"Missing text columns: {missing_text}")

    def combined_text(frame):
        return (
            frame[text_columns].astype(str).agg(" ".join, axis=1)
            .str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        )

    original_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=30000,
    )
    original_train_matrix = original_vectorizer.fit_transform(
        combined_text(train)
    )
    original_test_matrix = original_vectorizer.transform(combined_text(test))
    original_similarity = cosine_similarity(
        original_test_matrix, original_train_matrix
    )

    # Improved representation: separate title, conditions, and summary.
    title_train, title_test, title_vocab = make_matrix(
        normalize_text(train["title"]),
        normalize_text(test["title"]),
    )
    condition_train, condition_test, condition_vocab = make_matrix(
        normalize_text(train["conditions"]),
        normalize_text(test["conditions"]),
    )
    summary_train, summary_test, summary_vocab = make_matrix(
        normalize_text(train["brief_summary"]),
        normalize_text(test["brief_summary"]),
    )

    title_similarity = cosine_similarity(title_test, title_train)
    condition_similarity = cosine_similarity(
        condition_test, condition_train
    )
    summary_similarity = cosine_similarity(summary_test, summary_train)

    improved_similarity = (
        TITLE_WEIGHT * title_similarity
        + CONDITIONS_WEIGHT * condition_similarity
        + SUMMARY_WEIGHT * summary_similarity
    )

    results = []

    for test_position, case in enumerate(cases):
        row = case["row"]
        visible = case["visible"]
        hidden = case["hidden"]

        for model_name, similarity_matrix in [
            ("original_content_based", original_similarity),
            ("improved_content_based", improved_similarity),
        ]:
            similarities = similarity_matrix[test_position]

            # Only training records can supply recommendation keywords.
            neighbor_indices = [
                int(j)
                for j in np.argsort(similarities)[::-1]
                if similarities[j] >= (
                    MIN_SIMILARITY
                    if model_name == "improved_content_based"
                    else 0.0
                )
            ][:TOP_NEIGHBORS]

            scores = {}
            for neighbor_index in neighbor_indices:
                similarity = float(similarities[neighbor_index])
                if similarity <= 0:
                    continue

                for keyword in train.iloc[neighbor_index]["keyword_set"]:
                    if keyword not in visible:
                        scores[keyword] = scores.get(keyword, 0.0) + similarity

            recommendations = [
                keyword for keyword, _ in sorted(
                    scores.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:K]
            ]

            results.append({
                "nct_id": row["nct_id"],
                "model": model_name,
                "hidden_keywords": "; ".join(sorted(hidden)),
                "recommended_keywords": "; ".join(recommendations),
                "neighbor_count": len(neighbor_indices),
                "precision_at_5": precision_at_k(recommendations, hidden),
                "recall_at_5": recall_at_k(recommendations, hidden),
                "ndcg_at_5": ndcg_at_k(recommendations, hidden),
            })

    detail = pd.DataFrame(results)
    metrics = ["precision_at_5", "recall_at_5", "ndcg_at_5"]
    summary = detail.groupby("model")[metrics].mean()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)

    report = [
        "ORIGINAL VS IMPROVED CONTENT-BASED EVALUATION",
        "=" * 49,
        f"Eligible trials: {len(eligible)}",
        f"Training trials: {len(train)}",
        f"Test trials: {len(test)}",
        f"Seed: {SEED}",
        f"Improved title weight: {TITLE_WEIGHT}",
        f"Improved conditions weight: {CONDITIONS_WEIGHT}",
        f"Improved summary weight: {SUMMARY_WEIGHT}",
        f"Improved minimum similarity: {MIN_SIMILARITY}",
        f"Original TF-IDF vocabulary: {len(original_vectorizer.vocabulary_)}",
        f"Improved title vocabulary: {title_vocab}",
        f"Improved conditions vocabulary: {condition_vocab}",
        f"Improved summary vocabulary: {summary_vocab}",
        "",
        summary.to_string(float_format=lambda value: f"{value:.4f}"),
        "",
        "Limitations:",
        "- The two models use the same split and hidden labels.",
        "- Both models use training descriptions and training keywords only.",
        "- The improved weights and threshold are exploratory, not tuned.",
        "- Metrics evaluate recorded-keyword recovery, not clinical validity.",
    ]

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print("\nOriginal vs improved content evaluation completed!")
    print(f"Eligible trials: {len(eligible)}")
    print(f"Training trials: {len(train)}")
    print(f"Test trials: {len(test)}")
    print("\nMean metrics:")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nDetailed results: {DETAIL_PATH}")
    print(f"Report: {REPORT_PATH}")

if __name__ == "__main__":
    main()

