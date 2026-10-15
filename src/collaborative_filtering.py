from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "clinical_trials_cleaned.csv"
OUTPUT_PATH = ROOT / "results" / "collaborative_recommendations.csv"
REPORT_PATH = ROOT / "results" / "collaborative_report.txt"

RANDOM_SEED = 42
TOP_NEIGHBORS = 10
TOP_RECOMMENDATIONS = 10
HIDE_FRACTION = 0.30


def parse_keywords(value):
    if pd.isna(value) or not str(value).strip():
        return set()

    return {
        " ".join(part.lower().split())
        for part in str(value).split(";")
        if part.strip()
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH).fillna("")

    required = ["nct_id", "title", "original_keywords"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["keyword_set"] = df["original_keywords"].apply(parse_keywords)
    eligible = df[df["keyword_set"].map(len) >= 1].copy()
    eligible = eligible.reset_index(drop=True)

    if len(eligible) < 2:
        raise ValueError("Not enough trials with recorded keywords.")

    # Reproducible trial-level split.
    rng = np.random.default_rng(RANDOM_SEED)
    order = rng.permutation(len(eligible))
    test_count = max(1, round(len(eligible) * 0.20))
    test_positions = sorted(order[:test_count].tolist())
    train_positions = sorted(order[test_count:].tolist())

    train = eligible.iloc[train_positions].reset_index(drop=True)
    test = eligible.iloc[test_positions].reset_index(drop=True)

    # Vocabulary and interactions are built from training trials only.
    vocabulary = sorted({
        keyword
        for keyword_set in train["keyword_set"]
        for keyword in keyword_set
    })
    keyword_index = {
        keyword: index for index, keyword in enumerate(vocabulary)
    }

    interactions = np.zeros(
        (len(train), len(vocabulary)),
        dtype=np.float32
    )

    for row_index, keyword_set in enumerate(train["keyword_set"]):
        for keyword in keyword_set:
            interactions[row_index, keyword_index[keyword]] = 1.0

    rows = []

    for _, test_row in test.iterrows():
        all_keywords = test_row["keyword_set"]

        # Hide a reproducible portion of recorded keywords.
        if len(all_keywords) >= 2:
            keyword_list = sorted(all_keywords)
            hide_count = max(
                1, round(len(keyword_list) * HIDE_FRACTION)
            )
            hide_count = min(hide_count, len(keyword_list) - 1)

            hidden = set(
                rng.choice(
                    keyword_list,
                    size=hide_count,
                    replace=False
                ).tolist()
            )
            visible = all_keywords - hidden
        else:
            # A one-keyword trial has no valid keyword to hide while
            # retaining a visible profile, so use its full profile.
            hidden = set()
            visible = set(all_keywords)

        # Build the target's visible interaction vector.
        target_vector = np.zeros(
            (1, len(vocabulary)),
            dtype=np.float32
        )

        for keyword in visible:
            if keyword in keyword_index:
                target_vector[0, keyword_index[keyword]] = 1.0

        # Compare this target trial to each training trial.
        if target_vector.sum() > 0:
            similarities = cosine_similarity(
                target_vector, interactions
            ).ravel()
        else:
            similarities = np.zeros(len(train), dtype=np.float32)

        # Keep only positive-similarity neighbors.
        neighbor_indices = np.argsort(similarities)[::-1]
        neighbor_indices = [
            index for index in neighbor_indices
            if similarities[index] > 0
        ][:TOP_NEIGHBORS]

        # Similarity-weighted votes for training keywords.
        candidate_scores = {}

        for neighbor_index in neighbor_indices:
            score = float(similarities[neighbor_index])

            for keyword in train.iloc[neighbor_index]["keyword_set"]:
                if keyword in visible:
                    continue

                candidate_scores[keyword] = (
                    candidate_scores.get(keyword, 0.0) + score
                )

        recommendations = sorted(
            candidate_scores.items(),
            key=lambda item: (-item[1], item[0])
        )[:TOP_RECOMMENDATIONS]

        rows.append({
            "nct_id": test_row["nct_id"],
            "title": test_row["title"],
            "visible_keywords": "; ".join(sorted(visible)),
            "hidden_keywords_for_inspection": "; ".join(sorted(hidden)),
            "neighbor_count": len(neighbor_indices),
            "recommended_keywords": "; ".join(
                keyword for keyword, _ in recommendations
            ),
            "recommendation_scores": "; ".join(
                f"{score:.4f}" for _, score in recommendations
            ),
        })

    output = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    report = [
        "COLLABORATIVE FILTERING RECOMMENDATION REPORT",
        "=" * 50,
        f"Trials with recorded keywords: {len(eligible)}",
        f"Training trials: {len(train)}",
        f"Test trials: {len(test)}",
        f"Training vocabulary: {len(vocabulary)}",
        f"Interaction matrix shape: {interactions.shape}",
        f"Neighborhood size: {TOP_NEIGHBORS}",
        "",
        "Method:",
        "1. Build binary training trial-keyword interactions.",
        "2. Hide some recorded keywords for multi-keyword test trials.",
        "3. Compare visible target keywords with training trials.",
        "4. Rank unseen candidate keywords by neighbor similarity votes.",
        "",
        "Caution:",
        "This output is for inspecting recommendations.",
        "Use a separate fixed evaluation set for model comparisons.",
        "One-keyword test trials cannot provide a hidden-label evaluation.",
    ]

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print("\nCollaborative-filtering recommendations generated!")
    print(f"Keyword-labelled trials: {len(eligible)}")
    print(f"Training trials: {len(train)}")
    print(f"Test trials: {len(test)}")
    print(f"Training vocabulary: {len(vocabulary)}")
    print(f"Interaction matrix shape: {interactions.shape}")
    print(f"\nRecommendations saved to: {OUTPUT_PATH}")
    print(f"Report saved to: {REPORT_PATH}")

    print("\nFirst 5 recommendation examples:")
    for _, row in output.head(5).iterrows():
        print(f"\nTrial: {row['nct_id']} - {row['title']}")
        print(f"Visible keywords: {row['visible_keywords']}")
        print(f"Neighbors found: {row['neighbor_count']}")
        print(
            "Recommended keywords:",
            row["recommended_keywords"] or "(no recommendations)"
        )


if __name__ == "__main__":
    main()
