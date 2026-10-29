
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "clinical_trials_cleaned.csv"
RECOMMENDATIONS_PATH = ROOT / "results" / "content_based_recommendations.csv"
IMPROVED_BM25_PATH = ROOT / "results" / "improved_bm25_recommendations.csv"
MODEL_RESULTS_PATH = ROOT / "results" / "model_comparison_summary.csv"
BM25_EVAL_PATH = ROOT / "results" / "bm25_evaluation.csv"
IMPROVED_BM25_EVAL_PATH = ROOT / "results" / "improved_bm25_evaluation.csv"
BM25_REPORT_PATH = ROOT / "results" / "bm25_evaluation_report.txt"
IMPROVED_BM25_REPORT_PATH = ROOT / "results" / "improved_bm25_evaluation_report.txt"
RESULTS_DIR = ROOT / "results"

st.set_page_config(
    page_title="Clinical Trial Keyword Recommendation",
    page_icon="🧬",
    layout="wide",
)

@st.cache_data
def load_csv(path):
    if path.exists():
        return pd.read_csv(path).fillna("")
    return pd.DataFrame()

@st.cache_data
def load_text(path):
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""

st.title("🧬 Clinical Trial Keyword Recommendation")
st.caption(
    "Machine learning research prototype using ClinicalTrials.gov records"
)

if not DATA_PATH.exists():
    st.error(f"Clinical trial dataset not found: {DATA_PATH}")
    st.stop()

df = load_csv(DATA_PATH)
recommendations = load_csv(RECOMMENDATIONS_PATH)
improved_bm25 = load_csv(IMPROVED_BM25_PATH)
model_results = load_csv(MODEL_RESULTS_PATH)
bm25_eval = load_csv(BM25_EVAL_PATH)
improved_bm25_eval = load_csv(IMPROVED_BM25_EVAL_PATH)

overview, explorer, recommender, improved_recommender, evaluation = st.tabs([
    "🏠 Overview",
    "🔎 Trial Explorer",
    "🧠 Content-Based Recommendations",
    "✨ Improved BM25",
    "📊 Model Evaluation",
])

# ---------------------------------------------------------
# 1. OVERVIEW
# ---------------------------------------------------------
with overview:
    st.header("Project Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Clinical Trial Records", f"{len(df):,}")

    if "original_keywords" in df.columns:
        keyword_count = (
            df["original_keywords"].astype(str).str.strip().ne("").sum()
        )
    elif "keywords" in df.columns:
        keyword_count = df["keywords"].astype(str).str.strip().ne("").sum()
    else:
        keyword_count = 0

    c2.metric("Records with Keywords", f"{keyword_count:,}")
    c3.metric("Recommendation Methods", "5")

    st.subheader("About the Project")
    st.write(
        "This project explores keyword recommendation for clinical trial "
        "records using frequency-based, content-based, collaborative, hybrid, "
        "and BM25-based approaches."
    )

    st.subheader("Improved BM25")
    st.write(
        "The improved BM25 experiment combines text similarity with a "
        "smoothed inverse keyword-frequency weight. This is intended to "
        "give more importance to keywords found in fewer training trials."
    )

    st.info(
        "Educational research prototype only. Recommendations are exploratory "
        "and must not be used for medical decisions."
    )

# ---------------------------------------------------------
# 2. TRIAL EXPLORER
# ---------------------------------------------------------
with explorer:
    st.header("🔎 Explore Clinical Trials")

    search = st.text_input(
        "Search by trial ID, title, condition, or keyword",
        placeholder="For example: asthma, vitamin D, or NCT07733635",
        key="trial_search",
    )

    filtered = df.copy()

    if search.strip():
        searchable_columns = [
            col for col in [
                "nct_id", "title", "conditions", "keywords",
                "original_keywords", "brief_summary"
            ]
            if col in filtered.columns
        ]

        mask = pd.Series(False, index=filtered.index)

        for column in searchable_columns:
            mask |= filtered[column].astype(str).str.contains(
                search.strip(),
                case=False,
                na=False,
                regex=False,
            )

        filtered = filtered[mask]

    st.write(f"Matching records: **{len(filtered):,}**")

    display_columns = [
        col for col in [
            "nct_id", "title", "status", "study_type",
            "conditions", "keywords", "original_keywords"
        ]
        if col in filtered.columns
    ]

    st.dataframe(
        filtered[display_columns].head(200),
        use_container_width=True,
        hide_index=True,
    )

    if not filtered.empty and "nct_id" in filtered.columns:
        selected_id = st.selectbox(
            "View a trial's details",
            filtered["nct_id"].astype(str).tolist(),
            key="explorer_trial",
        )

        selected = filtered[
            filtered["nct_id"].astype(str) == selected_id
        ].iloc[0]

        st.subheader(str(selected.get("title", selected_id)))
        st.write("**Trial ID:**", selected_id)
        st.write("**Status:**", selected.get("status", "Not available"))
        st.write("**Study type:**", selected.get("study_type", "Not available"))
        st.write("**Conditions:**", selected.get("conditions", "Not available"))

        recorded = selected.get(
            "original_keywords",
            selected.get("keywords", "None recorded"),
        )
        st.write("**Recorded keywords:**", recorded or "None recorded")
        st.write("**Brief summary:**", selected.get("brief_summary", "Not available"))

# ---------------------------------------------------------
# 3. EXISTING CONTENT-BASED RECOMMENDATIONS
# ---------------------------------------------------------
with recommender:
    st.header("🧠 Content-Based Keyword Recommendations")

    if recommendations.empty:
        st.warning(
            "Content-based recommendations were not found. "
            "Check results/content_based_recommendations.csv."
        )
    elif "nct_id" not in recommendations.columns:
        st.error("The content-based recommendation file has no nct_id column.")
    else:
        st.write(
            "These are precomputed outputs from your existing content-based "
            "recommendation script."
        )

        options = recommendations["nct_id"].astype(str).tolist()

        selected_id = st.selectbox(
            "Choose a clinical trial",
            options,
            key="content_trial",
            format_func=lambda x: (
                str(
                    recommendations.loc[
                        recommendations["nct_id"].astype(str) == x,
                        "title",
                    ].iloc[0]
                )[:110] + f" ({x})"
            ),
        )

        result = recommendations[
            recommendations["nct_id"].astype(str) == selected_id
        ].iloc[0]

        st.subheader(str(result.get("title", selected_id)))
        st.write("**Trial ID:**", selected_id)

        existing = df[df["nct_id"].astype(str) == selected_id]

        if not existing.empty:
            recorded = str(
                existing.iloc[0].get("original_keywords", "")
            ).strip()
            st.write(
                "**Recorded keywords:**",
                recorded if recorded else "None recorded",
            )

        keyword_text = str(
            result.get("recommended_keywords", "")
        ).strip()

        if keyword_text:
            keywords = [
                x.strip() for x in keyword_text.split(";") if x.strip()
            ]
            st.success(f"{len(keywords)} precomputed keyword recommendations")
            st.write(", ".join(f"**{keyword}**" for keyword in keywords))
        else:
            st.warning("No recommendations were saved for this trial.")

        similar_ids = str(result.get("similar_trial_ids", "")).split(";")
        scores = str(result.get("similarity_scores", "")).split(";")
        similar_rows = []

        for trial_id, score in zip(similar_ids, scores):
            trial_id = trial_id.strip()
            if trial_id:
                similar_rows.append({
                    "Similar trial ID": trial_id,
                    "Similarity score": score.strip(),
                })

        if similar_rows:
            st.subheader("Similar clinical trials used")
            st.dataframe(
                pd.DataFrame(similar_rows),
                use_container_width=True,
                hide_index=True,
            )

        st.warning(
            "These recommendations are exploratory. They were generated "
            "from the collected dataset and are not independent evaluation results."
        )

# ---------------------------------------------------------
# 4. IMPROVED BM25 RECOMMENDATIONS
# ---------------------------------------------------------
with improved_recommender:
    st.header("✨ Improved BM25 Keyword Recommendations")

    st.write(
        "Improved BM25 combines BM25 text similarity with smoothed inverse "
        "keyword frequency. It gives more weight to keywords that appear "
        "in fewer trials."
    )

    if improved_bm25.empty:
        st.warning(
            "Improved BM25 recommendations were not found. Run "
            "src/improved_bm25_recommender.py first."
        )
    elif "nct_id" not in improved_bm25.columns:
        st.error("The Improved BM25 file has no nct_id column.")
    else:
        options = improved_bm25["nct_id"].astype(str).tolist()

        selected_id = st.selectbox(
            "Choose a clinical trial",
            options,
            key="improved_bm25_trial",
            format_func=lambda x: (
                str(
                    improved_bm25.loc[
                        improved_bm25["nct_id"].astype(str) == x,
                        "title",
                    ].iloc[0]
                )[:110] + f" ({x})"
            ),
        )

        result = improved_bm25[
            improved_bm25["nct_id"].astype(str) == selected_id
        ].iloc[0]

        st.subheader(str(result.get("title", selected_id)))
        st.write("**Trial ID:**", selected_id)

        recorded = str(result.get("existing_keywords", "")).strip()
        st.write(
            "**Recorded keywords:**",
            recorded if recorded else "None recorded",
        )

        keyword_text = str(
            result.get("recommended_keywords", "")
        ).strip()

        if keyword_text and keyword_text.lower() != "nan":
            keywords = [
                keyword.strip()
                for keyword in keyword_text.split(";")
                if keyword.strip()
            ]
            st.success(f"{len(keywords)} recommended keywords")
            st.write(", ".join(f"**{keyword}**" for keyword in keywords))
        else:
            st.info("No new keywords were recommended for this trial.")

        score_text = str(
            result.get("recommendation_scores", "")
        ).strip()

        if score_text and score_text.lower() != "nan":
            st.caption(
                "Recommendation scores in the saved output; "
                "higher scores indicate higher ranking within this model."
            )
            st.code(score_text)

        st.warning(
            "These suggestions are exploratory, not medical advice. "
            "This recommendation file was generated using the collected "
            "dataset and is separate from the held-out evaluation results."
        )

# ---------------------------------------------------------
# 5. MODEL EVALUATION
# ---------------------------------------------------------
with evaluation:
    st.header("📊 Model Evaluation")

    st.write(
        "These metrics measure recovery of recorded keywords in held-out "
        "test trials. They do not establish clinical relevance."
    )

    if not model_results.empty:
        st.subheader("Existing Model Comparison")
        st.dataframe(
            model_results,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("The existing model comparison summary CSV was not found or is empty.")

    st.subheader("BM25 vs Improved BM25")

    comparison_rows = []

    evaluation_files = [
        ("Original BM25", bm25_eval),
        ("Improved BM25", improved_bm25_eval),
    ]

    metric_columns = [
        "precision_at_5",
        "recall_at_5",
        "ndcg_at_5",
    ]

    metric_labels = {
        "precision_at_5": "Precision@5",
        "recall_at_5": "Recall@5",
        "ndcg_at_5": "NDCG@5",
    }

    for model_name, details in evaluation_files:
        if not details.empty:
            available = [
                col for col in metric_columns if col in details.columns
            ]

            if available:
                row = {"Model": model_name}

                for metric in available:
                    row[metric_labels[metric]] = pd.to_numeric(
                        details[metric], errors="coerce"
                    ).mean()

                comparison_rows.append(row)

    if comparison_rows:
        comparison = pd.DataFrame(comparison_rows)
        st.dataframe(
            comparison.style.format(
                {label: "{:.4f}" for label in metric_labels.values()}
            ),
            use_container_width=True,
            hide_index=True,
        )

        chart_columns = [
            label for label in metric_labels.values()
            if label in comparison.columns
        ]

        if chart_columns:
            st.subheader("Metric Comparison")
            chart_data = comparison.set_index("Model")[chart_columns]
            st.bar_chart(chart_data)
    else:
        st.info(
            "BM25 evaluation files were not found. Run the evaluation scripts "
            "to generate the comparison."
        )

    with st.expander("View Original BM25 Evaluation Report"):
        report = load_text(BM25_REPORT_PATH)
        if report:
            st.text(report)
        else:
            st.info("Original BM25 report not found.")

    with st.expander("View Improved BM25 Evaluation Report"):
        report = load_text(IMPROVED_BM25_REPORT_PATH)
        if report:
            st.text(report)
        else:
            st.info("Improved BM25 report not found.")

    graph_files = [
        ("Existing Precision@5 Comparison", "precision_at_5_comparison.png"),
        ("Existing Recall@5 Comparison", "recall_at_5_comparison.png"),
        ("Existing NDCG@5 Comparison", "ndcg_at_5_comparison.png"),
    ]

    for label, filename in graph_files:
        image_path = RESULTS_DIR / filename
        if image_path.exists():
            st.subheader(label)
            st.image(str(image_path), use_container_width=True)

    st.caption(
        "BM25 evaluation uses a held-out test set. Improved BM25's rarity "
        "weights are calculated from training trials only in the evaluation "
        "script. Small metric differences should not be interpreted as proof "
        "of a meaningful improvement without further testing."
    )
