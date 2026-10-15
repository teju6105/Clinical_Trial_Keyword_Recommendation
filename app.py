from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "clinical_trials_cleaned.csv"
RECOMMENDATIONS_PATH = ROOT / "results" / "content_based_recommendations.csv"
MODEL_RESULTS_PATH = ROOT / "results" / "model_comparison_summary.csv"
RESULTS_DIR = ROOT / "results"

st.set_page_config(
    page_title="Clinical Trial Keyword Recommendation",
    page_icon="🧬",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH).fillna("")

@st.cache_data
def load_recommendations():
    return pd.read_csv(RECOMMENDATIONS_PATH).fillna("")

@st.cache_data
def load_model_results():
    if MODEL_RESULTS_PATH.exists():
        return pd.read_csv(MODEL_RESULTS_PATH)
    return pd.DataFrame()

st.title("🧬 Clinical Trial Keyword Recommendation")
st.caption("Machine learning research prototype using ClinicalTrials.gov records")

if not DATA_PATH.exists() or not RECOMMENDATIONS_PATH.exists():
    st.error("Required project data files were not found. Check the data and results folders.")
    st.stop()

df = load_data()
recommendations = load_recommendations()
model_results = load_model_results()

overview, explorer, recommender, evaluation = st.tabs([
    "🏠 Overview",
    "🔎 Trial Explorer",
    "🧠 Keyword Recommendations",
    "📊 Model Evaluation"
])

with overview:
    st.header("Project Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Clinical Trial Records", f"{len(df):,}")
    c2.metric(
        "Records with Keywords",
        f"{(df['original_keywords'].astype(str).str.strip() != '').sum():,}"
        if "original_keywords" in df.columns else "Unavailable"
    )
    c3.metric("Recommendation Methods", "4")

    st.subheader("About the Project")
    st.write(
        "This project explores keyword recommendation for clinical trial records "
        "using frequency-based, content-based, collaborative, and hybrid approaches."
    )

    st.info(
        "Educational research prototype. Recommendations are exploratory and "
        "must not be used for medical decisions."
    )

with explorer:
    st.header("🔎 Explore Clinical Trials")

    search = st.text_input(
        "Search by trial ID, title, condition, or keyword",
        placeholder="For example: asthma, vitamin D, or NCT07733635"
    )

    filtered = df.copy()
    if search.strip():
        mask = pd.Series(False, index=filtered.index)
        for column in ["nct_id", "title", "conditions", "keywords"]:
            if column in filtered.columns:
                mask |= filtered[column].astype(str).str.contains(
                    search.strip(), case=False, na=False, regex=False
                )
        filtered = filtered[mask]

    st.write(f"Matching records: **{len(filtered):,}**")

    display_columns = [
        col for col in [
            "nct_id", "title", "status", "study_type",
            "conditions", "keywords"
        ] if col in filtered.columns
    ]

    st.dataframe(
        filtered[display_columns].head(200),
        use_container_width=True,
        hide_index=True
    )

    if not filtered.empty:
        selected_id = st.selectbox(
            "View a trial's details",
            filtered["nct_id"].astype(str).tolist()
        )
        selected = filtered[filtered["nct_id"].astype(str) == selected_id].iloc[0]

        st.subheader(str(selected.get("title", selected_id)))
        st.write("**Trial ID:**", selected_id)
        st.write("**Status:**", selected.get("status", "Not available"))
        st.write("**Study type:**", selected.get("study_type", "Not available"))
        st.write("**Conditions:**", selected.get("conditions", "Not available"))
        st.write("**Recorded keywords:**", selected.get("keywords", "None recorded"))
        st.write("**Brief summary:**", selected.get("brief_summary", "Not available"))

with recommender:
    st.header("🧠 Content-Based Keyword Recommendations")
    st.write(
        "These are precomputed outputs from your existing content-based "
        "recommendation script."
    )

    options = recommendations["nct_id"].astype(str).tolist()
    selected_id = st.selectbox(
        "Choose a clinical trial",
        options,
        format_func=lambda x: (
            str(recommendations.loc[
                recommendations["nct_id"].astype(str) == x, "title"
            ].iloc[0])[:110] + f" ({x})"
        )
    )

    result = recommendations[
        recommendations["nct_id"].astype(str) == selected_id
    ].iloc[0]

    st.subheader(str(result["title"]))
    st.write("**Trial ID:**", selected_id)

    existing = df[df["nct_id"].astype(str) == selected_id]
    if not existing.empty:
        recorded = str(existing.iloc[0].get("original_keywords", "")).strip()
        st.write("**Recorded keywords:**", recorded if recorded else "None recorded")

    keyword_text = str(result.get("recommended_keywords", "")).strip()
    if keyword_text:
        keywords = [x.strip() for x in keyword_text.split(";") if x.strip()]
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
                "Similarity score": score.strip()
            })

    if similar_rows:
        st.subheader("Similar clinical trials used")
        st.dataframe(
            pd.DataFrame(similar_rows),
            use_container_width=True,
            hide_index=True
        )

    st.warning(
        "Recommendations are exploratory. The saved recommendations were "
        "generated by fitting the content-based model on the collected dataset, "
        "so they are not independent evaluation results."
    )

with evaluation:
    st.header("📊 Model Evaluation")
    st.write("Saved evaluation results from your project.")

    if not model_results.empty:
        st.subheader("Model comparison")
        st.dataframe(model_results, use_container_width=True, hide_index=True)
    else:
        st.info("The model comparison summary CSV was not found or is empty.")

    graph_files = [
        ("Precision@5", "precision_at_5_comparison.png"),
        ("Recall@5", "recall_at_5_comparison.png"),
        ("NDCG@5", "ndcg_at_5_comparison.png"),
    ]

    for label, filename in graph_files:
        image_path = RESULTS_DIR / filename
        if image_path.exists():
            st.subheader(label)
            st.image(str(image_path), use_container_width=True)
        else:
            st.info(f"{filename} was not found.")

    st.caption(
        "Evaluation metrics measure recovery of recorded keywords in the "
        "project's evaluation setup; they do not establish clinical relevance."
    )
