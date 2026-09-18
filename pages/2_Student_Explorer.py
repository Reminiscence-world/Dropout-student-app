"""
pages/2_Student_Explorer.py — drill into one student.
Supports inspecting either the held-out test cohort or an uploaded batch.
"""

import sys
from pathlib import Path

# Add project root directory to Python path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
from utils import add_sidebar_logo
import pandas as pd
import plotly.express as px

from src.data_loader import load_and_clean_data
from src.model import train_model
from src.explain import (
    build_explainer,
    get_shap_values_for_test_set,
    explain_student,
    explanation_to_text,
)
from src.risk_tiers import assign_risk_tiers, TIER_CAVEAT
from src.interventions import suggest_interventions

add_sidebar_logo()

try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

st.markdown(
    """
<style>
/* Main app background */
.stApp {
    background-color: #DBE8F4;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #1E2A4A 0%,
        #243A73 100%
    );
}

/* Sidebar text */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: white !important;
}

/* Page names */
[data-testid="stSidebarNav"] span {
    color: white !important;
}

[data-testid="stSidebarNav"] a[aria-current="page"] {
    background-color: rgba(255,255,255,0.15) !important;
    border-radius: 10px;
}

/* Headers */
h1, h2, h3 {
    color: #1E2A4A !important;
}

/* Normal text */
p, label, div {
    color: #1E2A4A;
}

/* Selectboxes & Radio */
.stSelectbox > div > div {
    background-color: white;
    color: #1E2A4A;
    border-radius: 10px;
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #DCE6F2;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
}

/* Divider */
hr {
    border-color: #DCE6F2;
}

/* Plotly chart container */
[data-testid="stPlotlyChart"] {
    background-color: white;
    border-radius: 15px;
    padding: 10px;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="
padding:25px;
background:linear-gradient(135deg,#1E2A4A,#2E73B8);
border-radius:20px;
margin-bottom:20px;
">
<h1 style="color:white;">🎓 Student Explorer</h1>
<p style="color:white;">
Drill into one student: inspect predicted risk, SHAP explanations, and intervention suggestions.
</p>
</div>
""",
    unsafe_allow_html=True,
)

# --- Cohort Toggle Switch ---
has_upload = (
    "uploaded_batch" in st.session_state
    and st.session_state["uploaded_batch"] is not None
)

from pathlib import Path

ROSTER_FILE = Path("data/incremental_roster.csv")

# 1. Check if an active upload is in session_state
has_upload = (
    "uploaded_batch" in st.session_state
    and st.session_state["uploaded_batch"] is not None
)

# 2. If not in session_state, check if we have saved data on disk
if not has_upload and ROSTER_FILE.exists():
    saved_df = pd.read_csv(ROSTER_FILE)
    if not saved_df.empty:
        # Separate features from stored targets
        feature_cols = [
            c
            for c in saved_df.columns
            if c not in ["predicted_proba", "risk_tier"]
        ]
        explainer = build_explainer(results["model"], results["X_train"])
        shap_saved_df = get_shap_values_for_test_set(
            results["model"], explainer, saved_df[feature_cols]
        )

        st.session_state["uploaded_batch"] = {
            "features": saved_df[feature_cols],
            "scores": saved_df[["predicted_proba", "risk_tier"]],
            "shap_df": shap_saved_df,
            "verdicts": saved_df,
        }
        has_upload = True
data_source = st.radio(
    "Select Student Population to Inspect:",
    options=(
        ["Test-set cohort", "Uploaded batch"]
        if has_upload
        else ["Test-set cohort"]
    ),
    horizontal=True,
    help=(
        "Switch between the baseline validation cohort and newly uploaded CSV"
        " students."
        if has_upload
        else "Upload a CSV on the Dashboard page to unlock the uploaded batch view."
    ),
)

if not has_upload and data_source == "Test-set cohort":
    st.caption(
        "ℹ️ *Tip: You can upload a batch CSV on the Dashboard to explore custom"
        " student profiles here.*"
    )

# --- Load Selected Population Data ---
if data_source == "Uploaded batch" and has_upload:
    batch = st.session_state["uploaded_batch"]
    X_view = batch["features"]
    scores = batch["scores"]
    shap_df = batch["shap_df"]
    student_ids = X_view.index.tolist()
    tiers = scores["risk_tier"]
    probas = scores["predicted_proba"]
    actuals = None  # Uploaded data has no ground-truth label
else:
    df = load_and_clean_data()
    results = train_model(df)
    tiers, cutoffs = assign_risk_tiers(results["y_pred_proba"])
    explainer = build_explainer(results["model"], results["X_train"])
    shap_df = get_shap_values_for_test_set(
        results["model"], explainer, results["X_test"]
    )
    X_view = results["X_test"]
    student_ids = X_view.index.tolist()
    probas = results["y_pred_proba"]
    actuals = results["y_test"]

# --- Filter + Select ---
col_filter, col_select = st.columns([1, 2])

with col_filter:
    tier_filter = st.selectbox(
        "Filter by risk tier", options=["All", "Low", "Medium", "High"]
    )

if tier_filter == "All":
    visible_ids = student_ids
else:
    mask = [t == tier_filter for t in tiers]
    visible_ids = [sid for sid, keep in zip(student_ids, mask) if keep]

if not visible_ids:
    st.warning("No students match that filter.")
    st.stop()

with col_select:
    selected_id = st.selectbox(
        "Select student record",
        options=visible_ids,
        format_func=lambda x: f"Student ID / Row: {x}",
    )

position = X_view.index.get_loc(selected_id)

# --- Headline Metrics for Selected Student ---
tier_val = tiers.iloc[position] if hasattr(tiers, "iloc") else tiers[position]
proba_val = (
    probas.iloc[position] if hasattr(probas, "iloc") else probas[position]
)

if actuals is not None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk tier", tier_val)
    c2.metric("Predicted P(Dropout)", f"{float(proba_val):.2f}")
    act_val = (
        actuals.iloc[position]
        if hasattr(actuals, "iloc")
        else actuals[position]
    )
    c3.metric(
        "Actual outcome (Ground truth)",
        "Dropout" if act_val == 1 else "Not Dropout",
    )
else:
    c1, c2 = st.columns(2)
    c1.metric("Risk tier", tier_val)
    c2.metric("Predicted P(Dropout)", f"{float(proba_val):.2f}")

st.caption(TIER_CAVEAT)
st.divider()

# --- SHAP Explanation ---
st.subheader("Why the model flagged this student")
st.caption(
    "This is a property of the model's prediction, not a causal claim — see the"
    " Causal Insights page for confounder-adjusted associations."
)

full_explanation = explain_student(
    position, X_view, shap_df, top_n=len(X_view.columns)
)
top5_explanation = explain_student(position, X_view, shap_df, top_n=5)

explanation_df = pd.DataFrame(top5_explanation)
explanation_df["abs_shap"] = explanation_df["shap_value"].abs()
explanation_df = explanation_df.sort_values("abs_shap")

fig = px.bar(
    explanation_df,
    x="shap_value",
    y="feature",
    orientation="h",
    color="direction",
    color_discrete_map={
        "increases risk": "#2E73B8",
        "decreases risk": "#5EA4F3",
    },
    title="Top contributing factors (SHAP)",
    labels={"shap_value": "SHAP value (positive = pushes toward Dropout)"},
)
fig.update_layout(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#F8FBFF",
    margin=dict(l=20, r=20, t=60, b=20),
    font=dict(color="black", size=14),
    title_font=dict(color="black", size=18),
    xaxis=dict(
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        gridcolor="#E6EEF8",
    ),
    yaxis=dict(
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        gridcolor="#E6EEF8",
    ),
    legend=dict(font=dict(color="black")),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(explanation_to_text(top5_explanation))
st.divider()

# --- Intervention Suggestion ---
st.subheader("Suggested intervention")

student_row = X_view.iloc[position]
suggestions = suggest_interventions(
    full_explanation, tier_val, student_row=student_row
)

if suggestions:
    for s in suggestions:
        st.markdown(f"- {s}")
else:
    st.markdown(
        "No actionable, risk-increasing factor among this student's top"
        " contributors."
    )