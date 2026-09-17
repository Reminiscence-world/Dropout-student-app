"""
Ticket 10: pages/2_Student_Explorer.py — drill into one student.

Shows, for a single selected student: their predicted risk tier and
probability, their top SHAP-explained factors (chart + plain text), and
any intervention suggestion the rule table (Ticket 8) fires for them.

Same cohort convention as the Dashboard (Ticket 9): students shown here
are from the held-out test set, since that's the only group with
out-of-sample predictions.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import load_and_clean_data
from src.model import train_model
from src.explain import build_explainer, get_shap_values_for_test_set, explain_student, explanation_to_text
from src.risk_tiers import assign_risk_tiers, TIER_CAVEAT
from src.interventions import suggest_interventions

st.markdown("""
<div style="
padding:25px;
background:linear-gradient(135deg,#1E2A4A,#2E73B8);
border-radius:20px;
margin-bottom:20px;
">
<h1 style="color:white;">🎓 Student Explorer</h1>
<p style="color:white;">
Drill into one student from the demo cohort:
their predicted risk, explanation and intervention suggestions.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Main app background */
.stApp {
    background-color: #F4F8FC;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #DCE6F2;
}

/* Headers */
h1, h2, h3 {
    color: #1E2A4A !important;
}

/* Normal text */
p, label, div {
    color: #1E2A4A;
}

/* Selectboxes */
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
""", unsafe_allow_html=True)

st.markdown("""
<div style="
padding:25px;
background:linear-gradient(135deg,#1E2A4A,#2E73B8);
border-radius:20px;
margin-bottom:20px;
">
<h1 style="color:white;">🎓 Student Explorer</h1>
<p style="color:white;">
Drill into one student from the demo cohort:
their predicted risk, explanation and intervention suggestions.
</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Drill into one student from the demo cohort: their predicted risk, "
    "why the model flagged them, and any suggested intervention."
)

df = load_and_clean_data()
results = train_model(df)
tiers, cutoffs = assign_risk_tiers(results["y_pred_proba"])

explainer = build_explainer(results["model"], results["X_train"])
shap_df = get_shap_values_for_test_set(results["model"], explainer, results["X_test"])

X_test = results["X_test"]
student_ids = X_test.index.tolist()

# --- Filter + select ---
tier_filter = st.selectbox("Filter by risk tier", options=["All", "Low", "Medium", "High"])

if tier_filter == "All":
    visible_ids = student_ids
else:
    mask = tiers.values == tier_filter
    visible_ids = [sid for sid, keep in zip(student_ids, mask) if keep]

if not visible_ids:
    st.warning("No students match that filter.")
    st.stop()

selected_id = st.selectbox(
    "Select a student (type to search by ID)",
    options=visible_ids,
    format_func=lambda x: f"Student ID {x}",
)

position = X_test.index.get_loc(selected_id)

# --- Headline for this student ---
tier = tiers.iloc[position]
proba = results["y_pred_proba"][position]
actual = results["y_test"].iloc[position]

c1, c2, c3 = st.columns(3)
c1.metric("Risk tier", tier)
c2.metric("Predicted P(Dropout)", f"{proba:.2f}")
c3.metric("Actual outcome (this dataset)", "Dropout" if actual == 1 else "Not Dropout")
st.caption(TIER_CAVEAT)

st.divider()

# --- SHAP explanation ---
st.subheader("Why the model flagged this student")
st.caption(
    "This is a property of the model's prediction, not a causal claim — "
    "see the Causal Insights page for confounder-adjusted associations."
)

# 1. Full explanation across all features for intervention checks
full_explanation = explain_student(position, X_test, shap_df, top_n=len(X_test.columns))

# 2. Top-5 explanation strictly for clean visual chart display
top5_explanation = explain_student(position, X_test, shap_df, top_n=5)
explanation_df = pd.DataFrame(top5_explanation)
explanation_df["abs_shap"] = explanation_df["shap_value"].abs()
explanation_df = explanation_df.sort_values("abs_shap")

fig = px.bar(
    explanation_df,
    x="shap_value",
    y="feature",
    orientation="h",
    color="direction",
    color_discrete_map={"increases risk": "#2E73B8","decreases risk": "#5EA4F3"},
    title="Top contributing factors (SHAP)",
    labels={"shap_value": "SHAP value (positive = pushes toward Dropout)"},
)
fig.update_layout(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#F8FBFF",
    margin=dict(l=20, r=20, t=60, b=20),
    
    font=dict(
        color="black",
        size=14
    ),

    title_font=dict(
        color="black",
        size=18
    ),

    xaxis=dict(
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        gridcolor="#E6EEF8"
    ),

    yaxis=dict(
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        gridcolor="#E6EEF8"
    ),

    legend=dict(
        font=dict(color="black")
    )
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(explanation_to_text(top5_explanation))

st.divider()

# --- Intervention suggestion ---
st.subheader("Suggested intervention")

student_row = X_test.iloc[position]
suggestions = suggest_interventions(full_explanation, tier, student_row=student_row)

if suggestions:
    for s in suggestions:
        st.markdown(f"- {s}")
else:
    st.markdown("No actionable, risk-increasing factor among this student's top contributors.")
