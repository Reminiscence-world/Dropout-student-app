
"""
Ticket 9: pages/1_Dashboard.py — cohort-level overview.

Design note: the "cohort" shown on this page (and on Student Explorer /
Causal Insights in later tickets) is the held-out TEST SET from
model.py's train/test split — that's the only group of students we
have out-of-sample predicted probabilities for. There's no live roster
in this prototype, just the static UCI snapshot (see the About &
Limitations page — Ticket 13 — for why: Target is an end-of-program
snapshot, not real-time tracking). In a real deployment this page would
run the trained model over the current active student list instead.
"""

import streamlit as st
import plotly.express as px

from src.data_loader import load_and_clean_data
from src.model import train_model
from src.risk_tiers import assign_risk_tiers, tier_counts, TIER_CAVEAT

def load_css():
    with open("style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

st.set_page_config(page_title="Dashboard", layout="wide")

st.markdown("""
<div style='
padding:30px;
background:linear-gradient(135deg,#1E2A4A,#2E73B8);
border-radius:20px;
color:white;
margin-bottom:20px;
'>

<h1 style='color:white;'>📊 Cohort Dashboard</h1>

<p style='color:white;'>
Predicted dropout risk across the demo cohort.
Monitor students, identify risks early,
and improve retention outcomes.
</p>

</div>
""", unsafe_allow_html=True)

st.caption(
    "Predicted dropout risk across the demo cohort (this app's held-out "
    "test set). See 'About & Limitations' for why there's no live roster "
    "in this prototype."
)

df = load_and_clean_data()
results = train_model(df)
tiers, cutoffs = assign_risk_tiers(results["y_pred_proba"])

cohort = results["X_test"].reset_index(drop=True).copy()
cohort["predicted_proba"] = results["y_pred_proba"]
cohort["risk_tier"] = tiers.values
cohort["actual_dropout"] = results["y_test"].reset_index(drop=True)

# --- Headline counts ---
n_students = len(cohort)
n_high_risk = int((cohort["risk_tier"] == "High").sum())
avg_proba = cohort["predicted_proba"].mean()
actual_dropout_rate = cohort["actual_dropout"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Students in cohort", n_students)
c2.metric("High risk tier", n_high_risk, f"{n_high_risk / n_students:.0%} of cohort")
c3.metric("Avg. predicted P(Dropout)", f"{avg_proba:.2f}")
c4.metric("Actual dropout rate", f"{actual_dropout_rate:.0%}")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.info("🎯 AI-Powered Risk Detection")

with col2:
    st.success("📈 Retention Analytics")

with col3:
    st.warning("⚡ Early Intervention Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Risk tier distribution")
    st.caption(TIER_CAVEAT)
    counts_df = tier_counts(tiers)
    
    fig_tiers = px.bar(
        counts_df,
        x="tier",
        y="count",
        color="tier",
        color_discrete_sequence=[
            "#5EA4F3",
            "#2E73B8",
            "#1E2A4A"
        ]
    )

    st.plotly_chart(fig_tiers, use_container_width=True)

with col2:
    st.subheader("Average predicted risk by course")
    by_course = (
        cohort.groupby("course")["predicted_proba"]
        .mean()
        .reset_index()
        .sort_values("predicted_proba", ascending=False)
    )
    by_course["course"] = by_course["course"].astype(str)
    fig_course = px.bar(
        by_course,
        x="course",
        y="predicted_proba",
        color="predicted_proba",
        color_continuous_scale=[
            "#8BC34A",
            "#4CAF50",
            "#0F3D2E"
    ]
    )

    st.plotly_chart(fig_course, use_container_width=True)


