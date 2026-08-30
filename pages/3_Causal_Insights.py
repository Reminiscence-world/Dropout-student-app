"""
Ticket 11: pages/3_Causal_Insights.py — confounder-adjusted associations.

This page answers a DIFFERENT question than the SHAP explanations on
Student Explorer: not "why did the model flag THIS student" but "among
students who otherwise look similar on measured confounders, is this
actionable factor associated with a different dropout rate, across the
whole cohort?" Deliberately kept visually and verbally separate from
SHAP language — explicit comparison callout, distinct chart style, and
a standalone warning-style caveat — so nobody mistakes one for the
other or reads either as a randomized-experiment causal effect.
"""

import streamlit as st
import plotly.graph_objects as go

from src.data_loader import load_and_clean_data, build_binary_target
from src.causal import run_causal_adjustment, result_to_text, ASSUMPTION_CAVEAT
from src.variable_roles import CONFOUNDER

st.set_page_config(page_title="Causal Insights", layout="wide")

st.title("Causal Insights")
st.caption(
    "Confounder-adjusted associations for the 3 actionable variables — a "
    "cohort-level question, distinct from the per-student SHAP explanations "
    "on Student Explorer."
)

st.info(
    "**SHAP (Student Explorer) vs. Causal Adjustment (this page)**\n\n"
    "- **SHAP** answers: *\"why did the model flag THIS student?\"* — a "
    "property of the trained model's prediction for one person.\n"
    "- **Causal adjustment** answers: *\"among students who otherwise look "
    "similar on measured confounders, is this actionable factor associated "
    "with a different dropout rate, across the whole cohort?\"*\n\n"
    "Neither is a randomized-experiment causal effect. The estimates below "
    "specifically assume the confounder set captures the important "
    "differences between students."
)

df = load_and_clean_data()
df = build_binary_target(df)
causal_results = run_causal_adjustment(df)

st.divider()
st.subheader("Confounder-adjusted associations")
st.caption(
    f"Adjusted for: {', '.join(CONFOUNDER)}. Never computed for mediators "
    "(1st/2nd semester grades, units approved — too close to the outcome to "
    "support a causal claim) or fixed/demographic variables (never the "
    "'reason' for a recommendation)."
)

# --- Chart: adjusted coefficients with 95% confidence intervals ---
labels = list(causal_results.keys())
coefs = [causal_results[k]["coefficient"] * 100 for k in labels]
ci_low = [causal_results[k]["conf_int_low"] * 100 for k in labels]
ci_high = [causal_results[k]["conf_int_high"] * 100 for k in labels]
errors_plus = [h - c for h, c in zip(ci_high, coefs)]
errors_minus = [c - l for c, l in zip(coefs, ci_low)]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=coefs,
    y=labels,
    orientation="h",
    error_x=dict(type="data", symmetric=False, array=errors_plus, arrayminus=errors_minus),
    marker_color=["#d62728" if c > 0 else "#2ca02c" for c in coefs],
))
fig.update_layout(
    title="Adjusted association with dropout probability (percentage points, 95% CI)",
    xaxis_title="Percentage-point change in dropout probability",
    yaxis_title="",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

for actionable_var, result in causal_results.items():
    st.markdown(f"### {actionable_var}")
    st.write(result_to_text(actionable_var, result))
    st.markdown("")

st.divider()
st.warning(f"**Assumption behind every number on this page:** {ASSUMPTION_CAVEAT}")
