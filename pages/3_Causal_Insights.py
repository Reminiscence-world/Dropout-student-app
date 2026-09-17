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

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )
    
st.set_page_config(page_title="Causal Insights", layout="wide")

st.markdown("""
<style>

/* Sidebar gradient */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #1E2A4A 0%,
        #243A73 100%
    ) !important;
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

/* Navigation items */
[data-testid="stSidebarNav"] span {
    color: white !important;
}

/* Selected page */
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background-color: rgba(255,255,255,0.15) !important;
    border-radius: 10px;
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
<h1 style="color:white;">📊 Causal Insights</h1>
<p style="color:white;">
Confounder-adjusted associations across the cohort.
This is a cohort-level analysis and is distinct from the individual SHAP explanations.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
background:white;
padding:20px;
border-radius:15px;
border-left:5px solid #2E73B8;
box-shadow:0px 2px 6px rgba(0,0,0,0.05);
">

<b>SHAP (Student Explorer) vs. Causal Adjustment (this page)</b>

<br><br>

• <b>SHAP</b> answers:
<i>"why did the model flag THIS student?"</i>

<br><br>

• <b>Causal adjustment</b> answers:
<i>"among students who otherwise look similar on measured confounders, is this actionable factor associated with a different dropout rate?"</i>

</div>
""", unsafe_allow_html=True)

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
fig.update_layout(
    title="Adjusted association with dropout probability (percentage points, 95% CI)",
    xaxis_title="Percentage-point change in dropout probability",
    yaxis_title="",

    paper_bgcolor="white",
    plot_bgcolor="white",

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
        title_font=dict(color="black")
    ),

    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    )
)
fig.add_trace(go.Bar(
    x=coefs,
    y=labels,
    orientation="h",
    error_x=dict(type="data", symmetric=False, array=errors_plus, arrayminus=errors_minus),
    marker_color=["#2E73B8" if c > 0 else "#5EA4F3" for c in coefs],
))
st.plotly_chart(fig, use_container_width=True)

st.divider()

for actionable_var, result in causal_results.items():
    st.markdown(f"""
    <div style="
    background:white;
    padding:18px;
    border-radius:15px;
    margin-bottom:15px;
    box-shadow:0px 2px 6px rgba(0,0,0,0.05);
    ">
    <h3 style="color:#1E2A4A;">
        {actionable_var}
    </h3>
    <p style="color:black;">
        {result_to_text(actionable_var, result)}
    </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown(f"""
<div style="
background:#E7F0FB;
padding:20px;
border-radius:15px;
border-left:5px solid #2E73B8;
margin-top:15px;
">
<b>Assumption behind every number on this page:</b><br><br>
{ASSUMPTION_CAVEAT}
</div>
""", unsafe_allow_html=True)
