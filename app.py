"""
Ticket 14 (Polish Pass): app.py is now the app's landing page.

The Ticket 1-8 "does this backend module work" checks that used to live
here have all been superseded by dedicated pages — Dashboard, Student
Explorer, Causal Insights, and Model Evaluation each demonstrate the
same functionality live. (That old scratch content still exists at
pages/_Debug.py, hidden from the sidebar but reachable directly if you
ever want to re-run it.) This page's only job now is to be a clean
opener: the problem statement, plus a way into the rest of the app.
"""

import streamlit as st
from src.theme import APP_TITLE, APP_ICON, PAGE_ICONS


st.set_page_config(page_title="Home", page_icon=PAGE_ICONS["home"], layout="wide")

st.title(f"{APP_ICON} {APP_TITLE}")

st.markdown(
    "University advisors don't have a systematic way to know **which** "
    "students are at risk of dropping out early, or **what kind of "
    "support** to offer them. This tool is a decision-support prototype "
    "that, for each student: predicts a dropout-risk score, explains why "
    "the model flagged them, estimates how a small set of **actionable** "
    "factors relate to dropout (under stated assumptions), and suggests a "
    "fitting intervention — all in one local dashboard."
)

st.info(
    "Built on the UCI **\"Predict Students' Dropout and Academic Success\"** "
    "dataset (~4,400 students, one Portuguese institution). See **About & "
    "Limitations** for what this tool is — and isn't — before trusting any "
    "number in it."
)

st.divider()
st.subheader("Get started")

col1, col2, col3 = st.columns(3)
with col1:
    st.page_link("pages/1_Dashboard.py", label="Dashboard", icon=PAGE_ICONS["dashboard"])
    st.caption("Cohort-level risk overview.")
with col2:
    st.page_link("pages/2_Student_Explorer.py", label="Student Explorer", icon=PAGE_ICONS["student_explorer"])
    st.caption("Drill into one student's risk, explanation, and suggested intervention.")
with col3:
    st.page_link("pages/3_Causal_Insights.py", label="Causal Insights", icon=PAGE_ICONS["causal_insights"])
    st.caption("Confounder-adjusted associations for the actionable factors.")

col4, col5, col6 = st.columns(3)
with col4:
    st.page_link("pages/4_Model_Evaluation.py", label="Model Evaluation", icon=PAGE_ICONS["model_evaluation"])
    st.caption("How the model was chosen, and how well it performs.")
with col5:
    st.page_link("pages/5_About_Limitations.py", label="About & Limitations", icon=PAGE_ICONS["about"])
    st.caption("What this tool is, and isn't.")
with col6:
    st.page_link("pages/6_What_If_Simulator.py", label="What-If Simulator", icon=PAGE_ICONS["what_if"])
    st.caption("Edit a student's inputs and see how the prediction changes (stretch goal).")

