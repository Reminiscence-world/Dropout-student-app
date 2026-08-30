"""
Ticket 13: pages/5_About_Limitations.py — disclosures panel.

Static content page: the mandatory disclosures from spec section 12,
plus the fairness/privacy notes from section 13, written in plain
language for an advisor (not a data scientist) audience.
"""

import streamlit as st

st.set_page_config(page_title="About & Limitations", layout="wide")

st.title("About the Data & Model")
st.caption(
    "Read this before trusting any number elsewhere in this app. This page "
    "exists so nothing here is oversold."
)

st.divider()
st.subheader("Where the data comes from")
st.markdown(
    "- Source: the UCI **\"Predict Students' Dropout and Academic Success\"** "
    "dataset — roughly 4,400 students, all from **one institution in "
    "Portugal**. Results here may not generalize to other institutions, "
    "countries, or student populations.\n"
    "- This is **observational data**, not a randomized experiment. Nobody "
    "was randomly assigned to, say, be a scholarship holder — students "
    "self-selected or qualified into these situations for reasons the "
    "dataset doesn't fully capture.\n"
    "- The outcome (**Target**) is a **single end-of-program snapshot** — "
    "Dropout, Enrolled, or Graduate — recorded once, not a real-time feed "
    "tracking students during the semester. This app cannot tell you "
    "*when* during a term a student's risk changed."
)

st.divider()
st.subheader("What the model's outputs do — and don't — mean")
st.markdown(
    "- **Predicted risk** (Dashboard, Student Explorer) is exactly that — a "
    "prediction from a trained statistical model. It is not a diagnosis "
    "and not a guarantee.\n"
    "- **Risk tiers** (Low / Medium / High) are **model-based, illustrative "
    "thresholds** — tertiles of this test set's predicted probabilities. "
    "They are **not official, clinically validated, or institutionally "
    "approved** risk categories.\n"
    "- **SHAP explanations** (Student Explorer) describe why the *model* "
    "produced a given prediction for a given student. They are **not** a "
    "causal claim about what would happen if something about that student "
    "changed.\n"
    "- **Causal-adjustment associations** (Causal Insights) are **not proof "
    "of causation**. They assume the confounder set used in the regression "
    "captures the major differences between students — if an important "
    "confounder was left unmeasured, the adjusted association could still "
    "be biased. Treat these as suggestive associations, not effect sizes "
    "you can act on with certainty.\n"
    "- **Intervention suggestions** (Student Explorer) are phrased as "
    "*\"suggested based on this student's risk-associated factors\"* — "
    "never as a promise that acting on them will reduce a specific "
    "student's risk by any amount."
)

st.divider()
st.subheader("Fairness & privacy")
st.markdown(
    "- Demographic and fixed variables (gender, nationality, marital "
    "status, parents' occupation, etc.) may help the model predict more "
    "accurately, but they **never appear as the stated reason** for an "
    "intervention suggestion — only the 3 actionable variables (tuition "
    "status, scholarship status, debtor status) can trigger a "
    "suggestion.\n"
    "- This prototype has **no authentication or access control** — "
    "acceptable here because the underlying dataset is already public and "
    "anonymized. A real deployment handling actual student records would "
    "need **FERPA-compliant access controls** and **role-restricted "
    "advisor logins**."
)

st.divider()
st.subheader("What this tool is (and isn't)")
st.markdown(
    "This is a **decision-support prototype** to help an advisor prioritize "
    "outreach — not a production system, not a diagnostic tool, and not a "
    "substitute for an advisor's judgment. It has no database beyond a "
    "static CSV, no login, no multi-institution support, and doesn't write "
    "back to any system of record."
)
