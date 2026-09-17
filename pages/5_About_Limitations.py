"""
Ticket 13: pages/5_About_Limitations.py — disclosures panel.

Static content page: the mandatory disclosures from spec section 12,
plus the fairness/privacy notes from section 13, written in plain
language for an advisor (not a data scientist) audience.
"""

import streamlit as st

st.set_page_config(page_title="About & Limitations", layout="wide")

st.markdown(
    """
    <style>
        /* ---------- App background ---------- */
        .stApp {
            background:
                radial-gradient(circle at 1px 1px,
                    rgba(43, 105, 160, 0.12) 1px,
                    transparent 1px) 0 0 / 28px 28px,
                linear-gradient(180deg, #edf6fc 0%, #f7fbfe 100%);
            color: #172b4d;
        }

        /* ---------- White top header ---------- */
        header[data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.97);
            border-bottom: 1px solid #dce7f1;
        }

        /* ---------- Main content ---------- */
        .block-container {
            max-width: 1180px;
            padding-top: 3rem;
            padding-bottom: 4rem;
        }

        /* ---------- AETiON brand ---------- */
        .aetion-brand {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.7rem;
            font-weight: 700;
            letter-spacing: -0.04em;
            color: #17365d;
            margin-bottom: 2.2rem;
        }

        .aetion-brand span {
            color: #2f6fa3;
        }

        /* ---------- Hero ---------- */
        .hero {
            text-align: center;
            margin: 0 auto 2.5rem auto;
        }

        .hero h1 {
            color: #142b4b;
            font-size: 2.6rem;
            line-height: 1.15;
            font-weight: 750;
            letter-spacing: -0.035em;
            margin: 0 0 0.65rem 0;
        }

        .hero p {
            color: #63758a;
            font-size: 1.05rem;
            margin: 0;
        }

        /* ---------- Section cards ---------- */
        .info-card {
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid #dbe8f2;
            border-radius: 18px;
            padding: 1.45rem 1.6rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 8px 28px rgba(35, 76, 112, 0.07);
        }

        .info-card h3 {
            color: #183b63;
            font-size: 1.2rem;
            font-weight: 750;
            margin: 0 0 0.9rem 0;
        }

        .info-card p,
        .info-card li {
            color: #52677d;
            font-size: 0.96rem;
            line-height: 1.65;
        }

        .info-card ul {
            margin: 0;
            padding-left: 1.25rem;
        }

        .info-card li {
            margin-bottom: 0.65rem;
        }

        .info-card li:last-child {
            margin-bottom: 0;
        }

        /* ---------- Highlight card ---------- */
        .highlight-card {
            background: linear-gradient(135deg, #2c70a7 0%, #173b70 100%);
            border-radius: 20px;
            padding: 1.7rem 1.8rem;
            margin: 0.5rem 0 1.4rem 0;
            box-shadow: 0 12px 32px rgba(26, 67, 105, 0.18);
        }

        .highlight-card h3,
        .highlight-card p,
        .highlight-card li {
            color: white;
        }

        .highlight-card h3 {
            font-size: 1.25rem;
            margin: 0 0 0.75rem 0;
        }

        .highlight-card p,
        .highlight-card li {
            font-size: 0.96rem;
            line-height: 1.65;
        }

        /* ---------- Streamlit markdown cleanup ---------- */
        div[data-testid="stMarkdownContainer"] p {
            margin-bottom: 0.35rem;
        }

        /* Hide Streamlit's default decorative divider */
        hr {
            display: none;
        }

        /* Buttons/links use the same blue family */
        a {
            color: #286899 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="aetion-brand"><i>Aeti</i><span>ON</span></div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>About the Data &amp; Model</h1>
        <p>Understand what the model can tell you — and where its limits are.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
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
