"""
Ticket 13: pages/5_About_Limitations.py — disclosures panel.

Static content page: the mandatory disclosures from spec section 12,
plus the fairness/privacy notes from section 13, written in plain
language for an advisor (not a data scientist) audience.
"""

import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="About & Limitations",
    page_icon="ℹ️",
    layout="wide"
)


# ============================================================
# AETION COLOUR THEME
# ============================================================

st.markdown(
"""<style>

* ============================================================
   MAIN APPLICATION
   ============================================================ */

.stApp {
    background-color: #DBE8F4;
    color: #000000;
}


/* ============================================================
   MAIN CONTENT WIDTH
   ============================================================ */

.block-container {
    max-width: 1180px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}


/* ============================================================
   MAIN TEXT
   ============================================================ */

.stApp p,
.stApp li,
.stApp label,
.stApp span {
    color: #000000;
}

h1,
h2,
h3,
h4 {
    color: #000000 !important;
}


/* Captions */

[data-testid="stCaptionContainer"] {
    color: #5B6B7F !important;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

[data-testid="stSidebar"] {
    background-color: #263D73 !important;
}

[data-testid="stSidebar"] > div:first-child {
    background-color: #263D73 !important;
}


/* Sidebar text */

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #FFFFFF !important;
}


/* Sidebar navigation */

[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNav"] a span {
    color: #FFFFFF !important;
}


/* ============================================================
   HERO
   ============================================================ */

.about-hero {
    background: linear-gradient(
        110deg,
        #202F57 0%,
        #2E73B8 100%
    );

    border-radius: 20px;

    padding: 30px 34px;

    margin-bottom: 22px;

    box-shadow:
        0 8px 22px rgba(32, 47, 87, 0.18);
}

.about-hero h1 {
    color: #FFFFFF !important;

    font-size: 38px;

    font-weight: 700;

    margin: 0 0 8px 0;
}

.about-hero p {
    color: #FFFFFF !important;

    font-size: 16px;

    margin: 0;

    opacity: 0.92;
}


/* ============================================================
   INFORMATION CARDS
   ============================================================ */

.info-card {
    background-color: #FFFFFF;

    border-radius: 16px;

    border-left: 5px solid #2E73B8;

    padding: 22px 24px;

    margin-bottom: 20px;

    box-shadow:
        0 6px 18px rgba(32, 47, 87, 0.10);
}

.info-card h3 {
    color: #000000 !important;

    font-size: 21px;

    font-weight: 700;

    margin: 0 0 12px 0;
}

.info-card p {
    color: #000000 !important;

    font-size: 15px;

    line-height: 1.65;

    margin: 0 0 10px 0;
}

.info-card ul {
    margin: 0;

    padding-left: 22px;
}

.info-card li {
    color: #000000 !important;

    font-size: 15px;

    line-height: 1.65;

    margin-bottom: 10px;
}

.info-card li:last-child {
    margin-bottom: 0;
}


/* ============================================================
   HIGHLIGHT CARD
   ============================================================ */

.highlight-card {
    background: linear-gradient(
        135deg,
        #2E73B8 0%,
        #202F57 100%
    );

    border-radius: 18px;

    padding: 24px 26px;

    margin: 18px 0 22px 0;

    box-shadow:
        0 10px 28px rgba(32, 47, 87, 0.16);
}

.highlight-card h3 {
    color: #FFFFFF !important;

    font-size: 21px;

    font-weight: 700;

    margin: 0 0 10px 0;
}

.highlight-card p,
.highlight-card li {
    color: #FFFFFF !important;

    font-size: 15px;

    line-height: 1.65;
}

.highlight-card ul {
    margin: 0;

    padding-left: 22px;
}

.highlight-card li {
    margin-bottom: 9px;
}


/* ============================================================
   STREAMLIT MARKDOWN
   ============================================================ */

[data-testid="stMarkdownContainer"] p {
    margin-bottom: 0.5rem;
}


/* ============================================================
   DIVIDERS
   ============================================================ */

hr {
    border-color: #C9DDEC !important;
}


/* ============================================================
   LINKS
   ============================================================ */

a {
    color: #2E73B8 !important;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 768px) {

    .block-container {
        padding-top: 1.5rem;
    }

    .about-hero {
        padding: 24px;
    }

    .about-hero h1 {
        font-size: 30px;
    }

    .info-card {
        padding: 18px;
    }

}

</style>""",
unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
"""<div class="about-hero">
<h1>About the Data &amp; Model</h1>
<p>Understand what the model can tell you — and where its limits are.</p>
</div>""",
unsafe_allow_html=True
)


st.caption(
    "Read this before trusting any number elsewhere in this app. "
    "This page exists so nothing here is oversold."
)


# ============================================================
# WHERE THE DATA COMES FROM
# ============================================================

st.divider()

st.subheader("Where the Data Comes From")

st.markdown(
"""
<div class="info-card">
<ul>
<li>
<b>Source:</b> the UCI <b>"Predict Students' Dropout and Academic Success"</b>
dataset — roughly 4,400 students, all from <b>one institution in Portugal</b>.
Results here may not generalize to other institutions, countries, or student
populations.
</li>

<li>
This is <b>observational data</b>, not a randomized experiment. Nobody was
randomly assigned to, say, be a scholarship holder — students self-selected
or qualified into these situations for reasons the dataset doesn't fully
capture.
</li>

<li>
The outcome (<b>Target</b>) is a <b>single end-of-program snapshot</b> —
Dropout, Enrolled, or Graduate — recorded once, not a real-time feed tracking
students during the semester. This app cannot tell you <i>when</i> during a
term a student's risk changed.
</li>
</ul>
</div>
""",
unsafe_allow_html=True
)


# ============================================================
# MODEL OUTPUTS
# ============================================================

st.divider()

st.subheader("What the Model's Outputs Do — and Don't — Mean")

st.markdown(
"""
<div class="info-card">
<ul>

<li>
<b>Predicted risk</b> (Dashboard, Student Explorer) is exactly that —
a prediction from a trained statistical model. It is not a diagnosis
and not a guarantee.
</li>

<li>
<b>Risk tiers</b> (Low / Medium / High) are <b>model-based, illustrative
thresholds</b> — tertiles of this test set's predicted probabilities.
They are <b>not official, clinically validated, or institutionally approved</b>
risk categories.
</li>

<li>
<b>SHAP explanations</b> (Student Explorer) describe why the <i>model</i>
produced a given prediction for a given student. They are <b>not</b> a causal
claim about what would happen if something about that student changed.
</li>

<li>
<b>Causal-adjustment associations</b> (Causal Insights) are <b>not proof
of causation</b>. They assume the confounder set used in the regression
captures the major differences between students — if an important confounder
was left unmeasured, the adjusted association could still be biased.
Treat these as suggestive associations, not effect sizes you can act on
with certainty.
</li>

<li>
<b>Intervention suggestions</b> (Student Explorer) are phrased as
<i>"suggested based on this student's risk-associated factors"</i> —
never as a promise that acting on them will reduce a specific student's
risk by any amount.
</li>

</ul>
</div>
""",
unsafe_allow_html=True
)


# ============================================================
# FAIRNESS & PRIVACY
# ============================================================

st.divider()

st.subheader("Fairness & Privacy")

st.markdown(
"""
<div class="info-card">
<ul>

<li>
Demographic and fixed variables (gender, nationality, marital status,
parents' occupation, etc.) may help the model predict more accurately,
but they <b>never appear as the stated reason</b> for an intervention
suggestion — only the 3 actionable variables (tuition status, scholarship
status, debtor status) can trigger a suggestion.
</li>

<li>
This prototype has <b>no authentication or access control</b> — acceptable
here because the underlying dataset is already public and anonymized.
A real deployment handling actual student records would need
<b>FERPA-compliant access controls</b> and <b>role-restricted advisor
logins</b>.
</li>

</ul>
</div>
""",
unsafe_allow_html=True
)


# ============================================================
# WHAT THIS TOOL IS
# ============================================================

st.divider()

st.subheader("What This Tool Is — and Isn't")

st.markdown(
"""
<div class="highlight-card">
<h3>Decision-Support Prototype</h3>

<p>
This is a <b>decision-support prototype</b> to help an advisor prioritize
outreach — not a production system, not a diagnostic tool, and not a
substitute for an advisor's judgment.
</p>

<ul>
<li>It has no database beyond a static CSV.</li>
<li>It has no login within the Streamlit prototype.</li>
<li>It does not currently provide multi-institution support.</li>
<li>It does not write back to any system of record.</li>
</ul>

</div>
""",
unsafe_allow_html=True
)


# ============================================================
# FINAL DISCLAIMER
# ============================================================

st.info(
    "AETION is intended to support advisor decision-making, not replace it. "
    "Predictions and explanations should always be interpreted alongside "
    "real-world context and appropriate institutional processes."
)
