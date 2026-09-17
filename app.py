"""
Ticket 14 (Polish Pass): app.py is now the app's landing page.

The Ticket 1-8 "does this backend module work" checks that used to live
here have all been superseded by dedicated pages — Dashboard, Student
Explorer, Causal Insights, and Model Evaluation each demonstrate the
same functionality live. This page's only job now is to be a clean
opener: the problem statement, plus a way into the rest of the app.
"""

import streamlit as st
from src.theme import APP_TITLE, APP_ICON, PAGE_ICONS


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Home",
    page_icon=PAGE_ICONS["home"],
    layout="wide"
)


# ============================================================
# AETION COLOUR THEME
# ============================================================

st.markdown(
"""<style>

/* ============================================================
   MAIN APPLICATION BACKGROUND
   ============================================================ */

.stApp {
    background-color: #DBE8F4;
    color: #000000;
}


/* ============================================================
   MAIN CONTENT
   ============================================================ */

.block-container {
    max-width: 1180px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}


/* Main text */

.stApp p,
.stApp label,
.stApp span {
    color: #000000;
}


/* Headings */

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


/* Change "app" to visually display as "App" */

[data-testid="stSidebarNav"] {
    text-transform: capitalize;
}


/* ============================================================
   HERO
   ============================================================ */

.home-hero {
    background: linear-gradient(
        110deg,
        #202F57 0%,
        #2E73B8 100%
    );

    border-radius: 20px;

    padding: 30px 34px;

    margin-bottom: 24px;

    box-shadow:
        0 8px 22px rgba(32, 47, 87, 0.18);
}


/* Hero title */

.home-hero h1 {
    color: #FFFFFF !important;

    font-size: 38px;

    font-weight: 700;

    margin: 0 0 10px 0;
}


/* Hero description */

.home-hero p {
    color: #FFFFFF !important;

    font-size: 16px;

    line-height: 1.6;

    margin: 0;

    opacity: 0.94;
}


/* ============================================================
   PROBLEM STATEMENT CARD
   ============================================================ */

.problem-card {
    background-color: #FFFFFF;

    border-radius: 16px;

    border-left: 5px solid #2E73B8;

    padding: 22px 24px;

    margin-bottom: 20px;

    box-shadow:
        0 6px 18px rgba(32, 47, 87, 0.10);
}


.problem-card p {
    color: #000000 !important;

    font-size: 16px;

    line-height: 1.7;

    margin: 0;
}


/* ============================================================
   DATASET INFO CARD
   ============================================================ */

.dataset-card {
    background: linear-gradient(
        135deg,
        #2E73B8 0%,
        #202F57 100%
    );

    border-radius: 16px;

    padding: 20px 24px;

    margin-bottom: 26px;

    box-shadow:
        0 8px 22px rgba(32, 47, 87, 0.14);
}


.dataset-card p {
    color: #FFFFFF !important;

    font-size: 15px;

    line-height: 1.6;

    margin: 0;
}


/* ============================================================
   GET STARTED SECTION
   ============================================================ */

.section-title {
    color: #000000 !important;

    font-size: 25px;

    font-weight: 700;

    margin-bottom: 16px;
}


/* ============================================================
   PAGE LINK CARDS
   ============================================================ */

.page-link-card {
    background-color: #FFFFFF;

    border-radius: 16px;

    border: 1px solid #DCE6F2;

    padding: 18px 20px;

    min-height: 120px;

    box-shadow:
        0 5px 16px rgba(32, 47, 87, 0.08);

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}


.page-link-card:hover {
    transform: translateY(-3px);

    box-shadow:
        0 9px 22px rgba(32, 47, 87, 0.13);
}


/* ============================================================
   PAGE LINK BUTTONS
   ============================================================ */

.stPageLink a {
    color: #2E73B8 !important;

    font-weight: 700;
}


/* ============================================================
   INFO / ALERT
   ============================================================ */

[data-testid="stAlert"] {
    border-radius: 14px;
}


/* ============================================================
   DIVIDERS
   ============================================================ */

hr {
    border-color: #C9DDEC !important;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 768px) {

    .block-container {
        padding-top: 1.5rem;
    }

    .home-hero {
        padding: 24px;
    }

    .home-hero h1 {
        font-size: 30px;
    }

    .home-hero p {
        font-size: 15px;
    }

}

</style>""",
unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
f"""<div class="home-hero">
<h1>{APP_ICON} {APP_TITLE}</h1>
<p>AI-powered decision support to identify students at risk early, understand the factors behind their risk, and support timely intervention.</p>
</div>""",
unsafe_allow_html=True
)


# ============================================================
# PROBLEM STATEMENT
# ============================================================

st.markdown(
"""<div class="problem-card">
<p>University advisors don't have a systematic way to know <b>which</b>
students are at risk of dropping out early, or <b>what kind of support</b>
to offer them. This tool is a decision-support prototype that, for each
student: predicts a dropout-risk score, explains why the model flagged them,
estimates how a small set of <b>actionable</b> factors relate to dropout
(under stated assumptions), and suggests a fitting intervention — all in
one local dashboard.</p>
</div>""",
unsafe_allow_html=True
)


# ============================================================
# DATASET INFORMATION
# ============================================================

st.markdown(
"""<div class="dataset-card">
<p>Built on the UCI <b>"Predict Students' Dropout and Academic Success"</b>
dataset (~4,400 students, one Portuguese institution). See
<b>About &amp; Limitations</b> for what this tool is — and isn't —
before trusting any number in it.</p>
</div>""",
unsafe_allow_html=True
)


# ============================================================
# GET STARTED
# ============================================================

st.markdown(
"""<div class="section-title">
Get Started
</div>""",
unsafe_allow_html=True
)


# ============================================================
# FIRST ROW OF PAGES
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.page_link(
        "pages/1_Dashboard.py",
        label="Dashboard",
        icon=PAGE_ICONS["dashboard"]
    )

    st.caption(
        "Cohort-level risk overview."
    )


with col2:

    st.page_link(
        "pages/2_Student_Explorer.py",
        label="Student Explorer",
        icon=PAGE_ICONS["student_explorer"]
    )

    st.caption(
        "Drill into one student's risk, explanation, "
        "and suggested intervention."
    )


with col3:

    st.page_link(
        "pages/3_Causal_Insights.py",
        label="Causal Insights",
        icon=PAGE_ICONS["causal_insights"]
    )

    st.caption(
        "Confounder-adjusted associations for the "
        "actionable factors."
    )


# ============================================================
# SECOND ROW OF PAGES
# ============================================================

col4, col5, col6 = st.columns(3)


with col4:

    st.page_link(
        "pages/4_Model_Evaluation.py",
        label="Model Evaluation",
        icon=PAGE_ICONS["model_evaluation"]
    )

    st.caption(
        "How the model was chosen, and how well it performs."
    )


with col5:

    st.page_link(
        "pages/5_About_Limitations.py",
        label="About & Limitations",
        icon=PAGE_ICONS["about"]
    )

    st.caption(
        "What this tool is, and isn't."
    )


with col6:

    st.page_link(
        "pages/6_What_If_Simulator.py",
        label="What-If Simulator",
        icon=PAGE_ICONS["what_if"]
    )

    st.caption(
        "Edit a student's inputs and see how the prediction "
        "changes (stretch goal)."
    )
