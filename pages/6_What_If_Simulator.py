"""
Ticket 15 (Stretch goal): pages/6_What_If_Simulator.py — What-If Simulator.

Lets an advisor edit one student's ACTIONABLE and MEDIATOR inputs.

Re-runs ONLY the predictive model on the edited inputs — never the
causal-adjustment module.

This is a hypothetical run of the correlational predictive model and
does not prove that changing a factor would actually change a real
student's dropout risk.
"""

import streamlit as st

from src.data_loader import load_and_clean_data
from src.model import train_model
from src.risk_tiers import assign_risk_tiers, TIER_CAVEAT
from src.variable_roles import ACTIONABLE, MEDIATOR
from src.theme import PAGE_ICONS


st.set_page_config(
    page_title="What-If Simulator",
    page_icon=PAGE_ICONS["what_if"],
    layout="wide",
)

DISCLAIMER = (
    "**Hypothetical simulation using the predictive model.** Does not prove "
    "that changing this factor would actually reduce dropout risk."
)


def classify_tier(proba: float, cutoffs: dict) -> str:
    """Classify probability using the same test-set cutoffs."""
    if proba <= cutoffs["low_medium"]:
        return "Low"
    elif proba <= cutoffs["medium_high"]:
        return "Medium"
    return "High"


# ============================================================
# AETION VISUAL THEME
# ============================================================

st.markdown(
    """
<style>

    /* ========================================================
       GLOBAL MAIN AREA
       ======================================================== */

    .stApp {
        background-color: #DBE8F4;
    }

    /* Main content text */
    .stApp p,
    .stApp label,
    .stApp span,
    .stApp div {
        color: #000000;
    }

    /* Main headings */
    h1, h2, h3, h4 {
        color: #000000 !important;
    }

    /* Caption / secondary text */
    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: #5B6B7F !important;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background-color: #263D73 !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background-color: #263D73 !important;
    }

    /* ALL SIDEBAR TEXT WHITE */
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

    /* Sidebar navigation links */
    [data-testid="stSidebarNav"] a {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebarNav"] a span {
        color: #FFFFFF !important;
    }

    /* Sidebar selectbox */
    [data-testid="stSidebar"] [data-baseweb="select"] {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #FFFFFF !important;
    }

    /* Sidebar input */
    [data-testid="stSidebar"] input {
        color: #FFFFFF !important;
    }


    /* ========================================================
       TOP HEADER / HERO
       ======================================================== */

    .whatif-hero {
        background: linear-gradient(
            110deg,
            #202F57 0%,
            #2E73B8 100%
        );

        border-radius: 20px;

        padding: 32px 36px;

        margin-bottom: 22px;

        box-shadow: 0 8px 22px rgba(32, 47, 87, 0.18);
    }

    .whatif-hero h1 {
        color: #FFFFFF !important;
        font-size: 38px;
        font-weight: 700;
        margin: 0 0 8px 0;
    }

    .whatif-hero p {
        color: #FFFFFF !important;
        font-size: 16px;
        margin: 0;
        opacity: 0.92;
    }


    /* ========================================================
       SECTION HEADINGS
       ======================================================== */

    .section-title {
        color: #000000 !important;
        font-size: 24px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 6px;
    }


    /* ========================================================
       WHITE CARDS
       ======================================================== */

    .info-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 6px 18px rgba(32, 47, 87, 0.10);
        border-left: 5px solid #2E73B8;
        margin-bottom: 18px;
    }

    .info-card-title {
        color: #000000 !important;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .info-card-text {
        color: #5B6B7F !important;
        font-size: 14px;
    }


    /* ========================================================
       STREAMLIT INPUTS
       ======================================================== */

    [data-baseweb="select"] > div {
        border-color: #2E73B8 !important;
        background-color: #FFFFFF !important;
    }

    [data-baseweb="select"] * {
        color: #000000 !important;
    }

    /* Radio buttons */
    [data-baseweb="radio"] label {
        color: #000000 !important;
    }

    [data-baseweb="radio"] [aria-checked="true"] {
        background-color: #2E73B8 !important;
        border-color: #2E73B8 !important;
    }

    /* Sliders */
    [data-testid="stSlider"] {
        color: #2E73B8 !important;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        background-color: #2E73B8 !important;
        color: #FFFFFF !important;
        border: 1px solid #2E73B8 !important;
        border-radius: 8px;
    }

    .stButton > button:hover {
        background-color: #245F98 !important;
        border-color: #245F98 !important;
        color: #FFFFFF !important;
    }


    /* ========================================================
       METRICS
       ======================================================== */

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 18px;
        border-left: 5px solid #2E73B8;
        box-shadow: 0 6px 18px rgba(32, 47, 87, 0.10);
    }

    [data-testid="stMetricLabel"] {
        color: #000000 !important;
    }

    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #000000 !important;
    }


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {
        border-color: #C9DDEC !important;
    }


    /* ========================================================
       EXPANDER
       ======================================================== */

    [data-testid="stExpander"] {
        background-color: #FFFFFF;
        border: 1px solid #C9DDEC;
        border-radius: 12px;
    }

    [data-testid="stExpander"] summary {
        color: #000000 !important;
    }


    /* ========================================================
       ALERT / WARNING BOX
       ======================================================== */

    [data-testid="stAlert"] {
        background-color: #F3F8FC;
        border-color: #5EA4F3;
    }

    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span {
        color: #000000 !important;
    }


    /* ========================================================
       DATAFRAME
       ======================================================== */

    [data-testid="stDataFrame"] {
        background-color: #FFFFFF;
        border-radius: 12px;
    }


    /* ========================================================
       RESPONSIVE
       ======================================================== */

    @media (max-width: 768px) {

        .whatif-hero {
            padding: 24px;
        }

        .whatif-hero h1 {
            font-size: 30px;
        }

    }

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="whatif-hero">
        <h1>What-If Simulator</h1>
        <p>
            Explore how changes to actionable and early-warning student
            factors affect the predictive model's estimated dropout risk.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


st.warning(DISCLAIMER)


# ============================================================
# LOAD DATA + MODEL
# ============================================================

df = load_and_clean_data()
results = train_model(df)

tiers, cutoffs = assign_risk_tiers(results["y_pred_proba"])

X_test = results["X_test"]
student_ids = X_test.index.tolist()


# ============================================================
# STUDENT SELECTION
# ============================================================

st.markdown(
    '<div class="section-title">Select Student</div>',
    unsafe_allow_html=True,
)

selected_id = st.selectbox(
    "Select a student (type to search by ID)",
    options=student_ids,
    format_func=lambda x: f"Student ID {x}",
)

position = X_test.index.get_loc(selected_id)

original_row = X_test.iloc[[position]].copy()

original_proba = results["y_pred_proba"][position]
original_tier = classify_tier(original_proba, cutoffs)


# ============================================================
# EDIT INPUTS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">Edit this student\'s inputs</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Only actionable factors and early-warning factors are editable. "
    "Demographic and fixed variables remain unchanged."
)


edited_values = {}

col_a, col_b = st.columns(2)


# ============================================================
# ACTIONABLE FACTORS
# ============================================================

with col_a:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-card-title">Actionable Factors</div>
            <div class="info-card-text">
                Factors that may represent practical intervention levers.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for feature in ACTIONABLE:

        if feature not in original_row.columns:
            continue

        current_val = int(original_row[feature].iloc[0])

        label = feature.replace("_", " ").title()

        choice = st.radio(
            label,
            options=["No", "Yes"],
            index=current_val,
            horizontal=True,
            key=f"edit_{feature}",
        )

        edited_values[feature] = 1 if choice == "Yes" else 0


# ============================================================
# MEDIATOR / EARLY WARNING FACTORS
# ============================================================

with col_b:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-card-title">Early-Warning Factors</div>
            <div class="info-card-text">
                Academic indicators that can change over time and may
                provide early warning signals.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for feature in MEDIATOR:

        if feature not in original_row.columns:
            continue

        col_min = float(X_test[feature].min())
        col_max = float(X_test[feature].max())

        current_val = float(original_row[feature].iloc[0])

        label = feature.replace("_", " ").title()

        if "approved" in feature.lower():

            new_val = st.slider(
                label,
                min_value=int(col_min),
                max_value=int(col_max),
                value=int(current_val),
                step=1,
                key=f"edit_{feature}",
            )

        else:

            new_val = st.slider(
                label,
                min_value=col_min,
                max_value=col_max,
                value=current_val,
                step=0.1,
                key=f"edit_{feature}",
            )

        edited_values[feature] = new_val


# ============================================================
# FIXED VARIABLES
# ============================================================

with st.expander(
    "Other inputs used by the model — read-only"
):

    st.caption(
        "These variables remain fixed for this simulation and cannot "
        "be hypothetically changed."
    )

    other_cols = [
        c
        for c in original_row.columns
        if c not in ACTIONABLE and c not in MEDIATOR
    ]

    st.dataframe(
        original_row[other_cols],
        use_container_width=True,
    )


# ============================================================
# HYPOTHETICAL MODEL RUN
# ============================================================

hypothetical_row = original_row.copy()

for feature, value in edited_values.items():
    hypothetical_row[feature] = value


new_proba = results["model"].predict_proba(
    hypothetical_row
)[0, 1]

new_tier = classify_tier(
    new_proba,
    cutoffs,
)


# ============================================================
# RESULT
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">Simulation Result</div>',
    unsafe_allow_html=True,
)


c1, c2 = st.columns(2)


with c1:

    st.metric(
        "Original predicted P(Dropout)",
        f"{original_proba:.2f}",
    )

    st.caption(
        f"Original risk tier: **{original_tier}**"
    )


with c2:

    delta = new_proba - original_proba

    st.metric(
        "Hypothetical predicted P(Dropout)",
        f"{new_proba:.2f}",
        delta=f"{delta:+.2f}",
        delta_color="inverse",
    )

    st.caption(
        f"Hypothetical risk tier: **{new_tier}**"
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.caption(TIER_CAVEAT)

st.warning(DISCLAIMER)
