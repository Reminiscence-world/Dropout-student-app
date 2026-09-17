"""
Ticket 12: pages/4_Model_Evaluation.py — Model Evaluation.

Trains and evaluates Logistic Regression and Random Forest side by side
on the identical held-out test split used elsewhere in the app.

Highlights the model actually used by the application.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import load_and_clean_data
from src.model import evaluate_both_models


# ============================================================
# CONFIGURATION
# ============================================================

CHOSEN_MODEL = "Logistic Regression"

JUSTIFICATION = (
    "Random Forest's accuracy gain over Logistic Regression was not large "
    "enough to outweigh Logistic Regression's interpretability — per the "
    "project's selection rule, interpretability is the tiebreaker, not raw "
    "accuracy."
)


st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📊",
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
}


/* ============================================================
   MAIN PAGE TEXT
   ============================================================ */

.stApp p,
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


/* Captions / secondary text */

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


/* Sidebar text must remain WHITE */

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


/* Sidebar navigation text */

[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNav"] a span {
    color: #FFFFFF !important;
}


/* ============================================================
   HERO SECTION
   ============================================================ */

.model-hero {
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

.model-hero h1 {
    color: #FFFFFF !important;

    font-size: 38px;

    font-weight: 700;

    margin: 0 0 8px 0;
}

.model-hero p {
    color: #FFFFFF !important;

    font-size: 16px;

    margin: 0;

    opacity: 0.92;
}


/* ============================================================
   CHOSEN MODEL CARD
   ============================================================ */

.chosen-model-card {
    background-color: #FFFFFF;

    border-left: 6px solid #2E73B8;

    border-radius: 16px;

    padding: 20px 24px;

    margin-bottom: 24px;

    box-shadow:
        0 6px 18px rgba(32, 47, 87, 0.10);
}

.chosen-model-label {
    color: #5B6B7F !important;

    font-size: 13px;

    font-weight: 600;

    text-transform: uppercase;

    letter-spacing: 0.6px;

    margin-bottom: 4px;
}

.chosen-model-name {
    color: #000000 !important;

    font-size: 25px;

    font-weight: 700;

    margin-bottom: 6px;
}

.chosen-model-description {
    color: #5B6B7F !important;

    font-size: 14px;

    line-height: 1.5;
}


/* ============================================================
   METRIC CARDS
   ============================================================ */

[data-testid="stMetric"] {
    background-color: #FFFFFF;

    border-radius: 16px;

    padding: 18px 20px;

    border-left: 5px solid #2E73B8;

    box-shadow:
        0 6px 18px rgba(32, 47, 87, 0.10);
}

[data-testid="stMetricLabel"] {
    color: #5B6B7F !important;
}

[data-testid="stMetricValue"] {
    color: #000000 !important;
}

[data-testid="stMetricDelta"] {
    color: #000000 !important;
}


/* ============================================================
   DATAFRAME
   ============================================================ */

[data-testid="stDataFrame"] {
    background-color: #FFFFFF;

    border-radius: 14px;
}


/* ============================================================
   PLOTLY CHART
   ============================================================ */

[data-testid="stPlotlyChart"] {
    background-color: #FFFFFF;

    border-radius: 16px;

    padding: 10px;

    box-shadow:
        0 6px 18px rgba(32, 47, 87, 0.08);
}


/* ============================================================
   DIVIDERS
   ============================================================ */

hr {
    border-color: #C9DDEC !important;
}


/* ============================================================
   ALERTS
   ============================================================ */

[data-testid="stAlert"] {
    border-radius: 14px;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 768px) {

    .model-hero {
        padding: 24px;
    }

    .model-hero h1 {
        font-size: 30px;
    }

    .chosen-model-name {
        font-size: 21px;
    }

}

</style>""",
unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
"""<div class="model-hero">
<h1>📊 Model Evaluation</h1>
<p>Compare Logistic Regression and Random Forest on the same held-out test split and evaluate the model used by AETION.</p>
</div>""",
unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = load_and_clean_data()

eval_results = evaluate_both_models(df)


# ============================================================
# CHOSEN MODEL
# ============================================================

st.markdown(
f"""<div class="chosen-model-card">
<div class="chosen-model-label">Model selected for AETION</div>
<div class="chosen-model-name">✓ {CHOSEN_MODEL}</div>
<div class="chosen-model-description">{JUSTIFICATION}</div>
</div>""",
unsafe_allow_html=True
)


# ============================================================
# PERFORMANCE OVERVIEW
# ============================================================

st.subheader("Performance Overview")

st.caption(
    "A quick comparison of the two models on the same held-out test set."
)


lr = eval_results.get("Logistic Regression")
rf = eval_results.get("Random Forest")


if lr is not None and rf is not None:

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Logistic Regression Accuracy",
            f"{lr['accuracy']:.2%}"
        )

    with col2:
        st.metric(
            "Random Forest Accuracy",
            f"{rf['accuracy']:.2%}"
        )

    with col3:
        st.metric(
            "LR Dropout Recall",
            f"{lr['dropout_recall']:.2%}"
        )

    with col4:
        st.metric(
            "RF Dropout Recall",
            f"{rf['dropout_recall']:.2%}"
        )


# ============================================================
# METRICS SIDE BY SIDE
# ============================================================

st.divider()

st.subheader("Metrics Side by Side")

st.caption(
    "Both models are evaluated using the identical held-out test split."
)


summary_rows = []

for name, r in eval_results.items():

    summary_rows.append(
        {
            "Model": (
                name + " ✓ (chosen)"
                if name == CHOSEN_MODEL
                else name
            ),

            "Accuracy": r["accuracy"],

            "Dropout precision": r["dropout_precision"],

            "Dropout recall": r["dropout_recall"],

            "5-fold CV accuracy (mean)": r["cv_mean"],

            "5-fold CV accuracy (std)": r["cv_std"],
        }
    )


summary_df = (
    pd.DataFrame(summary_rows)
    .set_index("Model")
    .round(4)
)


st.dataframe(
    summary_df,
    use_container_width=True
)


# ============================================================
# MODEL PERFORMANCE COMPARISON
# ============================================================

st.divider()

st.subheader("Model Performance Comparison")

st.caption(
    "Higher values indicate stronger performance for the corresponding metric."
)


chart_df = summary_df.reset_index()[
    [
        "Model",
        "Accuracy",
        "Dropout precision",
        "Dropout recall"
    ]
]


chart_df_melted = chart_df.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score"
)


fig = px.bar(
    chart_df_melted,

    x="Metric",

    y="Score",

    color="Model",

    barmode="group",

    title="Accuracy vs. Dropout-Class Precision and Recall",

    color_discrete_sequence=[
        "#2E73B8",
        "#5EA4F3"
    ]
)


fig.update_layout(

    paper_bgcolor="#FFFFFF",

    plot_bgcolor="#F8FBFF",

    font=dict(
        color="#000000",
        size=13
    ),

    title_font=dict(
        color="#000000",
        size=18
    ),

    xaxis=dict(
        tickfont=dict(
            color="#000000"
        ),

        title_font=dict(
            color="#000000"
        ),

        gridcolor="#E6EEF8"
    ),

    yaxis=dict(
        tickfont=dict(
            color="#000000"
        ),

        title_font=dict(
            color="#000000"
        ),

        gridcolor="#E6EEF8",

        range=[0, 1]
    ),

    legend=dict(
        font=dict(
            color="#000000"
        ),

        bgcolor="#FFFFFF"
    ),

    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    )
)


fig.update_yaxes(
    tickformat=".0%"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


st.caption(
    "Dropout precision and recall are shown separately from overall "
    "accuracy to assess how effectively each model identifies the "
    "dropout class."
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

st.divider()

st.subheader("Confusion Matrices")

st.caption(
    "Comparison between the models' predictions and the actual outcomes "
    "in the held-out test set."
)


cols = st.columns(len(eval_results))


for col, (name, r) in zip(cols, eval_results.items()):

    with col:

        if name == CHOSEN_MODEL:

            st.markdown(
                f"### ✓ {name}"
            )

            st.caption(
                "Model used by AETION"
            )

        else:

            st.markdown(
                f"### {name}"
            )

            st.caption(
                "Alternative model"
            )


        cm_df = pd.DataFrame(
            r["confusion_matrix"],

            index=[
                "Actual: Not Dropout",
                "Actual: Dropout"
            ],

            columns=[
                "Predicted: Not Dropout",
                "Predicted: Dropout"
            ]
        )


        st.dataframe(
            cm_df,
            use_container_width=True
        )


# ============================================================
# WHY THESE METRICS MATTER
# ============================================================

st.divider()

st.subheader("Why These Metrics Matter")

st.markdown(
    """
**Accuracy** describes the proportion of all predictions that are correct.

**Dropout precision** measures how often students predicted as dropouts
actually belong to the dropout class.

**Dropout recall** measures how many of the actual dropout cases the
model identifies.
"""
)


# ============================================================
# FINAL NOTE
# ============================================================

st.info(
    "Model evaluation is performed on the held-out test data. "
    "The selected Logistic Regression model is used by AETION "
    "for its predictive workflow."
)
