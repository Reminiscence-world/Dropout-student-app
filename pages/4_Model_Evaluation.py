"""
Ticket 12: pages/4_Model_Evaluation.py — model comparison.

Trains and evaluates Logistic Regression and Random Forest side by
side, on the identical split used everywhere else in the app (via
model.get_train_test_split), and highlights which one the app actually
uses (chosen in Ticket 3) with a one-line justification.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import load_and_clean_data
from src.model import evaluate_both_models

CHOSEN_MODEL = "Logistic Regression"
JUSTIFICATION = (
    "Random Forest's accuracy gain over Logistic Regression was not large "
    "enough to outweigh Logistic Regression's interpretability — per the "
    "project's selection rule, interpretability is the tiebreaker, not raw "
    "accuracy."
)

st.set_page_config(page_title="Model Evaluation", layout="wide")

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F4F8FC;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #DCE6F2;
}

/* Normal page headings */
h2, h3 {
    color: #1E2A4A;
}

/* Metric cards / containers */
[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #DCE6F2;
    border-radius: 15px;
    padding: 15px;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 15px;
}

/* Plotly chart container */
[data-testid="stPlotlyChart"] {
    background-color: white;
    border-radius: 15px;
    padding: 10px;
}

/* Success box */
[data-testid="stAlert"] {
    border-radius: 15px;
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
<h1 style="color:white !important; margin-bottom:10px;">
📊 Model Evaluation
</h1>
<p style="color:white !important; margin:0;">
Logistic Regression vs. Random Forest, evaluated on the identical
held-out test split.
</p>
</div>
""", unsafe_allow_html=True)

df = load_and_clean_data()
eval_results = evaluate_both_models(df)

st.success(f"**Chosen model: {CHOSEN_MODEL}.** {JUSTIFICATION}")

st.divider()
st.subheader("Metrics side by side")

summary_rows = []
for name, r in eval_results.items():
    summary_rows.append({
        "Model": name + (" ✓ (chosen)" if name == CHOSEN_MODEL else ""),
        "Accuracy": r["accuracy"],
        "Dropout precision": r["dropout_precision"],
        "Dropout recall": r["dropout_recall"],
        "5-fold CV accuracy (mean)": r["cv_mean"],
        "5-fold CV accuracy (std)": r["cv_std"],
    })
summary_df = pd.DataFrame(summary_rows).set_index("Model").round(4)
st.dataframe(summary_df, use_container_width=True)

chart_df = summary_df.reset_index()[["Model", "Accuracy", "Dropout precision", "Dropout recall"]]
chart_df_melted = chart_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
fig = px.bar(
    chart_df_melted,
    x="Metric",
    y="Score",
    color="Model",
    barmode="group",
    title="Accuracy vs. Dropout-class precision/recall",
    color_discrete_sequence=["#2E73B8", "#5EA4F3"]
)
fig.update_layout(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#F8FBFF",

    font=dict(
        color="#1E2A4A",
        size=14
    ),

    title_font=dict(
        color="#1E2A4A",
        size=18
    ),

    xaxis=dict(
        tickfont=dict(color="#1E2A4A"),
        title_font=dict(color="#1E2A4A"),
        gridcolor="#E6EEF8"
    ),

    yaxis=dict(
        tickfont=dict(color="#1E2A4A"),
        title_font=dict(color="#1E2A4A"),
        gridcolor="#E6EEF8"
    ),

    legend=dict(
        font=dict(color="#1E2A4A")
    )
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Dropout-class precision/recall are shown separately from overall accuracy "
    "specifically to confirm neither model is just predicting the majority "
    "class (Graduate)."
)

st.divider()
st.subheader("Confusion matrices")

cols = st.columns(len(eval_results))
for col, (name, r) in zip(cols, eval_results.items()):
    with col:
        label = name + (" ✓ (chosen)" if name == CHOSEN_MODEL else "")
        st.markdown(f"**{label}**")
        cm_df = pd.DataFrame(
            r["confusion_matrix"],
            index=["Actual: Not Dropout", "Actual: Dropout"],
            columns=["Predicted: Not Dropout", "Predicted: Dropout"],
        )
        st.dataframe(cm_df, use_container_width=True)
