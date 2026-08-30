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

st.title("Model Evaluation")
st.caption("Logistic Regression vs. Random Forest, evaluated on the identical held-out test split.")

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
