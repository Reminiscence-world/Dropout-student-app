import json
from pathlib import Path

import streamlit as st


META_PATH = Path("src/artifacts/school_model_meta.json")


st.title("Model Evaluation")
st.write("Performance metrics for the School dropout prediction model.")

if not META_PATH.exists():
    st.error("Model evaluation information is unavailable.")
    st.stop()

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

st.subheader("Model Information")

st.write(f"**Model Version:** {meta['model_version']}")
st.write(f"**Positive Class:** {meta['positive_class']}")

st.divider()

metrics = meta["metrics"]

st.subheader("Performance Metrics")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    st.metric("Dropout Precision", f"{metrics['dropout_precision'] * 100:.2f}%")
    st.metric("Dropout Recall", f"{metrics['dropout_recall'] * 100:.2f}%")

with col2:
    st.metric("Dropout F1", f"{metrics['dropout_f1'] * 100:.2f}%")
    st.metric("CV F1 Mean", f"{metrics['cv_f1_mean'] * 100:.2f}%")
    st.metric("CV F1 Std", f"{metrics['cv_f1_std'] * 100:.3f}%")

st.divider()

st.subheader("Confusion Matrix")

cm = metrics["confusion_matrix"]

st.dataframe(
    {
        "Actual Dropout": cm[0],
        "Actual Enrolled": cm[1],
    },
    use_container_width=True,
)