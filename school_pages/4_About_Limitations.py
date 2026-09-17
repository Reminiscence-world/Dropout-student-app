import json
from pathlib import Path
import streamlit as st
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import render_school_hero

render_school_hero(
    title="ℹ️ About & Limitations",
    subtitle="Methodology notes, algorithmic boundaries, and ethical safeguards.",
    description="Interventions are transparent heuristic routing rules rather than causal guarantees. Demographic attributes are never used to justify interventions."
)

# Limitations text continues below...


META_PATH = Path("src/artifacts/school_model_meta.json")


st.title("About & Limitations")

st.subheader("School Dropout Prediction")

st.write(
    "This School application uses a trained machine-learning model "
    "to estimate dropout risk from the available student and school "
    "information."
)

st.subheader("Model")

if META_PATH.exists():
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    st.write(f"**Model Version:** {meta['model_version']}")
    st.write(f"**Positive Class:** {meta['positive_class']}")

st.subheader("Important Limitations")

st.write(
    "- Predictions are model-derived estimates, not certain outcomes."
)

st.write(
    "- Risk predictions should be reviewed alongside relevant student context."
)

st.write(
    "- The model does not establish causation."
)

st.write(
    "- A prediction should not be used as the sole basis for decisions "
    "about a student."
)

st.subheader("Responsible Use")

st.info(
    "Use the prediction as decision-support information and combine it "
    "with appropriate human review and contextual understanding."
)