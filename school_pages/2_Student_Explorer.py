import streamlit as st
import pandas as pd

from src.school_pipeline import load_school_data, clean_school_data
from src.inference import predict_school


st.title("👨‍🎓 Student Explorer")
st.caption("Explore an individual student's profile and dropout-risk prediction")

# -----------------------------
# Load dataset
# -----------------------------

df = load_school_data("data/SIH-Dataset.csv")
df = clean_school_data(df)

# Generate deterministic Student IDs
df = df.reset_index(drop=True)
df.insert(0, "Student_ID", range(1, len(df) + 1))

# -----------------------------
# Student selection
# -----------------------------

student_id = st.selectbox(
    "Select Student ID",
    options=df["Student_ID"].tolist(),
    index=0,
)

student = df.loc[df["Student_ID"] == student_id].iloc[0]

st.divider()

# -----------------------------
# Student profile
# -----------------------------

st.subheader(f"Student {student_id}")

col1, col2 = st.columns(2)

with col1:
    st.write("### Student Information")

    st.write(f"**Student ID:** {student['Student_ID']}")
    st.write(f"**Gender:** {student['Gender']}")
    st.write(f"**Caste:** {student['Caste']}")
    st.write(f"**Age:** {student['Age']}")
    st.write(f"**Standard:** {student['Standard']}")

with col2:
    st.write("### School Information")

    st.write(f"**School Type:** {student['School_Type']}")
    st.write(f"**Location:** {student['Location']}")
    st.write(f"**Infrastructure:** {student['Infrastructure']}")
    st.write(f"**Teaching Staff:** {student['Teaching_Staff']}")
    st.write(
        f"**Socioeconomic Status:** "
        f"{student['Socioeconomic_Status']}"
    )

st.divider()

# -----------------------------
# Actual dataset status
# -----------------------------

st.subheader("Recorded Student Status")

if student["Dropout_Status"] == "Dropout":
    st.warning("Recorded status: Dropout")
else:
    st.success("Recorded status: Enrolled")

st.divider()

# -----------------------------
# Backend prediction
# -----------------------------

if st.button("🔍 Analyze Student", use_container_width=True):

    # Only send model features.
    # Student_ID and Dropout_Status are NOT prediction features.
    input_data = {
        "School_Type": student["School_Type"],
        "Location": student["Location"],
        "Infrastructure": student["Infrastructure"],
        "Teaching_Staff": student["Teaching_Staff"],
        "Gender": student["Gender"],
        "Caste": student["Caste"],
        "Socioeconomic_Status": student["Socioeconomic_Status"],
        "Age": student["Age"],
        "Standard": student["Standard"],
    }

    with st.spinner("Analyzing student profile..."):
        result = predict_school(input_data)

    st.subheader("Dropout Risk Prediction")

    probability = result["risk_probability"]
    tier = result["risk_tier"]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Dropout Probability",
            f"{probability * 100:.1f}%"
        )

    with col2:
        st.metric(
            "Risk Level",
            tier
        )

    # Model information
    if result.get("model_loaded"):
        st.success(
            f"Model: {result.get('model_version', 'Unknown')} • "
            "Backend model loaded"
        )
    else:
        st.warning("Backend model is not loaded. Fallback prediction is being used.")

    # -----------------------------
    # Suggested support
    # -----------------------------

    st.subheader("Suggested Support")

    interventions = result.get("interventions", [])

    if interventions:
        for intervention in interventions:
            st.write(f"• {intervention}")
    else:
        st.write("No specific support recommendation available.")