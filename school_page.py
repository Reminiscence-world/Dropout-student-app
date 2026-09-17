import streamlit as st

from src.inference import predict_school


st.set_page_config(
    page_title="School Dashboard",
    page_icon="🏫",
)

st.title("School Dropout Risk")
st.write("Enter the student's information below.")

st.subheader("Student Information")

school_type = st.selectbox(
    "School Type",
    ["Government", "Private"],
)

location = st.selectbox(
    "Location",
    ["Rural", "Semi-Urban", "Urban"],
)

infrastructure = st.selectbox(
    "Infrastructure",
    ["Basic", "Excellent", "Good", "Poor"],
)

teaching_staff = st.selectbox(
    "Teaching Staff",
    ["Excellent", "Good", "Poor", "Unknown"],
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"],
)

caste = st.selectbox(
    "Caste",
    ["General", "OBC", "SC", "ST"],
)

socioeconomic_status = st.selectbox(
    "Socioeconomic Status",
    ["High", "Low", "Moderate"],
)

age = st.number_input(
    "Age",
    min_value=10,
    max_value=16,
    value=13,
    step=1,
)

standard = st.number_input(
    "Standard",
    min_value=5,
    max_value=12,
    value=7,
    step=1,
)


if st.button("Predict Dropout Risk"):

    student = {
        "School_Type": school_type,
        "Location": location,
        "Infrastructure": infrastructure,
        "Teaching_Staff": teaching_staff,
        "Gender": gender,
        "Caste": caste,
        "Socioeconomic_Status": socioeconomic_status,
        "Age": age,
        "Standard": standard,
    }

    if not (10 <= age <= 16 and 5 <= standard <= 12):
        st.error("Please check the student information and try again.")

    else:
        try:
            with st.spinner("Analyzing student profile…"):
                result = predict_school(student)

        except Exception:
            st.error(
                "Unable to reach the prediction service. Please try again."
            )
            st.stop()

        st.divider()

        st.subheader("Dropout Risk")

        risk_probability = result["risk_probability"]
        risk_tier = result["risk_tier"]
        risk_tier_label = result["risk_tier_label"]
        model_version = result["model_version"]
        model_loaded = result["model_loaded"]

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Risk Probability",
                f"{risk_probability * 100:.1f}%"
            )

        with col2:
            st.metric(
                "Risk Tier",
                risk_tier
            )

        st.caption(risk_tier_label)

        if model_loaded:
            st.caption(f"Model: {model_version}")
        else:
            st.warning(
                f"Demo / fallback mode — Model: {model_version}"
            )

        st.subheader("Suggested Support")

        for intervention in result["interventions"]:
            st.write(f"**{intervention['title']}**")
            st.write(intervention["rationale"])