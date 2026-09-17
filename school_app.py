import streamlit as st

from src.inference import predict_school


st.set_page_config(
    page_title="School Dashboard",
    page_icon="🏫",
)

st.title("School Dropout Risk")

st.write("School dashboard")

st.success("School inference engine connected successfully.")