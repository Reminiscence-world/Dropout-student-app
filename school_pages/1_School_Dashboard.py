import streamlit as st
import sys
from pathlib import Path

# Add root folder so utils can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import render_school_hero

render_school_hero(
    title="🏫 School Dropout Risk Dashboard",
    subtitle="Cohort-level overview of the school student dataset.",
    description="This tool provides early decision support to identify at-risk K-12 students, highlighting critical infrastructure, academic, and socio-economic indicators across schools.",
    scope_badge="Built on the SIH K-12 Dropout Dataset (~10,198 students). See About & Limitations for details."
)

# Rest of your existing cohort charts & metrics continue below...
import pandas as pd

from src.school_pipeline import load_school_data, clean_school_data


st.title("🏫 School Dashboard")
st.caption("Cohort-level overview of the school student dataset")

# Load and clean dataset
df = load_school_data("data/SIH-Dataset.csv")
df = clean_school_data(df)

# Add deterministic Student IDs
df = df.reset_index(drop=True)
df.insert(0, "Student_ID", range(1, len(df) + 1))

# -----------------------------
# Cohort overview
# -----------------------------

st.subheader("Cohort Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Students", len(df))

with col2:
    enrolled = (df["Dropout_Status"] == "Enrolled").sum()
    st.metric("Enrolled", enrolled)

with col3:
    dropout = (df["Dropout_Status"] == "Dropout").sum()
    st.metric("Dropout", dropout)

st.divider()

# -----------------------------
# Student status distribution
# -----------------------------

st.subheader("Student Status Distribution")

status_counts = df["Dropout_Status"].value_counts()

col1, col2 = st.columns(2)

with col1:
    st.bar_chart(status_counts)

with col2:
    st.write("### Status Breakdown")

    total = len(df)

    for status, count in status_counts.items():
        percentage = (count / total) * 100
        st.write(f"**{status}:** {count} students ({percentage:.1f}%)")

st.divider()

# -----------------------------
# Dataset overview
# -----------------------------

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Schools Types", df["School_Type"].nunique())

with col2:
    st.metric("Locations", df["Location"].nunique())

with col3:
    st.metric("Standards", df["Standard"].nunique())

st.info(
    "Student IDs are generated sequentially after data cleaning and are used "
    "only to identify students in the Student Explorer. They are not model features."
)