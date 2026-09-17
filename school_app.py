import streamlit as st


st.set_page_config(
    page_title="School Dashboard",
    page_icon="🏫",
)

pages = [
    st.Page(
        "school_pages/1_School_Dashboard.py",
        title="School Dashboard",
        icon="🏫",
    ),
    st.Page(
        "school_pages/2_Student_Explorer.py",
        title="Student Explorer",
        icon="👨‍🎓",
    ),
    st.Page(
        "school_pages/3_Model_Evaluation.py",
        title="Model Evaluation",
        icon="📊",
    ),
    st.Page(
        "school_pages/4_About_Limitations.py",
        title="About & Limitations",
        icon="ℹ️",
    ),
]

pg = st.navigation(pages)

pg.run()