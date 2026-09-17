import streamlit as st


def add_sidebar_logo():
    with st.sidebar:
        st.markdown(
            """
            <style>

            div.aetion-sidebar-logo {
                font-family: Georgia, "Times New Roman", serif;
                font-size: 1.8rem;
                font-weight: 700;
                letter-spacing: -0.04em;
                margin-top: -8px;
                margin-bottom: 25px;
                padding-left: 5px;
                line-height: 1.2;
            }

            div.aetion-sidebar-logo span.aeti {
                color: #B8860B !important;
            }

            div.aetion-sidebar-logo span.on {
                color: #202F57 !important;
            }

            </style>

            <div class="aetion-sidebar-logo">
                <span class="aeti"><i>Aeti</i></span><span class="on">ON</span>
            </div>
            """,
            unsafe_allow_html=True
        )
