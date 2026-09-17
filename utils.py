import streamlit as st


def add_sidebar_logo():
    st.markdown(
        """
        <style>

        /* =====================================================
           AETION SIDEBAR LOGO
           ===================================================== */

        .aetion-sidebar-logo {
            position: fixed;
            top: 18px;
            left: 27px;
            z-index: 999999;

            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.8rem;
            font-weight: 700;
            letter-spacing: -0.04em;
            line-height: 1.2;

            white-space: nowrap;
        }

        /* Aeti = dark gold */
        .aetion-sidebar-logo .aeti {
            color: #B8860B !important;
        }

        /* ON = pure black */
        .aetion-sidebar-logo .on {
            color: #000000 !important;
        }


        /* =====================================================
           PUSH SIDEBAR NAV DOWN SO IT DOES NOT OVERLAP LOGO
           ===================================================== */

        [data-testid="stSidebarNav"] {
            padding-top: 55px !important;
        }

        </style>


        <div class="aetion-sidebar-logo">
            <span class="aeti"><i>Aeti</i></span><span class="on">ON</span>
        </div>
        """,
        unsafe_allow_html=True
    )
