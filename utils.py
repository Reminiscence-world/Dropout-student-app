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

        /* AETI = dark yellow + italic */
        .aetion-sidebar-logo .aeti {
            color: #B8860B !important;
            font-style: italic !important;
        }

        /* ON = black + normal */
        .aetion-sidebar-logo .on {
            color: #4B5563 !important;
            font-style: normal !important;
        }


        /* =====================================================
           KEEP SIDEBAR NAV BELOW THE LOGO
           ===================================================== */

        [data-testid="stSidebarNav"] {
            padding-top: 55px !important;
        }

        </style>


        <div class="aetion-sidebar-logo">
            <span class="aeti">AETI</span><span class="on">ON</span>
        </div>

        """,
        unsafe_allow_html=True
    )
