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
            color: #5EA4F3 !important;
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


# =====================================================
# SCHOOL APP STYLING & HERO COMPONENTS
# =====================================================

def apply_school_theme():
    """Injects the corporate light-blue background, navy sidebar, and card styles."""
    add_sidebar_logo()
    
    st.markdown(
        """
        <style>
        /* Base page background & font color */
        .stApp {
            background-color: #EBF3FA !important;
            color: #0F172A !important;
        }

        /* Deep Navy Sidebar matching University app */
        section[data-testid="stSidebar"] {
            background-color: #1A365D !important;
        }
        section[data-testid="stSidebar"] * {
            color: #E2E8F0 !important;
        }

        /* Metric cards styling */
        div[data-testid="stMetricValue"] {
            color: #1E3A8A !important;
            font-weight: 700 !important;
        }
        div[data-testid="metric-container"] {
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 12px !important;
            padding: 16px !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
        }
        div[data-testid="stMetricLabel"] p {
            color: #475569 !important;
            font-weight: 600 !important;
        }

        /* Hero banner */
        .hero-banner {
            background: linear-gradient(135deg, #1E40AF 0%, #2563EB 50%, #3B82F6 100%);
            padding: 26px 32px;
            border-radius: 16px;
            color: #FFFFFF !important;
            margin-bottom: 20px;
            box-shadow: 0 4px 14px rgba(30, 64, 175, 0.18);
        }
        .hero-banner h1 {
            color: #FFFFFF !important;
            font-size: 28px !important;
            font-weight: 700 !important;
            margin: 0 0 8px 0 !important;
        }
        .hero-banner p {
            color: #DBEAFE !important;
            font-size: 15px !important;
            margin: 0 !important;
            line-height: 1.5;
        }

        /* White Highlight Card */
        .content-card-white {
            background-color: #FFFFFF;
            padding: 20px 24px;
            border-radius: 12px;
            border: 1px solid #CBD5E1;
            color: #334155;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
            font-size: 14.5px;
            line-height: 1.6;
        }

        /* Navy Scope Sub-card */
        .content-card-navy {
            background-color: #1E3A8A;
            padding: 14px 20px;
            border-radius: 10px;
            color: #E0F2FE;
            font-size: 14px;
            margin-bottom: 24px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_school_hero(title: str, subtitle: str, description: str = None, scope_badge: str = None):
    """Renders the top banner and card containers."""
    st.markdown(
        f"""
        <div class="hero-banner">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if description:
        st.markdown(
            f"""
            <div class="content-card-white">
                {description}
            </div>
            """,
            unsafe_allow_html=True
        )

    if scope_badge:
        st.markdown(
            f"""
            <div class="content-card-navy">
                {scope_badge}
            </div>
            """,
            unsafe_allow_html=True
        )