"""
Ticket 14 (Polish Pass): shared color/style constants so every page's
charts, callouts, and sidebar entries read consistently instead of each
page picking its own ad hoc colors and icons.
"""

APP_TITLE = "Causal Modeling for Early Student Dropout Interventions"
APP_ICON = "🎓"

# Shared by every "this pushes risk up / down" chart, whichever module
# produced the number (SHAP direction on Student Explorer, or the sign
# of a causal-adjustment coefficient on Causal Insights) — same colors,
# same meaning, wherever it shows up.
RISK_UP_COLOR = "#d62728"    # red — pushes toward more dropout risk
RISK_DOWN_COLOR = "#2ca02c"  # green — pushes toward less dropout risk

DIRECTION_COLORS = {
    "increases risk": RISK_UP_COLOR,
    "decreases risk": RISK_DOWN_COLOR,
}

# Risk TIER colors (Low/Medium/High) — a separate 3-way palette from the
# 2-way up/down colors above, since a tier isn't "up vs. down."
RISK_TIER_COLORS = {
    "Low": "#2ca02c",
    "Medium": "#f0ad4e",
    "High": "#d62728",
}

PAGE_ICONS = {
    "home": "🏠",
    "dashboard": "📊",
    "student_explorer": "🧑‍🎓",
    "causal_insights": "🔗",
    "model_evaluation": "📈",
    "about": "ℹ️",
    "what_if": "🧪",
}

import streamlit as st

def render_stat_card(title: str, value: str, subtitle: str, color_border: str = "#2563eb"):
    card_html = f"""
    <div style="
        background-color: #1e293b;
        border-left: 5px solid {color_border};
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    ">
        <p style="margin: 0; font-size: 13px; color: #94a3b8; font-weight: 600; text-transform: uppercase;">{title}</p>
        <p style="margin: 4px 0 0 0; font-size: 24px; font-weight: 700; color: #f8fafc;">{value}</p>
        <p style="margin: 4px 0 0 0; font-size: 12px; color: #64748b;">{subtitle}</p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)