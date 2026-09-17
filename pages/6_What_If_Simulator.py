"""
Ticket 15 (Stretch goal): pages/6_What_If_Simulator.py — What-If Simulator.

Lets an advisor edit one student's ACTIONABLE and MEDIATOR inputs — the
two buckets that plausibly represent something that could change for a
real student (tuition/scholarship/debtor status; grades and units
approved). CONFOUNDER and FIXED variables (age, gender, nationality,
parents' occupation, etc.) are shown as read-only context and are never
editable here — simulating "what if this student were a different
gender/nationality" isn't a meaningful lever and would undercut the
fairness framing used everywhere else in this app (spec section 13).

Re-runs ONLY the predictive model on the edited inputs — never the
causal-adjustment module (per spec section 10). A persistent disclaimer
is shown with every result: this is a hypothetical run of the
CORRELATIONAL predictive model, not proof that changing anything would
actually change a real student's risk.
"""

import streamlit as st

from src.data_loader import load_and_clean_data
from src.model import train_model
from src.risk_tiers import assign_risk_tiers, TIER_CAVEAT
from src.variable_roles import ACTIONABLE, MEDIATOR
from src.theme import PAGE_ICONS

st.set_page_config(page_title="What-If Simulator", page_icon=PAGE_ICONS["what_if"], layout="wide")

DISCLAIMER = (
    "**Hypothetical simulation using the predictive model.** Does not prove "
    "that changing this factor would actually reduce dropout risk."
)


def classify_tier(proba: float, cutoffs: dict) -> str:
    """Classify a single new probability using the SAME tertile cutoffs
    computed on the real test set (Ticket 7), rather than recomputing
    tertiles from a single hypothetical point."""
    if proba <= cutoffs["low_medium"]:
        return "Low"
    elif proba <= cutoffs["medium_high"]:
        return "Medium"
    return "High"


st.title("What-If Simulator")
st.caption(
    "Edit a student's actionable and early-warning inputs to see how the "
    "PREDICTIVE model's output would change. This does not re-run the "
    "causal-adjustment module and is not a causal claim."
)

# --- AETION-inspired visual theme ---
st.markdown("""
<style>
    /* Main page background */
    .stApp {
        background-color: #DBE8F4;
        color: #17243A;
    }

    /* Main content text */
    .stApp, .stApp p, .stApp label, .stApp span, .stApp div {
        color: #17243A;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #17243A !important;
    }

    /* Secondary/caption text */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #5B6B7F !important;
    }

    /* Blue accent for interactive elements */
    [data-baseweb="select"] > div,
    .stSelectbox > div > div {
        border-color: #2E73B8 !important;
    }

            /* Sidebar - white text */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] * {
            color: #FFFFFF !important;
        }
        
        /* Sidebar input text */
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            color: #FFFFFF !important;
        }
        
        /* Sidebar selectbox text */
        [data-testid="stSidebar"] [data-baseweb="select"] * {
            color: #FFFFFF !important;
        }
        
        /* Sidebar captions / secondary text */
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #FFFFFF !important;
        }
        
        /* Sidebar buttons */
        [data-testid="stSidebar"] .stButton > button {
            color: #FFFFFF !important;
        }
            
            /* Buttons / primary accents */
            .stButton > button {
                background-color: #2E73B8;
                color: #FFFFFF !important;
                border: 1px solid #2E73B8;
            }
        
            .stButton > button:hover {
                background-color: #245F98;
                border-color: #245F98;
            }
        
            /* Dividers */
            hr {
                border-color: #C9DDEC !important;
            }
        
            /* Metric values */
            [data-testid="stMetricValue"] {
                color: #17243A !important;
            }
        
            /* Radio/slider accents */
            [data-baseweb="radio"] [aria-checked="true"] {
                background-color: #2E73B8 !important;
                border-color: #2E73B8 !important;
            }
        
            /* Warning boxes: keep them compatible with the light blue theme */
            [data-testid="stAlert"] {
                background-color: #F3F8FC;
                border-color: #5EA4F3;
            }
        </style>
""", unsafe_allow_html=True)


st.warning(DISCLAIMER)

df = load_and_clean_data()
results = train_model(df)
tiers, cutoffs = assign_risk_tiers(results["y_pred_proba"])

X_test = results["X_test"]
student_ids = X_test.index.tolist()

selected_id = st.selectbox(
    "Select a student (type to search by ID)",
    options=student_ids,
    format_func=lambda x: f"Student ID {x}",
)
position = X_test.index.get_loc(selected_id)

original_row = X_test.iloc[[position]].copy()  # single-row DataFrame
original_proba = results["y_pred_proba"][position]
original_tier = classify_tier(original_proba, cutoffs)

st.divider()
st.subheader("Edit this student's inputs")
st.caption(
    "Only actionable factors (tuition, scholarship, debtor status) and "
    "early-warning factors (grades, units approved) are editable here — "
    "confounders and demographic variables are shown as fixed context "
    "below and are never something to hypothetically change."
)

edited_values = {}
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Actionable factors**")
    for feature in ACTIONABLE:
        if feature not in original_row.columns:
            continue
        current_val = int(original_row[feature].iloc[0])
        label = feature.replace("_", " ").title()
        choice = st.radio(
            label, options=["No", "Yes"], index=current_val, horizontal=True, key=f"edit_{feature}"
        )
        edited_values[feature] = 1 if choice == "Yes" else 0

with col_b:
    st.markdown("**Early-warning factors (mediators)**")
    for feature in MEDIATOR:
        if feature not in original_row.columns:
            continue
        col_min = float(X_test[feature].min())
        col_max = float(X_test[feature].max())
        current_val = float(original_row[feature].iloc[0])
        label = feature.replace("_", " ").title()

        if "approved" in feature:
            new_val = st.slider(
                label, min_value=int(col_min), max_value=int(col_max),
                value=int(current_val), step=1, key=f"edit_{feature}",
            )
        else:
            new_val = st.slider(
                label, min_value=col_min, max_value=col_max,
                value=current_val, step=0.1, key=f"edit_{feature}",
            )
        edited_values[feature] = new_val

with st.expander("Other inputs used by the model (read-only, fixed for this simulation)"):
    other_cols = [c for c in original_row.columns if c not in ACTIONABLE and c not in MEDIATOR]
    st.dataframe(original_row[other_cols])

# --- Build the hypothetical row and re-run ONLY the predictive model ---
hypothetical_row = original_row.copy()
for feature, value in edited_values.items():
    hypothetical_row[feature] = value

new_proba = results["model"].predict_proba(hypothetical_row)[0, 1]
new_tier = classify_tier(new_proba, cutoffs)

st.divider()
st.subheader("Result")

c1, c2 = st.columns(2)
with c1:
    st.metric("Original predicted P(Dropout)", f"{original_proba:.2f}")
    st.caption(f"Risk tier: {original_tier}")
with c2:
    delta = new_proba - original_proba
    st.metric(
        "Hypothetical predicted P(Dropout)",
        f"{new_proba:.2f}",
        delta=f"{delta:+.2f}",
        delta_color="inverse",  # a drop in predicted dropout is the "good" direction
    )
    st.caption(f"Risk tier: {new_tier}")

st.caption(TIER_CAVEAT)
st.warning(DISCLAIMER)
