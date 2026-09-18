"""
pages/1_Dashboard.py — cohort-level overview & batch scoring.
"""
from src.scoring import (
    read_uploaded_csv,
    validate_schema,
    find_missing_values,
    coerce_numeric,
    score_students,
    append_to_cumulative_roster,  # <-- Added
)
import sys
from pathlib import Path

# Add project root directory to Python path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
from utils import add_sidebar_logo
import pandas as pd
import plotly.express as px

from src.data_loader import load_and_clean_data, standardize_columns
from src.model import train_model
from src.risk_tiers import assign_risk_tiers, tier_counts, TIER_CAVEAT
from src.scoring import (
    read_uploaded_csv,
    validate_schema,
    find_missing_values,
    coerce_numeric,
    score_students,
)
from src.explain import (
    build_explainer,
    get_shap_values_for_test_set,
    explain_student,
)
from src.interventions import suggest_interventions


def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


st.set_page_config(page_title="Dashboard", layout="wide")
load_css()
add_sidebar_logo()

st.markdown(
    """
<div style="
padding:25px;
background:linear-gradient(135deg,#1E2A4A,#2E73B8);
border-radius:20px;
margin-bottom:20px;
">
<h1 style='color:white;'>📊 Cohort Dashboard</h1>
<p style='color:white;'>
Predicted dropout risk across the demo cohort.
Monitor students, identify risks early,
and improve retention outcomes.
</p>
</div>
""",
    unsafe_allow_html=True,
)

st.caption(
    "Predicted dropout risk across the demo cohort (this app's held-out "
    "test set). See 'About & Limitations' for why there's no live roster "
    "in this prototype."
)

df = load_and_clean_data()
results = train_model(df)
tiers, cutoffs = assign_risk_tiers(results["y_pred_proba"])

cohort = results["X_test"].reset_index(drop=True).copy()
cohort["predicted_proba"] = results["y_pred_proba"]
cohort["risk_tier"] = tiers.values
cohort["actual_dropout"] = results["y_test"].reset_index(drop=True)

# --- Headline counts ---
n_students = len(cohort)
n_high_risk = int((cohort["risk_tier"] == "High").sum())
avg_proba = cohort["predicted_proba"].mean()
actual_dropout_rate = cohort["actual_dropout"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Students in cohort", n_students)
c2.metric("High risk tier", n_high_risk, f"{n_high_risk / n_students:.0%} of cohort")
c3.metric("Avg. predicted P(Dropout)", f"{avg_proba:.2f}")
c4.metric("Actual dropout rate", f"{actual_dropout_rate:.0%}")

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.info("🎯 AI-Powered Risk Detection")
with col2:
    st.success("📈 Retention Analytics")
with col3:
    st.warning("⚡ Early Intervention Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Risk tier distribution")
    st.caption(TIER_CAVEAT)
    counts_df = tier_counts(tiers)

    fig_tiers = px.bar(
        counts_df,
        x="tier",
        y="count",
        color="tier",
        color_discrete_sequence=["#5EA4F3", "#2E73B8", "#1E2A4A"],
    )
    fig_tiers.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        title_font=dict(color="black"),
        xaxis=dict(tickfont=dict(color="black"), title_font=dict(color="black")),
        yaxis=dict(tickfont=dict(color="black"), title_font=dict(color="black")),
        legend=dict(font=dict(color="black")),
    )
    st.plotly_chart(fig_tiers, use_container_width=True)

with col2:
    st.subheader("Average predicted risk by course")
    by_course = (
        cohort.groupby("course")["predicted_proba"]
        .mean()
        .reset_index()
        .sort_values("predicted_proba", ascending=False)
    )
    by_course["course"] = by_course["course"].astype(str)
    fig_course = px.bar(
        by_course,
        x="course",
        y="predicted_proba",
        color="predicted_proba",
        color_continuous_scale=["#8BC34A", "#4CAF50", "#0F3D2E"],
    )
    fig_course.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        title_font=dict(color="black"),
        xaxis=dict(tickfont=dict(color="black"), title_font=dict(color="black")),
        yaxis=dict(tickfont=dict(color="black"), title_font=dict(color="black")),
        legend=dict(font=dict(color="black")),
    )
    st.plotly_chart(fig_course, use_container_width=True)

# ==========================================================================
# File-upload scoring
# ==========================================================================
st.markdown(
    """
<div style="
padding:25px;
background:linear-gradient(135deg,#1E2A4A,#2E73B8);
border-radius:20px;
margin-top:30px;
margin-bottom:20px;
">
<h1 style='color:white;'>📤 Score New Student Data</h1>
<p style='color:white;'>
Upload a CSV of current students to get a per-student risk verdict
from the model above — no retraining, no causal-adjustment module,
just a prediction.
</p>
</div>
""",
    unsafe_allow_html=True,
)

st.caption(
    "Expected format: same columns as the training data, but no Target "
    "column, since these students' outcomes aren't known yet."
)

uploaded_file = st.file_uploader("Upload student data (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        raw_upload_df = read_uploaded_csv(uploaded_file)
    except Exception as e:
        st.error(f"Couldn't read that file as a CSV: {e}")
        st.stop()

    cleaned_upload_df = standardize_columns(raw_upload_df)
    feature_order = list(results["X_train"].columns)
    missing, extra = validate_schema(cleaned_upload_df, feature_order)

    if missing:
        st.error(
            "This file is missing columns the model needs, so it can't be "
            f"scored: {', '.join(missing)}"
        )
        st.stop()

    if extra:
        st.warning(f"Ignoring columns not used by the model: {', '.join(extra)}")

    missing_report = find_missing_values(cleaned_upload_df, feature_order)
    if not missing_report.empty:
        st.error(
            "This file has missing values in required columns — fix these and re-upload:"
        )
        st.dataframe(missing_report, use_container_width=True)
        st.stop()

    numeric_df, bad_cells = coerce_numeric(cleaned_upload_df, feature_order)
    if bad_cells:
        st.error(
            "This file has non-numeric values in required columns — fix these and re-upload:"
        )
        st.dataframe(pd.DataFrame(bad_cells), use_container_width=True)
        st.stop()

    scores_df, ordered_features = score_students(
        numeric_df, results["model"], feature_order, cutoffs
    )
    scores_df, ordered_features = score_students(
        numeric_df, results["model"], feature_order, cutoffs
        )

    append_to_cumulative_roster(ordered_features, scores_df)

    # Compute SHAP explanations for the uploaded batch
    explainer = build_explainer(results["model"], results["X_train"])
    shap_upload_df = get_shap_values_for_test_set(
        results["model"], explainer, ordered_features
    )
    # Compute SHAP explanations for the uploaded batch
    explainer = build_explainer(results["model"], results["X_train"])
    shap_upload_df = get_shap_values_for_test_set(
        results["model"], explainer, ordered_features
    )

    verdict_rows = []
    for pos in range(len(ordered_features)):
        explanation = explain_student(pos, ordered_features, shap_upload_df, top_n=3)
        tier = scores_df["risk_tier"].iloc[pos]
        student_row = ordered_features.iloc[pos]
        suggestions = suggest_interventions(explanation, tier, student_row=student_row)
        top_factors_text = "; ".join(
            f"{e['feature']} ({e['direction']})" for e in explanation
        )
        verdict_rows.append(
            {
                "row": ordered_features.index[pos],
                "predicted_proba": round(
                    float(scores_df["predicted_proba"].iloc[pos]), 3
                ),
                "risk_tier": tier,
                "top_factors": top_factors_text,
                "suggested_interventions": (
                    "; ".join(suggestions) if suggestions else "None"
                ),
            }
        )

    verdicts_df = pd.DataFrame(verdict_rows)

    # Cache into session_state for Student Explorer
    st.session_state["uploaded_batch"] = {
        "features": ordered_features,
        "scores": scores_df,
        "shap_df": shap_upload_df,
        "verdicts": verdicts_df,
    }

    st.success(f"Successfully scored {len(scores_df)} students!")

    # Summary metrics
    n_high_upload = int((scores_df["risk_tier"] == "High").sum())
    uc1, uc2, uc3 = st.columns(3)
    uc1.metric("Students scored", len(scores_df))
    uc2.metric(
        "High risk",
        n_high_upload,
        f"{n_high_upload / len(scores_df):.0%} of upload",
    )
    uc3.metric(
        "Avg. predicted P(Dropout)", f"{scores_df['predicted_proba'].mean():.2f}"
    )

    upload_counts_df = tier_counts(pd.Series(scores_df["risk_tier"]))
    fig_upload = px.bar(
        upload_counts_df,
        x="tier",
        y="count",
        color="tier",
        color_discrete_map={
            "Low": "#5EA4F3",
            "Medium": "#2E73B8",
            "High": "#1E2A4A",
        },
        category_orders={"tier": ["Low", "Medium", "High"]},
    )
    fig_upload.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        title_font=dict(color="black"),
        xaxis=dict(tickfont=dict(color="black"), title_font=dict(color="black")),
        yaxis=dict(tickfont=dict(color="black"), title_font=dict(color="black")),
        legend=dict(font=dict(color="black")),
    )
    st.plotly_chart(fig_upload, use_container_width=True)
    st.caption(TIER_CAVEAT)

    st.subheader("📋 Per-student verdicts")
    st.dataframe(verdicts_df, use_container_width=True)

    csv_bytes = verdicts_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download verdicts as CSV",
        data=csv_bytes,
        file_name="student_risk_verdicts.csv",
        mime="text/csv",
    )

    st.info(
        "💡 **Deep Dive Ready**: Head over to the **Student Explorer** tab in the sidebar to inspect individual SHAP factor breakdowns and intervention plans for this uploaded batch!"
    )
else:
    # Clear state if file was removed
    if "uploaded_batch" in st.session_state:
        del st.session_state["uploaded_batch"]
    st.info("Upload a CSV to score a new batch of students.")