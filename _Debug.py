"""
Hidden debug page (Ticket 14 polish pass).

This is the original Ticket 1-8 scratch page that used to live at
app.py — kept here, unchanged, in case you want to re-run the
step-by-step module checks again. It's excluded from the sidebar nav
because the filename starts with an underscore, but still reachable by
navigating to it directly (append /_Debug to the app's URL).

Every check here has since been superseded by a real page: Dashboard
(Ticket 9), Student Explorer (Ticket 10), Causal Insights (Ticket 11),
Model Evaluation (Ticket 12) all demonstrate the same functionality
live. This page is just kept for quick backend sanity-checking.
"""

import streamlit as st
import plotly.express as px
from src.data_loader import (
    load_and_clean_data,
    check_target_categories,
    check_missing_values,
    build_binary_target,
)
from src.model import train_model
from src.explain import build_explainer, get_shap_values_for_test_set, explain_student, explanation_to_text
from src.causal import run_causal_adjustment, result_to_text
from src.risk_tiers import assign_risk_tiers, tier_counts, TIER_CAVEAT
from src.interventions import suggest_interventions

st.set_page_config(page_title="Debug", page_icon="🛠️", layout="wide")

st.title("Debug: per-ticket module checks")
st.caption(
    "Hidden from the sidebar on purpose — this is the original build-time "
    "scratch page (Tickets 1-8), kept for quick backend sanity-checking."
)

try:
    df = load_and_clean_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.success(f"Loaded and cleaned data/students.csv — shape: {df.shape[0]} rows x {df.shape[1]} columns")

st.subheader("First 5 rows (standardized column names)")
st.dataframe(df.head())

col1, col2 = st.columns(2)

with col1:
    st.subheader("Missing values")
    missing_report = check_missing_values(df)
    if missing_report.empty:
        st.success("No missing values found in any column.")
    else:
        st.warning(f"{len(missing_report)} column(s) have missing values:")
        st.dataframe(missing_report)

with col2:
    st.subheader("Target category check")
    target_report = check_target_categories(df)
    if target_report["ok"]:
        st.success(target_report["message"])
    else:
        st.error(target_report["message"])
    st.write("Categories found:", target_report["found_categories"])

st.subheader("Target distribution")
target_counts = df["target"].value_counts().reset_index()
target_counts.columns = ["target", "count"]
fig = px.bar(target_counts, x="target", y="count", title="Students by outcome (Target)")
st.plotly_chart(fig, use_container_width=True)

with st.expander("All columns (standardized names)"):
    st.write(list(df.columns))

st.divider()
st.subheader("Ticket 4 check: model training")

results = train_model(df)

st.success(f"Logistic Regression trained. Test-set accuracy: {results['accuracy']:.4f}")
st.caption(
    "Compare this number to the 'Test accuracy' printed for Logistic Regression "
    "by train_and_evaluate.py — it should match, since both use the same split "
    "(random_state=42, test_size=0.2, stratified) and the same feature/target setup."
)

st.divider()
st.subheader("Ticket 5 check: SHAP explanations for sample students")
st.caption(
    "This answers 'why did the model flag this student?' — a property of the "
    "model's prediction, not a causal claim. (Causal-adjustment estimates come "
    "in Ticket 6, and only for the 3 actionable variables.)"
)

explainer = build_explainer(results["model"], results["X_train"])
shap_df = get_shap_values_for_test_set(results["model"], explainer, results["X_test"])

n_sample = min(3, len(results["X_test"]))
for position in range(n_sample):
    actual = results["y_test"].iloc[position]
    predicted_proba = results["y_pred_proba"][position]
    explanation = explain_student(position, results["X_test"], shap_df, top_n=3)

    st.markdown(
        f"**Student (test-set position {position})** — "
        f"actual: {'Dropout' if actual == 1 else 'Not Dropout'}, "
        f"predicted P(Dropout): {predicted_proba:.2f}"
    )
    st.markdown(explanation_to_text(explanation))
    st.markdown("")

st.divider()
st.subheader("Ticket 6 check: causal-adjustment (confounder-controlled) associations")
st.caption(
    "This answers a different question than SHAP above: 'among students who "
    "otherwise look similar on measured confounders, is this actionable factor "
    "associated with a different dropout rate?' Computed only for the 3 "
    "actionable variables — never for mediators, confounders, or fixed variables."
)

df_with_target = build_binary_target(df)
causal_results = run_causal_adjustment(df_with_target)

for actionable_var, result in causal_results.items():
    st.markdown(f"**{actionable_var}**")
    st.write(result_to_text(actionable_var, result))
    st.markdown("")

st.divider()
st.subheader("Ticket 7 check: risk tiers")

tiers, cutoffs = assign_risk_tiers(results["y_pred_proba"])
st.caption(TIER_CAVEAT)
st.write(
    f"Cutoffs on this test set — Low/Medium boundary: {cutoffs['low_medium']:.3f}, "
    f"Medium/High boundary: {cutoffs['medium_high']:.3f}"
)

counts_df = tier_counts(tiers)
st.dataframe(counts_df)

fig_tiers = px.bar(counts_df, x="tier", y="count", title="Test-set students by risk tier")
st.plotly_chart(fig_tiers, use_container_width=True)

st.divider()
st.subheader("Ticket 8 check: intervention suggestions")
st.caption(
    "Suggestions only ever come from the 3 actionable variables, only when "
    "they're pushing this student's risk up — never from confounders or "
    "fixed/demographic variables. Framed as 'suggested based on this "
    "student's risk-associated factors,' never as a guaranteed effect."
)

# Filter test-set students who have actionable financial risk factors
actionable_mask = (results["X_test"]["tuition_fees_up_to_date"] == 0) | (results["X_test"]["debtor"] == 1)
sample_positions = [i for i, val in enumerate(actionable_mask) if val][:3]
# Fall back to first 3 if none found
if not sample_positions:
    sample_positions = list(range(min(3, len(results["X_test"]))))

for position in sample_positions:
    actual = results["y_test"].iloc[position]
    predicted_proba = results["y_pred_proba"][position]
    tier = tiers.iloc[position]
    explanation = explain_student(position, results["X_test"], shap_df, top_n=5)
    suggestions = suggest_interventions(explanation, tier)

    st.markdown(
        f"**Student (test-set position {position})** — "
        f"actual: {'Dropout' if actual == 1 else 'Not Dropout'}, "
        f"predicted P(Dropout): {predicted_proba:.2f}, risk tier: {tier}"
    )
    if suggestions:
        for s in suggestions:
            st.markdown(f"- {s}")
    else:
        st.markdown("- No actionable, risk-increasing factor among this student's top contributors.")
    st.markdown("")
