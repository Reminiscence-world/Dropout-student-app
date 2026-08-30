"""
Ticket 5: explain.py — SHAP-based per-student explanations.

Answers: "why did the model flag this student?" This is a property of
the model's PREDICTION, not a causal claim. (Causal claims are Ticket
6's job, and only for the 3 actionable variables — mediators and
confounders never get a causal-adjustment estimate, per the spec's
variable-bucketing table, even though they're fair game here for
explaining the prediction.)
"""

import streamlit as st
import shap
import pandas as pd


@st.cache_resource
def build_explainer(_model, X_train):
    """
    Build a SHAP LinearExplainer against the fitted Logistic Regression
    step of the pipeline.

    We explain the classifier in SCALED feature space (i.e. after the
    pipeline's StandardScaler), since that's the space the linear model
    actually operates in — SHAP values computed on raw units would be
    dominated by whichever features happen to have larger numeric
    ranges (e.g. "admission grade" vs a 0/1 flag), not by which
    features actually matter.

    The leading underscore on `_model` tells Streamlit's cache not to
    try to hash the sklearn Pipeline object (it isn't meaningfully
    hashable); X_train is still used to decide when to rebuild.
    """
    scaler = _model.named_steps["scaler"]
    clf = _model.named_steps["clf"]
    X_train_scaled = scaler.transform(X_train)
    explainer = shap.LinearExplainer(clf, X_train_scaled)
    return explainer


@st.cache_data
def get_shap_values_for_test_set(_model, _explainer, X_test):
    """
    Compute SHAP values for every row in X_test.

    Returns a DataFrame shaped exactly like X_test (same columns, same
    row index), where each cell is "how much did this feature, for this
    student, push the prediction toward Dropout (positive) or away from
    it (negative)."
    """
    scaler = _model.named_steps["scaler"]
    X_test_scaled = scaler.transform(X_test)
    shap_values = _explainer.shap_values(X_test_scaled)
    return pd.DataFrame(shap_values, columns=X_test.columns, index=X_test.index)


def explain_student(student_position: int, X_test: pd.DataFrame, shap_df: pd.DataFrame, top_n: int = 3):
    """
    Return the top contributing features for one student.

    student_position: integer POSITION (0-based, via .iloc) into X_test
    / shap_df — not a raw student ID. Ticket 10 (Student Explorer) will
    handle mapping a chosen student to a position.

    Returns a list of dicts sorted by |SHAP value| descending, e.g.:
      {"feature": "tuition_fees_up_to_date", "student_value": 0,
       "shap_value": 0.82, "direction": "increases risk"}
    """
    shap_row = shap_df.iloc[student_position]
    student_row = X_test.iloc[student_position]

    top_features = shap_row.abs().sort_values(ascending=False).head(top_n).index

    explanation = []
    for feature in top_features:
        shap_val = shap_row[feature]
        explanation.append({
            "feature": feature,
            "student_value": student_row[feature],
            "shap_value": shap_val,
            "direction": "increases risk" if shap_val > 0 else "decreases risk",
        })
    return explanation


def explanation_to_text(explanation: list) -> str:
    """Turn explain_student()'s output into a plain-English bullet list."""
    lines = []
    for item in explanation:
        lines.append(
            f"- **{item['feature']}** = {item['student_value']} "
            f"→ {item['direction']} (SHAP = {item['shap_value']:+.3f})"
        )
    return "\n".join(lines)
