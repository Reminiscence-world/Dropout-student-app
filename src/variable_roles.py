"""
Central variable-role bucketing, per spec section 4 ("Variable Roles /
Causal Bucketing"). This bucketing is the backbone of every
causal-honesty decision in the app — every module that touches these
variables (causal.py, interventions.py, the Student Explorer and Causal
Insights pages) must import from here rather than redefining its own
list, so the buckets can't drift out of sync across the app.

Names below are the standardized (snake_case) column names produced by
data_loader.standardize_columns() — i.e. what you get from
load_and_clean_data(), not the raw UCI headers.

Per the spec's table:
  - ACTIONABLE   (treatment): may be used for prediction, SHAP, causal
                  adjustment, AND may trigger an intervention suggestion.
  - MEDIATOR     (early-warning): may be used for prediction and SHAP,
                  but NEVER gets a causal-adjustment estimate — it sits
                  too close to the outcome (could be a symptom of the
                  same underlying issue rather than a lever) — and never
                  triggers an intervention suggestion.
  - CONFOUNDER: may be used for prediction and SHAP (flagged
                  non-actionable in the UI), and is the adjustment set
                  used when estimating an actionable variable's
                  effect — never itself treated as a treatment, never
                  triggers an intervention suggestion.
  - FIXED        (demographic-only): may be used for prediction and SHAP
                  (flagged non-actionable), but is NOT part of the
                  confounder/adjustment set and never triggers an
                  intervention suggestion. Per spec section 13, these
                  must never appear as "the reason" for a recommendation.
"""

import pandas as pd

ACTIONABLE = [
    "tuition_fees_up_to_date",
    "scholarship_holder",
    "debtor",
]

MEDIATOR = [
    "curricular_units_1st_sem_grade",
    "curricular_units_2nd_sem_grade",
    "curricular_units_1st_sem_approved",
    "curricular_units_2nd_sem_approved",
]

CONFOUNDER = [
    "age_at_enrollment",
    "admission_grade",
    "previous_qualification",
    "mothers_qualification",
    "fathers_qualification",
    "mothers_occupation",
    "fathers_occupation",
    "displaced",
    "gender",
]

FIXED = [
    "nacionality",  # UCI's original spelling — kept so it matches the real column
    "marital_status",
]


def resolve_columns(df: pd.DataFrame, wanted: list) -> list:
    """
    Return only the entries of `wanted` that actually exist as columns
    in df, preserving order. Column names/availability can vary slightly
    across dataset re-exports (e.g. whether a grade sub-column is
    present), so every module using these buckets calls this instead of
    assuming every name exists — avoids a hard crash over one missing
    column, at the cost of silently narrowing the set, which is why
    callers should surface `confounders_used` / similar back to the UI.
    """
    return [c for c in wanted if c in df.columns]
#this is edited
