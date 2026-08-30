"""
Ticket 6: causal.py — confounder-adjusted association estimates.

Answers a DIFFERENT question than SHAP (explain.py): "among students
who otherwise look similar on the measured confounders, is this
actionable factor associated with a different dropout rate?"

This is explicitly NOT a randomized-experiment causal effect — it's an
observational, confounder-adjusted association that assumes the
confounder set adequately captures the relevant differences between
students. That assumption must travel with every number this module
produces (see ASSUMPTION_CAVEAT below).

Per the spec's variable-bucketing table, this is computed ONLY for the
3 ACTIONABLE variables — never for mediators, confounders, or fixed
variables.

Modeling decision flagged explicitly (not spelled out column-by-column
in the spec): we use OLS (a linear probability model) rather than
logistic regression for the adjustment. Spec section 7 just says
"regression adjustment (statsmodels)" without naming OLS vs Logit.
OLS was chosen because its coefficient reads directly as "X
percentage-point change in dropout probability" — the plain-language
interpretation the UI needs — whereas a logistic coefficient would need
exponentiating into an odds ratio to mean anything to an advisor. This
is the standard choice for treatment-effect-style regression adjustment
in observational studies (a "linear probability model").
"""

import statsmodels.api as sm
import pandas as pd

from src.variable_roles import ACTIONABLE, CONFOUNDER, resolve_columns

ASSUMPTION_CAVEAT = (
    "This estimate assumes we have measured the important confounders. "
    "It is a confounder-adjusted association from observational data, "
    "not a randomized-experiment causal effect."
)


def run_causal_adjustment(df: pd.DataFrame) -> dict:
    """
    For each of the 3 actionable variables, fit:
        dropout_binary ~ actionable_var + confounder_1 + ... + confounder_k

    `df` must already have a `dropout_binary` column (see
    data_loader.build_binary_target) and the standardized column names
    from data_loader.load_and_clean_data.

    Returns a dict keyed by actionable variable name, each value a dict:
      {
        "coefficient": float,        # adjusted association, in probability points (e.g. 0.15 = +15pp)
        "p_value": float,
        "conf_int_low": float,
        "conf_int_high": float,
        "n_obs": int,
        "confounders_used": [...],   # columns actually controlled for
        "assumption_caveat": ASSUMPTION_CAVEAT,
      }
    """
    if "dropout_binary" not in df.columns:
        raise ValueError(
            "run_causal_adjustment expects a 'dropout_binary' column — "
            "call data_loader.build_binary_target(df) first."
        )

    actionable_cols = resolve_columns(df, ACTIONABLE)
    confounder_cols = resolve_columns(df, CONFOUNDER)

    results = {}
    for actionable_var in actionable_cols:
        predictor_cols = [actionable_var] + confounder_cols
        X = df[predictor_cols].copy()
        X = sm.add_constant(X)
        y = df["dropout_binary"]

        fitted = sm.OLS(y, X).fit()

        coef = fitted.params[actionable_var]
        p_value = fitted.pvalues[actionable_var]
        ci_low, ci_high = fitted.conf_int().loc[actionable_var]

        results[actionable_var] = {
            "coefficient": coef,
            "p_value": p_value,
            "conf_int_low": ci_low,
            "conf_int_high": ci_high,
            "n_obs": int(fitted.nobs),
            "confounders_used": confounder_cols,
            "assumption_caveat": ASSUMPTION_CAVEAT,
        }

    return results


def result_to_text(actionable_var: str, result: dict) -> str:
    """Plain-language, judge-safe summary of one actionable variable's adjusted association."""
    direction = "higher" if result["coefficient"] > 0 else "lower"
    significance = (
        "statistically significant at the 0.05 level"
        if result["p_value"] < 0.05
        else "not statistically significant at the 0.05 level"
    )
    return (
        f"After adjusting for {', '.join(result['confounders_used'])}, "
        f"**{actionable_var}** is associated with a "
        f"{abs(result['coefficient']) * 100:.1f} percentage-point {direction} "
        f"dropout rate (95% CI: {result['conf_int_low'] * 100:.1f} to "
        f"{result['conf_int_high'] * 100:.1f} pts, p={result['p_value']:.3f}, "
        f"{significance}, n={result['n_obs']}). {result['assumption_caveat']}"
    )
