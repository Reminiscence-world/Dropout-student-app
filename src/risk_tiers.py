"""
Ticket 7: risk_tiers.py — tertile-based Low/Medium/High risk labeling.

Cutoffs are derived from the test set's own predicted-probability
distribution (data-driven tertiles: bottom third = Low, middle third =
Medium, top third = High) — NOT fixed clinical or institutional
thresholds. Per spec section 8, the UI must always label these as
"model-based / illustrative thresholds," never as official university
risk categories.
"""

import pandas as pd

TIER_LABELS = ["Low", "Medium", "High"]

TIER_CAVEAT = (
    "These are model-based / illustrative thresholds — tertiles of this "
    "test set's predicted dropout probability — not official university "
    "risk categories."
)


def assign_risk_tiers(y_pred_proba):
    """
    Assign Low/Medium/High tiers based on tertiles of the given predicted
    probabilities.

    Returns (tiers, cutoffs):
      tiers   - pandas Series of "Low"/"Medium"/"High", one per input value.
      cutoffs - dict {"low_medium": float, "medium_high": float} with the
                two tertile boundary probabilities, so the UI can show
                exactly where the lines were drawn.
    """
    proba = pd.Series(y_pred_proba).reset_index(drop=True)

    tiers, bin_edges = pd.qcut(proba, q=3, labels=TIER_LABELS, retbins=True)

    cutoffs = {
        "low_medium": bin_edges[1],
        "medium_high": bin_edges[2],
    }

    return tiers, cutoffs


def tier_counts(tiers) -> pd.DataFrame:
    """Tier counts as a small DataFrame, ordered Low/Medium/High for charting."""
    counts = pd.Series(tiers).value_counts().reindex(TIER_LABELS).fillna(0).astype(int)
    return pd.DataFrame({"tier": TIER_LABELS, "count": counts.values})


def classify_tier_single(proba: float, cutoffs: dict) -> str:
    """
    Classify ONE new predicted probability into Low/Medium/High using
    PRE-COMPUTED cutoffs (e.g. from assign_risk_tiers on the real test
    set) — used for a single new point (a What-If Simulator prediction)
    rather than recomputing tertiles, which would be meaningless with
    n=1.
    """
    if proba <= cutoffs["low_medium"]:
        return "Low"
    elif proba <= cutoffs["medium_high"]:
        return "Medium"
    return "High"


def classify_tiers_batch(proba_array, cutoffs: dict):
    """
    Classify a batch of NEW predicted probabilities (e.g. an admin's
    uploaded file) against the SAME frozen cutoffs from the real test
    set, rather than computing fresh tertiles from the new batch —
    a differently-sized or differently-shaped batch would otherwise
    produce tiers that aren't comparable to the rest of the app.
    Returns a numpy array of tier label strings, same length as input.
    """
    return pd.Series(proba_array).apply(lambda p: classify_tier_single(p, cutoffs)).values
