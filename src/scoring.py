"""
Admin file-upload scoring: run the already-trained predictive model
against a NEW batch of students an admin uploads (e.g. this term's
active roster), producing a per-student verdict (risk tier, top
contributing factors, suggested intervention) — score only, never
retrains anything, never touches the causal-adjustment module.

The uploaded file is expected to have the SAME feature columns as the
UCI training data (raw UCI headers, e.g. "Tuition fees up to date") —
but no Target column, since these are current students with an unknown
outcome. Column names are standardized the same way as the training
data (data_loader.standardize_columns) before anything else happens.
"""

import io
import pandas as pd

from src.risk_tiers import classify_tiers_batch


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """
    Read an admin-uploaded CSV (a Streamlit UploadedFile). Tries
    semicolon delimiting first (the UCI dataset's native format), falls
    back to auto-detected delimiter — mirrors data_loader.load_raw_data's
    handling of the original file, since admins may re-export via Excel
    (comma) or keep the UCI-native format (semicolon).
    """
    raw_bytes = uploaded_file.getvalue()
    df = pd.read_csv(io.BytesIO(raw_bytes), sep=";")
    if df.shape[1] < 5:
        df = pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine="python")
    return df


def validate_schema(df: pd.DataFrame, expected_columns: list):
    """
    Confirm the (already standardized) uploaded DataFrame has every
    feature column the model was trained on.

    Returns (missing, extra):
      missing - required columns not present in the upload (blocks scoring)
      extra   - columns present but not used by the model (dropped, warning only)
    """
    uploaded_cols = set(df.columns)
    expected_cols = set(expected_columns)
    missing = sorted(expected_cols - uploaded_cols)
    extra = sorted(uploaded_cols - expected_cols)
    return missing, extra


def find_missing_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Rows/columns with missing data among the required feature columns."""
    subset = df[columns]
    na_mask = subset.isna()
    if not na_mask.values.any():
        return pd.DataFrame(columns=["row", "column"])
    rows, cols = na_mask.values.nonzero()
    return pd.DataFrame({
        "row": rows,
        "column": [columns[c] for c in cols],
    })


def coerce_numeric(df: pd.DataFrame, columns: list):
    """
    Coerce required feature columns to numeric. Returns (clean_df,
    bad_cells) where bad_cells is a list of dicts describing any cell
    that couldn't be converted — scoring can't proceed until these are
    fixed, since the model expects purely numeric input.
    """
    clean_df = df.copy()
    bad_cells = []
    for col in columns:
        original = clean_df[col]
        converted = pd.to_numeric(original, errors="coerce")
        newly_bad = converted.isna() & original.notna()
        if newly_bad.any():
            for row in original.index[newly_bad]:
                bad_cells.append({"row": row, "column": col, "value": original.loc[row]})
        clean_df[col] = converted
    return clean_df, bad_cells


def score_students(input_df: pd.DataFrame, model, feature_order: list, cutoffs: dict):
    """
    Run the trained model on a validated, cleaned, numeric DataFrame.

    input_df must already have exactly `feature_order`'s columns present
    (drop any extras before calling this) and no missing/non-numeric
    values.

    Reindexes to `feature_order` explicitly before predicting — the
    fitted scaler/classifier depend on column ORDER as well as names,
    so this guards against a silently-misaligned prediction if the
    upload's column order differs from the order used at training time.

    Returns (scores_df, ordered_features):
      scores_df       - DataFrame indexed like input_df with
                         predicted_proba and risk_tier columns
      ordered_features - input_df reindexed to feature_order (this is
                         what SHAP/explain functions should be run on,
                         so their column order matches what the model
                         and explainer expect)
    """
    ordered = input_df[feature_order]
    proba = model.predict_proba(ordered)[:, 1]
    tiers = classify_tiers_batch(proba, cutoffs)

    scores_df = pd.DataFrame({
        "predicted_proba": proba,
        "risk_tier": tiers,
    }, index=input_df.index)

    return scores_df, ordered
from pathlib import Path

ROSTER_FILE = Path("data/incremental_roster.csv")

def append_to_cumulative_roster(ordered_features: pd.DataFrame, scores_df: pd.DataFrame) -> pd.DataFrame:
    """Combines features and scores, appends to CSV on disk, and returns full cumulative table."""
    ROSTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Merge features with their computed predictions
    current_batch = ordered_features.copy()
    current_batch["predicted_proba"] = scores_df["predicted_proba"]
    current_batch["risk_tier"] = scores_df["risk_tier"]
    
    if ROSTER_FILE.exists():
        existing_df = pd.read_csv(ROSTER_FILE)
        combined = pd.concat([existing_df, current_batch], ignore_index=True).drop_duplicates()
    else:
        combined = current_batch
        
    combined.to_csv(ROSTER_FILE, index=False)
    return combined