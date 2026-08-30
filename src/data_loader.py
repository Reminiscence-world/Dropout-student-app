"""
Ticket 1: minimal data loading.

Just reads data/students.csv into a pandas DataFrame. No cleaning logic
yet — that's Ticket 2. Kept deliberately small so we can confirm the raw
file loads correctly before touching it.
"""

import os
import re
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "students.csv")

EXPECTED_TARGETS = {"Dropout", "Enrolled", "Graduate"}


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the UCI 'Predict Students' Dropout and Academic Success' CSV.

    The file UCI serves is usually semicolon-delimited, but we don't want
    the app to break if someone re-saves it as a comma CSV (e.g. after
    opening it in Excel). We try semicolon first, then fall back to
    letting pandas sniff the delimiter.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Couldn't find {path}. Did you download the UCI dataset and "
            f"save it as data/students.csv? (See Ticket 0.)"
        )

    df = pd.read_csv(path, sep=";")

    # Sanity check: a correctly-delimited read should have ~35+ columns.
    # If we only got 1-2 columns back, the separator guess was wrong.
    if df.shape[1] < 5:
        df = pd.read_csv(path, sep=None, engine="python")

    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn UCI's raw column headers (e.g. "Mother's qualification",
    "Curricular units 1st sem (grade)", "Daytime/evening attendance\t")
    into consistent snake_case names we can rely on everywhere else in
    the app: mothers_qualification, curricular_units_1st_sem_grade,
    daytime_evening_attendance.

    We deliberately do NOT rename to something unrecognizable — keeping
    the words intact (just lowercased/underscored) makes it easy to map
    back to the spec's variable-bucket table by eye.
    """
    df = df.copy()

    def clean_name(name: str) -> str:
        name = name.strip()
        name = name.replace("'", "")          # Mother's -> Mothers
        name = re.sub(r"[^0-9a-zA-Z]+", "_", name)  # non-alnum -> _
        name = re.sub(r"_+", "_", name)             # collapse repeats
        name = name.strip("_").lower()
        return name

    df.columns = [clean_name(c) for c in df.columns]
    return df


def check_target_categories(df: pd.DataFrame, target_col: str = "target") -> dict:
    """
    Confirm the outcome variable has exactly the 3 expected categories
    (Dropout, Enrolled, Graduate). Returns a small report dict rather
    than raising, so the app can display what it found even if
    something's off, instead of crashing.
    """
    if target_col not in df.columns:
        return {
            "ok": False,
            "message": f"Column '{target_col}' not found in data.",
            "found_categories": [],
            "value_counts": {},
        }

    found = set(df[target_col].dropna().unique())
    value_counts = df[target_col].value_counts().to_dict()

    ok = found == EXPECTED_TARGETS
    if ok:
        message = "Target has exactly the 3 expected categories: Dropout, Enrolled, Graduate."
    else:
        unexpected = found - EXPECTED_TARGETS
        missing = EXPECTED_TARGETS - found
        parts = []
        if unexpected:
            parts.append(f"unexpected categories found: {sorted(unexpected)}")
        if missing:
            parts.append(f"expected categories missing: {sorted(missing)}")
        message = "Target categories don't match expectations — " + "; ".join(parts)

    return {
        "ok": ok,
        "message": message,
        "found_categories": sorted(found),
        "value_counts": value_counts,
    }


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame of columns with missing values (count + percent).
    Empty DataFrame means "no missing values" — the caller can check
    `.empty` to decide which message to show.
    """
    counts = df.isna().sum()
    counts = counts[counts > 0].sort_values(ascending=False)
    if counts.empty:
        return pd.DataFrame(columns=["column", "missing_count", "missing_pct"])

    report = pd.DataFrame(
        {
            "column": counts.index,
            "missing_count": counts.values,
            "missing_pct": (counts.values / len(df) * 100).round(2),
        }
    ).reset_index(drop=True)
    return report


def load_and_clean_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Ticket 2 entry point: load the raw CSV and apply cleaning
    (column-name standardization only — we do NOT drop or impute rows
    here, since the spec doesn't call for it and the dataset is known
    to be complete; check_missing_values() below is how we verify that
    assumption rather than assume it silently).
    """
    df = load_raw_data(path)
    df = standardize_columns(df)
    return df


def build_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a dropout_binary column: 1 = Dropout, 0 = Enrolled or Graduate.

    Modeling decision (made in Ticket 3, confirmed with you): everything
    downstream — predicted probability of Dropout, risk tiers, the
    causal-adjustment regression — is framed around Dropout vs not, so
    we collapse the original 3-class Target down to binary here rather
    than doing it separately in each script/module.
    """
    df = df.copy()
    df["dropout_binary"] = (df["target"] == "Dropout").astype(int)
    return df


def get_features_and_target(df: pd.DataFrame):
    """
    Split a cleaned + binary-labeled DataFrame into X (features) and y
    (dropout_binary). Per the spec's variable-bucketing table, every
    bucket (actionable, early-warning/mediator, confounder,
    fixed/demographic) may be used for PREDICTION — the causal
    restrictions only kick in later, for SHAP framing and the
    causal-adjustment module. So the predictive model uses every column
    except the two target columns.
    """
    drop_cols = ["target", "dropout_binary"]
    X = df.drop(columns=drop_cols)
    y = df["dropout_binary"]
    return X, y