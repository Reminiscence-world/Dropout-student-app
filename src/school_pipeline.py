"""
Ticket 1: load, validate, and prepare the K-12 School dropout dataset.

This file does NOT train a model. It only:
  1. loads data/SIH-Dataset.csv
  2. checks the data looks the way we expect (right columns, sane values)
  3. splits it into features (X) and target (y)
  4. builds (but does not yet fit) the preprocessing step that will later
     turn text categories into numbers a model can use

Training the model is Ticket 2 — a separate step, on purpose, so any data
problems get caught here, in a small and easy place, rather than buried
inside a training run.

Run this file directly to see a plain-language summary:

    python src/school_pipeline.py
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------------------------
# Constants: the single source of truth for "which columns are allowed".
# ---------------------------------------------------------------------------

# The 9 approved features. Nothing outside this list is ever used as a
# model input.
SCHOOL_CATEGORICAL_FEATURES = [
    "School_Type",
    "Location",
    "Infrastructure",
    "Teaching_Staff",
    "Gender",
    "Caste",
    "Socioeconomic_Status",
]
SCHOOL_NUMERIC_FEATURES = ["Age", "Standard"]
SCHOOL_FEATURES = SCHOOL_CATEGORICAL_FEATURES + SCHOOL_NUMERIC_FEATURES

# The answer we're trying to predict.
SCHOOL_TARGET = "Dropout_Status"
SCHOOL_TARGET_VALUES = {"Dropout", "Enrolled"}

# Columns that must NEVER be used as a model feature, and why:
#   - Dropout_Reason: only known AFTER a student has already dropped out.
#     Using it would be target leakage (the model would look great in
#     testing and be useless in real use, since this field is blank for
#     every currently-enrolled student). We simply never select it.
#   - any "index"-style column: not real information about a student,
#     just a row number.
SCHOOL_EXCLUDED_COLUMNS = ["Dropout_Reason", "index"]


# ---------------------------------------------------------------------------
# Step 1: load
# ---------------------------------------------------------------------------

def load_school_data(path: str = "data/SIH-Dataset.csv") -> pd.DataFrame:
    """Load the school dataset. Comma-delimited (pandas' default), unlike
    the university dataset which is semicolon-delimited."""
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Step 2: validate
# ---------------------------------------------------------------------------

def validate_school_data(df: pd.DataFrame) -> None:
    """Check the data looks the way we expect. Raises a clear error and
    stops immediately if something is wrong, rather than silently
    continuing with bad assumptions.

    Returns nothing — if this function doesn't raise, the data is good
    enough to proceed.
    """
    required_columns = SCHOOL_FEATURES + [SCHOOL_TARGET]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required column(s) in the dataset: {missing_columns}. "
            f"Expected columns: {required_columns}"
        )

    unexpected_targets = set(df[SCHOOL_TARGET].dropna().unique()) - SCHOOL_TARGET_VALUES
    if unexpected_targets:
        raise ValueError(
            f"'{SCHOOL_TARGET}' contains unexpected value(s): {unexpected_targets}. "
            f"Expected only: {SCHOOL_TARGET_VALUES}"
        )

    if df[SCHOOL_TARGET].isna().any():
        n_missing = df[SCHOOL_TARGET].isna().sum()
        raise ValueError(
            f"'{SCHOOL_TARGET}' has {n_missing} missing value(s). "
            f"Every row needs a known outcome to train on."
        )


def report_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return a per-feature-column count of missing values, for visibility.
    This does NOT drop or fill anything — just reports, so nothing is
    silently swallowed. (Blanks in Dropout_Reason are expected and fine;
    that column is intentionally excluded, so it isn't checked here.)"""
    return df[SCHOOL_FEATURES].isna().sum()


# ---------------------------------------------------------------------------
# Step 3: split features and target
# ---------------------------------------------------------------------------

def get_school_features_and_target(df: pd.DataFrame):
    """Split into X (the 9 approved feature columns only) and y (the
    target). Because X is built from SCHOOL_FEATURES specifically, it is
    structurally impossible to accidentally include Dropout_Reason or
    Dropout_Status as a feature."""
    X = df[SCHOOL_FEATURES].copy()
    y = df[SCHOOL_TARGET].copy()
    return X, y


# ---------------------------------------------------------------------------
# Step 4: build (but do not fit) the preprocessor
# ---------------------------------------------------------------------------

def build_school_preprocessor() -> ColumnTransformer:
    """Build the preprocessing step that will later turn the categorical
    columns into numbers (one-hot encoding) and pass the numeric columns
    through unchanged. Returned UNFITTED — fitting happens during training
    in Ticket 2, not here."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                SCHOOL_CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", SCHOOL_NUMERIC_FEATURES),
        ]
    )


# ---------------------------------------------------------------------------
# Run directly for a plain-language summary
# ---------------------------------------------------------------------------

def main():
    df = load_school_data()
    print(f"Loaded {len(df)} rows from data/SIH-Dataset.csv")

    validate_school_data(df)
    print("All required columns present.")

    target_counts = df[SCHOOL_TARGET].value_counts().to_dict()
    print(f"Dropout_Status values: {target_counts}")

    missing = report_missing_values(df)
    print("Missing values per feature column:")
    for col, count in missing.items():
        print(f"  {col}: {count}")

    X, y = get_school_features_and_target(df)
    print(f"Features (X) shape: {X.shape}  |  Target (y) shape: {y.shape}")

    preprocessor = build_school_preprocessor()
    print(
        f"Preprocessor built successfully "
        f"({len(SCHOOL_CATEGORICAL_FEATURES)} categorical, "
        f"{len(SCHOOL_NUMERIC_FEATURES)} numeric columns)."
    )

    # Quick end-to-end sanity check: does the preprocessor actually run
    # against a few real rows without error? (Still not "training" —
    # fit_transform here is just proving the wiring works.)
    preprocessor.fit_transform(X.head(5))
    print("Preprocessor successfully transformed a 5-row sample. Ticket 1 checks passed.")


if __name__ == "__main__":
    main()
