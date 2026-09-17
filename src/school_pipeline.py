"""
School dropout data pipeline.

This module loads, cleans, validates, prepares, trains, and exports
the K-12 school dropout model.

This is separate from the existing university dropout model.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
)


# ---------------------------------------------------------------------------
# Required public interface
# ---------------------------------------------------------------------------

SCHOOL_CATEGORICAL = [
    "School_Type",
    "Location",
    "Infrastructure",
    "Teaching_Staff",
    "Gender",
    "Caste",
    "Socioeconomic_Status",
]

SCHOOL_NUMERIC = [
    "Age",
    "Standard",
]

SCHOOL_TARGET = "Dropout_Status"

SCHOOL_DROP = [
    "Dropout_Reason",
]


# ---------------------------------------------------------------------------
# Backward-compatible names used by the existing code
# ---------------------------------------------------------------------------

SCHOOL_CATEGORICAL_FEATURES = SCHOOL_CATEGORICAL
SCHOOL_NUMERIC_FEATURES = SCHOOL_NUMERIC

SCHOOL_FEATURES = SCHOOL_CATEGORICAL + SCHOOL_NUMERIC

SCHOOL_TARGET_VALUES = {
    "Dropout",
    "Enrolled",
}

SCHOOL_EXCLUDED_COLUMNS = [
    "Dropout_Reason",
    "index",
]


RANDOM_STATE = 42
POSITIVE_LABEL = "Dropout"


# ---------------------------------------------------------------------------
# Step 1: Load
# ---------------------------------------------------------------------------

def load_school_data(
    path: str = "data/SIH-Dataset.csv",
) -> pd.DataFrame:
    """Load the K-12 school dropout dataset."""
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Step 2: Clean
# ---------------------------------------------------------------------------

def clean_school_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the approved cleaning policy.

    1. Copy the dataframe so the original data is never modified.
    2. Invalid Teaching_Staff values are recoded to 'Unknown'.
    3. Rows missing Location are dropped.
    4. Dropout_Reason is never used as a model feature.
    """
    cleaned = df.copy()

    valid_teaching_staff = {
        "Good",
        "Poor",
        "Excellent",
    }

    invalid_teaching_staff = ~cleaned["Teaching_Staff"].isin(
        valid_teaching_staff
    )

    cleaned.loc[
        invalid_teaching_staff,
        "Teaching_Staff",
    ] = "Unknown"

    cleaned = cleaned.dropna(
        subset=["Location"]
    ).copy()

    return cleaned


# ---------------------------------------------------------------------------
# Step 3: Validate
# ---------------------------------------------------------------------------

def validate_school_data(df: pd.DataFrame) -> None:
    """Check that the cleaned dataset has the expected structure and values."""

    required_columns = SCHOOL_FEATURES + [SCHOOL_TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required column(s) in the dataset: "
            f"{missing_columns}. "
            f"Expected columns: {required_columns}"
        )

    unexpected_targets = (
        set(df[SCHOOL_TARGET].dropna().unique())
        - SCHOOL_TARGET_VALUES
    )

    if unexpected_targets:
        raise ValueError(
            f"'{SCHOOL_TARGET}' contains unexpected value(s): "
            f"{unexpected_targets}. "
            f"Expected only: {SCHOOL_TARGET_VALUES}"
        )

    if df[SCHOOL_TARGET].isna().any():
        n_missing = df[SCHOOL_TARGET].isna().sum()

        raise ValueError(
            f"'{SCHOOL_TARGET}' has {n_missing} missing value(s). "
            f"Every row needs a known outcome to train on."
        )


def report_missing_values(
    df: pd.DataFrame,
) -> pd.Series:
    """Return missing-value counts for approved feature columns."""
    return df[SCHOOL_FEATURES].isna().sum()


# ---------------------------------------------------------------------------
# Step 4: Split features and target
# ---------------------------------------------------------------------------

def get_school_features_and_target(df: pd.DataFrame):
    """
    Return only the approved model features and the target.

    Dropout_Reason is intentionally excluded to prevent target leakage.
    """
    X = df[SCHOOL_FEATURES].copy()
    y = df[SCHOOL_TARGET].copy()

    return X, y


# ---------------------------------------------------------------------------
# Step 5: Build preprocessing
# ---------------------------------------------------------------------------

def build_school_preprocessor() -> ColumnTransformer:
    """
    Build the unfitted preprocessing step.

    Categorical columns use OneHotEncoder with
    handle_unknown='ignore'.

    Numeric columns are passed through unchanged.
    """
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                SCHOOL_CATEGORICAL,
            ),
            (
                "numeric",
                "passthrough",
                SCHOOL_NUMERIC,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Required public function: build_school_pipeline
# ---------------------------------------------------------------------------

def build_school_pipeline() -> Pipeline:
    """
    Build the complete school-model Pipeline.

    The preprocessing and Random Forest classifier are bundled
    together so inference uses the exact same transformations
    as training.
    """

    preprocessor = build_school_preprocessor()

    classifier = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


# ---------------------------------------------------------------------------
# Required public function: train_school_model
# ---------------------------------------------------------------------------

def train_school_model(
    df: pd.DataFrame,
) -> tuple[Pipeline, dict]:
    """
    Train the school dropout model and return:

        (fitted_pipeline, metrics)

    The input dataframe is cleaned before training.
    """

    cleaned_df = clean_school_data(df)

    validate_school_data(cleaned_df)

    X, y = get_school_features_and_target(cleaned_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_school_pipeline()

    pipeline.fit(
        X_train,
        y_train,
    )

    y_pred = pipeline.predict(X_test)

    classifier = pipeline.named_steps["classifier"]
    classes = list(classifier.classes_)

    positive_class_index = classes.index(
        POSITIVE_LABEL
    )

    y_prob = pipeline.predict_proba(
        X_test
    )[:, positive_class_index]

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    dropout_precision = precision_score(
        y_test,
        y_pred,
        pos_label=POSITIVE_LABEL,
    )

    dropout_recall = recall_score(
        y_test,
        y_pred,
        pos_label=POSITIVE_LABEL,
    )

    dropout_f1 = f1_score(
        y_test,
        y_pred,
        pos_label=POSITIVE_LABEL,
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=sorted(SCHOOL_TARGET_VALUES),
    )

    f1_scorer = make_scorer(
        f1_score,
        pos_label=POSITIVE_LABEL,
    )

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring=f1_scorer,
    )

    risk_thresholds = {
        "low_medium": float(
            pd.Series(y_prob).quantile(1 / 3)
        ),
        "medium_high": float(
            pd.Series(y_prob).quantile(2 / 3)
        ),
    }

    metrics = {
        "accuracy": float(accuracy),
        "dropout_precision": float(dropout_precision),
        "dropout_recall": float(dropout_recall),
        "dropout_f1": float(dropout_f1),
        "confusion_matrix": cm.tolist(),
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "risk_thresholds": risk_thresholds,
        "cleaned_df": cleaned_df,
    }

    return pipeline, metrics


# ---------------------------------------------------------------------------
# Step 6: Export trained model + metadata
# ---------------------------------------------------------------------------

def export_school_model(
    pipeline,
    metrics,
    out_dir: str = "src/artifacts",
) -> None:
    """
    Export the fitted school model and its metadata sidecar.
    """

    out_path = Path(out_dir)
    out_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        out_path / "school_model.joblib"
    )

    metadata_path = (
        out_path / "school_model_meta.json"
    )

    # Save fitted pipeline.
    joblib.dump(
        pipeline,
        model_path,
    )

    # Get actual fitted class ordering.
    classifier = pipeline.named_steps[
        "classifier"
    ]

    classes = list(
        classifier.classes_
    )

    if "Dropout" not in classes:
        raise ValueError(
            "Expected 'Dropout' to be one "
            "of the fitted model classes."
        )

    positive_class_index = classes.index(
        "Dropout"
    )

    cleaned_df = metrics["cleaned_df"]

    categorical_values = {
        column: sorted(
            cleaned_df[column]
            .dropna()
            .unique()
            .tolist()
        )
        for column in SCHOOL_CATEGORICAL
    }

    numeric_ranges = {
        column: [
            float(
                cleaned_df[column].min()
            ),
            float(
                cleaned_df[column].max()
            ),
        ]
        for column in SCHOOL_NUMERIC
    }

    metadata = {
        "model_version": "school_dropout_v1",
        "trained_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "positive_class": "Dropout",
        "positive_class_index": positive_class_index,
        "feature_order": SCHOOL_FEATURES,
        "categorical_values": categorical_values,
        "numeric_ranges": numeric_ranges,
        "risk_thresholds": metrics[
            "risk_thresholds"
        ],
        "metrics": {
            "accuracy": metrics["accuracy"],
            "dropout_precision": metrics[
                "dropout_precision"
            ],
            "dropout_recall": metrics[
                "dropout_recall"
            ],
            "dropout_f1": metrics[
                "dropout_f1"
            ],
            "confusion_matrix": metrics[
                "confusion_matrix"
            ],
            "cv_f1_mean": metrics[
                "cv_f1_mean"
            ],
            "cv_f1_std": metrics[
                "cv_f1_std"
            ],
        },
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved fitted pipeline to {model_path}"
    )

    print(
        f"Saved model metadata to {metadata_path}"
    )


# ---------------------------------------------------------------------------
# Existing diagnostic entry point
# ---------------------------------------------------------------------------

def main():
    """Run basic pipeline diagnostics."""

    df = load_school_data()

    print(
        f"Loaded {len(df)} rows "
        f"from data/SIH-Dataset.csv"
    )

    validate_school_data(df)

    print(
        "All required columns present."
    )

    target_counts = (
        df[SCHOOL_TARGET]
        .value_counts()
        .to_dict()
    )

    print(
        f"Dropout_Status values: "
        f"{target_counts}"
    )

    missing = report_missing_values(df)

    print(
        "Missing values per feature column:"
    )

    for column, count in missing.items():
        print(
            f"  {column}: {count}"
        )

    X, y = get_school_features_and_target(
        df
    )

    print(
        f"Features (X) shape: {X.shape} | "
        f"Target (y) shape: {y.shape}"
    )

    preprocessor = (
        build_school_preprocessor()
    )

    print(
        "Preprocessor built successfully "
        f"({len(SCHOOL_CATEGORICAL)} categorical, "
        f"{len(SCHOOL_NUMERIC)} numeric columns)."
    )

    preprocessor.fit_transform(
        X.head(5)
    )

    print(
        "Preprocessor successfully "
        "transformed a 5-row sample."
    )


if __name__ == "__main__":
    main()