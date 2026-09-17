import json
from pathlib import Path

import joblib
import pandas as pd

from src.school_pipeline import (
    SCHOOL_CATEGORICAL_FEATURES,
    SCHOOL_NUMERIC_FEATURES,
    SCHOOL_FEATURES,
    SCHOOL_TARGET,
    clean_school_data,
    get_school_features_and_target,
    load_school_data,
    build_school_preprocessor,
    validate_school_data,
)


DATA_PATH = "data/SIH-Dataset.csv"
MODEL_PATH = Path("src/artifacts/school_model.joblib")
METADATA_PATH = Path("src/artifacts/school_model_meta.json")


def test_school_data_loads():
    df = load_school_data(DATA_PATH)

    assert len(df) > 0
    assert SCHOOL_TARGET in df.columns

    for column in SCHOOL_FEATURES:
        assert column in df.columns


def test_dropout_reason_is_not_a_feature():
    df = load_school_data(DATA_PATH)
    cleaned = clean_school_data(df)

    X, y = get_school_features_and_target(cleaned)

    assert "Dropout_Reason" not in X.columns
    assert SCHOOL_TARGET not in X.columns

    assert list(X.columns) == SCHOOL_FEATURES
    assert len(X) == len(y)


def test_cleaning_removes_missing_locations():
    df = load_school_data(DATA_PATH)

    cleaned = clean_school_data(df)

    assert cleaned["Location"].isna().sum() == 0
    assert len(cleaned) <= len(df)


def test_cleaning_recodes_invalid_teaching_staff():
    df = load_school_data(DATA_PATH)

    cleaned = clean_school_data(df)

    valid_values = {
        "Good",
        "Poor",
        "Excellent",
        "Unknown",
    }

    assert set(cleaned["Teaching_Staff"].unique()).issubset(
        valid_values
    )


def test_validation_passes_after_cleaning():
    df = load_school_data(DATA_PATH)
    cleaned = clean_school_data(df)

    validate_school_data(cleaned)


def test_preprocessor_handles_unknown_categories():
    df = load_school_data(DATA_PATH)
    cleaned = clean_school_data(df)

    X, _ = get_school_features_and_target(cleaned)

    preprocessor = build_school_preprocessor()

    preprocessor.fit(X)

    unknown_row = X.iloc[[0]].copy()

    unknown_row["Teaching_Staff"] = "A_CATEGORY_THAT_DOES_NOT_EXIST"

    transformed = preprocessor.transform(unknown_row)

    assert transformed.shape[0] == 1


def test_saved_school_artifact_exists():
    assert MODEL_PATH.exists(), (
        "school_model.joblib does not exist. "
        "Run: python train_school_model.py"
    )


def test_saved_school_metadata_exists():
    assert METADATA_PATH.exists(), (
        "school_model_meta.json does not exist. "
        "Run: python train_school_model.py"
    )


def test_saved_artifact_accepts_raw_student_row():
    assert MODEL_PATH.exists(), (
        "school_model.joblib does not exist. "
        "Run: python train_school_model.py"
    )

    assert METADATA_PATH.exists(), (
        "school_model_meta.json does not exist. "
        "Run: python train_school_model.py"
    )

    pipeline = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    row = {}

    for column in metadata["feature_order"]:
        if column == "Age":
            row[column] = 12
        elif column == "Standard":
            row[column] = 7
        else:
            row[column] = metadata["categorical_values"][column][0]

    input_df = pd.DataFrame([row])

    probabilities = pipeline.predict_proba(input_df)

    assert probabilities.shape[0] == 1
    assert probabilities.shape[1] == 2
    assert 0.0 <= probabilities[0][metadata["positive_class_index"]] <= 1.0


def test_metadata_has_required_fields():
    assert METADATA_PATH.exists(), (
        "school_model_meta.json does not exist. "
        "Run: python train_school_model.py"
    )

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    required_fields = {
        "model_version",
        "trained_at",
        "positive_class",
        "positive_class_index",
        "feature_order",
        "categorical_values",
        "numeric_ranges",
        "risk_thresholds",
        "metrics",
    }

    assert required_fields.issubset(metadata.keys())

    assert metadata["positive_class"] == "Dropout"
    assert metadata["feature_order"] == SCHOOL_FEATURES

    for column in SCHOOL_CATEGORICAL_FEATURES:
        assert metadata["categorical_values"][column]

    for column in SCHOOL_NUMERIC_FEATURES:
        assert len(metadata["numeric_ranges"][column]) == 2