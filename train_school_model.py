"""
Train and evaluate the K-12 school dropout model.

This is separate from the existing university dropout model.

The script:
1. loads and cleans the school dataset
2. performs a stratified 80/20 train-test split
3. trains a Random Forest inside a preprocessing Pipeline
4. evaluates the held-out test set
5. performs 5-fold cross-validation on the training set
6. exports the fitted pipeline and metadata
"""

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

from src.school_pipeline import (
    load_school_data,
    clean_school_data,
    validate_school_data,
    get_school_features_and_target,
    build_school_preprocessor,
    export_school_model,
    SCHOOL_TARGET_VALUES,
)


RANDOM_STATE = 42
POSITIVE_LABEL = "Dropout"


# ---------------------------------------------------------------------------
# Build model pipeline
# ---------------------------------------------------------------------------

def build_school_model_pipeline() -> Pipeline:
    """
    Bundle preprocessing + Random Forest into one Pipeline.
    """

    preprocessor: ColumnTransformer = build_school_preprocessor()

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


# ---------------------------------------------------------------------------
# Main training flow
# ---------------------------------------------------------------------------

def main():
    # --- Load ---------------------------------------------------------------

    df_raw = load_school_data()

    print(
        f"Loaded {len(df_raw)} raw rows "
        f"from data/SIH-Dataset.csv"
    )

    # --- Clean --------------------------------------------------------------

    df = clean_school_data(df_raw)

    valid_teaching_staff = {
        "Good",
        "Poor",
        "Excellent",
    }

    n_recoded = (
        ~df_raw["Teaching_Staff"].isin(valid_teaching_staff)
    ).sum()

    n_dropped = (
        len(df_raw)
        - len(df_raw.dropna(subset=["Location"]))
    )

    print("Applied data-cleaning policy:")
    print(
        f"  Teaching_Staff values recoded to 'Unknown': "
        f"{n_recoded}"
    )
    print(
        f"  Rows dropped for missing Location:          "
        f"{n_dropped}"
    )
    print(f"Rows after cleaning: {len(df)}")

    # --- Validate -----------------------------------------------------------

    validate_school_data(df)

    X, y = get_school_features_and_target(df)

    print(
        f"Dropout_Status values: "
        f"{y.value_counts().to_dict()}"
    )

    print()

    # --- Stratified 80/20 split --------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"Train set: {len(X_train)} rows | "
        f"Test set: {len(X_test)} rows"
    )

    print()

    # --- Train --------------------------------------------------------------

    pipeline = build_school_model_pipeline()

    pipeline.fit(X_train, y_train)

    # --- Evaluate held-out test set ----------------------------------------

    y_pred = pipeline.predict(X_test)

    # Get the probability belonging specifically to Dropout.
    classifier = pipeline.named_steps["classifier"]
    classes = list(classifier.classes_)

    positive_class_index = classes.index(POSITIVE_LABEL)

    y_prob = pipeline.predict_proba(X_test)[:, positive_class_index]

    acc = accuracy_score(
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

    report = classification_report(
        y_test,
        y_pred,
    )

    # --- Risk thresholds ----------------------------------------------------

    # The thresholds are the 1/3 and 2/3 tertiles of the
    # held-out test-set Dropout probabilities.
    risk_thresholds = {
        "low_medium": float(
            pd.Series(y_prob).quantile(1 / 3)
        ),
        "medium_high": float(
            pd.Series(y_prob).quantile(2 / 3)
        ),
    }

    # --- Print test metrics -------------------------------------------------

    print("=" * 70)
    print("SCHOOL MODEL — RANDOM FOREST — TEST SET RESULTS")
    print("=" * 70)

    print(f"Accuracy:                 {acc:.4f}")
    print(
        f"Dropout class precision:  "
        f"{dropout_precision:.4f}"
    )
    print(
        f"Dropout class recall:     "
        f"{dropout_recall:.4f}"
    )
    print(
        f"Dropout class F1:         "
        f"{dropout_f1:.4f}"
    )

    print()

    print(
        "Confusion matrix "
        f"(rows=actual, cols=predicted, "
        f"order={sorted(SCHOOL_TARGET_VALUES)}):"
    )

    print(cm)

    print()

    print("Full classification report:")
    print(report)

    # --- 5-fold cross-validation --------------------------------------------

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

    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())

    print("=" * 70)
    print(
        "5-FOLD CROSS-VALIDATION "
        "(F1, Dropout class, on training data)"
    )
    print("=" * 70)

    print(
        f"Scores per fold: "
        f"{cv_scores.round(4).tolist()}"
    )

    print(
        f"Mean F1: {cv_mean:.4f} "
        f"(+/- {cv_std:.4f})"
    )

    print()

    # --- Export model + metadata -------------------------------------------

    metrics = {
        "accuracy": float(acc),
        "dropout_precision": float(dropout_precision),
        "dropout_recall": float(dropout_recall),
        "dropout_f1": float(dropout_f1),
        "confusion_matrix": cm.tolist(),
        "cv_f1_mean": cv_mean,
        "cv_f1_std": cv_std,
        "risk_thresholds": risk_thresholds,

        # Used internally by export_school_model() to generate
        # categorical_values and numeric_ranges.
        "cleaned_df": df,
    }

    export_school_model(
        pipeline,
        metrics,
        out_dir="src/artifacts",
    )

    print()
    print("Ticket 4 model export completed successfully.")


if __name__ == "__main__":
    main()