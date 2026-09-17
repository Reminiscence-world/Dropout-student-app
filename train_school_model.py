"""
Ticket 2: train and evaluate the school dropout model.

Uses everything Ticket 1 built (src/school_pipeline.py) to load and prepare
the data, then trains a Random Forest, evaluates it honestly, and saves the
complete fitted pipeline (preprocessing + model, bundled together) so a
later step can hand it a raw student record and get a prediction back
without re-implementing any encoding logic.

This is completely separate from the existing university model
(train_and_evaluate.py / src/data_loader.py) — nothing there is touched
or imported here.

Run from the project root:

    python train_school_model.py
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

from src.school_pipeline import (
    load_school_data,
    clean_school_data,
    validate_school_data,
    get_school_features_and_target,
    build_school_preprocessor,
    SCHOOL_TARGET_VALUES,
)

RANDOM_STATE = 42
POSITIVE_LABEL = "Dropout"  # the class we care most about catching
ARTIFACT_PATH = "src/artifacts/school_model.joblib"


def build_school_model_pipeline() -> Pipeline:
    """Bundle preprocessing + Random Forest into one Pipeline, so the
    saved artifact can accept raw student records directly."""
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


def main():
    # --- Load, clean (see DATA_CLEANING.md), validate, split -----------
    df_raw = load_school_data()
    print(f"Loaded {len(df_raw)} raw rows from data/SIH-Dataset.csv")

    df = clean_school_data(df_raw)
    n_recoded = (~df_raw["Teaching_Staff"].isin({"Good", "Poor", "Excellent"})).sum()
    n_dropped = len(df_raw) - len(df_raw.dropna(subset=["Location"]))
    print(f"Applied data-cleaning policy (see DATA_CLEANING.md):")
    print(f"  Teaching_Staff values recoded to 'Unknown': {n_recoded}")
    print(f"  Rows dropped for missing Location:          {n_dropped}")
    print(f"Rows after cleaning: {len(df)}")

    validate_school_data(df)
    X, y = get_school_features_and_target(df)

    print(f"Dropout_Status values: {y.value_counts().to_dict()}")
    print()

    # --- Stratified 80/20 split -------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train set: {len(X_train)} rows  |  Test set: {len(X_test)} rows")
    print()

    # --- Train ---------------------------------------------------------
    pipeline = build_school_model_pipeline()
    pipeline.fit(X_train, y_train)

    # --- Evaluate on the held-out test set ------------------------------
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    dropout_precision = precision_score(y_test, y_pred, pos_label=POSITIVE_LABEL)
    dropout_recall = recall_score(y_test, y_pred, pos_label=POSITIVE_LABEL)
    dropout_f1 = f1_score(y_test, y_pred, pos_label=POSITIVE_LABEL)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(SCHOOL_TARGET_VALUES))
    report = classification_report(y_test, y_pred)

    print("=" * 70)
    print("SCHOOL MODEL — RANDOM FOREST — TEST SET RESULTS")
    print("=" * 70)
    print(f"Accuracy:                 {acc:.4f}")
    print(f"Dropout class precision:  {dropout_precision:.4f}")
    print(f"Dropout class recall:     {dropout_recall:.4f}")
    print(f"Dropout class F1:         {dropout_f1:.4f}")
    print()
    print(f"Confusion matrix (rows=actual, cols=predicted, order={sorted(SCHOOL_TARGET_VALUES)}):")
    print(cm)
    print()
    print("Full classification report:")
    print(report)

    # --- 5-fold cross-validation on the training set --------------------
    f1_scorer = make_scorer(f1_score, pos_label=POSITIVE_LABEL)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=f1_scorer)
    print("=" * 70)
    print("5-FOLD CROSS-VALIDATION (F1, Dropout class, on training data)")
    print("=" * 70)
    print(f"Scores per fold: {cv_scores.round(4).tolist()}")
    print(f"Mean F1: {cv_scores.mean():.4f}  (+/- {cv_scores.std():.4f})")
    print()

    # --- Save the complete fitted pipeline ------------------------------
    joblib.dump(pipeline, ARTIFACT_PATH)
    print(f"Saved fitted pipeline to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
