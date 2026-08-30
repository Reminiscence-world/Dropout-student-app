"""
Ticket 4: model.py — wraps the chosen model into a reusable, cached
function for use inside the Streamlit app.

Logistic Regression was selected in Ticket 3 over Random Forest per the
spec's selection rule (prefer Logistic Regression unless Random Forest
is meaningfully more accurate — interpretability is the tiebreaker, not
just raw accuracy).
"""

import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

from src.data_loader import build_binary_target, get_features_and_target

RANDOM_STATE = 42


def get_train_test_split(df):
    """
    Shared split logic used by every model-training function in this
    module, so Logistic Regression (the app's chosen model) and any
    comparison model (Random Forest, for Ticket 12's Model Evaluation
    page) are always evaluated on the EXACT same split — same rows,
    same random_state — rather than two separately-called
    train_test_splits that could silently drift out of sync.
    """
    df = build_binary_target(df)
    X, y = get_features_and_target(df)
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


@st.cache_resource
def train_model(df):
    """
    Train Logistic Regression on an already-cleaned DataFrame (i.e. the
    output of data_loader.load_and_clean_data()).

    Cached with @st.cache_resource so the model is trained once per app
    session rather than retrained on every page load / widget
    interaction. Streamlit hashes the input DataFrame to know when to
    retrain, so if the underlying data changes, this reruns
    automatically.

    Returns a dict:
      model          - fitted sklearn Pipeline (StandardScaler + LogisticRegression)
      X_train, X_test, y_train, y_test - the split used for training/evaluation
      y_pred         - predicted class (0/1) on the test set
      y_pred_proba   - predicted probability of Dropout (class 1) on the test set
      accuracy       - test-set accuracy (float)
    """
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "accuracy": accuracy,
    }


@st.cache_resource
def evaluate_both_models(df):
    """
    Ticket 12: train + evaluate BOTH candidate models (Logistic
    Regression and Random Forest) on the identical split used
    everywhere else in the app, for the Model Evaluation page's
    side-by-side comparison. Mirrors Ticket 3's standalone script, but
    runs inside Streamlit (cached) instead of needing a subprocess call
    out to train_and_evaluate.py.

    Returns a dict keyed by model name, each value a dict with:
      accuracy, confusion_matrix, dropout_precision, dropout_recall,
      cv_mean, cv_std
    """
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    results = {}
    for name, candidate_model in candidates.items():
        candidate_model.fit(X_train, y_train)
        y_pred = candidate_model.predict(X_test)
        cv_scores = cross_val_score(candidate_model, X_train, y_train, cv=5, scoring="accuracy")

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "dropout_precision": precision_score(y_test, y_pred, pos_label=1),
            "dropout_recall": recall_score(y_test, y_pred, pos_label=1),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

    return results