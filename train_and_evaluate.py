"""
Ticket 3: standalone model training + evaluation script.

NOT part of the Streamlit app yet — this is a plain script you run from
the terminal to compare Logistic Regression vs Random Forest and decide
which one to carry forward. Ticket 4 wraps the winner into model.py.

Run from the project root:
    python train_and_evaluate.py

Modeling decision made here (flagging it explicitly, since it's not
spelled out column-by-column in the spec):
  Target is Dropout / Enrolled / Graduate (3 classes), but the spec's
  downstream needs (predicted probability of Dropout, risk tiers,
  causal-adjustment regression on Dropout) all point to a BINARY
  target: Dropout vs Not-Dropout (Enrolled + Graduate collapsed
  together). So that's what we train on here. If you'd rather keep
  all 3 classes, say so and we'll switch before Ticket 4.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
)

from src.data_loader import load_and_clean_data, build_binary_target, get_features_and_target

RANDOM_STATE = 42


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Not Dropout", "Dropout"])
    dropout_precision = precision_score(y_test, y_pred, pos_label=1)
    dropout_recall = recall_score(y_test, y_pred, pos_label=1)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

    print("=" * 70)
    print(f"{name}")
    print("=" * 70)
    print(f"Test accuracy:               {acc:.4f}")
    print(f"Dropout class precision:     {dropout_precision:.4f}")
    print(f"Dropout class recall:        {dropout_recall:.4f}")
    print(f"5-fold CV accuracy:          {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print()
    print("Confusion matrix (rows=actual, cols=predicted, order=[Not Dropout, Dropout]):")
    print(cm)
    print()
    print("Full classification report:")
    print(report)
    print()

    return {
        "name": name,
        "accuracy": acc,
        "dropout_precision": dropout_precision,
        "dropout_recall": dropout_recall,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }


def main():
    df = load_and_clean_data()
    df = build_binary_target(df)

    print(f"Loaded {len(df)} students. Dropout rate: {df['dropout_binary'].mean():.1%}")
    print()

    X, y = get_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Logistic Regression needs scaled features to converge well and to
    # make its coefficients comparable; Random Forest doesn't care about
    # scale, but running it through an identical-shaped pipeline keeps
    # the comparison clean.
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])

    rf = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    )

    results = []
    results.append(evaluate_model("Logistic Regression", logreg, X_train, X_test, y_train, y_test))
    results.append(evaluate_model("Random Forest", rf, X_train, X_test, y_train, y_test))

    print("=" * 70)
    print("SIDE-BY-SIDE SUMMARY")
    print("=" * 70)
    summary = pd.DataFrame(results).set_index("name")
    print(summary.round(4))
    print()
    print("Selection rule from spec: prefer Logistic Regression unless Random")
    print("Forest is MEANINGFULLY more accurate (interpretability is the")
    print("tiebreaker, not raw accuracy). Compare the numbers above and decide.")


if __name__ == "__main__":
    main()