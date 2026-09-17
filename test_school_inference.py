"""
Ticket 3: smallest possible inference test for the saved school model.

This does NOT train or modify anything. It only proves the saved
src/artifacts/school_model.joblib can accept a raw student record (the
9 approved features, nothing else) and return a usable prediction.

Run from the project root:

    python test_school_inference.py
"""

import joblib
import pandas as pd

from src.school_pipeline import SCHOOL_FEATURES

MODEL_PATH = "src/artifacts/school_model.joblib"


# Real category values, taken directly from data/SIH-Dataset.csv (nothing
# invented). Ages 10-16, Standards 5-12 are the dataset's actual ranges.
SAMPLE_STUDENTS = [
    {
        "label": "Student 1 (realistic)",
        "profile": {
            "School_Type": "Government",
            "Location": "Rural",
            "Infrastructure": "Poor",
            "Teaching_Staff": "Poor",
            "Gender": "Female",
            "Caste": "SC",
            "Age": 14,
            "Standard": 8,
            "Socioeconomic_Status": "Low",
        },
    },
    {
        "label": "Student 2 (realistic)",
        "profile": {
            "School_Type": "Private",
            "Location": "Urban",
            "Infrastructure": "Excellent",
            "Teaching_Staff": "Good",
            "Gender": "Male",
            "Caste": "General",
            "Age": 12,
            "Standard": 6,
            "Socioeconomic_Status": "High",
        },
    },
    {
        "label": "Student 3 (realistic)",
        "profile": {
            "School_Type": "Government",
            "Location": "Semi-Urban",
            "Infrastructure": "Good",
            "Teaching_Staff": "Excellent",
            "Gender": "Male",
            "Caste": "OBC",
            "Age": 13,
            "Standard": 7,
            "Socioeconomic_Status": "Moderate",
        },
    },
    {
        "label": "EDGE CASE (not a real student) — unknown category value",
        "profile": {
            "School_Type": "Government",
            "Location": "Rural",
            "Infrastructure": "Poor",
            # "Homeschool" is not a category the model was trained on.
            # This exists only to prove OneHotEncoder(handle_unknown="ignore")
            # doesn't crash on an unseen value.
            "Teaching_Staff": "Homeschool",
            "Gender": "Female",
            "Caste": "SC",
            "Age": 14,
            "Standard": 8,
            "Socioeconomic_Status": "Low",
        },
    },
]


def main():
    pipeline = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
    print()

    for case in SAMPLE_STUDENTS:
        # Build a 1-row DataFrame using only the 9 approved features, in
        # the same order the model expects. Dropout_Status, Dropout_Reason,
        # and index are never part of this dict, so they can't leak in.
        row = pd.DataFrame([case["profile"]])[SCHOOL_FEATURES]

        predicted_class = pipeline.predict(row)[0]
        probabilities = pipeline.predict_proba(row)[0]
        class_labels = pipeline.classes_
        dropout_index = list(class_labels).index("Dropout")
        dropout_probability = probabilities[dropout_index]

        assert 0.0 <= dropout_probability <= 1.0, "Probability out of [0, 1] range!"

        print("=" * 70)
        print(case["label"])
        print("=" * 70)
        print("Input profile:")
        for k, v in case["profile"].items():
            print(f"  {k}: {v}")
        print(f"Predicted class:      {predicted_class}")
        print(f"Dropout probability:  {dropout_probability:.4f}")
        print(f"Valid range check:    {'PASS' if 0.0 <= dropout_probability <= 1.0 else 'FAIL'}")
        print()


if __name__ == "__main__":
    main()
