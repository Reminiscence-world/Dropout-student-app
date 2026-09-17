"""
Track B: dual inference engines.

Provides:
- predict_university()
- predict_school()
- health()

The school engine uses Track A's serialized pipeline when available.
The university engine currently uses a deterministic fallback scorer
until a university artifact is available.

Artifacts are loaded lazily and cached.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.interventions import recommend_school, recommend_university

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

SCHOOL_MODEL_PATH = ARTIFACT_DIR / "school_model.joblib"
SCHOOL_META_PATH = ARTIFACT_DIR / "school_model_meta.json"

# Lazy caches. Nothing is loaded during module import.
_school_model = None
_school_meta = None
_school_load_attempted = False

_university_model = None
_university_meta = None
_university_load_attempted = False


def _load_school_artifact():
    """Load the school model and metadata once, on first use."""
    global _school_model
    global _school_meta
    global _school_load_attempted

    if _school_load_attempted:
        return _school_model, _school_meta

    _school_load_attempted = True

    try:
        import joblib

        _school_model = joblib.load(SCHOOL_MODEL_PATH)

        with SCHOOL_META_PATH.open("r", encoding="utf-8") as handle:
            _school_meta = json.load(handle)

        logger.info("School model loaded successfully.")

    except Exception as exc:
        logger.warning(
            "School model unavailable; using fallback: %s",
            exc,
        )
        _school_model = None
        _school_meta = None

    return _school_model, _school_meta


def _load_university_artifact():
    """
    Load a university artifact if one exists.

    Track A only supplies the school artifact, so the university engine
    safely falls back until a university model is available.
    """
    global _university_model
    global _university_meta
    global _university_load_attempted

    if _university_load_attempted:
        return _university_model, _university_meta

    _university_load_attempted = True

    # Possible future artifact names.
    # None are required for Track A.
    model_path = ARTIFACT_DIR / "university_model.joblib"
    meta_path = ARTIFACT_DIR / "university_model_meta.json"

    if not model_path.exists() or not meta_path.exists():
        logger.warning(
            "University model artifact not found; using fallback."
        )
        return None, None

    try:
        import joblib

        _university_model = joblib.load(model_path)

        with meta_path.open("r", encoding="utf-8") as handle:
            _university_meta = json.load(handle)

        logger.info("University model loaded successfully.")

    except Exception as exc:
        logger.warning(
            "University model unavailable; using fallback: %s",
            exc,
        )
        _university_model = None
        _university_meta = None

    return _university_model, _university_meta


def _risk_tier(
    probability: float,
    thresholds: dict[str, Any] | None,
) -> str:
    """Convert probability into binary Low/High risk."""

    return "High" if probability >= 0.5 else "Low"


def _common_response(
    *,
    probability: float,
    model_loaded: bool,
    model_version: str,
    thresholds: dict[str, Any] | None,
    interventions: list[dict],
) -> dict:
    """Build the common response envelope."""

    probability = max(
        0.0,
        min(1.0, float(probability)),
    )

    risk_tier = _risk_tier(
        probability,
        thresholds,
    )

    return {
        "risk_probability": probability,
        "risk_tier": risk_tier,
        "model_loaded": model_loaded,
        "model_version": model_version,
        "interventions": interventions,
    }


def _school_fallback_probability(input_dict: dict) -> float:
    """
    Deterministic school fallback scorer.

    Starts at 0.15 and adds fixed increments for known
    risk-associated values.
    """
    score = 0.15

    if input_dict.get("Infrastructure") == "Poor":
        score += 0.20

    if input_dict.get("Teaching_Staff") in {
        "Poor",
        "Inadequate",
    }:
        score += 0.15

    if input_dict.get("Socioeconomic_Status") == "Low":
        score += 0.15

    if input_dict.get("Location") == "Rural":
        score += 0.10

    age = input_dict.get("Age")
    standard = input_dict.get("Standard")

    if isinstance(age, (int, float)) and isinstance(
        standard,
        (int, float),
    ):
        if age >= standard + 8:
            score += 0.15

    return max(
        0.02,
        min(0.95, score),
    )


def _university_fallback_probability(input_dict: dict) -> float:
    """Deterministic university fallback scorer."""
    score = 0.15

    if (
        input_dict.get("debtor") is True
        or input_dict.get("debtor") == 1
    ):
        score += 0.20

    if input_dict.get("tuition_fees_up_to_date") is False:
        score += 0.15

    if input_dict.get("scholarship_holder") is False:
        score += 0.10

    return max(
        0.02,
        min(0.95, score),
    )


def _prepare_school_input(input_dict: dict) -> dict:
    """Remove unsupported target/leakage fields without mutating input."""
    cleaned = dict(input_dict)

    cleaned.pop(
        "Dropout_Reason",
        None,
    )

    return cleaned


def _model_probability(
    model,
    meta: dict,
    input_dict: dict,
) -> float:
    """Run predict_proba and select the positive class from metadata."""

    cleaned = dict(input_dict)

    feature_order = meta["feature_order"]

    row = {
        feature: cleaned.get(feature)
        for feature in feature_order
    }

    frame = pd.DataFrame([row])

    positive_index = int(
        meta["positive_class_index"]
    )

    return float(
        model.predict_proba(frame)[0][positive_index]
    )


def predict_school(input_dict: dict) -> dict:
    """Predict school dropout risk."""

    if not isinstance(input_dict, dict):
        raise TypeError(
            "input_dict must be a dictionary"
        )

    cleaned = _prepare_school_input(input_dict)

    model, meta = _load_school_artifact()

    if model is not None and meta is not None:
        try:
            probability = _model_probability(
                model,
                meta,
                cleaned,
            )

            thresholds = meta.get(
                "risk_thresholds"
            )

            version = meta.get(
                "model_version",
                "unknown",
            )

            risk_tier = _risk_tier(
                probability,
                thresholds,
            )

            interventions = recommend_school(
                cleaned,
                risk_tier,
            )

            return _common_response(
                probability=probability,
                model_loaded=True,
                model_version=version,
                thresholds=thresholds,
                interventions=interventions,
            )

        except Exception as exc:
            logger.warning(
                "School model prediction failed; using fallback: %s",
                exc,
            )

    probability = _school_fallback_probability(
        cleaned
    )

    risk_tier = _risk_tier(
        probability,
        None,
    )

    interventions = recommend_school(
        cleaned,
        risk_tier,
    )

    return _common_response(
        probability=probability,
        model_loaded=False,
        model_version="mock-v1",
        thresholds=None,
        interventions=interventions,
    )


def predict_university(input_dict: dict) -> dict:
    """Predict university dropout risk."""

    if not isinstance(input_dict, dict):
        raise TypeError(
            "input_dict must be a dictionary"
        )

    cleaned = dict(input_dict)

    cleaned.pop(
        "Dropout_Reason",
        None,
    )

    model, meta = _load_university_artifact()

    if model is not None and meta is not None:
        try:
            probability = _model_probability(
                model,
                meta,
                cleaned,
            )

            thresholds = meta.get(
                "risk_thresholds"
            )

            version = meta.get(
                "model_version",
                "unknown",
            )

            risk_tier = _risk_tier(
                probability,
                thresholds,
            )

            interventions = recommend_university(
                cleaned,
                risk_tier,
            )

            return _common_response(
                probability=probability,
                model_loaded=True,
                model_version=version,
                thresholds=thresholds,
                interventions=interventions,
            )

        except Exception as exc:
            logger.warning(
                "University model prediction failed; using fallback: %s",
                exc,
            )

    probability = _university_fallback_probability(
        cleaned
    )

    risk_tier = _risk_tier(
        probability,
        None,
    )

    interventions = recommend_university(
        cleaned,
        risk_tier,
    )

    return _common_response(
        probability=probability,
        model_loaded=False,
        model_version="mock-v1",
        thresholds=None,
        interventions=interventions,
    )


def health() -> dict:
    """Report whether each inference engine has a live artifact."""

    school_model, school_meta = _load_school_artifact()
    university_model, university_meta = _load_university_artifact()

    return {
        "school": {
            "model_loaded": (
                school_model is not None
                and school_meta is not None
            ),
            "model_version": (
                school_meta.get("model_version")
                if school_meta is not None
                else "mock-v1"
            ),
        },
        "university": {
            "model_loaded": (
                university_model is not None
                and university_meta is not None
            ),
            "model_version": (
                university_meta.get("model_version")
                if university_meta is not None
                else "mock-v1"
            ),
        },
    }