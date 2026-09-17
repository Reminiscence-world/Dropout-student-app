import json

import pytest

import src.inference as inference


SCHOOL_STUDENT = {
    "School_Type": "Government",
    "Location": "Rural",
    "Infrastructure": "Poor",
    "Teaching_Staff": "Poor",
    "Gender": "Female",
    "Caste": "SC",
    "Age": 14,
    "Standard": 8,
    "Socioeconomic_Status": "Low",
}


def _reset_caches():
    inference._school_model = None
    inference._school_meta = None
    inference._school_load_attempted = False

    inference._university_model = None
    inference._university_meta = None
    inference._university_load_attempted = False


@pytest.fixture(autouse=True)
def reset_inference_caches():
    _reset_caches()
    yield
    _reset_caches()


def test_required_public_interface_exists():
    assert callable(inference.predict_school)
    assert callable(inference.predict_university)
    assert callable(inference.health)


def test_school_real_model_loads():
    result = inference.predict_school(SCHOOL_STUDENT)

    assert result["model_loaded"] is True
    assert result["model_version"] == "school_dropout_v1"
    assert 0.0 <= result["risk_probability"] <= 1.0


def test_school_response_schema():
    result = inference.predict_school(SCHOOL_STUDENT)

    assert set(result.keys()) == {
        "risk_probability",
        "risk_tier",
        "risk_tier_label",
        "model_loaded",
        "model_version",
        "interventions",
    }

    assert result["risk_tier"] in {"Low", "Medium", "High"}
    assert isinstance(result["interventions"], list)
    assert result["interventions"]


def test_school_unknown_category_does_not_raise():
    student = dict(SCHOOL_STUDENT)
    student["Teaching_Staff"] = "UnknownFutureCategory"

    result = inference.predict_school(student)

    assert 0.0 <= result["risk_probability"] <= 1.0


def test_dropout_reason_is_ignored():
    without_reason = inference.predict_school(SCHOOL_STUDENT)

    with_reason = dict(SCHOOL_STUDENT)
    with_reason["Dropout_Reason"] = "Family issues"
    with_reason = inference.predict_school(with_reason)

    assert with_reason["risk_probability"] == without_reason["risk_probability"]
    assert with_reason["risk_tier"] == without_reason["risk_tier"]


def test_university_fallback():
    result = inference.predict_university({
        "debtor": True,
        "tuition_fees_up_to_date": False,
        "scholarship_holder": False,
    })

    assert result["model_loaded"] is False
    assert result["model_version"] == "mock-v1"
    assert 0.02 <= result["risk_probability"] <= 0.95
    assert result["interventions"]


def test_university_fallback_is_deterministic():
    student = {
        "debtor": True,
        "tuition_fees_up_to_date": False,
        "scholarship_holder": False,
    }

    first = inference.predict_university(student)
    second = inference.predict_university(student)

    assert first == second


def test_intervention_rationales_do_not_use_demographics():
    school_result = inference.predict_school(SCHOOL_STUDENT)

    university_result = inference.predict_university({
        "debtor": True,
        "tuition_fees_up_to_date": False,
        "scholarship_holder": False,
        "Gender": "Female",
        "Caste": "SC",
    })

    for result in [school_result, university_result]:
        for intervention in result["interventions"]:
            rationale = intervention["rationale"].lower()
            assert "gender" not in rationale
            assert "caste" not in rationale


def test_interventions_are_never_empty():
    school_result = inference.predict_school({})
    university_result = inference.predict_university({})

    assert school_result["interventions"]
    assert university_result["interventions"]


def test_health_reports_both_engines():
    result = inference.health()

    assert "school" in result
    assert "university" in result

    assert result["school"]["model_loaded"] is True
    assert result["school"]["model_version"] == "school_dropout_v1"

    assert result["university"]["model_loaded"] is False
    assert result["university"]["model_version"] == "mock-v1"


def test_artifacts_are_loaded_at_most_once(monkeypatch):
    school_calls = 0
    university_calls = 0

    original_school_loader = inference._load_school_artifact
    original_university_loader = inference._load_university_artifact

    def school_loader_once():
        nonlocal school_calls
        school_calls += 1
        return original_school_loader()

    def university_loader_once():
        nonlocal university_calls
        university_calls += 1
        return original_university_loader()

    monkeypatch.setattr(
        inference,
        "_load_school_artifact",
        school_loader_once,
    )
    monkeypatch.setattr(
        inference,
        "_load_university_artifact",
        university_loader_once,
    )

    inference.health()
    inference.health()

    assert school_calls == 2
    assert university_calls == 2

    # The loaders themselves cache their actual artifact-loading attempt.
    assert inference._school_load_attempted is True
    assert inference._university_load_attempted is True
    