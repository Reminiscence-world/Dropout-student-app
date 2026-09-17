"""
Track B: deterministic intervention recommendations.

These are routing heuristics, not ML predictions or proven treatments.
Demographic attributes such as Gender and Caste are never used as
recommendation rationales.
"""


def _is_high_or_medium(risk_tier: str) -> bool:
    return risk_tier in {"Medium", "High"}


def recommend_school(input_dict: dict, risk_tier: str) -> list[dict]:
    """Return deterministic school-support recommendations."""
    recommendations = []

    socioeconomic = input_dict.get("Socioeconomic_Status")
    infrastructure = input_dict.get("Infrastructure")
    teaching_staff = input_dict.get("Teaching_Staff")
    location = input_dict.get("Location")
    age = input_dict.get("Age")
    standard = input_dict.get("Standard")

    if socioeconomic == "Low":
        recommendations.append({
            "code": "FIN_SUPPORT",
            "title": "Fee waiver / scholarship eligibility review",
            "rationale": (
                "This student's profile shows an associated risk factor "
                "in the financial-support area."
            ),
        })

    if infrastructure == "Poor":
        recommendations.append({
            "code": "INFRA_ESCALATE",
            "title": "Flag facility gap to district authority",
            "rationale": (
                "This student's profile shows an associated risk factor "
                "in the infrastructure area."
            ),
        })

    if teaching_staff in {"Poor", "Inadequate"}:
        recommendations.append({
            "code": "STAFF_GAP",
            "title": "Staffing / teaching-support escalation",
            "rationale": (
                "This student's profile shows an associated risk factor "
                "in the teaching-support area."
            ),
        })

    if location == "Rural" and _is_high_or_medium(risk_tier):
        recommendations.append({
            "code": "ACCESS_SUPPORT",
            "title": "Transport or residential access review",
            "rationale": (
                "This student's profile shows an associated risk factor "
                "in the access-support area."
            ),
        })

    if isinstance(age, (int, float)) and isinstance(standard, (int, float)):
        expected_age = standard + 6
        if age >= expected_age + 2:
            recommendations.append({
                "code": "AGE_GAP_REVIEW",
                "title": "Over-age-for-grade: remedial/bridge review",
                "rationale": (
                    "This student's profile shows an associated risk factor "
                    "in the age-for-grade area."
                ),
            })

    if risk_tier == "High" and not recommendations:
        recommendations.append({
            "code": "GENERIC_COUNSEL",
            "title": "Priority counselling outreach",
            "rationale": (
                "This student's profile shows an associated risk factor "
                "that warrants priority counselling outreach."
            ),
        })

    if not recommendations:
        recommendations.append({
            "code": "GENERAL_SUPPORT",
            "title": "General student support review",
            "rationale": (
                "This student's profile does not identify a specific "
                "routing heuristic, so a general support review is suggested."
            ),
        })

    return recommendations


def recommend_university(input_dict: dict, risk_tier: str) -> list[dict]:
    """Return deterministic university-support recommendations."""
    recommendations = []

    debtor = input_dict.get("debtor")
    tuition_due = input_dict.get("tuition_fees_up_to_date")
    scholarship_holder = input_dict.get("scholarship_holder")

    if debtor is True or debtor == 1:
        recommendations.append({
            "code": "FIN_SUPPORT",
            "title": "Financial counseling referral",
            "rationale": (
                "This student's profile shows an associated risk factor "
                "in the financial-support area."
            ),
        })

    if tuition_due is False or tuition_due == 0:
        recommendations.append({
            "code": "PAYMENT_REVIEW",
            "title": "Financial aid / payment plan review",
            "rationale": (
                "This student's profile shows an associated risk factor "
                "in the tuition-payment area."
            ),
        })

    if scholarship_holder is False and risk_tier == "High":
        recommendations.append({
            "code": "SCHOLARSHIP_REVIEW",
            "title": "Scholarship eligibility review",
            "rationale": (
                "This student's profile shows an associated risk factor "
                "in the financial-support area."
            ),
        })

    if risk_tier == "High" and not recommendations:
        recommendations.append({
            "code": "GENERIC_COUNSEL",
            "title": "Priority counselling outreach",
            "rationale": (
                "This student's profile shows an associated risk factor "
                "that warrants priority counselling outreach."
            ),
        })

    if not recommendations:
        recommendations.append({
            "code": "GENERAL_SUPPORT",
            "title": "General student support review",
            "rationale": (
                "This student's profile does not identify a specific "
                "routing heuristic, so a general support review is suggested."
            ),
        })

    return recommendations
