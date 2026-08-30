"""
Ticket 8: interventions.py — rule table mapping each actionable
variable, when it shows up as a risk-increasing contributor to a
student's SHAP explanation, to a suggested intervention.

Per spec section 9's mandatory phrasing rule: every suggestion is
framed as "suggested based on this student's risk-associated factors"
— never as a promise like "this will reduce their risk by X%." Per
spec section 13, demographic/fixed variables must NEVER appear as the
reason for a recommendation — this module only ever looks at actionable
variables, and only when they're pushing risk up for this student.
"""

from src.variable_roles import ACTIONABLE

INTERVENTION_RULES = {
    "tuition_fees_up_to_date": "Financial aid / payment plan review",
    "debtor": "Financial counseling referral",
    "scholarship_holder": "Scholarship eligibility review",
}

# scholarship_holder only triggers a suggestion when the student is
# ALSO in the High risk tier — per spec section 9's rule is "not a
# scholarship holder + high risk", not "not a scholarship holder" alone.
SCHOLARSHIP_REQUIRES_HIGH_RISK = "scholarship_holder"


def suggest_interventions(explanation: list, risk_tier: str, student_row=None) -> list:
    """
    explanation: output of explain.explain_student() — a list of dicts
      with "feature", "student_value", "shap_value", "direction".
    risk_tier: "Low" / "Medium" / "High" for this student (from
      risk_tiers.assign_risk_tiers).
    student_row: pandas Series representing the student's record (optional).

    Returns a list of suggestion strings, each already phrased per the
    spec's mandatory wording.
    """
    suggestions = []

    # 1. Actionable financial levers from SHAP explanation
    for item in explanation:
        feature = item["feature"]
        is_risk_increasing = item["direction"] == "increases risk"

        if feature not in ACTIONABLE or not is_risk_increasing:
            continue

        if feature == SCHOLARSHIP_REQUIRES_HIGH_RISK and risk_tier != "High":
            continue

        rule_text = INTERVENTION_RULES.get(feature)
        if rule_text:
            formatted_text = f"{rule_text} — suggested based on this student's risk-associated factors."
            if formatted_text not in suggestions:
                suggestions.append(formatted_text)

    # 2. Academic Support Lever (only trigger for Medium & High risk tiers)
    if student_row is not None and risk_tier in ["Medium", "High"]:
        col_approved = [c for c in student_row.index if "approved" in c.lower() and "2nd" in c.lower()]
        col_enrolled = [c for c in student_row.index if "enrolled" in c.lower() and "2nd" in c.lower()]

        if col_approved and col_enrolled:
            approved = student_row[col_approved[0]]
            enrolled = student_row[col_enrolled[0]]

            if enrolled > 0 and approved < enrolled:
                academic_text = "Academic tutoring / credit load counseling — suggested based on this student's risk-associated factors."
                if academic_text not in suggestions:
                    suggestions.append(academic_text)

    return suggestions