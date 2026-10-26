from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

from .preprocessing import normalize_text
from .routing import get_department_for_issue, get_default_severity

SAFETY_PATTERNS = {
    "Allergen Risk": ["allergen", "allergy", "contains nuts", "gluten", "cross contamination"],
    "Food Contamination": ["contamination", "contaminated", "mold", "fungus", "spoiled food", "food poisoning"],
    "Hygiene": ["hygiene", "dirty", "unclean", "washroom", "toilet", "garbage", "cockroach", "spill", "smell bad"],
    "Unsafe Product": ["unsafe", "dangerous", "hazard", "broken glass", "electrical shock", "slip hazard"],
    "Staff Misconduct": ["misconduct", "harassment", "inappropriate", "abuse", "rude staff", "touching", "threatened"],
    "Customer Safety": ["customer safety", "unsafe for customers", "injury", "hurt", "accident"],
}

# Allergen terms that are only meaningful when the complaint is about a missing or
# wrong *declaration*. A bare ingredient mention is not a safety incident: "milk
# packets are spoiling" is a refrigeration failure, not an allergen risk. These
# terms therefore require a nearby declaration context word.
CONTEXTUAL_SAFETY_PATTERNS: dict[str, dict[str, list[str]]] = {
    "Allergen Risk": {
        "terms": ["milk", "egg", "nuts", "nut", "peanut", "soy", "sesame", "mustard"],
        "context": [
            "label",
            "labelled",
            "labeled",
            "labelling",
            "labeling",
            "ingredient",
            "ingredients",
            "packaging",
            "allergy",
            "allergen",
            "warning",
            "declared",
            "mentioned",
            "mention",
            "written",
            "printed",
            "print",
            "listed",
            "states",
            "says",
            "information",
            "info",
        ],
    }
}

LOW_SEVERITY_POSITIVE = ["good", "great", "nice", "helpful", "excellent", "thanks", "thank you", "pleasant"]


@dataclass(slots=True)
class PolicyDecision:
    severity: str
    actionable: bool
    department: str
    reason: str
    human_review_required: bool
    matched_rules: list[str]
    policy_source: str = "policy_rules"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class PolicyResult:
    category: str
    decision: PolicyDecision

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["decision"] = self.decision.to_dict()
        return payload


def _match_rules(text: str, patterns: dict[str, list[str]]) -> list[str]:
    normalized = normalize_text(text)
    matches: list[str] = []
    for label, keywords in patterns.items():
        if any(keyword in normalized for keyword in keywords):
            matches.append(label)
    return matches


def _contextual_match(normalized: str, spec: dict[str, list[str]]) -> bool:
    """Return True when a contextual safety term appears near a declaration cue.

    Both the term and the context word must be present in the same complaint for
    the rule to fire, which keeps bare ingredient mentions (a spoiling milk
    packet) from being read as an allergen declaration failure.
    """
    if not any(term in normalized for term in spec.get("terms", [])):
        return False
    return any(context in normalized for context in spec.get("context", []))


def _match_safety_patterns(text: str) -> list[str]:
    """Match safety rules, applying context requirements where configured."""
    normalized = normalize_text(text)
    matches: list[str] = []
    for label, keywords in SAFETY_PATTERNS.items():
        if any(keyword in normalized for keyword in keywords):
            matches.append(label)
            continue
        spec = CONTEXTUAL_SAFETY_PATTERNS.get(label)
        if spec and _contextual_match(normalized, spec):
            matches.append(label)
    return matches


def determine_severity(
    text: str,
    issue_category: str | None = None,
    baseline_severity: str | None = None,
) -> tuple[str, list[str], bool, str, str]:
    """Deterministic policy-first severity assignment."""
    normalized = normalize_text(text)
    matches = _match_safety_patterns(normalized)
    human_review_required = False
    reason_parts: list[str] = []

    if matches:
        if len(matches) > 1:
            human_review_required = True
        severity = "HIGH"
        if "Staff Misconduct" in matches:
            department = "HR"
            reason_parts.append("Potential staff misconduct identified.")
        elif "Allergen Risk" in matches:
            department = "Merchandising"
            reason_parts.append("Allergen-related safety concern detected.")
        elif "Food Contamination" in matches:
            department = "Store Operations"
            reason_parts.append("Possible food contamination or spoilage detected.")
        elif "Hygiene" in matches:
            department = "Store Operations"
            reason_parts.append("Hygiene issue flagged as a store safety concern.")
        elif "Unsafe Product" in matches:
            department = "Store Operations"
            reason_parts.append("Potential unsafe product hazard detected.")
        else:
            department = "Store Operations"
            reason_parts.append("Customer safety concern detected.")
        if len(matches) > 1:
            reason_parts.append("Ambiguous safety signal flagged for human verification.")
        return severity, matches, human_review_required, department, " ".join(reason_parts)

    if any(token in normalized for token in LOW_SEVERITY_POSITIVE):
        department = get_department_for_issue(issue_category or "Customer Service Delay")
        return "LOW", [], False, department, "Positive or low-risk feedback detected."

    severity = (baseline_severity or get_default_severity(issue_category or "")).upper()
    if severity not in {"LOW", "MEDIUM", "HIGH"}:
        severity = "MEDIUM"
    department = get_department_for_issue(issue_category or "")
    if severity == "HIGH" and not issue_category:
        reason = "High severity retained from operational policy."
    else:
        reason = f"Severity aligned to policy default for {issue_category or 'general feedback'}."
    return severity, [], False, department, reason


def evaluate_policy(text: str, issue_category: str | None = None) -> PolicyDecision:
    severity, matches, human_review_required, department, reason = determine_severity(text, issue_category)
    actionable = severity in {"MEDIUM", "HIGH"} or bool(matches)
    if not matches and severity == "LOW":
        actionable = False
    return PolicyDecision(
        severity=severity,
        actionable=actionable,
        department=department,
        reason=reason,
        human_review_required=human_review_required,
        matched_rules=matches,
    )


def should_escalate(text: str) -> bool:
    return evaluate_policy(text).severity == "HIGH"
