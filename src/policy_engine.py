from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

from .preprocessing import contains_keyword, normalize_text
from .routing import get_department_for_issue, get_default_severity

# Negation words: if these appear within 3 tokens before a safety keyword, the match is negated
NEGATORS = [
    "no", "not", "without", "never", "neither", "nor",
    "nahi", "nahin", "mat",  # Hindi/Hinglish
    "illai", "illa",  # Tamil
    "kaadu", "kadhu", "ledu",  # Telugu
    "nako",  # Marathi
]

# Positive context words: if present in same text, likely a compliment not a complaint
POSITIVE_CONTEXT_WORDS = [
    "good", "great", "nice", "helpful", "excellent", "thanks", "thank you", "pleasant",
    "clean", "hygienic", "saaf", "fresh", "well maintained", "neat", "tidy",
    "appreciate", "appreciated", "happy", "satisfied", "wonderful", "perfect",
    "fine", "ok", "okay", "correct", "proper", "appropriate",  # Added more positive indicators
]

# Neutral hygiene words: only escalate when paired with negative qualifiers
NEUTRAL_HYGIENE_WORDS = ["washroom", "toilet", "bathroom", "restroom", "garbage", "trash", "spill"]

# Negative qualifiers that turn neutral hygiene words into safety issues
NEGATIVE_QUALIFIERS = [
    "dirty", "smelly", "unclean", "overflowing", "overflow", "blocked", "broken",
    "filthy", "disgusting", "stink", "smell bad", "smells bad", "stinks",
    "clogged", "leaking", "leak", "not working", "not cleaned",
]

# Safety patterns that always escalate (no negation checking for these)
ALWAYS_ESCALATE_PATTERNS = {
    "Staff Misconduct": ["misconduct", "harassment", "inappropriate", "inappropriately", "abuse", "rude staff", "touching", "threatened"],
    "Food Contamination": ["contamination", "contaminated", "mold", "fungus", "spoiled food", "food poisoning", "cockroach", "pest", "insect in food"],
}

# Safety patterns that can be negated or need context
SAFETY_PATTERNS = {
    "Allergen Risk": ["allergen", "allergy", "contains nuts", "gluten", "cross contamination"],
    "Food Contamination": ["spoil", "spoiled", "spoiled milk", "expired", "rotten", "stale"],
    "Hygiene": ["hygiene", "dirty", "unclean", "smell bad", "smells bad", "filthy", "disgusting"],
    "Unsafe Product": ["unsafe", "dangerous", "hazard", "broken glass", "electrical shock", "slip hazard"],
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

LOW_SEVERITY_POSITIVE = POSITIVE_CONTEXT_WORDS


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
    """Match rule labels with whole-word semantics.

    Patterns are compiled once per keyword (preprocessing.keyword_pattern), so
    short keywords such as "spill" or "slip" cannot match inside unrelated
    words, and repeated evaluation does not recompile regexes.
    """
    normalized = normalize_text(text)
    matches: list[str] = []
    for label, keywords in patterns.items():
        if any(contains_keyword(normalized, keyword) for keyword in keywords if keyword):
            matches.append(label)
    return matches


def _contextual_match(normalized: str, spec: dict[str, list[str]]) -> bool:
    """Return True when a contextual safety term appears near a declaration cue.

    Both the term and the context word must be present in the same complaint for
    the rule to fire, which keeps bare ingredient mentions (a spoiling milk
    packet) from being read as an allergen declaration failure.
    """
    if not any(contains_keyword(normalized, term) for term in spec.get("terms", [])):
        return False
    return any(contains_keyword(normalized, context) for context in spec.get("context", []))


def _is_negated_simple(text: str, keyword: str) -> bool:
    """Simple negation check: does text contain both keyword and a negator NEAR each other?

    Proximity-based: negator must appear within 5 words before the keyword.
    Special case: "allergen not mentioned" is a REAL complaint, not a negation.
    """
    normalized = normalize_text(text)

    # Check if keyword is present
    if not contains_keyword(normalized, keyword):
        return False

    # Special case: "allergen/label not mentioned/declared" is a REAL complaint, not negation
    if keyword in ["allergen", "allergy"]:
        complaint_phrases = ["not mentioned", "not declared", "not labeled", "not labelled",
                            "not listed", "not written", "not printed", "not stated"]
        if any(phrase in normalized for phrase in complaint_phrases):
            return False  # This is a real complaint, not a negation

    # Proximity check: negator must be within 5 words before keyword
    words = normalized.split()
    keyword_words = keyword.split()

    # Find where keyword appears
    for i in range(len(words) - len(keyword_words) + 1):
        if all(words[i + j] == keyword_words[j] for j in range(len(keyword_words))):
            # Found keyword at position i, check 5 words before
            start = max(0, i - 5)
            window = words[start:i]

            for neg in NEGATORS:
                if neg in window:
                    return True

    return False


def _has_positive_context(text: str) -> bool:
    """Check if text contains positive context words indicating a compliment, not complaint."""
    normalized = normalize_text(text)
    return any(contains_keyword(normalized, word) for word in POSITIVE_CONTEXT_WORDS)


def _neutral_hygiene_needs_qualifier(text: str) -> bool:
    """Check if neutral hygiene words (washroom, toilet, garbage) are paired with negative qualifiers.

    Returns True only if:
    - A neutral hygiene word is present, AND
    - A negative qualifier is also present

    This prevents "washroom was clean" from escalating while catching "washroom dirty".
    """
    normalized = normalize_text(text)

    has_neutral_word = any(contains_keyword(normalized, word) for word in NEUTRAL_HYGIENE_WORDS)
    if not has_neutral_word:
        return False

    has_negative = any(contains_keyword(normalized, qual) for qual in NEGATIVE_QUALIFIERS)
    return has_negative


def _match_safety_patterns(text: str) -> list[str]:
    """Match safety rules with negation and positive context awareness.

    Always-escalate patterns (staff misconduct, food contamination with pests) are never negated.
    Other safety patterns can be negated ("no allergen issue") or occur in positive context
    ("washroom was very clean"). Neutral hygiene words only escalate when paired with
    negative qualifiers.
    """
    normalized = normalize_text(text)
    matches: list[str] = []

    # Check positive context first - if present, likely a compliment
    has_positive = _has_positive_context(text)

    # Always-escalate patterns: never negated
    for label, keywords in ALWAYS_ESCALATE_PATTERNS.items():
        for keyword in keywords:
            if keyword and contains_keyword(normalized, keyword):
                matches.append(label)
                break

    # Other safety patterns: check for negation and positive context
    for label, keywords in SAFETY_PATTERNS.items():
        for keyword in keywords:
            if not keyword:
                continue
            if not contains_keyword(normalized, keyword):
                continue

            # Skip if negated
            if _is_negated_simple(text, keyword):
                continue

            # Skip if positive context present (applies to all safety patterns)
            if has_positive:
                continue

            matches.append(label)
            break

    # Handle neutral hygiene words: only escalate with negative qualifiers
    if "Hygiene" not in matches:  # Don't double-add
        for neutral_word in NEUTRAL_HYGIENE_WORDS:
            if contains_keyword(normalized, neutral_word):
                # Only escalate if negative qualifier is present AND not positive context
                if _neutral_hygiene_needs_qualifier(text) and not has_positive:
                    matches.append("Hygiene")
                    break

    # Check contextual patterns (allergen risk with declaration context)
    spec = CONTEXTUAL_SAFETY_PATTERNS.get("Allergen Risk")
    if spec and _contextual_match(normalized, spec) and "Allergen Risk" not in matches:
        # Don't escalate if negated or positive context
        if not _is_negated_simple(text, "allergen") and not has_positive:
            matches.append("Allergen Risk")

    # Also check for "allergen" keyword with "label" context (e.g., "allergen not mentioned on label")
    if "Allergen Risk" not in matches:
        if contains_keyword(normalized, "allergen") or contains_keyword(normalized, "allergy"):
            # Check if label/packaging context exists
            label_context_words = ["label", "packaging", "ingredient", "mentioned", "declared", "listed"]
            if any(contains_keyword(normalized, ctx) for ctx in label_context_words):
                # This is an allergen labeling complaint, not a negation
                if not has_positive:
                    matches.append("Allergen Risk")

    return list(set(matches))  # Remove duplicates


def determine_severity(
    text: str,
    issue_category: str | None = None,
    baseline_severity: str | None = None,
) -> tuple[str, list[str], bool, str, str]:
    """Deterministic policy-first severity assignment."""
    normalized = normalize_text(text)
    matches = _match_safety_patterns(text)  # Pass original text, not normalized
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

    if any(contains_keyword(normalized, token) for token in LOW_SEVERITY_POSITIVE):
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
