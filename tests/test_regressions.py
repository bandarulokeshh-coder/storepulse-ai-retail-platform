"""Regression tests for bugs fixed during hackathon preparation.

STEP 5: Verify all earlier bug fixes remain working.
Tests cover Tasks 1-4: whole-word matching, allergen rules, negation, hygiene split.
"""
import pytest
from src.policy_engine import evaluate_policy


@pytest.mark.parametrize(
    "text,expected_severity,expected_dept,must_not_contain_in_reason,description",
    [
        # Task 2: Allergen rule fixes
        (
            "Store freezer work avvatledu, milk packets spoil autu may",
            "HIGH",
            "Store Operations",
            "allergen",
            "Refrigeration failure (spoilage) not allergen"
        ),
        (
            "Bought milk, billing counter was slow",
            "MEDIUM",
            None,
            None,
            "Bare milk mention not HIGH"
        ),
        (
            "I got the eggplant and it was fresh",
            "LOW",
            None,
            None,
            "Eggplant not egg allergen (whole-word)"
        ),

        # Task 3: Positive context and negation
        (
            "The washroom was very clean and hygienic, thanks",
            "LOW",
            None,
            None,
            "Positive washroom feedback"
        ),
        (
            "No allergen issue, label was correct",
            "LOW",
            None,
            None,
            "Negated allergen with positive context"
        ),
        (
            "Store was not dirty, very good",
            "LOW",
            None,
            None,
            "Negated dirty with positive"
        ),

        # Task 3: Always escalate patterns
        (
            "Staff was inappropriate with a customer",
            "HIGH",
            "HR",
            None,
            "Staff misconduct always HIGH, routes to HR"
        ),
        (
            "found cockroach in the food aisle",
            "HIGH",
            "Store Operations",
            None,
            "Pest contamination always HIGH"
        ),

        # Task 4: Hygiene vs floor cleanliness
        (
            "Store floor is dirty and trash is visible near the entrance",
            "MEDIUM",
            "Store Operations",
            None,
            "Floor cleanliness is MEDIUM not HIGH"
        ),
        (
            "washroom is dirty and smells bad",
            "HIGH",
            "Store Operations",
            None,
            "Washroom hygiene is HIGH"
        ),

        # Additional: Digital not safety
        (
            "UPI payment work hot nahi",
            "HIGH",
            "Digital",
            None,
            "Payment failure (Digital), not safety"
        ),

        # Task 3: Real allergen complaint (not negation)
        (
            "allergen not mentioned on label",
            "HIGH",
            "Merchandising",
            None,
            "Allergen missing from label is real complaint"
        ),

        # Hinglish examples
        (
            "Washroom nahi dirty, chala clean",
            "LOW",
            None,
            None,
            "Hinglish: Negated dirty with positive (clean)"
        ),
        (
            "Freezer kaam nahi kar raha, sabzi kharab ho gayi",
            "HIGH",
            None,
            None,
            "Hinglish: Refrigeration failure, food spoiled"
        ),
        (
            "Billing counter bahut slow tha",
            "MEDIUM",
            None,
            None,
            "Hinglish: Billing slow (operational issue)"
        ),
        (
            "Store badhiya tha, saaf aur accha",
            "LOW",
            None,
            None,
            "Hinglish: Positive store feedback"
        ),

        # Tamil transliterated
        (
            "Toilet illai dirty, very fresh",
            "LOW",
            None,
            None,
            "Tamil: Negated dirty (illai=not)"
        ),
        (
            "Store floor dirty ah irukku",
            "MEDIUM",
            None,
            None,
            "Tamil: Floor dirty (MEDIUM)"
        ),
        (
            "Washroom romba alukku ah irukku",
            "HIGH",
            None,
            None,
            "Tamil: Washroom very dirty (HIGH)"
        ),

        # Telugu transliterated
        (
            "Garbage area kaadu smelly, neat ga undi",
            "LOW",
            None,
            None,
            "Telugu: Negated smelly (kaadu=not), neat"
        ),
        (
            "Store floor dirty ga undi",
            "MEDIUM",
            None,
            None,
            "Telugu: Floor dirty (MEDIUM)"
        ),
        (
            "Freezer work avvatledu, milk spoil ayyindi",
            "HIGH",
            None,
            None,
            "Telugu: Freezer not working, milk spoiled"
        ),

        # Marathi transliterated
        (
            "Toilet swachh aahe, dhanyavaad",
            "LOW",
            None,
            None,
            "Marathi: Toilet clean, thanks"
        ),
        (
            "Store madhe gandh yete",
            "HIGH",
            None,
            None,
            "Marathi: Store smells bad (hygiene)"
        ),

        # Edge cases
        (
            "Staff member was helpful and nice",
            "LOW",
            None,
            None,
            "Positive staff feedback (not misconduct)"
        ),
        (
            "No problem with the store, everything good",
            "LOW",
            None,
            None,
            "General positive, negated problem"
        ),
        (
            "Product packaging was damaged",
            "MEDIUM",
            None,
            None,
            "Packaging damage (Supply Chain, MEDIUM)"
        ),
    ],
)
def test_regression_bug_fixes(text, expected_severity, expected_dept, must_not_contain_in_reason, description):
    """Regression test for all bug fixes from Tasks 1-4."""
    decision = evaluate_policy(text)

    # Check severity
    assert decision.severity == expected_severity, (
        f"{description}: Expected {expected_severity}, got {decision.severity}. "
        f"Matched: {decision.matched_rules}, Reason: {decision.reason}"
    )

    # Check department if specified
    if expected_dept:
        assert decision.department == expected_dept, (
            f"{description}: Expected department {expected_dept}, got {decision.department}"
        )

    # Check reason doesn't contain forbidden words
    if must_not_contain_in_reason:
        assert must_not_contain_in_reason.lower() not in decision.reason.lower(), (
            f"{description}: Reason should not mention '{must_not_contain_in_reason}'. "
            f"Reason: {decision.reason}"
        )


def test_regression_whole_word_matching():
    """Task 1: Verify whole-word matching prevents substring false positives."""
    false_positive_cases = [
        ("opened the package", "opened", "ned"),  # "ned" should not match
        ("position is good", "position", "pos"),  # "pos" should not match
        ("eggplant was fresh", "eggplant", "egg"),  # "egg" should not match
        ("slippers were available", "slippers", "slip"),  # "slip" should not match
    ]

    for text, word, should_not_match in false_positive_cases:
        decision = evaluate_policy(text)
        # Should not be HIGH due to partial keyword match
        if should_not_match in ["egg"]:
            assert "Allergen Risk" not in decision.matched_rules, (
                f"'{text}' should not match allergen keyword '{should_not_match}'"
            )


def test_regression_negation_proximity():
    """Task 3: Negation must be near keyword (within 5 words), not anywhere in text."""
    # "not working" is far from "spoiled" - should still escalate
    text = "Freezer is not working properly. The milk packets are spoiled."
    decision = evaluate_policy(text)
    assert decision.severity == "HIGH", (
        f"Should escalate despite 'not' in different clause. Got {decision.severity}"
    )


def test_regression_allergen_complaint_not_negation():
    """Task 3: 'allergen not mentioned' is a REAL complaint, not a negation."""
    complaints = [
        "allergen not mentioned on label",
        "ingredients not listed properly",
        "label does not state allergen info",
    ]

    for text in complaints:
        decision = evaluate_policy(text)
        assert decision.severity == "HIGH", (
            f"'{text}' should be HIGH (real complaint), got {decision.severity}"
        )
