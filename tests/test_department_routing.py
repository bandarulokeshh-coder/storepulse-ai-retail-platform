"""Test consistent department routing across analyzer and alerts.

Task 5: Ensure department assignment is consistent regardless of where
the issue appears (AI Feedback Analyzer vs Emerging Alerts table).
"""
import pytest
from src.policy_engine import evaluate_policy
from src.routing import get_department_for_issue


def test_department_consistency_for_issue_categories():
    """Department should come from taxonomy for non-safety issues."""
    test_cases = [
        ("Refrigeration Failure", "Store Operations"),
        ("Equipment Failure", "Store Operations"),
        ("Stock Shortage", "Supply Chain"),
        ("Delivery Delay", "Supply Chain"),
        ("Incorrect Price Label", "Merchandising"),
        ("Promotion Mismatch", "Merchandising"),
        ("Billing Issue", "Digital"),
        ("App Payment Failure", "Digital"),
        ("Checkout Delay", "Store Operations"),
        ("Staff Courtesy", "HR"),
        ("Security Concern", "Store Operations"),
    ]

    for issue_category, expected_dept in test_cases:
        dept_from_routing = get_department_for_issue(issue_category)
        assert dept_from_routing == expected_dept, (
            f"{issue_category}: routing says {dept_from_routing}, expected {expected_dept}"
        )


def test_safety_pattern_department_overrides():
    """Safety patterns override department: Staff Misconduct→HR, Allergen→Merchandising, others→Store Ops."""
    test_cases = [
        ("Staff was inappropriate with customer", "HR", "Staff Misconduct"),
        ("Staff threatened me", "HR", "Staff Misconduct"),
        ("Allergen not mentioned on label", "Merchandising", "Allergen Risk"),
        ("Contains nuts not labeled", "Merchandising", "Allergen Risk"),
        ("Found cockroach in food", "Store Operations", "Food Contamination"),
        ("Washroom is dirty and blocked", "Store Operations", "Washroom Hygiene"),
        ("Product is unsafe and dangerous", "Store Operations", "Unsafe Product"),
    ]

    for text, expected_dept, expected_pattern in test_cases:
        decision = evaluate_policy(text)
        assert decision.department == expected_dept, (
            f"'{text}': got {decision.department}, expected {expected_dept}. "
            f"Matched: {decision.matched_rules}"
        )
        assert expected_pattern in decision.matched_rules, (
            f"'{text}': expected to match {expected_pattern}, got {decision.matched_rules}"
        )


def test_refrigeration_failure_consistent_department():
    """Refrigeration Failure should always route to Store Operations, never Merchandising.

    This was the specific issue mentioned in Task 5.
    """
    text = "Freezer not working, milk packets spoiling"

    # Via policy engine
    decision = evaluate_policy(text, issue_category="Refrigeration Failure")
    assert decision.department == "Store Operations", (
        f"Policy engine routed to {decision.department}, expected Store Operations"
    )

    # Via routing module
    dept = get_department_for_issue("Refrigeration Failure")
    assert dept == "Store Operations", (
        f"Routing module returned {dept}, expected Store Operations"
    )


def test_department_from_taxonomy_not_hardcoded():
    """Non-safety issues should get department from taxonomy, not hardcoded in policy engine."""
    # These should use taxonomy, not policy overrides
    text_category_pairs = [
        ("Payment keeps failing", "App Payment Failure", "Digital"),
        ("Out of stock", "Stock Shortage", "Supply Chain"),
        ("Wrong price on shelf", "Incorrect Price Label", "Merchandising"),
    ]

    for text, category, expected_dept in text_category_pairs:
        decision = evaluate_policy(text, issue_category=category)
        assert decision.department == expected_dept, (
            f"'{text}' ({category}): got {decision.department}, expected {expected_dept}"
        )
