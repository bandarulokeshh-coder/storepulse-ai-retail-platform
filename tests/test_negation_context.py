"""Test negation and positive context handling in policy engine.

Task 3: Prevent false positives from positive statements and handle negation.
Focus on the most common false positive cases.
"""
import pytest
from src.policy_engine import evaluate_policy, determine_severity


def test_positive_washroom_feedback_not_escalated():
    """Positive washroom/hygiene feedback should not be HIGH."""
    positive_cases = [
        "The washroom was very clean and hygienic, thanks",
        "Toilet is clean and well maintained",
        "Store floor is neat and garbage bins are saaf",
    ]

    for text in positive_cases:
        decision = evaluate_policy(text)
        assert decision.severity == "LOW", f"'{text}' should be LOW, got {decision.severity}"
        assert "Hygiene" not in decision.matched_rules, f"Should not match Hygiene: {decision.matched_rules}"


def test_negated_hygiene_with_positive_context():
    """Negated safety keywords with positive context should not escalate."""
    cases = [
        "Store was not dirty, very good",
        "Washroom nahi dirty, chala clean",
        "Toilet illai dirty, very fresh",
        "Garbage area kaadu smelly, neat ga undi",
    ]

    for text in cases:
        decision = evaluate_policy(text)
        assert decision.severity == "LOW", f"'{text}' should be LOW, got {decision.severity}"


def test_staff_misconduct_always_escalates():
    """Staff misconduct always escalates regardless of context."""
    cases = [
        "Staff was inappropriate with a customer",
        "Found cockroach in the food aisle",
        "Staff threatened me",
    ]

    for text in cases:
        decision = evaluate_policy(text)
        assert decision.severity == "HIGH", f"'{text}' should always be HIGH, got {decision.severity}"


def test_hygiene_with_negative_qualifiers_escalates():
    """Washroom/toilet hygiene issues WITH negative qualifiers should escalate to HIGH.

    Task 4 policy: General floor cleanliness is MEDIUM (Store Cleanliness).
    Only washroom/toilet hygiene escalates to HIGH.
    """
    high_cases = [
        "Washroom is dirty and smells bad",
        "Toilet is blocked and overflowing",
        "Garbage is overflowing and smelly",
    ]

    for text in high_cases:
        decision = evaluate_policy(text)
        assert decision.severity == "HIGH", f"'{text}' should be HIGH, got {decision.severity}"
        assert "Washroom Hygiene" in decision.matched_rules or "Food Contamination" in decision.matched_rules

    # Task 4: General floor cleanliness should be MEDIUM, not HIGH
    floor_cases = [
        "Store floor is dirty and trash is visible",
        "Floor is messy and needs cleaning",
    ]

    for text in floor_cases:
        decision = evaluate_policy(text, issue_category="Store Cleanliness")
        assert decision.severity == "MEDIUM", f"'{text}' should be MEDIUM (Store Cleanliness), got {decision.severity}"


def test_refrigeration_failure_routes_to_store_operations_not_merchandising():
    """Task 2 requirement: spoiled milk is refrigeration failure, not allergen risk."""
    text = "Store freezer work avvatledu, milk packets spoil autu may"
    decision = evaluate_policy(text, issue_category="Refrigeration Failure")

    assert decision.severity == "HIGH", f"Expected HIGH, got {decision.severity}"
    assert decision.department == "Store Operations", f"Expected Store Operations, got {decision.department}"
    assert "Allergen" not in decision.reason, f"Reason should not mention allergen: {decision.reason}"


def test_milk_without_spoilage_not_high():
    """Task 2 requirement: bare milk mention should not escalate."""
    text = "Bought milk, billing counter was slow"
    decision = evaluate_policy(text)

    assert decision.severity != "HIGH", f"Should not be HIGH for bare milk mention, got {decision.severity}"
    assert "Allergen Risk" not in decision.matched_rules


def test_eggplant_not_egg_allergen():
    """Task 2 requirement: 'eggplant' should not match 'egg' allergen (whole-word matching)."""
    text = "I got the eggplant and it was fresh"
    decision = evaluate_policy(text)

    assert decision.severity != "HIGH", f"Should not be HIGH for eggplant, got {decision.severity}"
    assert "Allergen Risk" not in decision.matched_rules


def test_food_contamination_patterns():
    """Food contamination with pests/mold should always escalate."""
    cases = [
        ("Food contamination, found mold on bread", "HIGH"),
        ("Milk packets are spoiled, freezer work avvatledu", "HIGH"),
    ]

    for text, expected in cases:
        decision = evaluate_policy(text)
        assert decision.severity == expected, f"'{text}' should be {expected}, got {decision.severity}"


def test_allergen_label_error():
    """Allergen not mentioned on label is a REAL complaint, should escalate."""
    text = "Allergen not mentioned on label"
    decision = evaluate_policy(text)

    assert decision.severity == "HIGH", f"Should be HIGH for allergen label error, got {decision.severity}"


def test_unsafe_product():
    """Unsafe product should escalate."""
    text = "Product is unsafe and dangerous"
    decision = evaluate_policy(text)

    assert decision.severity == "HIGH", f"Should be HIGH for unsafe product, got {decision.severity}"


def test_no_stock_not_safety_issue():
    """No stock should not be treated as allergen/contamination."""
    text = "No stock of milk"
    decision = evaluate_policy(text)

    assert "Allergen Risk" not in decision.matched_rules
    assert "Food Contamination" not in decision.matched_rules
