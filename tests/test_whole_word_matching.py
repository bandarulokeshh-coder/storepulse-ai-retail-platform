"""Whole-word keyword matching (Task 1).

Short keywords such as "ned", "pos", "pani", "slip" and "egg" used to match
inside unrelated words ("opened", "position", "panipuri", "slipper",
"eggplant") because rules used plain substring containment.
"""
from __future__ import annotations

import pytest

from src.classifier import FeedbackAnalyzer
from src.policy_engine import evaluate_policy
from src.preprocessing import contains_keyword
from src.routing import get_department_for_issue


@pytest.mark.parametrize(
    "text,keyword",
    [
        ("opened", "ned"),
        ("position", "pos"),
        ("eggplant", "egg"),
        ("panipuri", "pani"),
        ("slipper", "slip"),
        ("smokey room", "smoke"),  # "smoke" is not a keyword here; boundary sanity
    ],
)
def test_contains_keyword_rejects_substrings(text: str, keyword: str) -> None:
    assert contains_keyword(text, keyword) is False


@pytest.mark.parametrize(
    "text,keyword",
    [
        ("the pos machine is broken", "pos"),
        ("stock ned led", "ned"),
        ("washroom pani everywhere", "pani"),
        ("found cockroach", "cockroach"),
    ],
)
def test_contains_keyword_accepts_whole_words(text: str, keyword: str) -> None:
    assert contains_keyword(text, keyword) is True


def test_classifier_does_not_match_short_keywords_inside_words() -> None:
    analyzer = FeedbackAnalyzer()
    _, _, equipment_matches = analyzer._score_rules("The manager was positioned near the entrance.")
    assert "pos" not in equipment_matches

    category, _, stock_matches = analyzer._score_rules("We opened the cartons this morning.")
    assert "ned" not in stock_matches
    assert category != "Stock Shortage"


def test_classifier_still_matches_real_short_keyword_words() -> None:
    analyzer = FeedbackAnalyzer()
    category, _, _ = analyzer._score_rules("The pos machine is not working.")
    assert category == "Equipment Failure"


def test_policy_does_not_treat_eggplant_as_egg_declaration_risk() -> None:
    # With substring matching, "eggplant" supplied the contextual "egg" term
    # and "label" supplied the declaration cue -> false Allergen Risk.
    decision = evaluate_policy("The eggplant label was wrong on the pack.")
    assert "Allergen Risk" not in decision.matched_rules
    assert decision.severity != "HIGH"


def test_routing_fallback_matches_whole_words_only() -> None:
    # "opened" contains the Stock Shortage keyword "ned"; the fallback must
    # not route on it.
    department = get_department_for_issue("Some Unknown Label", "opened the boxes for restock")
    assert department == "Store Operations"

    department = get_department_for_issue("Some Unknown Label", "shelf is empty and out of stock")
    assert department == "Supply Chain"
