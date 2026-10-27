from src.classifier import FeedbackAnalyzer
from src.policy_engine import evaluate_policy


def test_safety_escalation_allergen_label_is_high():
    decision = evaluate_policy("Incorrect allergen label on packaged food.", "Allergen Label Error")
    assert decision.severity == "HIGH"
    assert decision.department == "Merchandising"
    assert decision.actionable is True


def test_unknown_feedback_does_not_fabricate_critical_issue():
    analyzer = FeedbackAnalyzer()
    result = analyzer.analyze("Good shopping experience.")
    assert result.severity == "LOW"
    assert result.actionable is False
    assert result.issue_category in {"Customer Praise", "General Feedback"}


def test_missing_or_empty_text_is_handled_safely():
    analyzer = FeedbackAnalyzer()
    result = analyzer.analyze("")
    assert result.issue_category == "General Feedback"
    assert result.human_review_required is True


def test_bare_ingredient_mention_is_not_an_allergen_escalation():
    """A refrigeration complaint that mentions milk is not an allergen risk.

    The bare word "milk" must not trigger Allergen Risk, otherwise every
    spoiling-milk complaint escalates to Merchandising as a safety incident.
    Note: Spoilage escalates to HIGH severity for food safety, but this is
    for spoilage (Food Contamination), not allergens.
    """
    decision = evaluate_policy("The freezer is not working and milk packets are spoiling.")
    assert "Allergen Risk" not in decision.matched_rules
    assert "Food Contamination" in decision.matched_rules
    assert decision.severity == "HIGH"


def test_ingredient_with_declaration_context_still_escalates():
    """An ingredient mention tied to labelling language is a real allergen risk."""
    decision = evaluate_policy("The label says no nuts, but the ingredient list includes nuts.")
    assert "Allergen Risk" in decision.matched_rules
    assert decision.severity == "HIGH"


def test_allergen_keyword_alone_still_escalates():
    decision = evaluate_policy("Wrong allergen label on the packet.")
    assert "Allergen Risk" in decision.matched_rules
    assert decision.severity == "HIGH"


def test_hinglish_spoiling_milk_is_refrigeration_high_not_allergen():
    """Spoiling milk is a refrigeration failure, not an allergen incident.

    Must escalate (food contamination/spoilage policy) to HIGH with Store
    Operations owning it, and the reason must never mention allergens.
    """
    from src.classifier import FeedbackAnalyzer

    text = "Store freezer work avvatledu, milk packets spoil autu may"
    decision = evaluate_policy(text, "Refrigeration Failure")
    assert decision.severity == "HIGH"
    assert decision.department == "Store Operations"
    assert "allergen" not in decision.reason.lower()
    assert "Allergen Risk" not in decision.matched_rules

    result = FeedbackAnalyzer().analyze(text)
    assert result.issue_category == "Refrigeration Failure"
    assert result.severity == "HIGH"
    assert result.department == "Store Operations"


def test_bare_milk_mention_in_unrelated_text_is_not_high():
    decision = evaluate_policy("Bought milk, billing counter was slow")
    assert decision.severity != "HIGH"
    assert "Allergen Risk" not in decision.matched_rules


def test_eggplant_is_not_high():
    decision = evaluate_policy("I got the eggplant and it was fresh")
    assert decision.severity != "HIGH"
    assert "Allergen Risk" not in decision.matched_rules
