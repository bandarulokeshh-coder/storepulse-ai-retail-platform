import pandas as pd

from src.classifier import FeedbackAnalyzer
from src.synthetic_data import BASE_WEIGHTS, TEMPLATES


def _labeled_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feedback_text": [
                "The freezer is not working and milk packets are spoiling.",
                "Fridge temperature is too high; the frozen section is warm.",
                "Cold storage failed again and products are getting spoiled.",
                "The label says no nuts, but the ingredient list includes nuts.",
                "Incorrect allergen information printed on the food packet.",
                "Allergen warning missing from the packaged item.",
            ],
            "issue_category": [
                "Refrigeration Failure",
                "Refrigeration Failure",
                "Refrigeration Failure",
                "Allergen Label Error",
                "Allergen Label Error",
                "Allergen Label Error",
            ],
            "severity": ["HIGH"] * 6,
            "actionable": ["True"] * 6,
        }
    )


def test_training_succeeds_and_reports_no_errors():
    """Training must not silently fail.

    scikit-learn 1.7 removed `multi_class` from LogisticRegression; the old
    pipeline raised TypeError which was swallowed, leaving the analyzer in
    rule-based fallback mode with no visible cause.
    """
    analyzer = FeedbackAnalyzer()
    analyzer.fit_from_dataframe(_labeled_frame())

    assert analyzer.model_bundle.errors == []
    assert analyzer.model_bundle.issue_model is not None
    assert "issue_type" in analyzer.model_bundle.trained_targets


def test_trained_model_predicts_issue_category():
    analyzer = FeedbackAnalyzer()
    analyzer.fit_from_dataframe(_labeled_frame())
    result = analyzer.analyze("The freezer is not working and milk packets are spoiling.")
    assert result.issue_category == "Refrigeration Failure"


def test_every_weighted_category_has_templates():
    """Every generated category needs its own templates.

    Categories without templates used to fall back to the Customer Praise
    text, which silently mislabelled roughly a third of the dataset.
    """
    missing = [category for category in BASE_WEIGHTS if category not in TEMPLATES]
    assert missing == []


def test_every_template_category_has_all_languages():
    from src.synthetic_data import LANGUAGES

    incomplete = {
        category: [language for language in LANGUAGES if language not in languages]
        for category, languages in TEMPLATES.items()
        if any(language not in languages for language in LANGUAGES)
    }
    assert incomplete == {}
