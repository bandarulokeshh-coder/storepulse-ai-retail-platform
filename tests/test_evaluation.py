import pandas as pd

from src.classifier import FeedbackAnalyzer
from src.emerging_detector import detect_emerging_incidents
from src.evaluation import (
    MISSING_LABELS_MESSAGE,
    evaluate_emerging_detection,
    evaluate_feedback_classification,
)


def _weekly_rows(weekly_counts: list[int], other_count: int = 15) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-05")
    for idx, count in enumerate(weekly_counts):
        week_start = base + pd.Timedelta(weeks=idx)
        for item in range(other_count):
            rows.append(
                {
                    "store_id": "ST-042",
                    "region": "Visakhapatnam",
                    "issue_category": "Customer Praise",
                    "severity": "LOW",
                    "actionable": False,
                    "timestamp": week_start + pd.Timedelta(days=1, hours=item),
                    "feedback_text": "Great shopping experience.",
                }
            )
        for item in range(count):
            rows.append(
                {
                    "store_id": "ST-042",
                    "region": "Visakhapatnam",
                    "issue_category": "Refrigeration Failure",
                    "severity": "HIGH",
                    "actionable": True,
                    "timestamp": week_start + pd.Timedelta(days=2, hours=item),
                    "feedback_text": f"Freezer complaint {idx}-{item}",
                }
            )
    return pd.DataFrame(rows)


def _alerts(df: pd.DataFrame) -> pd.DataFrame:
    return detect_emerging_incidents(df, minimum_support=2, growth_threshold=2.5, spike_threshold=3.0)


def test_escalation_precision_requires_ground_truth_labels():
    df = _weekly_rows([0, 0, 1, 2, 4, 7])
    alerts = _alerts(df)
    assert not alerts.empty
    result = evaluate_emerging_detection(df, alerts, actionability_labels_available=False)
    assert result["escalation_precision"] is None
    assert result["escalation_precision_note"] is not None
    assert "Ground-truth labels" in result["escalation_precision_note"]


def test_escalation_precision_computed_when_labels_exist():
    df = _weekly_rows([0, 0, 1, 2, 4, 7])
    alerts = _alerts(df)
    assert not alerts.empty
    result = evaluate_emerging_detection(df, alerts, actionability_labels_available=True)
    assert result["escalation_precision"] is not None
    assert 0.0 <= result["escalation_precision"] <= 1.0
    assert result["escalation_precision_note"] is None


def test_classification_metrics_hidden_without_ground_truth_flags():
    df = _weekly_rows([0, 0, 1, 2, 4, 7]).head(30)
    df["issue_category"] = "Refrigeration Failure"
    df["severity"] = "HIGH"
    analyzer = FeedbackAnalyzer()
    result = evaluate_feedback_classification(
        df,
        analyzer,
        ground_truth={"issue_type": False, "severity": False, "actionability": False},
    )
    for key in ["issue_type", "severity", "actionability"]:
        assert result[key]["available"] is False
        assert result[key]["metrics"] is None
        assert result[key]["message"] == MISSING_LABELS_MESSAGE


def test_classification_metrics_computed_when_labels_exist():
    df = _weekly_rows([0, 0, 1, 2, 4, 7]).head(30)
    df["issue_category"] = "Refrigeration Failure"
    df["severity"] = "HIGH"
    analyzer = FeedbackAnalyzer()
    result = evaluate_feedback_classification(
        df,
        analyzer,
        ground_truth={"issue_type": True, "severity": True, "actionability": True},
    )
    assert result["issue_type"]["available"] is True
    assert result["issue_type"]["metrics"].n_samples == 30
    assert result["issue_type"]["demo_only"] is True