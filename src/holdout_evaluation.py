"""Held-out evaluation with rule vs ML vs hybrid comparison.

Task 6: Complete ML integration with train/test split and mode comparison.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.classifier import FeedbackAnalyzer


def train_with_holdout(
    df: pd.DataFrame,
    analyzer: FeedbackAnalyzer,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train analyzer on 80% split, return train and test sets.

    Returns:
        (train_df, test_df): Stratified splits for held-out evaluation
    """
    if df.empty or "feedback_text" not in df.columns:
        return df, pd.DataFrame()

    valid_df = df.loc[df["feedback_text"].fillna("").astype(str).str.strip().ne("")].copy()
    if len(valid_df) < 10:
        analyzer.fit_from_dataframe(valid_df)
        return valid_df, pd.DataFrame()

    # Stratify on issue_category if available
    stratify_col = None
    if "issue_category" in valid_df.columns and valid_df["issue_category"].nunique() >= 2:
        stratify_col = valid_df["issue_category"]

    try:
        train_df, test_df = train_test_split(
            valid_df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )
    except ValueError:
        # Stratification failed
        train_df, test_df = train_test_split(
            valid_df,
            test_size=test_size,
            random_state=random_state
        )

    # Train only on training set
    analyzer.fit_from_dataframe(train_df)

    # Store split info in model bundle
    if hasattr(analyzer, 'model_bundle') and analyzer.model_bundle:
        analyzer.model_bundle.train_size = len(train_df)
        analyzer.model_bundle.test_size = len(test_df)
        analyzer.model_bundle.random_state = random_state

    return train_df, test_df


def evaluate_classifier_modes(
    test_df: pd.DataFrame,
    analyzer: FeedbackAnalyzer,
    target: str = "issue_category"
) -> dict:
    """Compare rule-only vs ML-only vs hybrid classifier performance.

    Args:
        test_df: Held-out test set with ground truth labels
        analyzer: Trained analyzer
        target: Target to evaluate (issue_category, severity, actionability)

    Returns:
        Dictionary with metrics for each mode
    """
    if test_df.empty or target not in test_df.columns:
        return {
            "available": False,
            "message": f"Ground truth '{target}' not available in test set"
        }

    # Get ground truth
    y_true = test_df[target].fillna("").astype(str)
    valid_mask = y_true.ne("")
    y_true = y_true[valid_mask]
    test_texts = test_df.loc[valid_mask, "feedback_text"].tolist()

    if len(y_true) < 5:
        return {
            "available": False,
            "message": f"Insufficient test samples for '{target}' ({len(y_true)} < 5)"
        }

    results = {"available": True, "n_samples": len(y_true), "modes": {}}

    # Map target to AnalysisResult attribute
    attr_map = {
        "issue_category": "issue_category",
        "severity": "severity",
        "actionable": "actionable"
    }
    attr_name = attr_map.get(target, "issue_category")

    # RULE-ONLY mode
    analyzer.set_classifier_mode("rule")
    y_pred_rule = []
    for text in test_texts:
        result = analyzer.analyze(text)
        value = getattr(result, attr_name, "")
        # Convert boolean to string for actionable
        y_pred_rule.append(str(value) if not isinstance(value, str) else value)

    results["modes"]["rule"] = {
        "accuracy": accuracy_score(y_true, y_pred_rule),
        "macro_f1": f1_score(y_true, y_pred_rule, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred_rule, average="weighted", zero_division=0),
    }

    # ML-ONLY mode
    analyzer.set_classifier_mode("ml")
    y_pred_ml = []
    for text in test_texts:
        result = analyzer.analyze(text)
        value = getattr(result, attr_name, "")
        y_pred_ml.append(str(value) if not isinstance(value, str) else value)

    results["modes"]["ml"] = {
        "accuracy": accuracy_score(y_true, y_pred_ml),
        "macro_f1": f1_score(y_true, y_pred_ml, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred_ml, average="weighted", zero_division=0),
    }

    # HYBRID mode (default)
    analyzer.set_classifier_mode(None)  # Reset to auto/hybrid
    y_pred_hybrid = []
    for text in test_texts:
        result = analyzer.analyze(text)
        value = getattr(result, attr_name, "")
        y_pred_hybrid.append(str(value) if not isinstance(value, str) else value)

    results["modes"]["hybrid"] = {
        "accuracy": accuracy_score(y_true, y_pred_hybrid),
        "macro_f1": f1_score(y_true, y_pred_hybrid, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred_hybrid, average="weighted", zero_division=0),
    }

    return results


def compare_all_targets(
    test_df: pd.DataFrame,
    analyzer: FeedbackAnalyzer
) -> dict:
    """Compare rule/ML/hybrid across all available targets.

    Returns:
        Dictionary with comparison results for each target
    """
    targets = ["issue_category", "severity", "actionable"]
    comparison = {}

    for target in targets:
        if target in test_df.columns:
            comparison[target] = evaluate_classifier_modes(test_df, analyzer, target)
        else:
            comparison[target] = {
                "available": False,
                "message": f"'{target}' not in test data"
            }

    return comparison
