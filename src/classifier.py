from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .embeddings import SEMANTIC_MATCH_THRESHOLD, best_semantic_match, build_category_prototypes, ensure_embedding_model
from .policy_engine import evaluate_policy
from .preprocessing import contains_keyword, normalize_text
from .routing import category_keywords, get_department_for_issue, get_suggested_action

GENERAL_FEEDBACK_CATEGORY = "General Feedback"
PRAISE_CATEGORY = "Customer Praise"

POSITIVE_HINTS = ["good", "great", "excellent", "nice", "happy", "helpful", "love", "thanks", "thank you"]
NEGATIVE_HINTS = ["bad", "issue", "problem", "broken", "failed", "delay", "dirty", "wrong", "not working"]


@dataclass(slots=True)
class AnalysisResult:
    issue_category: str
    severity: str
    actionable: bool
    department: str
    suggested_action: str
    explanation: str
    method: str
    confidence: float | None = None
    rule_matches: list[str] | None = None
    human_review_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_category": self.issue_category,
            "severity": self.severity,
            "actionable": self.actionable,
            "department": self.department,
            "suggested_action": self.suggested_action,
            "explanation": self.explanation,
            "method": self.method,
            "confidence": self.confidence,
            "rule_matches": self.rule_matches or [],
            "human_review_required": self.human_review_required,
        }


@dataclass(slots=True)
class ModelBundle:
    issue_model: Pipeline | None = None
    issue_labels: list[str] | None = None
    severity_model: Pipeline | None = None
    severity_labels: list[str] | None = None
    actionable_model: Pipeline | None = None
    actionable_labels: list[str] | None = None
    trained_on_demo_data: bool = False
    notes: str = "rule_based"
    errors: list[str] | None = None

    @property
    def trained_targets(self) -> list[str]:
        targets = []
        if self.issue_model is not None:
            targets.append("issue_type")
        if self.severity_model is not None:
            targets.append("severity")
        if self.actionable_model is not None:
            targets.append("actionability")
        return targets


class FeedbackAnalyzer:
    def __init__(self, taxonomy_path: str | None = None, use_embeddings: bool = False):
        self.taxonomy_path = taxonomy_path
        self.keywords = category_keywords(taxonomy_path) if taxonomy_path else category_keywords()
        self.model_bundle = ModelBundle()
        self.use_embeddings = use_embeddings
        self._embed_prototypes: dict[str, np.ndarray] | None = None

    def set_use_embeddings(self, enabled: bool) -> None:
        """Enable/disable multilingual embedding assistance (falls back to rules if unavailable)."""
        self.use_embeddings = bool(enabled)

    def reload_taxonomy(self) -> None:
        """Re-read the issue taxonomy after a configuration update."""
        self.keywords = category_keywords(self.taxonomy_path) if self.taxonomy_path else category_keywords()
        self._embed_prototypes = None

    @staticmethod
    def _build_pipeline() -> Pipeline:
        # `multi_class` was removed in scikit-learn 1.7; the solver now handles
        # multiclass automatically, so passing it raises TypeError.
        return Pipeline(
            steps=[
                ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=8000)),
                ("classifier", LogisticRegression(max_iter=500, class_weight="balanced")),
            ]
        )

    def _fit_target(self, training_df: pd.DataFrame, column: str) -> Pipeline | None:
        """Train a TF-IDF + Logistic Regression model for one labeled target column."""
        labels = training_df[column].astype(str)
        if labels.nunique() < 2:
            return None
        model = self._build_pipeline()
        model.fit(training_df["feedback_text"].astype(str), labels)
        return model

    def fit_from_dataframe(self, df: pd.DataFrame) -> None:
        """Train supervised models for every target that has usable labels.

        Spec: when labeled data is available, support training for issue type,
        severity, and actionability. Missing or unusable labels are skipped
        silently for that target (the rule engine remains the fallback).

        Training errors are recorded in `self.model_bundle.errors` instead of
        being swallowed, so a failed training run is visible in the UI rather
        than silently degrading to the rule engine.
        """
        if "feedback_text" not in df.columns:
            return
        training_df = df.loc[df["feedback_text"].fillna("").astype(str).str.strip().ne("")].copy()
        if training_df.empty:
            return

        trained_targets: list[str] = []
        failed = False
        errors: list[str] = []

        if "issue_category" in training_df.columns and training_df["issue_category"].nunique() >= 2:
            try:
                model = self._fit_target(training_df, "issue_category")
                if model is not None:
                    self.model_bundle.issue_model = model
                    self.model_bundle.issue_labels = sorted(training_df["issue_category"].astype(str).unique().tolist())
                    trained_targets.append("issue_type")
            except Exception as exc:
                failed = True
                errors.append(f"issue_type: {exc.__class__.__name__}: {exc}")

        if "severity" in training_df.columns and training_df["severity"].nunique() >= 2:
            try:
                model = self._fit_target(training_df, "severity")
                if model is not None:
                    self.model_bundle.severity_model = model
                    self.model_bundle.severity_labels = sorted(training_df["severity"].astype(str).unique().tolist())
                    trained_targets.append("severity")
            except Exception as exc:
                failed = True
                errors.append(f"severity: {exc.__class__.__name__}: {exc}")

        if "actionable" in training_df.columns and training_df["actionable"].astype(str).nunique() >= 2:
            try:
                model = self._fit_target(training_df, "actionable")
                if model is not None:
                    self.model_bundle.actionable_model = model
                    self.model_bundle.actionable_labels = sorted(training_df["actionable"].astype(str).unique().tolist())
                    trained_targets.append("actionability")
            except Exception as exc:
                failed = True
                errors.append(f"actionability: {exc.__class__.__name__}: {exc}")

        self.model_bundle.errors = errors

        if trained_targets:
            self.model_bundle.trained_on_demo_data = True
            self.model_bundle.notes = "trained_tfidf_logreg(" + ", ".join(trained_targets) + ")"
        elif failed:
            self.model_bundle.notes = "rule_based_fallback"

    def _score_rules(self, text: str) -> tuple[str, float, list[str]]:
        normalized = normalize_text(text)
        best_category = GENERAL_FEEDBACK_CATEGORY
        best_score = 0.0
        matches: list[str] = []
        for category, keywords in self.keywords.items():
            hits = [keyword for keyword in keywords if keyword and contains_keyword(normalized, keyword)]
            score = float(len(hits))
            if score > best_score:
                best_score = score
                best_category = category
                matches = hits
            elif score == best_score and score > 0 and category != best_category:
                # Prefer more specific categories with safety relevance.
                if any(token in category.lower() for token in ["safety", "contamination", "allergen", "hygiene", "misconduct"]):
                    best_category = category
                    matches = hits
        if best_score == 0:
            if any(contains_keyword(normalized, token) for token in POSITIVE_HINTS) and not any(
                contains_keyword(normalized, token) for token in NEGATIVE_HINTS
            ):
                return PRAISE_CATEGORY, 0.15, []
            return GENERAL_FEEDBACK_CATEGORY, 0.0, []
        confidence = min(0.95, 0.35 + (best_score * 0.2))
        return best_category, confidence, matches

    def _score_with_embeddings(self, text: str) -> tuple[str | None, float | None]:
        """Semantic category match via multilingual sentence embeddings.

        Returns (None, None) when embeddings are disabled or the model is
        unavailable, so callers transparently fall back to the rule engine.
        """
        if not self.use_embeddings:
            return None, None
        model = ensure_embedding_model()
        if model is None:
            return None, None
        if self._embed_prototypes is None:
            prototype_texts = {
                category: f"{category}. {', '.join(keywords)}"
                for category, keywords in self.keywords.items()
                if keywords
            }
            self._embed_prototypes = build_category_prototypes(model, prototype_texts)
        return best_semantic_match(model, self._embed_prototypes, text)

    def _predict_with_model(self, text: str) -> tuple[str, float]:
        if self.model_bundle.issue_model is None:
            raise RuntimeError("model_not_available")
        proba = self.model_bundle.issue_model.predict_proba([text])[0]
        classes = list(self.model_bundle.issue_model.classes_)
        idx = int(np.argmax(proba))
        return str(classes[idx]), float(proba[idx])

    def analyze(self, text: str, issue_category_hint: str | None = None) -> AnalysisResult:
        original_text = text or ""
        normalized = normalize_text(original_text)
        if not normalized:
            decision = evaluate_policy(original_text, issue_category_hint)
            return AnalysisResult(
                issue_category=GENERAL_FEEDBACK_CATEGORY,
                severity="LOW",
                actionable=False,
                department=decision.department,
                suggested_action="Request more detail from the reporter.",
                explanation="Missing or empty feedback cannot be classified reliably.",
                method="rule_based",
                confidence=None,
                rule_matches=[],
                human_review_required=True,
            )

        rule_category, rule_confidence, rule_matches = self._score_rules(original_text)
        model_category = None
        model_confidence = None
        used_model = False

        # Multilingual embeddings only fill gaps when keyword rules are weak,
        # so transparent rule matches always take precedence.
        embed_category = None
        embed_similarity = None
        if self.use_embeddings and rule_category == GENERAL_FEEDBACK_CATEGORY:
            embed_category, embed_similarity = self._score_with_embeddings(original_text)

        if self.model_bundle.issue_model is not None:
            try:
                model_category, model_confidence = self._predict_with_model(original_text)
                used_model = True
            except Exception:
                used_model = False

        if (
            rule_category == GENERAL_FEEDBACK_CATEGORY
            and embed_category
            and embed_similarity is not None
            and embed_similarity >= SEMANTIC_MATCH_THRESHOLD
        ):
            issue_category = embed_category
            confidence = None
            method = "embedding_assisted"
        elif rule_category == GENERAL_FEEDBACK_CATEGORY and used_model and model_category:
            issue_category = model_category
            confidence = model_confidence
            method = "ml_fallback"
        elif rule_category == GENERAL_FEEDBACK_CATEGORY:
            issue_category = rule_category
            confidence = None
            method = "rule_based"
        else:
            issue_category = rule_category
            confidence = rule_confidence
            method = "hybrid" if used_model else "rule_based"

        decision = evaluate_policy(original_text, issue_category)
        actionable = decision.actionable
        department = decision.department or get_department_for_issue(issue_category, original_text)
        suggested_action = get_suggested_action(issue_category)

        if issue_category == PRAISE_CATEGORY:
            severity = "LOW"
            actionable = False
            department = "Customer Experience"
            suggested_action = "Share the positive feedback with the store team."
            explanation = "Positive language detected; no operational issue was identified."
            return AnalysisResult(
                issue_category=issue_category,
                severity=severity,
                actionable=actionable,
                department=department,
                suggested_action=suggested_action,
                explanation=explanation,
                method=method,
                confidence=confidence,
                rule_matches=rule_matches,
                human_review_required=False,
            )

        # Respect policy-first escalations even if the classifier predicts something else.
        severity = decision.severity
        explanation_parts = []
        if rule_matches:
            explanation_parts.append(f"Keyword matches: {', '.join(rule_matches[:4])}.")
        if decision.matched_rules:
            explanation_parts.append(f"Policy trigger(s): {', '.join(decision.matched_rules)}.")
        if decision.human_review_required:
            explanation_parts.append("Ambiguous safety language flagged for human review.")
        if method == "embedding_assisted" and embed_category and embed_similarity is not None:
            explanation_parts.append(
                f"Multilingual embedding matched '{embed_category}' with semantic similarity {embed_similarity:.2f} "
                "(similarity score, not a calibrated confidence)."
            )
        if confidence is not None and used_model and method not in {"rule_based", "embedding_assisted"}:
            explanation_parts.append(f"Model confidence: {confidence:.2f}.")
        if not explanation_parts:
            explanation_parts.append("No strong rule match; operating with fallback classification.")

        return AnalysisResult(
            issue_category=issue_category,
            severity=severity,
            actionable=actionable,
            department=department,
            suggested_action=suggested_action,
            explanation=" ".join(explanation_parts),
            method=method,
            confidence=confidence if issue_category != GENERAL_FEEDBACK_CATEGORY else None,
            rule_matches=rule_matches,
            human_review_required=decision.human_review_required,
        )

    def batch_analyze(self, texts: list[str]) -> pd.DataFrame:
        rows = [self.analyze(text).to_dict() for text in texts]
        return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def default_analyzer() -> FeedbackAnalyzer:
    return FeedbackAnalyzer()
