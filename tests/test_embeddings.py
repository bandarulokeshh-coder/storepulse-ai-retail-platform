import numpy as np

from src import embeddings as embeddings_module
from src.classifier import FeedbackAnalyzer


class StubModel:
    """Deterministic encoder for tests: substring-based vectors, no downloads."""

    _MAP = {
        "freezer": [1.0, 0.0],       # matches the Refrigeration Failure prototype
        "chiller": [1.0, 0.0],       # matches only via embeddings (not a rule keyword)
    }

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        vectors = []
        for text in texts:
            lowered = str(text).lower()
            vector = [0.0, 1.0]
            for key, value in self._MAP.items():
                if key in lowered:
                    vector = value
                    break
            array = np.array(vector, dtype=float)
            norm = float(np.linalg.norm(array))
            if norm > 0:
                array = array / norm
            vectors.append(array)
        return np.array(vectors)


def test_embeddings_disabled_by_default_uses_rules():
    analyzer = FeedbackAnalyzer()
    assert analyzer.use_embeddings is False
    result = analyzer.analyze("Store lo freezer work avvatledu, milk packets spoil avtunnayi!")
    assert result.issue_category == "Refrigeration Failure"
    assert result.method == "rule_based"


def test_embedding_unavailable_falls_back_to_rules(monkeypatch):
    import src.classifier as classifier_module

    monkeypatch.setattr(classifier_module, "ensure_embedding_model", lambda *args, **kwargs: None)
    analyzer = FeedbackAnalyzer(use_embeddings=True)
    # No keyword match and no usable embedding model: must not crash or fabricate an issue.
    result = analyzer.analyze("aj ka din bahut boring tha")
    assert result.issue_category == "General Feedback"
    assert result.method == "rule_based"
    assert result.confidence is None


def test_embedding_assisted_classification_when_rules_are_weak(monkeypatch):
    import src.classifier as classifier_module

    monkeypatch.setattr(classifier_module, "ensure_embedding_model", lambda *args, **kwargs: StubModel())
    analyzer = FeedbackAnalyzer(use_embeddings=True)
    # Controlled keywords so "chiller" has no rule match but does match the
    # "Refrigeration Failure. freezer, ..." prototype embedding.
    analyzer.keywords = {
        "Refrigeration Failure": ["freezer"],
        "Stock Shortage": ["out of stock"],
    }
    result = analyzer.analyze("the chiller box stopped chilling")
    assert result.issue_category == "Refrigeration Failure"
    assert result.method == "embedding_assisted"
    assert result.confidence is None  # no fabricated confidence for semantic matching
    assert "semantic similarity" in result.explanation


def test_rule_matches_take_precedence_over_embeddings(monkeypatch):
    import src.classifier as classifier_module

    monkeypatch.setattr(classifier_module, "ensure_embedding_model", lambda *args, **kwargs: StubModel())
    analyzer = FeedbackAnalyzer(use_embeddings=True)
    result = analyzer.analyze("freezer is not working and milk is spoiling")
    assert result.issue_category == "Refrigeration Failure"
    assert result.method == "rule_based"


def test_embedding_status_reports_without_loading():
    # Never trigger a real model download inside the test suite.
    status = embeddings_module.embedding_status()
    assert status.available is False or status.available is True
    assert isinstance(status.reason, str) and status.reason
    assert status.to_dict()["model_name"] == embeddings_module.DEFAULT_MODEL_NAME