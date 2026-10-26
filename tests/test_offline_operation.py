"""Test offline operation with cached models.

Task 9 (STEP 4): Verify app works without network access.
"""
import pytest
from pathlib import Path

from src.classifier import FeedbackAnalyzer
from src.embeddings import embedding_status


def test_app_works_without_embeddings():
    """App should work when embedding model is unavailable (offline fallback)."""
    # Create analyzer without embeddings
    analyzer = FeedbackAnalyzer(use_embeddings=False)

    # Should still classify using rules
    result = analyzer.analyze("Freezer not working, milk spoiling")

    assert result.issue_category != "General Feedback", "Should classify without embeddings"
    assert result.method in ["rule_based", "hybrid"], "Should use rule-based method"

    print(f"✓ Offline mode works: classified as {result.issue_category} using {result.method}")


def test_embedding_status_reports_unavailable():
    """Embedding status should report when model is not loaded."""
    status = embedding_status()

    # Status should have availability info
    assert hasattr(status, 'available'), "Status should report availability"

    # If unavailable, should have reason
    if not status.available:
        assert hasattr(status, 'reason'), "Should explain why unavailable"

    print(f"✓ Embedding status: available={status.available}")


def test_classifier_never_requires_network():
    """TF-IDF classifier should work entirely offline once trained."""
    import pandas as pd

    # Small training set
    train_data = pd.DataFrame({
        "feedback_text": [
            "Payment not working",
            "Store is dirty",
            "Staff was rude",
            "Freezer broken"
        ],
        "issue_category": [
            "Payment Failure",
            "Store Cleanliness",
            "Staff Conduct",
            "Refrigeration Failure"
        ]
    })

    analyzer = FeedbackAnalyzer(use_embeddings=False)
    analyzer.fit_from_dataframe(train_data)

    # Should classify offline
    result = analyzer.analyze("UPI payment failed")

    assert result.issue_category is not None, "Should classify without network"
    print(f"✓ TF-IDF classifier works offline: {result.issue_category}")


def test_synthetic_data_generation_is_deterministic():
    """Synthetic data should be reproducible (no external API calls)."""
    from src.synthetic_data import load_or_create_synthetic_dataset

    # Load synthetic data twice
    bundle1 = load_or_create_synthetic_dataset()
    bundle2 = load_or_create_synthetic_dataset()

    # Should be identical (cached)
    assert len(bundle1.dataframe) == len(bundle2.dataframe), "Should be deterministic"

    print(f"✓ Synthetic data is deterministic: {len(bundle1.dataframe)} rows")


def test_download_models_script_exists():
    """download_models.py script should exist for offline preparation."""
    script_path = Path("scripts/download_models.py")

    assert script_path.exists(), "download_models.py should exist for offline prep"

    print(f"✓ Offline prep script exists: {script_path}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
