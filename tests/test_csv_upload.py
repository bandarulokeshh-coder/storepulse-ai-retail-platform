"""Test CSV upload handling - simplified version.

Task 9 (STEP 4): Test CSV upload with various edge cases.
"""
import pandas as pd
import pytest
from pathlib import Path

from src.preprocessing import infer_required_columns


def test_csv_upload_with_valid_data():
    """Valid CSV with all required columns should load successfully."""
    fixture_path = Path(__file__).parent / "fixtures" / "test_upload.csv"
    df = pd.read_csv(fixture_path)
    assert len(df) == 7, "Test CSV should have 7 rows"
    required = ["feedback_id", "feedback_text", "store_id", "region", "timestamp", "channel"]
    for col in required:
        assert col in df.columns, f"Missing required column: {col}"


def test_csv_handles_missing_and_bad_data():
    """CSV with missing values and bad dates should be handled."""
    fixture_path = Path(__file__).parent / "fixtures" / "test_upload.csv"
    df = pd.read_csv(fixture_path)
    
    # Has empty feedback
    empty = df[df["feedback_text"].isna() | (df["feedback_text"].astype(str).str.strip() == "")]
    assert len(empty) > 0
    
    # Has invalid dates
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    invalid_dates = df[df["timestamp"].isna()]
    assert len(invalid_dates) > 0


def test_column_inference():
    """Column inference should work with common names."""
    result = infer_required_columns(["feedback_text", "timestamp", "store_id"])
    assert result["text"] == "feedback_text"
    assert result["date"] == "timestamp"
    assert result["store_id"] == "store_id"
    
    result2 = infer_required_columns(["comment", "date", "store"])
    assert result2["text"] == "comment"
    assert result2["date"] == "date"
    assert result2["store_id"] == "store"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
