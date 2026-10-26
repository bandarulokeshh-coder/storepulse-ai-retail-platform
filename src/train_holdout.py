"""Train ML classifiers with proper train/test split and evaluation.

Task 6: Implement held-out evaluation to avoid training on test data.
"""
from sklearn.model_selection import train_test_split
import pandas as pd
from classifier import FeedbackAnalyzer


def train_with_held_out_split(
    df: pd.DataFrame,
    analyzer: FeedbackAnalyzer,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train models on 80% of data, return train and test splits for evaluation.

    Returns:
        (train_df, test_df): Stratified splits for held-out evaluation
    """
    if df.empty or "feedback_text" not in df.columns:
        return df, pd.DataFrame()

    # Filter valid rows
    valid_df = df.loc[df["feedback_text"].fillna("").astype(str).str.strip().ne("")].copy()
    if len(valid_df) < 10:  # Too small to split
        analyzer.fit_from_dataframe(valid_df)
        return valid_df, pd.DataFrame()

    # Stratified split on issue_category if available
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
        # Stratification failed (e.g., some classes have only 1 sample)
        train_df, test_df = train_test_split(
            valid_df,
            test_size=test_size,
            random_state=random_state
        )

    # Train only on training set
    analyzer.fit_from_dataframe(train_df)

    return train_df, test_df


if __name__ == "__main__":
    # Demo: train and evaluate with held-out split
    import sys
    sys.path.insert(0, ".")

    analyzer = FeedbackAnalyzer()

    # Load synthetic data
    df = pd.read_csv("data/synthetic_feedback.csv")

    print(f"Total labeled data: {len(df)} rows")

    # Train with 80/20 split
    train_df, test_df = train_with_held_out_split(df, analyzer, test_size=0.2, random_state=42)

    print(f"Train set: {len(train_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    print(f"Trained targets: {analyzer.model_bundle.trained_targets}")

    if not test_df.empty:
        # Evaluate on held-out test set
        from evaluation import evaluate_feedback_classification

        metrics = evaluate_feedback_classification(test_df, analyzer, max_rows=None)

        print("\n=== Held-out Test Set Evaluation ===")
        for target, result in metrics.items():
            if result["available"]:
                m = result["metrics"]
                print(f"\n{target}:")
                print(f"  Accuracy: {m['accuracy']:.3f}")
                print(f"  Macro F1: {m['macro_f1']:.3f}")
                print(f"  Weighted F1: {m['weighted_f1']:.3f}")
                print(f"  Samples: {m['n_samples']}")
