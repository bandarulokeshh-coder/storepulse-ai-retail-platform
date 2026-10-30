"""CLI detector for causal re-runs at specific dates."""
import pandas as pd
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add r5 package to path
sys.path.insert(0, str(Path(__file__).parent))

import dataio
from classifier import VoiceOfStoreClassifier


def detect(feedback_path: str, as_of_date: str, output_path: str):
    """
    Generate predictions for feedback up to as_of_date.

    Args:
        feedback_path: Path to feedback.csv file
        as_of_date: Date in YYYY-MM-DD format (causal cutoff)
        output_path: Path to output predictions CSV
    """
    # Parse as_of date
    cutoff = pd.to_datetime(as_of_date)

    # Load feedback
    feedback = pd.read_csv(feedback_path)
    feedback['created_at'] = pd.to_datetime(feedback['created_at'])

    # Filter to as_of date (causal constraint)
    feedback_filtered = feedback[feedback['created_at'] <= cutoff].copy()

    if len(feedback_filtered) == 0:
        # No data up to this date, return empty predictions
        predictions = pd.DataFrame(columns=['feedback_id', 'issue_type_id', 'severity', 'actionable'])
        predictions.to_csv(output_path, index=False)
        return

    # Load data from same folder as feedback
    data_folder = Path(feedback_path).parent
    feedback_all, labels_train, taxonomy, stores, weeks = dataio.load_pack(data_folder)

    # Filter training labels to only use data up to as_of (causal constraint)
    feedback_all['created_at'] = pd.to_datetime(feedback_all['created_at'])
    train_feedback_ids = feedback_all[feedback_all['created_at'] <= cutoff]['feedback_id']
    labels_train_filtered = labels_train[labels_train['feedback_id'].isin(train_feedback_ids)]

    if len(labels_train_filtered) == 0:
        # No training data available, cannot train classifier
        # Return NI02 (unclassified) for all
        predictions = pd.DataFrame({
            'feedback_id': feedback_filtered['feedback_id'],
            'issue_type_id': 'NI02',
            'severity': 1,
            'actionable': 0
        })
        predictions.to_csv(output_path, index=False)
        return

    # Train classifier on filtered training data
    clf = VoiceOfStoreClassifier()
    clf.train(labels_train_filtered, feedback_all, taxonomy)

    # Predict for feedback up to as_of date
    issues, severities, actionables = clf.predict(feedback_filtered['text'].tolist())

    # Build predictions
    predictions = pd.DataFrame({
        'feedback_id': feedback_filtered['feedback_id'],
        'issue_type_id': issues,
        'severity': severities,
        'actionable': actionables
    })

    # Save predictions
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Causal detector for R5 Voice of Store Engine')
    parser.add_argument('--feedback', required=True, help='Path to feedback.csv')
    parser.add_argument('--as-of', required=True, help='Cutoff date YYYY-MM-DD')
    parser.add_argument('--out', required=True, help='Output predictions CSV path')

    args = parser.parse_args()

    detect(args.feedback, args.as_of, args.out)
