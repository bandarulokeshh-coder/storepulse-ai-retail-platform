"""Compute the before/after metrics table for the hackathon demo.

All numbers are computed on the SYNTHETIC demonstration dataset unless a
labeled upload path is explicitly used. Run: python tools/metrics_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.classifier import FeedbackAnalyzer
from src.emerging_detector import detect_emerging_incidents
from src.evaluation import evaluate_emerging_detection
from src.policy_engine import evaluate_policy
from src.synthetic_data import load_or_create_synthetic_dataset


def _policy_highs(df: pd.DataFrame) -> tuple[list[bool], list[bool]]:
    policy_high = []
    label_high = []
    for text, severity in zip(df["feedback_text"].astype(str), df["severity"].astype(str)):
        policy_high.append(evaluate_policy(text, None).severity == "HIGH")
        label_high.append(severity.upper() == "HIGH")
    return policy_high, label_high


def main() -> int:
    bundle = load_or_create_synthetic_dataset()
    df = bundle.dataframe.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["actionable"] = df["actionable"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    print(f"dataset_rows={len(df)}")

    # --- classification metrics (full synthetic set, current analyzer) ---
    analyzer = FeedbackAnalyzer()
    analyzer.fit_from_dataframe(df)
    print(f"classifier_mode={analyzer.model_bundle.notes} trained_targets={analyzer.model_bundle.trained_targets}")
    if analyzer.model_bundle.errors:
        print(f"training_errors={analyzer.model_bundle.errors}")

    sample = df if len(df) <= 600 else df.sample(600, random_state=42)
    preds = sample["feedback_text"].astype(str).apply(analyzer.analyze)
    y_issue = sample["issue_category"].astype(str).tolist()
    p_issue = [r.issue_category for r in preds]
    y_sev = sample["severity"].astype(str).str.upper().tolist()
    p_sev = [r.severity.upper() for r in preds]
    y_act = sample["actionable"].astype(bool).astype(str).tolist()
    p_act = [str(bool(r.actionable)) for r in preds]
    print(f"issue_type_macro_f1={f1_score(y_issue, p_issue, average='macro', zero_division=0):.3f} "
          f"accuracy={accuracy_score(y_issue, p_issue):.3f} n={len(sample)}")
    print(f"severity_macro_f1={f1_score(y_sev, p_sev, average='macro', zero_division=0):.3f} "
          f"accuracy={accuracy_score(y_sev, p_sev):.3f}")
    print(f"actionability_macro_f1={f1_score(y_act, p_act, average='macro', zero_division=0):.3f} "
          f"accuracy={accuracy_score(y_act, p_act):.3f}")

    # --- escalation consistency (policy vs synthetic labels, row level) ---
    policy_high, label_high = _policy_highs(df)
    policy_high = np.asarray(policy_high)
    label_high = np.asarray(label_high)
    tp = int((policy_high & label_high).sum())
    precision = tp / max(int(policy_high.sum()), 1)
    recall = tp / max(int(label_high.sum()), 1)
    print(f"escalation_precision_row={precision:.3f} escalation_recall_row={recall:.3f} "
          f"policy_high={int(policy_high.sum())} label_high={int(label_high.sum())}")

    # --- alerts ---
    alerts = detect_emerging_incidents(df)
    print(f"alerts_total={len(alerts)}")
    emerging = evaluate_emerging_detection(df, alerts)
    print(f"seeded_incidents_detected={emerging['true_incidents_detected']}/{emerging['total_known_incidents']} "
          f"median_lead_days={emerging['median_detection_lead_time_days']} "
          f"escalation_precision_alert={emerging['escalation_precision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
