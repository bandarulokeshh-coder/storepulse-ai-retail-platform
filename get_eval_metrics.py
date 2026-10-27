"""Generate evaluation metrics for README."""
from sklearn.model_selection import train_test_split
from src.synthetic_data import load_or_create_synthetic_dataset
from src.classifier import FeedbackAnalyzer
from src.evaluation import compute_classification_metrics

# Load synthetic data
bundle = load_or_create_synthetic_dataset()
df = bundle.dataframe

print(f"Total synthetic rows: {len(df)}")
print(f"Dataset range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print()

# 80/20 split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['severity'])
print(f"Train: {len(train_df)}, Test: {len(test_df)}")
print()

# Rule-based evaluation
print("=" * 60)
print("RULE-BASED CLASSIFIER (no ML training)")
print("=" * 60)
rule_analyzer = FeedbackAnalyzer()
rule_analyzer.set_classifier_mode("rule")

predictions = test_df["feedback_text"].astype(str).apply(rule_analyzer.analyze)
pred_issue = [r.issue_category for r in predictions]
pred_severity = [r.severity for r in predictions]
pred_actionable = [str(r.actionable) for r in predictions]

true_issue = test_df["issue_category"].astype(str).tolist()
true_severity = test_df["severity"].astype(str).tolist()
true_actionable = test_df["actionable"].astype(bool).astype(str).tolist()

issue_metrics = compute_classification_metrics(true_issue, pred_issue)
severity_metrics = compute_classification_metrics(true_severity, pred_severity)
actionable_metrics = compute_classification_metrics(true_actionable, pred_actionable)

print(f"Issue Category - Accuracy: {issue_metrics.accuracy:.3f}, Macro F1: {issue_metrics.macro_f1:.3f}")
print(f"Severity - Accuracy: {severity_metrics.accuracy:.3f}, Macro F1: {severity_metrics.macro_f1:.3f}")
print(f"Actionability - Accuracy: {actionable_metrics.accuracy:.3f}, Macro F1: {actionable_metrics.macro_f1:.3f}")
print()

# ML-trained evaluation
print("=" * 60)
print("ML-TRAINED CLASSIFIER (TF-IDF + Logistic Regression)")
print("=" * 60)
ml_analyzer = FeedbackAnalyzer()
ml_analyzer.fit_from_dataframe(train_df)
ml_analyzer.set_classifier_mode("ml")

predictions = test_df["feedback_text"].astype(str).apply(ml_analyzer.analyze)
pred_issue = [r.issue_category for r in predictions]
pred_severity = [r.severity for r in predictions]
pred_actionable = [str(r.actionable) for r in predictions]

issue_metrics = compute_classification_metrics(true_issue, pred_issue)
severity_metrics = compute_classification_metrics(true_severity, pred_severity)
actionable_metrics = compute_classification_metrics(true_actionable, pred_actionable)

print(f"Issue Category - Accuracy: {issue_metrics.accuracy:.3f}, Macro F1: {issue_metrics.macro_f1:.3f}")
print(f"Severity - Accuracy: {severity_metrics.accuracy:.3f}, Macro F1: {severity_metrics.macro_f1:.3f}")
print(f"Actionability - Accuracy: {actionable_metrics.accuracy:.3f}, Macro F1: {actionable_metrics.macro_f1:.3f}")
print()

# Hybrid evaluation (policy-first, ML fallback)
print("=" * 60)
print("HYBRID CLASSIFIER (policy-first safety, ML for edge cases)")
print("=" * 60)
hybrid_analyzer = FeedbackAnalyzer()
hybrid_analyzer.fit_from_dataframe(train_df)
hybrid_analyzer.set_classifier_mode("hybrid")

predictions = test_df["feedback_text"].astype(str).apply(hybrid_analyzer.analyze)
pred_issue = [r.issue_category for r in predictions]
pred_severity = [r.severity for r in predictions]
pred_actionable = [str(r.actionable) for r in predictions]

issue_metrics = compute_classification_metrics(true_issue, pred_issue)
severity_metrics = compute_classification_metrics(true_severity, pred_severity)
actionable_metrics = compute_classification_metrics(true_actionable, pred_actionable)

print(f"Issue Category - Accuracy: {issue_metrics.accuracy:.3f}, Macro F1: {issue_metrics.macro_f1:.3f}")
print(f"Severity - Accuracy: {severity_metrics.accuracy:.3f}, Macro F1: {severity_metrics.macro_f1:.3f}")
print(f"Actionability - Accuracy: {actionable_metrics.accuracy:.3f}, Macro F1: {actionable_metrics.macro_f1:.3f}")
