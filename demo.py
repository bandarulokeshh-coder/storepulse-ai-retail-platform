#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.getcwd())

# 1. Load synthetic demo data
from src.synthetic_data import load_or_create_synthetic_dataset
bundle = load_or_create_synthetic_dataset()
df = bundle.dataframe
print(f"Loaded {len(df)} feedback records")

# 2. Classify sample feedback
from src.classifier import default_analyzer
analyzer = default_analyzer()
samples = [
    "Store freezer not working, milk packets spoil!",
    "Wrong allergen label on food packet.",
    "Customer praise for friendly staff.",
    "Staff misconduct at checkout counter."
]
print("\nClassification demo:")
for i, text in enumerate(samples, 1):
    result = analyzer.analyze(text)
    print(f"{i}. '{text[:50]}...'")
    print(f"   Category: {result.issue_category}, Severity: {result.severity}, Actionable: {result.actionable}, Dept: {result.department}, Method: {result.method}")

# 3. Policy-based safety escalation
from src.policy_engine import evaluate_policy, should_escalate
print("\nPolicy escalation demo:")
for text in ["Milk packet has nuts but no allergen label", "Generic positive feedback"]:
    decision = evaluate_policy(text)
    print(f"'{text[:50]}...' -> Severity: {decision.severity}, Escalates: {should_escalate(text)}, Dept: {decision.department}, Reason: {decision.reason}")

# 4. Emerging incident detection
from src.emerging_detector import detect_emerging_incidents, summarize_alerts
alerts = detect_emerging_incidents(df, minimum_support=2, growth_threshold=2.5, spike_threshold=3.0)
print(f"\nEmerging alerts demo: {len(alerts)} alerts detected")
if not alerts.empty:
    top = alerts.head(2)
    for _, a in top.iterrows():
        print(f"Store {a.store_id} | {a.issue_category} | Growth: {a.growth_indicator:.2f}x | Count: {a.current_count}")

# 5. Duplicate detection
from src.duplicate_detector import detect_duplicate_groups, duplicate_summary_message
dups = detect_duplicate_groups(df)
print(f"\nDuplicate groups demo: {len(dups)} groups found")
if not dups.empty:
    print(f"  {duplicate_summary_message(dups)}")

# 6. Department routing preview
from src.routing import load_taxonomy, get_department_for_issue, get_suggested_action
taxonomy = load_taxonomy()
print("\nDepartment routing demo:")
for cat in ["refrigeration failure", "allergen", "payment failure"]:
    dept = get_department_for_issue(cat)
    action = get_suggested_action(cat)
    print(f"'{cat}' -> Dept: {dept}, Action: {action}")

print("\nDemo complete - all core features working!")