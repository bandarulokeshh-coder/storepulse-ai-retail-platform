#!/usr/bin/env python3
"""
StorePulse AI - Live Demo Script
TalentAP Mega Hackathon 2026 - Problem Statement R5
Turning Store Chatter Into Action: The Voice-of-Store Engine

Run: python live_demo_script.py
"""
import sys, os
sys.path.insert(0, os.getcwd())

import time
from pathlib import Path

print("\n" + "=" * 70)
print("STOREPULSE AI - LIVE DEMO")
print("=" * 70)
print("\nProject: TalentAP Mega Hackathon 2026 - Problem Statement R5")
print("         Turning Store Chatter Into Action: The Voice-of-Store Engine")
print("-" * 70)

# ---- PROJECT OVERVIEW ----
print("\n[1] PROJECT OVERVIEW")
print("-" * 70)
print("""
StorePulse AI turns unstructured retail feedback into operational action.
The platform processes multilingual store feedback and delivers:

• Issue classification (multilingual: English, Hinglish, Telugu, Tamil, Marathi)
• Policy-first safety escalation (allergens, contamination, hygiene, misconduct)
• Department routing (Store Operations, Supply Chain, Merchandising, Digital, HR)
• Emerging incident detection (early-warning store-level spike detection)
• Weekly regional action briefs (PDF/CSV reports with evidence quotes)
• Cross-channel duplicate detection
• Evaluation & model insights
""")

# ---- TECH STACK ----
print("\n[2] TECHNOLOGY STACK")
print("-" * 70)
print("""
Frontend/Framework:    Streamlit (app.py) + Next.js 16 (React 19)
Data Processing:       Pandas, NumPy
NLP/ML:                Sentence-Transformers (optional), Scikit-learn
Rule-based engine:     Transparent keyword + policy patterns
Visualization:         Plotly
Database:              SQLite (demo) + PostgreSQL (Drizzle ORM)
Reporting:             ReportLab (PDF), Pandas CSV
""")

# ---- KEY FEATURES DEMO ----
print("\n[3] DEMONSTRATION OF KEY FEATURES")
print("-" * 70)

# 1. Classification
print("\n✦ 3.1 AI Feedback Analyzer")
print("    Classifies multilingual feedback in real-time")
print("    Demonstrating with sample feedback:")

from src.classifier import default_analyzer
analyzer = default_analyzer()
samples = [
    "Store freezer not working, milk packets spoil avtunnayi!",
    "Wrong allergen label on food packet.",
    "Customer praise for friendly staff.",
    "Staff misconduct at checkout counter."
]
for i, text in enumerate(samples, 1):
    result = analyzer.analyze(text)
    print(f"       {i}. [{result.issue_category.upper():20s}] Severity={result.severity:5s} "
          f"Actionable={result.actionable} Dept={result.department:15s} Method={result.method}")

# 2. Policy escalation
print("\n✦ 3.2 Policy-First Safety Escalation")
print("    Safety categories always escalate to HIGH severity")
from src.policy_engine import evaluate_policy, should_escalate
test_texts = [
    "Milk packet has nuts but no allergen label",
    "Staff harassment at store entrance",
    "Generic positive feedback from customer"
]
for text in test_texts:
    decision = evaluate_policy(text)
    print(f"       '{text[:50]}...' → Severity={decision.severity} "
          f"Escalates={should_escalate(text)} Dept={decision.department}")

# 3. Emerging alerts
print("\n✦ 3.3 Emerging Incident Detection")
print("    Early-warning system based on normalized complaint rates")
from src.emerging_detector import detect_emerging_incidents
from src.synthetic_data import load_or_create_synthetic_dataset
bundle = load_or_create_synthetic_dataset()
df = bundle.dataframe
alerts = detect_emerging_incidents(df, minimum_support=2, growth_threshold=2.5, spike_threshold=3.0)
print(f"       {len(alerts)} alerts detected from {len(df)} records")
if not alerts.empty:
    top3 = alerts.head(3)
    for _, a in top3.iterrows():
        print(f"       • Store {a.store_id} | {a.issue_category:25s} "
              f"Growth={a.growth_indicator:.2f}x Count={a.current_count}")

# 4. Department routing
print("\n✦ 3.4 Automatic Department Routing")
from src.routing import load_taxonomy, get_department_for_issue, get_suggested_action
taxonomy = load_taxonomy()
categories = ["refrigeration failure", "allergen", "payment failure", "staff misconduct"]
for cat in categories:
    dept = get_department_for_issue(cat)
    action = get_suggested_action(cat)
    print(f"       '{cat}' → Department: {dept:20s} Action: {action}")

# 5. Duplicate detection
print("\n✦ 3.5 Cross-Channel Duplicate Detection")
from src.duplicate_detector import detect_duplicate_groups, duplicate_summary_message
dups = detect_duplicate_groups(df)
print(f"       {len(dups)} potential duplicate groups identified")
if not dups.empty:
    print(f"       {duplicate_summary_message(dups)}")

# 6. Weekly reports
print("\n✦ 3.6 Weekly Regional Action Briefs")
print("    Generates downloadable PDF/CSV reports per region")
print("    Includes: regional overview, top 5 priority issues, action suggestions,")

# ---- USER INTERFACE ----
print("\n[4] INTERACTIVE UI (Streamlit at localhost:8501)")
print("-" * 70)
print("""
The live app features a sidebar with:
• Navigation between 7 pages: Dashboard, AI Analyzer, Emerging Alerts,
  Weekly Reports, Store Analytics, Evaluation, Settings/About
• Filter controls: Region, Store, Channel, Severity, Issue Category, Date Range
• KPI metrics: Total feedback, High-severity issues, Emerging incidents,
  Stores monitored, Actionable %, Departments requiring attention
• Demo mode: 4 judge-ready scenarios that run through live application logic
• Settings: Toggle embeddings, upload official datasets, replace taxonomy
• Download buttons: PDF and CSV reports per region
• Page-specific: confusion matrices, model insights, trend charts
""")

# ---- CONCLUSION ----
print("\n[5] DEMO SUMMARY")
print("-" * 70)
print("""StorePulse AI demonstrates how transparent rule-based AI can solve
real retail operations problems without paid APIs. All classification
decisions are explainable: every alert shows which rules matched,
which policies triggered, and which department owns the issue.

Key hackathon values:
• Multilingual support (English, Hinglish, transliterated Indian languages)
• Policy-first severity (safety categories → HIGH automatically)
• Emerging detection with robust baselines (median/MAD)
• Zero-history store safety (handles new stores with low volume)
• Full explainability (no fabricated confidence scores)
• Works offline with synthetic demo data clearly marked""")

print("\n" + "=" * 70)
print("DEMO COMPLETE - Thank you for watching!")

# Keep the Streamlit app running note
print("\nNOTE: The Streamlit app is running at http://localhost:8501")
print("     Use the interactive UI to explore filters, datasets, and reports.")
print("=" * 70 + "\n")