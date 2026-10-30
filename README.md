# StorePulse AI
**Turning Store Feedback Into Intelligent Action**

StorePulse AI is a demo-ready retail intelligence platform built for the **TalentAP Mega Hackathon 2026** problem statement:

> **R5 — Turning Store Chatter Into Action: The Voice-of-Store Engine**

The app processes multilingual retail feedback, detects operational issues, assigns department owners, flags policy-driven safety escalations, surfaces emerging incidents, and generates weekly regional action briefs.

## R5 Problem Statement

**Quick summary:**
- 39,769 feedback records from 60 stores over 26 weeks
- 3,000 labeled examples (weeks 1-13) for training
- 3 hidden emerging incidents to detect early
- 33 issue types with policy-defined severity floors

## Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Dataset Format](#dataset-format)
- [How Classification Works](#how-classification-works)
- [How Emerging Detection Works](#how-emerging-detection-works)
- [Limitations](#limitations)
- [Testing Instructions](#testing-instructions)
- [Safety Escalation Policy](#safety-escalation-policy)
- [Held-out Evaluation](#held-out-evaluation)
- [Team Roles](#team-roles)
- [Pre-built vs Built During Hackathon](#pre-built-vs-built-during-hackathon)
- [Windows + VS Code Quick Start](#windows--vscode-quick-start)
- [Notes for Judges](#notes-for-judges)


---

## Project Overview
Retail organizations receive large volumes of unstructured feedback from customer reviews, in-app comments, staff notes, and call-centre transcripts. StorePulse AI turns this feedback into operational action by:

- classifying issues from multilingual text
- applying policy-first safety escalation rules
- routing issues to the correct department
- detecting rising store-level incident patterns early
- generating weekly regional reports
- highlighting evidence-backed explanations and duplicate reports

The platform is designed to run locally and does **not require any paid API**.

---

## Key Features

### 1) Executive Dashboard

- Total feedback analyzed
- High-severity issues
- Emerging incidents
- Stores monitored
- Actionable feedback percentage
- Departments requiring attention
- Weekly top issues
- Feedback trend charts
- Issue distribution charts
- Department workload charts
- Filters for region, store, date range, channel, severity, and category

### 2) Multilingual AI Feedback Analyzer
Supports:

- English
- Hinglish
- Common transliterated Telugu
- Transliterated Tamil
- Transliterated Marathi
- Common spelling mistakes and retail shorthand

The analyzer uses:

- a transparent rule-based baseline
- optional lightweight ML training when labeled data is available
- policy-first safety escalation for allergen, hygiene, contamination, misconduct, and unsafe-product issues

### 3) Policy-First Safety Escalation
Any feedback mentioning safety-related concerns such as:

- allergens
- food contamination
- hygiene
- staff misconduct
- harassment
- unsafe products

is escalated to **HIGH** severity.

### 4) Emerging Incident Detection
A store-level early-warning system that monitors:

- complaint counts
- normalized complaint rates
- historical baselines
- growth rates
- robust median/MAD spikes

It can detect incidents early even in small stores with low complaint volume.

### 5) Weekly Regional Action Brief
For each region the app can generate:

- regional overview
- top 5 priority issues
- action suggestions
- evidence quotes
- downloadable PDF and CSV reports

### 6) Automatic Department Routing
Operational issues are routed to:

- Store Operations
- Supply Chain
- Merchandising
- Digital
- HR

### 7) Evidence-Based Explainability
Every key alert explains:

- why it triggered
- which rule or pattern matched
- which department owns the issue
- supporting feedback quotes from the actual dataset

### 8) Cross-Channel Duplicate Detection
The app groups probable duplicate incidents reported across multiple channels while keeping raw records intact.

### 9) Evaluation & Model Insights
Shows demonstration metrics for:

- issue type
- severity
- actionability
- emerging detection lead-time

---

## Technology Stack

### Frontend / App Framework

- **Streamlit** — main demo dashboard (`app.py`)
- **Primary demo app** — the primary demo application
- Next.js 16 (React 19) — supporting web infrastructure (optional, not used in main demo)

### Data Processing

- Pandas
- NumPy

### NLP / ML

- Sentence-Transformers — multilingual sentence embeddings (`paraphrase-multilingual-MiniLM-L12-v2`)
- Scikit-learn — TF-IDF + Logistic Regression classifiers
- Rule-based multilingual fallback for Hinglish and romanized Telugu, Tamil, Marathi

### Visualization

- Plotly

### Database

- SQLite — demo data storage (`data/storepulse.sqlite`, auto-regenerated)

**Note**: PostgreSQL/Drizzle ORM configuration exists for future Next.js integration but is not used in the current Streamlit demo.

### Reporting

- ReportLab for PDF exports
- Pandas CSV exports

---

## Architecture

```text
storepulse-ai-retail-platform/
|-- app.py                          # Main Streamlit dashboard
|-- requirements.txt
|-- README.md
|-- FINAL_REPORT.md
|-- demo.py                         # Live demo runner
|-- get_eval_metrics.py             # Evaluation metrics script
|-- data/
|   |-- storepulse.sqlite           # Auto-generated demo database
|   `-- synthetic_feedback.csv      # Raw synthetic dataset
|-- src/
|   |-- preprocessing.py            # Text normalization, multilingual mappings
|   |-- classifier.py               # FeedbackAnalyzer (rule/ML/hybrid modes)
|   |-- embeddings.py               # Sentence-transformers semantic matching
|   |-- policy_engine.py            # Safety escalation rules
|   |-- routing.py                  # Department routing, taxonomy loading
|   |-- emerging_detector.py        # Store-level incident early warning
|   |-- weekly_report.py            # Regional action brief generation
|   |-- duplicate_detector.py       # Cross-channel duplicate grouping
|   |-- evaluation.py               # Classification & emerging metrics
|   |-- holdout_evaluation.py       # Train/test split utilities
|   |-- synthetic_data.py           # Dataset generation (12,519 rows)
|   `-- storage.py                  # SQLite persistence layer
|-- config/
|   `-- issue_taxonomy.json         # Issue categories, keywords, departments
|-- assets/
|   |-- logo.png
|   `-- generate_logo.py
|-- scripts/
|   `-- download_models.py          # Offline model caching
|-- tools/
|   `-- metrics_report.py           # Metrics aggregation
`-- tests/
    |-- test_policy_engine.py       # Safety escalation tests
    |-- test_emerging_detector.py   # Alert detection tests
    |-- test_routing.py             # Department routing tests
    |-- test_regressions.py         # Bug fix regression suite (30 cases)
    |-- test_embeddings.py          # Semantic matching tests
    |-- test_evaluation.py          # Metrics calculation tests
    |-- test_whole_word_matching.py # Keyword boundary tests
    |-- test_classifier_training.py # ML training tests
    |-- test_csv_upload.py          # Upload handling tests
    |-- test_department_routing.py  # Department consistency tests
    |-- test_alert_consistency.py   # Dashboard/alerts page sync tests
    `-- smoke_all_pages.py          # Full app integration test
```

---

## Installation Instructions

### 1) Create and activate a virtual environment

#### Windows PowerShell
```bash
python -m venv .venv
```

```bash
.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt
```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate.bat
```

#### macOS / Linux
```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Start the Streamlit app
```bash
streamlit run app.py
```

### Optional: run tests
```bash
pytest -q
```

---

## Dataset Format

### Synthetic Demonstration Data
The app generates a labeled synthetic dataset with 12,519 feedback records spanning 12 weeks (Aug 11 - Oct 27, 2026). The dataset includes:

- 50 retail stores across 4 regions (Mumbai, Bengaluru, Hyderabad, Chennai)
- 4 channels (In-app Rating, Store Staff Note, Online Review, Call Center)
- Multilingual feedback (English, Hinglish, romanized Telugu/Tamil/Marathi)
- 6 embedded incident scenarios for early-warning system testing
- Labeled ground truth for issue category, severity, actionability, and department

The dataset is marked as **Synthetic Demonstration Data**.

### Required fields

- `feedback_id`
- `store_id`
- `region`
- `timestamp`
- `channel`
- `feedback_text`
- `issue_category`
- `severity`
- `actionable`
- `department`

### Additional fields

- `dataset_mode`
- `incident_tag`
- `known_incident_flag`
- `language_hint`
- `week_start`

### Official Dataset Upload
You can upload a CSV and map columns for:

- feedback text
- date
- store ID
- region
- channel

Optional uploads:

- store master
- taxonomy
- labeled training data

Missing values and invalid dates are handled safely.

---

## How Classification Works

StorePulse AI uses a transparent layered approach:

1. **Normalization**
   - lowercasing
   - punctuation cleanup
   - transliteration-friendly replacements
   - multilingual token normalization

2. **Rule-based classification**
   - keyword and policy pattern matches
   - multilingual retail terms
   - safety escalation rules

3. **Optional multilingual embeddings**
   - when available, a local sentence-transformers multilingual model
     (`paraphrase-multilingual-MiniLM-L12-v2`) provides semantic category
     matching for feedback where keyword rules are weak
   - embeddings only fill gaps: transparent keyword rules always take precedence
   - if the model cannot be loaded (offline, missing package), the app shows a
     labeled rule-based fallback and never crashes
   - semantic similarity scores are shown as similarities, never as
     fabricated classification confidences
   - the toggle lives in **Settings / About** (default: enabled, loads lazily
     on first use)

4. **Optional supervised ML**
   - if labeled data is available, the app trains lightweight
     TF-IDF + Logistic Regression models for **issue type, severity, and
     actionability**
   - the policy engine remains authoritative for safety-related severity
   - if ML is unavailable, the app falls back to the rule engine without crashing

5. **Policy-first severity logic**
   - allergen, hygiene, contamination, harassment, staff misconduct, and unsafe products are forced to HIGH severity

The app does **not** invent confidence percentages for pure rule-based decisions.

---

## How Emerging Detection Works

The early-warning system calculates, for each store and issue category:

- weekly issue count
- normalized complaint rate
- historical median baseline
- median absolute deviation (MAD)
- safe growth ratios

### Default alert conditions

- `minimum_support = 3` (raised from 2 to reduce alert fatigue on low-volume noise)
- `growth_threshold = 2.5` (current rate must be 2.5× the historical baseline)
- `spike_threshold = 3.0` (current rate must be 3 MAD above baseline for spike detection)
- `lookback_weeks = 6` (compare current week against 6-week historical window)
- `minimum_store_week_total = 5` (stores need at least 5 feedback items per week for reliable rate normalization)

### Alert sensitivity

Measured on the synthetic demo dataset (6 seeded incidents):

| minimum_support | Alerts | Seeded incidents caught | Median lead time (days) |
| --- | --- | --- | --- |
| 2 | 1,511 | 6/6 | 7.0 |
| 3 | 570 | 4/6 | 14.0 |
| 4 | 225 | 2/6 | 10.5 |
| 5 | 82 | 2/6 | 17.5 |

### Core formula
```text
normalized_rate = issue_count / max(total_store_feedback, 1)
```

The detector compares the current week against the previous 4–6 weeks using a robust baseline. It handles zero-history stores safely and can still detect small but meaningful spikes.

---

## Limitations

- The default demo dataset is synthetic and clearly labeled as such.
- **Synthetic labels were updated on Oct 26, 2026** to align with the refined safety escalation policy: 301 Store Cleanliness rows changed from HIGH to MEDIUM. See `data/LABEL_CHANGES.md` for details.
- **Improved escalation precision reflects this label correction**, not a model improvement. The policy now correctly distinguishes washroom hygiene (HIGH) from floor cleanliness (MEDIUM).
- Rule-based classification is transparent but less powerful than a fully trained multilingual model.
- Multilingual support covers common Hinglish and romanized Indian languages. Transliterated keywords added during testing include Hindi "kharab" (spoiled), Tamil "alukku" (dirty), and payment failure patterns.
- Optional sentence-transformers support depends on local installation and model availability.
- Some evaluation results are demo-only unless real ground-truth labels are uploaded.
- Duplicate detection is heuristic and intended for operations review, not automatic merging.

---

## Testing Instructions

Run the automated tests:

```bash
pytest -q
```

Run the full-app page check (executes every Streamlit page headlessly and
fails on any exception):

```bash
python tests/smoke_all_pages.py
```

Included scenarios:

- Safety escalation for allergen issues
- Department routing for payment failures
- Emerging refrigeration spike detection
- Low-volume store trend detection
- Missing data handling
- Unknown feedback should not fabricate a critical issue
- Embedding-unavailable fallback to the rule baseline
- Embedding-assisted classification for weak rule matches
- Escalation precision only computed when ground-truth labels exist
- Classification metrics hidden when ground-truth labels are missing

---

## Safety Escalation Policy

StorePulse AI applies a **policy-first safety escalation** approach that routes critical safety concerns to HIGH severity:

### Always HIGH Severity:

- **Food contamination** with pests (cockroaches, insects, mold, fungus)
- **Spoiled or expired food** (spoiling, spoiled, rotten, stale food): detected even in transliterated feedback ("kharab", "spoil autu", "spoil ayyindi")
- **Washroom/toilet hygiene issues** (dirty, blocked, overflowing, smelly)
- **Allergen label errors** (allergen not mentioned, incorrect labeling, ingredient not listed on label)
- **Staff misconduct** (inappropriate behavior, harassment, threats)
- **Unsafe products** (hazardous, dangerous, broken glass)

### MEDIUM Severity:

- **General floor/shelf cleanliness** (dirty floor, trash visible, messy aisles)
  - Even with "dirty" keyword, floor cleanliness stays MEDIUM unless food/pests involved
- **Stock shortages, delivery delays, price errors** (operational issues)

### Policy Implementation:

- Safety patterns **override** ML predictions
- Negation detection prevents false positives ("washroom was clean" → LOW, not HIGH)
- Positive context words (clean, fresh, thanks) prevent escalation
- Neutral hygiene words (washroom, toilet) only escalate with negative qualifiers

This split (Tasks 3-4, Oct 26 2026) reduced false HIGH escalations on routine cleaning issues while preserving detection of true safety concerns.

---

## Held-out Evaluation

Performance on 12,519 labeled synthetic feedback records spanning 12 weeks (Aug 11 - Oct 27, 2026). After deduplication by unique normalized feedback text, the dataset resolves to **1,314 unique texts**, and the held-out evaluation uses a train/test split of unique texts (80/20, stratified by severity):

### Rule-Based Classifier (no ML training)

- **Issue Category** - Accuracy: 0.688, Macro F1: 0.690
- **Severity** - Accuracy: 0.890, Macro F1: 0.878
- **Actionability** - Accuracy: 0.928, Macro F1: 0.884

### ML-Trained (TF-IDF + Logistic Regression)

- **Issue Category** - Accuracy: 0.992, Macro F1: 0.983
- **Severity** - Accuracy: 0.992, Macro F1: 0.991
- **Actionability** - Accuracy: 0.992, Macro F1: 0.989

### Hybrid (Policy-First Safety + ML Fallback)

- **Issue Category** - Accuracy: 0.840, Macro F1: 0.838
- **Severity** - Accuracy: 0.924, Macro F1: 0.918
- **Actionability** - Accuracy: 0.951, Macro F1: 0.926

**Notes:**

- Rule-based classifier is transparent and requires no training data, but has lower accuracy on ambiguous cases
- ML-trained achieves near-perfect issue categorization on this synthetic dataset (may not generalize to real-world noise)
- Hybrid mode uses policy rules for safety escalation and ML for operational issues, balancing safety guarantees with learned patterns
- All metrics measured on a held-out test set of **263 unique texts** not seen during training, computed on a split by unique normalized text to avoid leakage

---

## Team Roles

**Team Members**: [Fill in actual names and roles before submission]

Example roles:

- **Full-stack developer:** Streamlit app, UI, uploads, exports
- **ML/NLP engineer:** multilingual normalization, rule engine, ML training
- **Data scientist:** incident detection, evaluation metrics, report scoring
- **UX designer:** dashboard layout, cards, filters, report styling

---

## Pre-built vs Built During Hackathon

### Pre-built (before Oct 30, 2026):
**All work done on Oct 26, 2026 counts as PRE-BUILT preparation**, not hackathon work.

- Initial Streamlit app structure (`app.py`)
- Basic issue taxonomy (`config/issue_taxonomy.json`)
- Synthetic data generator (`src/synthetic_data.py`): 12,519 rows
- Rule-based classifier framework (`src/classifier.py`)
- Emerging incident detector (`src/emerging_detector.py`)
- Weekly report generator (`src/weekly_report.py`)
- Basic policy engine (`src/policy_engine.py`)
- Next.js/Postgres integration (unused in demo)

**Bug fixes and improvements (Oct 26, 2026 - PRE-BUILT):**

- Whole-word keyword matching (prevents "eggplant" matching "egg")
- Allergen rule refinement ("milk" mention ≠ allergen risk)
- Negation and positive context handling ("washroom was clean" → LOW)
- Hygiene vs floor cleanliness policy split (dirty floor → MEDIUM)
- Department routing consistency tests
- Train/test split utility (held-out evaluation framework)
- Alert fatigue reduction (minimum_support=3, volume thresholds)
- Real trend calculations (week-over-week rate changes)
- Demo reliability scripts (offline model caching, launcher scripts)
- **Synthetic label update**: 301 Store Cleanliness rows changed from HIGH to MEDIUM to match policy
- Comprehensive documentation and repo cleanup

All changes are tracked in the repository history (Oct 26, 2026).

### Changes Made During the Hackathon (Oct 30-31, 2026):
**[This section to be filled in during the actual hackathon event]**

Examples of what might be added:

- Live demo refinements based on judge feedback
- Additional test cases from real scenarios
- Performance optimizations discovered during demo
- UI/UX improvements from user testing
- Bug fixes found during live presentation

---

## Windows + VS Code Quick Start

1. Open the project folder in **VS Code**.
2. Open **Terminal** in VS Code.
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
4. Activate it:
   ```bash
   .venv\Scripts\activate
   ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Launch the app:
   ```bash
   streamlit run app.py
   ```
7. Open the local URL Streamlit prints in the terminal.

If Streamlit does not open automatically, copy the local URL into Chrome or Edge.

---

## Notes for Judges

- Demo Mode is available from the dashboard.
- All demo records are synthetic and labeled accordingly.
- The project is designed to work locally without paid APIs.
- Weekly reports and emerging alerts use actual processed records from the loaded dataset.
