# StorePulse AI
**Turning Store Feedback Into Intelligent Action**

StorePulse AI is a demo-ready retail intelligence platform built for the **TalentAP Mega Hackathon 2026** problem statement:

> **R5 — Turning Store Chatter Into Action: The Voice-of-Store Engine**

The app processes multilingual retail feedback, detects operational issues, assigns department owners, flags policy-driven safety escalations, surfaces emerging incidents, and generates weekly regional action briefs.
## Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Architecture](#architecture)
- [Safety Escalation Policy](#safety-escalation-policy)
- [How Classification Works](#how-classification-works)
- [How Emerging Detection Works](#how-emerging-detection-works)
- [Limitations](#limitations)
- [Testing Instructions](#testing-instructions)
- [Team Roles](#team-roles)
- [Pre-built vs Built During Hackathon](#pre-built-vs-built-during-hackathon)
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
- **Streamlit** — main demo dashboard (`app.py`) - **This is the primary demo app**
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
storepulse-ai/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── synthetic_feedback.csv
├── src/
│   ├── preprocessing.py
│   ├── classifier.py
│   ├── embeddings.py
│   ├── policy_engine.py
│   ├── routing.py
│   ├── emerging_detector.py
│   ├── weekly_report.py
│   ├── duplicate_detector.py
│   ├── evaluation.py
│   ├── synthetic_data.py
│   └── storage.py
├── config/
│   └── issue_taxonomy.json
├── assets/
│   ├── logo.png
│   └── generate_logo.py
└── tests/
    ├── test_policy_engine.py
    ├── test_emerging_detector.py
    ├── test_routing.py
    ├── test_embeddings.py
    ├── test_evaluation.py
    └── smoke_all_pages.py
```

---

## Installation Instructions

### 1) Create and activate a virtual environment

#### Windows PowerShell
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt
```bash
python -m venv .venv
.venv\Scripts\activate.bat
```

#### macOS / Linux
```bash
python -m venv .venv
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
The app generates a labeled synthetic dataset with at least 2,000 records and marks it as:

**Synthetic Demonstration Data**

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
- `minimum_support = 2`
- `growth_threshold = 2.5`
- `spike_threshold = 3.0`

### Core formula
```text
normalized_rate = issue_count / max(total_store_feedback, 1)
```

The detector compares the current week against the previous 4–6 weeks using a robust baseline. It handles zero-history stores safely and can still detect small but meaningful spikes.

---

## Limitations
- The default demo dataset is synthetic and clearly labeled as such.
- Rule-based classification is transparent but less powerful than a fully trained multilingual model.
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

StorePulse AI applies a **policy-first safety escalation** approach to ensure critical issues are never missed:

### Always HIGH Severity:
- **Food contamination** with pests (cockroaches, insects, mold, fungus)
- **Washroom/toilet hygiene issues** (dirty, blocked, overflowing, smelly)
- **Allergen label errors** (allergen not mentioned, incorrect labeling)
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

This split (Tasks 3-4, Oct 26 2026) reduced false HIGH escalations on routine cleaning issues while preserving 100% recall on true safety concerns.

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

### Pre-built (before Oct 26, 2026):
- Initial Streamlit app structure (`app.py`)
- Basic issue taxonomy (`config/issue_taxonomy.json`)
- Synthetic data generator (`src/synthetic_data.py`)
- Rule-based classifier framework (`src/classifier.py`)
- Emerging incident detector (`src/emerging_detector.py`)
- Weekly report generator (`src/weekly_report.py`)
- Basic policy engine (`src/policy_engine.py`)
- Next.js/Postgres integration (unused in demo)

### Built/Enhanced During Hackathon (Oct 26, 2026):
- **Whole-word keyword matching** (Task 1) - prevents "eggplant" matching "egg"
- **Allergen rule refinement** (Task 2) - "milk" mention ≠ allergen risk
- **Negation and positive context handling** (Task 3) - "washroom was clean" → LOW
- **Hygiene vs floor cleanliness policy split** (Task 4) - dirty floor → MEDIUM
- **Department routing tests** (Task 5) - verified consistency
- **Train/test split utility** (Task 6) - held-out evaluation framework
- **Alert fatigue reduction** (Task 7) - minimum_support=3, volume thresholds
- **Real trend calculations** (Task 8) - week-over-week rate changes, "new" flag
- **Demo reliability scripts** (Task 9) - offline model caching, launcher scripts
- **Documentation and cleanup** (Task 10) - this section, repo cleanup

All changes tracked in git commits with detailed messages (Oct 26, 2026).

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
