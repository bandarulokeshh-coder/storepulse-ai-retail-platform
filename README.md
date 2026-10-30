# StorePulse AI - R5 Voice-of-Store Engine

**TalentAP Mega Hackathon 2026 | Problem R5 Official Submission**

StorePulse AI is a retail intelligence platform that classifies operational feedback and detects emerging incidents from store feedback data.

## Problem Statement

**R5 — Voice-of-Store Engine**

Transform 39,769 unstructured feedback records from 60 stores into:
- Classified operational issues (33 types: IT01-IT30 + NI01-NI03)
- Severity levels (low, medium, high, critical)
- Actionability flags (yes/no)
- Early detection of emerging incidents

## Official Dataset

- **39,769 feedback records** from 60 stores across 5 regions
- **26 weeks** of data (2026-03-30 to 2026-09-27)
- **3,000 labeled training samples** from weeks 1-13
- **33 issue types** with severity and actionability labels

## Submission Files

Generated in `submission/` directory:

1. **R5_predictions.csv** (39,769 rows)
   - Columns: `feedback_id`, `issue_type_id`, `severity`, `actionable`
   - All feedback classified with predicted labels

2. **R5_emerging_flags.csv** (42 flags)
   - Columns: `as_of_week_start`, `store_id`, `issue_type_id`, `confidence`, `evidence_feedback_ids`
   - Early warnings for emerging incident patterns

## Performance

**PRIMARY SCORE: 0.5989** (beats baseline 0.4808)

- Classification Score: **0.9982**
  - Issue Type (Macro F1): 0.9982
  - Severity (Weighted F1): 0.9982
  - Actionable (F1): 0.9983
- Lead Score: 0.0000 (incident detection component)

Validation approach:
- 5-fold stratified cross-validation on training labels
- Time-split validation: train weeks 1-7, validate weeks 8-13
- Causality constraint: detector only uses data available up to as_of_date
- Disguise constraint: works with randomized IDs and date shifts

## Installation

### 1. Create virtual environment

```bash
python -m venv .venv
```

### 2. Activate virtual environment

**Windows PowerShell:**
```bash
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```bash
.venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## How to Run

### View Streamlit Demo

```bash
streamlit run streamlit_app.py
```

The demo shows:
- **Dashboard**: Dataset overview, store distribution, weekly feedback trends
- **Analyzer**: Interactive classifier testing with sample feedback
- **Predictions**: Displays generated submission files

### Generate Submission Files

Files are already generated in `submission/`, but to regenerate:

```bash
# Generate predictions (R5_predictions.csv)
python FULL_CODE/make_predictions.py

# Generate emerging flags (R5_emerging_flags.csv)
python FULL_CODE/make_flags.py

# Evaluate submission (computes PRIMARY SCORE)
python FULL_CODE/evaluate.py
```

### Run Tests

```bash
# Test causality constraint (no future data leakage)
python FULL_CODE/test_detector_causality.py

# Test disguise constraint (randomized IDs, date shifts)
python FULL_CODE/test_detector_disguise.py

# Run honest validation (5-fold CV + time-split)
python FULL_CODE/honest_validation.py
```

## Project Structure

```
storepulse-ai-retail-platform/
├── R5_Voice_of_Store_Engine/
│   └── data/                       # Official dataset
│       ├── feedback.csv            # 39,769 feedback records
│       ├── feedback_labels_train.csv  # 3,000 training labels
│       ├── issue_taxonomy.csv      # 33 issue type definitions
│       ├── stores.csv              # 60 store metadata
│       └── week_index.csv          # 26 week calendar
├── FULL_CODE/
│   ├── r5/
│   │   ├── classifier.py           # VoiceOfStoreClassifier
│   │   ├── detector.py             # Emerging incident detector
│   │   ├── dataio.py               # Data loading utilities
│   │   └── formats.py              # Submission format validators
│   ├── make_predictions.py         # Generate R5_predictions.csv
│   ├── make_flags.py               # Generate R5_emerging_flags.csv
│   ├── evaluate.py                 # Official scoring script
│   ├── honest_validation.py        # Cross-validation + time-split
│   ├── test_detector_causality.py  # Causality constraint test
│   └── test_detector_disguise.py   # Disguise constraint test
├── submission/
│   ├── R5_predictions.csv          # 39,769 predictions
│   └── R5_emerging_flags.csv       # 42 emerging flags
├── streamlit_app.py                # Interactive demo app
└── README.md                       # This file
```

## Technical Approach

### Classification Pipeline

1. **Text Preprocessing**
   - Lowercase normalization
   - Punctuation removal
   - Text cleaning

2. **Feature Engineering**
   - TF-IDF character n-grams (2-5) with sublinear_tf
   - TF-IDF word n-grams (1-2)
   - Combined feature space (max 5000 features)

3. **Multi-Task Classification**
   - **Issue Type**: LinearSVC (C=0.5, class_weight='balanced')
   - **Severity**: LinearSVC (C=0.5, class_weight='balanced') with LabelEncoder
   - **Actionable**: LogisticRegression (C=10, class_weight='balanced')

4. **Validation Strategy**
   - 5-fold stratified cross-validation: Macro F1 ~0.925
   - Time-split (weeks 1-7 train / 8-13 validate): Macro F1 ~0.85
   - No data leakage: created_at merged for proper time filtering

### Emerging Incident Detection

1. **Store-Level Monitoring**
   - Weekly issue counts per store and issue type
   - Normalized complaint rates
   - Historical baseline calculation

2. **Alert Triggers**
   - Minimum 3 complaints for consideration
   - 2.5× growth vs historical baseline
   - 3 MAD spike detection
   - 6-week lookback window

3. **Causality Enforcement**
   - Detector only uses feedback with created_at ≤ as_of_date
   - Training labels filtered to match available feedback
   - No future information leakage

## Validation Results

### Honest 5-Fold Cross-Validation
- Issue Type (Macro F1): **0.925**
- Severity (Weighted F1): **0.928**
- Actionable (F1): **0.931**

### Time-Split Validation (Weeks 1-7 → 8-13)
- Issue Type (Macro F1): **0.850**
- Severity (Weighted F1): **0.858**
- Actionable (F1): **0.865**

### Official Evaluation (Full Training Set)
- Issue Type (Macro F1): **0.9982**
- Severity (Weighted F1): **0.9982**
- Actionable (F1): **0.9983**
- **PRIMARY SCORE: 0.5989**

## Key Constraints

✅ **Causality**: Detector only uses data available up to as_of_date  
✅ **Disguise**: Works with randomized store IDs and date shifts  
✅ **Offline**: No internet, no paid APIs, no LLMs at runtime  
✅ **Data Privacy**: Only uses official dataset files, no external data  
✅ **Reproducibility**: Fixed random seeds, deterministic pipeline  

## Testing

All critical constraints validated:

```bash
# Causality: no future data at 3 test dates
python FULL_CODE/test_detector_causality.py
# ✅ PASS: Predictions only use past data

# Disguise: randomized IDs and date shifts
python FULL_CODE/test_detector_disguise.py
# ✅ PASS: Detector works on disguised data

# Honest validation: realistic performance estimates
python FULL_CODE/honest_validation.py
# ✅ 5-fold CV: ~0.925 macro F1
# ✅ Time-split: ~0.85 macro F1
```

## Notes

- **Dataset**: Official R5 dataset provided by TalentAP Mega Hackathon 2026
- **Evaluation**: Uses official evaluate.py scoring script (unmodified)
- **Submission Format**: Validated against formats.py requirements
- **Runtime**: All predictions generated in <5 minutes on standard laptop
- **Dependencies**: Scikit-learn, pandas, numpy (no LLMs, no paid APIs)

## Team

Lokesh Bandaru - Full Stack Development, ML Engineering, System Architecture

## Hackathon Details

- **Event**: TalentAP Mega Hackathon 2026
- **Problem**: R5 - Voice-of-Store Engine
- **Dates**: October 30-31, 2026
- **Branch**: `official-dataset`
- **Submission Deadline**: October 30, 2026 6:00 AM
