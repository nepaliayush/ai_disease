# Multi-Model Disease Prediction

A web application that predicts risk for four chronic diseases — **diabetes, heart
disease, liver disease, and chronic kidney disease** — by fusing four clinical
machine-learning models with a symptom-based triage model using **late
(decision-level) fusion**.

- **Backend:** Python + FastAPI (REST API, CORS-enabled)
- **Frontend:** Next.js (single sectioned form → single results view)
- **Explainability:** SHAP on every clinical prediction
- **Ethics:** public anonymized datasets only, subgroup bias evaluation reported

---

## Architecture

```
┌────────────────────────────┐        ┌─────────────────────────────┐
│  Next.js (port 3000)       │  HTTP  │  FastAPI (port 8000)         │
│  one unified form          │ ─────> │  POST /predict               │
│  sections: clinical labs + │        │    ├─ 4 clinical models      │
│  symptom checklist         │  JSON  │    │    (each P(disease|X))  │
│  results: fused risk cards │ <───── │    ├─ 1 symptom triage model │
│  clinical/symptom split    │        │    │    (multi-class dist)    │
│  SHAP explanations         │        │    └─ decision-level fusion  │
└────────────────────────────┘        └─────────────────────────────┘
```

### Prediction pipeline (per disease)

```
clinical_risk       = deployed clinical model's P(disease | clinical data)
symptom_relevance   = Σ weight · P(symptom-model class) over classes mapped
                      to that disease category
fused_risk          = 0.7 × clinical_risk + 0.3 × symptom_relevance
overall assessment  = mean(fused_risk) over the four diseases
                     (NOT the mean of raw clinical scores)
```

Each response returns, per disease: the fused risk percentage, a transparency
breakdown of the clinical vs. symptom component, the SHAP explanation of the
clinical model, and the top conditions the symptom model saw.

---

## Getting started

### 1. Backend

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Download the public datasets (UCI / Kaggle mirrors)
.venv/bin/python -m scripts.download_data

# Clean the raw data into processed CSVs
.venv/bin/python -m training.prepare_datasets

# Train the 4 clinical models (comparison + deployment) and the symptom model
.venv/bin/python -m training.train_clinical
.venv/bin/python -m training.train_symptom

# Optional: bias / fairness report across age & sex subgroups
.venv/bin/python -m training.evaluate_bias

# Run the API
.venv/bin/uvicorn app.main:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev          # -> http://localhost:3000
```

The frontend calls `http://localhost:8000` by default. If your backend runs on a
different host/port, set `NEXT_PUBLIC_API_URL`:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8001 npm run dev
```

The UI is built with **shadcn/ui** (Tailwind CSS) and ships with **dark and light
mode** — toggle via the sun/moon button in the header. Light mode is clean and
minimal (document-like); dark mode is a **neon-blue theme** on deep navy (glowing
accents, radial top glow). The page uses the **full viewport width** with a
multi-column field grid, so the large clinical form stays compact.

Backend CORS allows any `localhost`/`127.0.0.1` port during development; extend
via the `CORS_ORIGINS` environment variable (comma-separated) for other
deployments.

> Tip: `./run.sh` starts both services for you (backend + frontend on free
> ports, with the correct `NEXT_PUBLIC_API_URL` and `CORS_ORIGINS`).

### 3. Tests

```bash
cd backend
.venv/bin/python -m pytest tests/ -q
```

---

## Data (all public and anonymized)

| Dataset | Source | Target |
| --- | --- | --- |
| Pima Indians Diabetes | UCI | diabetes (0/1) |
| Heart Disease (Cleveland) | UCI | heart disease (0/1) |
| Indian Liver Patient Dataset | UCI | liver disease (0/1) |
| Chronic Kidney Disease | UCI | CKD (0/1) |
| Disease–Symptom (132 symptoms × diseases) | public Kaggle mirror | 42/43 disease classes |

**Symptom dataset augmentation.** The base symptom dataset has no general
"chronic kidney disease" or "liver disease" classes. These are added as
balanced, knowledge-based signature classes (120 rows each) whose symptom sets
are derived from public medical symptom knowledge and, for CKD, from the
symptom-related attributes of the UCI CKD dataset (pedal edema → `swollen_legs`,
poor appetite → `loss_of_appetite`, anemia → `fatigue`, diabetes → `polyuria`).
Row instances are random partial subsets of each signature to mimic realistic
presentations. This is documented in `training/prepare_datasets.py`.

---

## Model development

**Clinical models (one per disease).** Five families are compared per disease
with a **stratified 80:20 split** and **10-fold stratified cross-validation**
(repeated over 3 shuffle seeds): Logistic Regression, SVM (RBF), Random Forest,
XGBoost, and MLP.

Preprocessing pipeline (fitted on training splits only, no leakage):
1. **Missing-value imputation** (median; physiologically impossible zeros in Pima
   are treated as missing)
2. **IQR outlier detection + clipping** (Tukey whiskers, `Q1 − 1.5·IQR` /
   `Q3 + 1.5·IQR`)
3. **log1p transform** of heavily right-skewed liver markers (bilirubin, liver
   enzymes), so the deployed liver model does not overreact to long tails
4. **Feature scaling** — Min–Max by default, StandardScaler also searchable
5. **SMOTE / SMOTEENN** oversampling of the minority class (run inside every CV
   fold via an imblearn pipeline, and on the training split for the deployed
   model)

Metrics: accuracy, **balanced accuracy**, precision, recall, F1, specificity,
ROC-AUC, plus **per-class precision/recall/F1** and a **confusion matrix** on
the held-out test set. The deployed model is selected by **cross-validated
ROC-AUC** (more honest on these small datasets than a single held-out metric).
Non-linear deployed models are **Platt-calibrated** on a held-out calibration
subset so displayed probabilities are calibrated (see
`app/preprocess.CalibratedClassifier`). The deployed decision threshold is
optimized for **F1** by default, but for the imbalanced liver set (71% positive)
it is optimized for **balanced accuracy** so the deployed model does not just
predict the majority class.

Headline metrics are reported as **mean ± spread over repeated 80:20 hold-out
splits** (5 seeds) and **repeated 10-fold CV** (3 repeats), so a single lucky
split cannot produce a flat 100% (which happens on the trivially-separable CKD
set). Diabetes, heart and liver get per-family **hyperparameter tuning**
(RandomizedSearchCV over model params, scaler choice, and SMOTE/SMOTEENN).
Heart tuning alone moved the deployed model from 79.3% → 82.0% accuracy and
89.0% → 90.7% AUC on the honest repeated hold-out; feature selection was tested
(chi² / mutual information) and *hurt* this dataset, so it is not used.

Full comparison tables are written to `backend/reports/clinical_comparison.csv`.

**Symptom model.** A Random Forest multi-class classifier over the symptom
dataset (accuracy / macro-F1 reported). The 132-symptom vocabulary is binarized;
unchecked symptoms are 0. The public dataset is near-perfectly separable
(typical for this knowledge-based data), so it acts as a deterministic triage
mapper rather than a noisy learner.

### Fusion weight justification (0.7 / 0.3)

The 0.7/0.3 split is a **reasoned default**, not a tuned value:

- **Clinical data is objective and measured** — each clinical model is trained
  directly on that disease's own clinical dataset and its output is a direct
  risk estimate, so it carries the larger weight.
- **Symptoms are subjective and self-reported**, with large inter-individual
  variation; the triage signal is a coarse category-level hint, so it carries
  the smaller weight.

**Limitation / future work:** the weights are not empirically optimized. Given a
validation set with ground-truth outcomes for both modalities, the weight could
be tuned (e.g., grid search on 0.5–0.9 maximizing fused ROC-AUC). This is
documented in the API response under `methodology.weight_justification`.

### Lifestyle adjustment (smoking & alcohol)

The four clinical datasets do **not** contain smoking or alcohol features, so
these cannot feed the trained models directly. Instead the unified form collects
`smoking_status` (never / occasional / daily) and `alcohol_consumption`
(none / light / moderate / heavy), and the backend applies an **explicit,
conservative, additive modifier** on top of each fused risk, clamped to [0, 1]:

- `liver_disease`: alcohol is the strongest modifier (up to +0.12 heavy)
- `heart_disease`: smoking is the strongest modifier (up to +0.04 daily)
- `diabetes` / `ckd`: smaller modifiers for both

The exact per-disease deltas live in `app/config.LIFESTYLE_ADJUSTMENT`. This is a
**reasoned rule-based heuristic from published risk-factor evidence — not a model
output** — and it is fully disclosed: the API returns `lifestyle_adjustment`
per disease, the methodology notes it, and the UI shows the applied value
("Lifestyle adjustment (smoking + alcohol): +X%") on every risk card.

---

## Explainability & ethics

- **SHAP** is computed for every clinical prediction. Tree models use an
  interventional `TreeExplainer`, linear models use `LinearExplainer`, and other
  families fall back to `KernelExplainer` on a background subsample. The API
  returns the base risk plus the top features and their directional
  contributions.
- **Transparency in the UI:** every risk card shows exactly how much of the
  score came from labs/clinical data vs. self-reported symptoms
  (`clinical_share_pct` / `symptom_share_pct`) and the separately disclosed
  lifestyle adjustment.
- **UI:** full-width shadcn/ui interface with a clean light theme and a
  neon-blue dark theme (dark/light toggle); results show a risk gauge, per-disease
  summary pills, contribution breakdowns and SHAP explanations.
- **Bias evaluation** (`training/evaluate_bias.py`) reports per-subgroup
  accuracy/recall/specificity/ROC-AUC plus **demographic-parity** and
  **equal-opportunity** gaps across age groups (all four diseases) and sex
  (heart, liver). Findings from the current run:
  - The liver model over-predicts disease (specificity ≈ 0 on the test split) —
    the ILPD dataset is 71% positive and noisy; flagged as a calibration/ethics
    caveat.
  - CKD parity gap is driven largely by the true base-rate rise with age, not
    by the model alone.
  - **Ethnicity is not recorded in any of the four public datasets** — subgroup
    analysis by ethnicity is therefore not possible and is a documented
    limitation.
- Only public, anonymized datasets are used; no personal data is collected by
  the application.

---

## API

| Endpoint | Description |
| --- | --- |
| `GET /` | Service info + deployed model names |
| `GET /health` | Readiness |
| `GET /metadata` | Form schema (sections, fields, symptom checklist, weights, deployed models) — the frontend builds its form from this |
| `POST /predict` | Full prediction: `{ "clinical": {…}, "symptoms": [ … ] }` → per-disease fused risks, breakdown, SHAP, overall average |

---

## Project layout

```
backend/
  app/                 FastAPI app (main, schemas, fields, preprocess,
                       model_store, pipeline, fusion, shap_engine, symptoms)
  training/            prepare_datasets, train_clinical, train_symptom,
                       evaluate_bias
  scripts/             download_data
  tests/               API tests
  data/                raw/ + processed/ (generated)
  models/              deployed artifacts (joblib)
  reports/             comparison + bias reports
frontend/
  app/                 Next.js pages (page.js, layout, globals.css)
  components/          Field, FormSection, SymptomChecklist, DiseaseCard,
                       OverallCard, RiskBar, ShapChart, theme-provider,
                       theme-toggle
  components/ui/       shadcn/ui primitives (button, card, input, select, …)
  lib/                 api client + helpers
```

## Limitations & future work

- Small public datasets ⇒ models are demonstrations, not clinical-grade tools.
- Fusion weights are a reasoned default (see above) — empirical tuning is future
  work.
- The unified form maps a single "blood pressure" / "glucose" value onto each
  dataset's corresponding feature even though the original datasets measured
  them in slightly different contexts (diastolic vs. resting systolic vs. BP at
  admission; fasting vs. random glucose). This is a documented simplification.
- No ethnicity metadata in any dataset; sex is only available for heart and
  liver datasets.