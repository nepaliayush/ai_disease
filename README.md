# Multi-Model Disease Prediction

A web application that predicts risk for four chronic diseases — **diabetes, heart
disease, liver disease, and chronic kidney disease** — by fusing four clinical
machine-learning models with a symptom-based triage model using **late
(decision-level) fusion**.

- **Backend:** Python + FastAPI (REST API, CORS-enabled)
- **Frontend:** Next.js (single sectioned form → single results view)
- **Explainability:** SHAP on every clinical prediction
- **Ethics:** public anonymized datasets only, subgroup bias evaluation reported
