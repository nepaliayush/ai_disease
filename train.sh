#!/usr/bin/env bash
# Train (or retrain) the full disease-prediction model stack.
#
# Steps run, in order:
#   1. (optional) Download raw public datasets        --with-download
#   2. Clean raw data into processed CSVs (incl. engineered features)
#   3. Train the 4 clinical models (tuning + comparison + deployment)
#   4. Train the symptom triage model
#   5. (optional) Bias / fairness evaluation           --with-bias
#
# Usage:
#   ./train.sh                # prepare + train clinical + train symptom
#   ./train.sh heart_disease  # retrain only one clinical model (heart)
#   ./train.sh --with-download
#   ./train.sh --with-bias
#   ./train.sh --with-download --with-bias
#
# Valid disease names: diabetes | heart_disease | liver_disease | ckd
#
# After training, restart the API so it loads the new artifacts:
#   ./run.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT/backend"
PY="$BACKEND_DIR/.venv/bin/python"

WITH_DOWNLOAD=0
WITH_BIAS=0
DISEASE=""

for arg in "$@"; do
  case "$arg" in
    --with-download) WITH_DOWNLOAD=1 ;;
    --with-bias) WITH_BIAS=1 ;;
    --help|-h)
      sed -n '2,20p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    diabetes|heart_disease|liver_disease|ckd) DISEASE="$arg" ;;
    *) echo "[train] unknown option: $arg (see --help)" >&2; exit 1 ;;
  esac
done

# ---- 1. Sanity checks -------------------------------------------------------
if [[ ! -x "$PY" ]]; then
  echo "[train] ERROR: backend venv not found. Run:"
  echo "        cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

echo "[train] backend at $BACKEND_DIR"
cd "$BACKEND_DIR"

# ---- 2. (optional) Download raw datasets ------------------------------------
if [[ "$WITH_DOWNLOAD" -eq 1 ]]; then
  echo "[train] downloading raw datasets ..."
  "$PY" -m scripts.download_data
else
  if ! ls "$BACKEND_DIR"/data/raw/*.csv >/dev/null 2>&1; then
    echo "[train] no raw data found. Re-run with --with-download"
    exit 1
  fi
fi

# ---- 3. Clean raw data into processed CSVs ----------------------------------
# (only needed for a full run; a single-disease retrain uses existing data)
if [[ -z "$DISEASE" ]]; then
  echo "[train] preparing datasets (clean + engineered features) ..."
  "$PY" -m training.prepare_datasets
fi

# ---- 4. Train clinical models -----------------------------------------------
if [[ -n "$DISEASE" ]]; then
  echo "[train] training clinical model for '$DISEASE' ..."
  "$PY" -m training.train_clinical "$DISEASE"
else
  echo "[train] training clinical models (this takes a few minutes) ..."
  "$PY" -m training.train_clinical
fi

# ---- 5. Train symptom triage model ------------------------------------------
# (only on a full run — a single-disease retrain leaves the symptom model alone)
if [[ -z "$DISEASE" ]]; then
  echo "[train] training symptom triage model ..."
  "$PY" -m training.train_symptom
fi

# ---- 6. (optional) Bias / fairness report ------------------------------------
if [[ "$WITH_BIAS" -eq 1 ]]; then
  echo "[train] running bias evaluation ..."
  "$PY" -m training.evaluate_bias
fi

echo
echo "  ==============================================="
echo "   Training complete."
echo "   Artifacts: backend/models/*.joblib"
echo "   Reports:   backend/reports/*"
echo "   Restart the API to load the new models: ./run.sh"
echo "  ==============================================="