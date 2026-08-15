#!/usr/bin/env bash
# Run the full disease-prediction app: FastAPI backend + Next.js frontend.
#
# Usage:
#   ./run.sh                 # use default ports (8000/3000), fall back if busy
#   API_PORT=8001 UI_PORT=3001 ./run.sh   # override ports
#
# Stop with Ctrl+C (stops both servers).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT/backend"
FRONTEND_DIR="$ROOT/frontend"
PY="$BACKEND_DIR/.venv/bin/python"
UVICORN="$BACKEND_DIR/.venv/bin/uvicorn"

# ---- 1. Sanity checks -------------------------------------------------------
if [[ ! -x "$UVICORN" ]]; then
  echo "[run] ERROR: backend venv not found. Run:"
  echo "      cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "[run] ERROR: frontend deps missing. Run: cd frontend && npm install"
  exit 1
fi

if ! ls "$BACKEND_DIR"/models/*.joblib >/dev/null 2>&1; then
  echo "[run] ERROR: no trained models found in backend/models/."
  echo "      Train them first:"
  echo "      cd backend && .venv/bin/python -m scripts.download_data && \\"
  echo "        .venv/bin/python -m training.prepare_datasets && \\"
  echo "        .venv/bin/python -m training.train_clinical && \\"
  echo "        .venv/bin/python -m training.train_symptom"
  exit 1
fi

# ---- 2. Pick ports ----------------------------------------------------------
port_free() { ! (ss -ltn 2>/dev/null | grep -q "[:.]$1 "); }

API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-3000}"
while ! port_free "$API_PORT"; do API_PORT=$((API_PORT + 1)); done
while ! port_free "$UI_PORT"; do UI_PORT=$((UI_PORT + 1)); done

API_URL="http://localhost:$API_PORT"

echo "[run] backend on port $API_PORT, frontend on port $UI_PORT"

# ---- 3. Start backend -------------------------------------------------------
echo "[run] starting FastAPI backend ..."
cd "$BACKEND_DIR"
CORS_ORIGINS="http://localhost:$UI_PORT,http://127.0.0.1:$UI_PORT" \
  "$UVICORN" app.main:app --host 0.0.0.0 --port "$API_PORT" &
BACK_PID=$!

# ---- 4. Wait for backend readiness ------------------------------------------
echo -n "[run] waiting for backend"
for _ in $(seq 1 60); do
  if curl -sf -m 2 "$API_URL/health" >/dev/null 2>&1; then
    echo " - up"
    break
  fi
  echo -n "."
  sleep 1
done
if ! kill -0 "$BACK_PID" 2>/dev/null; then
  echo
  echo "[run] backend failed to start. Check backend log output above."
  exit 1
fi

# ---- 5. Start frontend ------------------------------------------------------
# Clear the compiled cache so NEXT_PUBLIC_API_URL is re-inlined with the chosen
# backend port (stale `.next` output otherwise bakes in the previous URL).
echo "[run] starting Next.js frontend ..."
rm -rf "$FRONTEND_DIR/.next"
cd "$FRONTEND_DIR"
NEXT_PUBLIC_API_URL="$API_URL" npx next dev -p "$UI_PORT" &
FRONT_PID=$!

# ---- 6. Cleanup on exit -----------------------------------------------------
cleanup() {
  echo
  echo "[run] stopping servers ..."
  kill "$FRONT_PID" "$BACK_PID" 2>/dev/null || true
  wait "$FRONT_PID" "$BACK_PID" 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

echo
echo "  ==============================================="
echo "   Disease prediction app is running"
echo "   UI:       http://localhost:$UI_PORT"
echo "   API:      $API_URL"
echo "   API docs: $API_URL/docs"
echo "   Press Ctrl+C to stop."
echo "  ==============================================="
echo

wait