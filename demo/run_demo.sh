#!/usr/bin/env bash
# Run the pipeline demo web server
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"

echo "=================================="
echo " MoE Quantization Pipeline Demo"
echo " http://${HOST}:${PORT}"
echo "=================================="

if ! python3 -c "import fastapi" 2>/dev/null; then
  echo "Installing demo dependencies..."
  pip install -r requirements.txt -q
fi

exec uvicorn server:app --host "$HOST" --port "$PORT" --reload
