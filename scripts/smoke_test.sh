#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${MOTIONSPEECH_PYTHON:-python}
exec "$PYTHON_BIN" "$REPO_ROOT/scripts/run_smoke.py" \
  --data-root "${MOTIONSPEECH_DATA_ROOT:?Set MOTIONSPEECH_DATA_ROOT}" \
  --work-root "${MOTIONSPEECH_SMOKE_ROOT:-$REPO_ROOT/outputs/smoke-ten}" "$@"
