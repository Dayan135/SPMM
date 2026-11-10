#!/bin/bash
set -eo pipefail
source ~/.bashrc
conda activate fem
set -u
: "${ROOT:?Set isolated sweep root}"
RUN_ROOT="$ROOT/eran_run"
export PYTHONPATH="$ROOT/deps:$ROOT/torch_fast_spdmm:$ROOT/tests/torch_fast_spdmm:$ROOT"
cd "$ROOT"
python scripts/research_panel/eran_confirm_report.py "$RUN_ROOT"
tar -czf "$RUN_ROOT/raw-confirmation-20261013.tar.gz" -C "$RUN_ROOT" \
    confirm confirmation_manifest.json logs
sha256sum "$RUN_ROOT/raw-confirmation-20261013.tar.gz" > \
    "$RUN_ROOT/raw-confirmation-20261013.sha256"
python - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / 'confirm'
statuses = []
for allocation in (1, 2, 3):
    path = root / f'alloc{allocation}' / 'status.json'
    statuses.append(json.loads(path.read_text()).get('status') if path.exists() else 'missing')
if statuses != ['complete', 'complete', 'complete']:
    raise SystemExit(f'confirmation incomplete: {statuses}')
PY
echo "ERAN_CONFIRMATION_FINALIZED $RUN_ROOT"
