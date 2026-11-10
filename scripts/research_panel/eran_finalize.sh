#!/bin/bash
# Run after the GPU screening array terminates, including failures. Originals
# remain untouched; the coverage audit identifies every incomplete cell.
set -eo pipefail
source ~/.bashrc
conda activate fem
set -u
: "${ROOT:?Set isolated sweep root}"
RUN_ROOT="$ROOT/eran_run"
export PYTHONPATH="$ROOT/deps:$ROOT/torch_fast_spdmm:$ROOT/tests/torch_fast_spdmm:$ROOT"
cd "$ROOT"
python scripts/research_panel/full_sweep_report.py "$RUN_ROOT"
python scripts/research_panel/eran_analyze.py "$RUN_ROOT" --out "$RUN_ROOT/analysis"
ALL_COMPLETE=0
if python -c 'import json,sys; p=sys.argv[1]; s=json.load(open(p)); sys.exit(0 if all(r["status"]=="complete" for r in s) else 1)' "$RUN_ROOT/aggregate/status.json"; then
    ALL_COMPLETE=1
    python scripts/research_panel/eran_confirm_manifest.py "$RUN_ROOT"
fi
EXTRA=()
if [ -f "$RUN_ROOT/confirmation_manifest.json" ]; then
    EXTRA+=(confirmation_manifest.json)
fi
tar -czf "$RUN_ROOT/raw-results-20261013-recovery.tar.gz" -C "$RUN_ROOT" \
    results logs smoke4 sweep.json source_hashes.txt aggregate analysis "${EXTRA[@]}" \
    -C "$ROOT" labels
sha256sum "$RUN_ROOT/raw-results-20261013-recovery.tar.gz" > "$RUN_ROOT/raw-results-20261013-recovery.sha256"
echo "ERAN_SCREENING_FINALIZED $RUN_ROOT"
test "$ALL_COMPLETE" -eq 1
