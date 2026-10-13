#!/bin/bash
set -euo pipefail
: "${ROOT:?Set isolated sweep root}"
cd "$ROOT"
python3 scripts/research_panel/full_sweep_report.py "$ROOT"
