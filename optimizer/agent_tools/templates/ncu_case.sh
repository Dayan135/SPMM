#!/bin/bash
set -euo pipefail
use {{BUILD_DIR}} {{SRC_DIR}}
mkdir -p {{RUN_DIR}}/ncu
# Replace MAT, F and KERNEL before submitting; the filename must match the metadata.
NCU_FLAGS='{{NCU_FLAGS}}'
{{NCU}} $NCU_FLAGS -o {{RUN_DIR}}/ncu/MAT_FF \
  python -m benchmark.agent_profile --kernel KERNEL --mat MAT --F F \
  --out {{RUN_DIR}}/ncu/MAT_FF --commit {{SHA}}
