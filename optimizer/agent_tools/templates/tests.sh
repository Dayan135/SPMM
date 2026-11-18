#!/bin/bash
set -euo pipefail
use {{BUILD_DIR}} {{SRC_DIR}}
# These nine test files exist in the candidate source tree.
python -m pytest test_pbr.py test_csr.py test_boo.py test_boo_edges.py \
  test_boo_shape.py test_kcp_shape.py test_boo_fpt.py test_boo_nnzflat_fpt.py \
  test_cusparse_direct.py test_timed_variants.py --boo-require-cuda -q -p no:cacheprovider
echo "AGENT_TESTS_PASSED {{SHA}}"
