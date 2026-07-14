"""Ensures the repo root (parent of the `optimizer` package) is importable
regardless of how pytest is invoked -- these tests are pure Python/SQLite,
no GPU/torch_fast_spdmm dependency, unlike tests/torch_fast_spdmm/ which
relies on `optimizer`'s sibling package being pip-installed instead.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
