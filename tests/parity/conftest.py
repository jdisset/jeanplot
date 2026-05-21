"""Parity-test conftest.

Puts this directory on sys.path so sibling test files can import from
`_parity_lib` (the SSOT for parity test helpers).
"""

import sys
from pathlib import Path

PARITY_DIR = Path(__file__).parent
if str(PARITY_DIR) not in sys.path:
    sys.path.insert(0, str(PARITY_DIR))
