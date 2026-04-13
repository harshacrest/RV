"""Shared path configuration for core framework files."""

import sys
from pathlib import Path

# core/ directory
CORE_DIR = Path(__file__).parent

# autoresearch_regime/ root
AUTORESEARCH_DIR = CORE_DIR.parent

# RV project root
RV_ROOT = AUTORESEARCH_DIR.parent

# Ensure core/ is on sys.path so sibling imports work (from X import Y)
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

# Ensure RV root is on path for pipeline imports
if str(RV_ROOT) not in sys.path:
    sys.path.insert(0, str(RV_ROOT))
