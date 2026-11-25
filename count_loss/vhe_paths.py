"""
vhe_paths -- the only machine-specific settings of this repository.
Every location can be set with an environment variable; the defaults point
inside the repository (released tables in work/, data you obtain in data/).
"""

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get("VHE_COUNT_WORK", os.path.join(REPO, "work"))            # runs, eval CSVs, detector tables, counter
ALIGNED = os.environ.get("VHE_ALIGNED_DIR", os.path.join(REPO, "data", "aligned"))  # registered SRS/H&E stacks (not released)
RAW_DATA = os.environ.get("VHE_RAW_DATA", os.path.join(REPO, "data", "raw"))   # folder with fsSRS/ and H&E/ (not released)
V2_DIR = os.environ.get("VHE_V2_DIR", os.path.join(REPO, "registration"))       # registration pipeline and its results/


def work(*parts):
    return os.path.join(WORK, *parts)
