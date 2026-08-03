"""ComplaintIQ: shared utilities for EDA and modeling notebooks."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# The one true seed, so any sample is reproducible run-to-run.
RANDOM_STATE: int = 42

# House plotting theme: clean grid + colorblind-safe palette, used everywhere.
sns.set_theme(
    style="whitegrid", palette="colorblind", context="notebook", font_scale=1.1
)
plt.rcParams["figure.figsize"] = (10, 6)
np.random.seed(RANDOM_STATE)

# Convenience: import these so notebooks can do `from complaintiq import sns, plt, RANDOM_STATE`
__all__ = ["RANDOM_STATE", "np", "pd", "plt", "sns"]


def print_versions() -> None:
    """Print Python and library versions (for notebook setup cells)."""
    print("python :", sys.version.split()[0])
    print("numpy  :", np.__version__)
    print("pandas :", pd.__version__)
    print("seaborn:", sns.__version__)
