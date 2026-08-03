"""Stratified sampling utilities for ComplaintIQ."""

from __future__ import annotations

from typing import Any

import pandas as pd


def stratified_pandas(
    split_sdf: Any,
    n_target: int,
    target_col: str = "monetary_relief",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Proportional per-class stratified sample of a Spark DataFrame, returned as pandas.

    Same fraction drawn from each class preserves the base rate. Useful for drawing
    fixed-size train/test folds from class-imbalanced Spark frames.

    Args:
        split_sdf: Spark DataFrame with a target column.
        n_target: Target sample size (may not be reached if Spark frame is smaller).
        target_col: Name of the target column for stratification (default "monetary_relief").
        random_state: Seed for reproducible sampling.

    Returns:
        pd.DataFrame: Stratified sample as pandas DataFrame.
    """
    total_rows = split_sdf.count()
    sampling_fraction = min(1.0, n_target / total_rows) if total_rows else 0.0

    # Sample each class at the same fraction to preserve class proportions.
    class_fractions = {0: sampling_fraction, 1: sampling_fraction}
    return split_sdf.sampleBy(
        target_col, fractions=class_fractions, seed=random_state
    ).toPandas()
