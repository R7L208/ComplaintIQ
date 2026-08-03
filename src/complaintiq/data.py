"""Data loading, path resolution, and schema validation for ComplaintIQ."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F


def resolve_data_dir() -> Path:
    """
    Resolve the data directory for both local and Databricks execution.

    When running on Databricks, returns /Volumes/workspace/complaintiq/data if it exists.
    Otherwise, falls back to ../data (local development setup).

    Returns:
        Path: The resolved data directory.
    """
    volume_path = Path("/Volumes/workspace/complaintiq/data")
    if volume_path.exists():
        return volume_path
    return Path("..") / "data"


def load_parquet_with_schema_check(
    spark, narrative_only: bool = False
) -> tuple[SparkDataFrame, int]:
    """
    Load the complaints Parquet with schema validation.

    Args:
        spark: SparkSession
        narrative_only: If True, load complaints_narrative_only.parquet.
                       If False, load complaints.parquet.

    Returns:
        tuple: (Spark DataFrame, number of rows).

    Raises:
        FileNotFoundError: If neither file exists.
        AssertionError: If columns don't match expected schema.
    """
    data_dir = resolve_data_dir()

    expected_columns = [
        "complaint_id",
        "date_received",
        "date_sent_to_company",
        "product",
        "sub_product",
        "issue",
        "sub_issue",
        "complaint_text",
        "company",
        "state",
        "zip_code",
        "tags",
        "submitted_via",
        "timely_response",
        "company_response_to_consumer",
        "complaint_text_length",
        "has_narrative",
        "monetary_relief",
    ]

    # Prefer narrative_only if requested; fall back to full.
    candidates = []
    if narrative_only:
        candidates.append(data_dir / "complaints_narrative_only.parquet")
    candidates.append(data_dir / "complaints.parquet")

    parquet_path = next((p for p in candidates if p.exists()), None)
    if parquet_path is None:
        raise FileNotFoundError(
            f"No Parquet found in {data_dir}. Run `make parquet` first."
        )

    print(f"Loading: {parquet_path.name}")

    # Schema check: must match include/complaintiq/schema.h
    file_cols = spark.read.parquet(str(parquet_path)).columns
    missing_cols = set(expected_columns) - set(file_cols)
    unexpected_cols = set(file_cols) - set(expected_columns)
    assert not missing_cols and not unexpected_cols, (
        f"Parquet columns don't match schema.h. Missing: {missing_cols}; "
        f"unexpected: {unexpected_cols}. Regenerate with `make parquet`."
    )
    print(f"Schema OK: {len(file_cols)} columns match schema.h")

    df = spark.read.parquet(str(parquet_path))
    n_rows = df.count()
    print(f"Loaded {n_rows:,} rows x {len(df.columns)} columns")

    return df, n_rows


def chronological_split(
    sdf: SparkDataFrame, train_frac: float = 0.80, random_state: int = 42
) -> tuple[SparkDataFrame, SparkDataFrame]:
    """
    Split Spark DataFrame chronologically at a percentile of date_received.

    This split is leakage-safe: train on older complaints, evaluate on recent ones.
    Within each split, the positive rate reflects the data's natural distribution.

    Args:
        sdf: Spark DataFrame with a 'date_received' column (must be Date type or string ISO).
        train_frac: Fractional cutoff for train window (default 0.80).
        random_state: Seed for approxQuantile (unused here; kept for API compatibility).

    Returns:
        tuple: (train_sdf, test_sdf) both without the temporary 'epoch' column.
    """
    # Convert to date and drop nulls.
    sdf = sdf.withColumn("date_received", F.to_date("date_received"))
    sdf = sdf.dropna(subset=["date_received"])

    # Use days since epoch to compute quantile chronologically (avoids time zone issues).
    sdf = sdf.withColumn("epoch", F.datediff("date_received", F.lit("1970-01-01")))
    cutoff_epoch = sdf.approxQuantile("epoch", [train_frac], 0.001)[0]
    train_sdf = sdf.filter(F.col("epoch") <= cutoff_epoch).drop("epoch")
    test_sdf = sdf.filter(F.col("epoch") > cutoff_epoch).drop("epoch")

    print(f"Chronological split at {train_frac:.0%}:")
    print(f"  train rows: {train_sdf.count():,}")
    print(f"  test rows:  {test_sdf.count():,}")

    return train_sdf, test_sdf
