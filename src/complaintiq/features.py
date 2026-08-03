"""Feature engineering utilities for ComplaintIQ."""

from __future__ import annotations

from typing import Any

import numpy as np


def downcast_sparse_int32(matrix: Any) -> Any:
    """
    Downcast sparse matrix indices to int32 for sklearn compatibility.

    sklearn rejects sparse matrices with 64-bit indices after hstack() or vectorize().
    Convert scipy sparse matrix to CSR format and downcast indices and indptr to int32.

    Args:
        matrix: scipy.sparse matrix (any format).

    Returns:
        scipy.sparse csr_matrix with int32 indices and indptr.
    """
    csr_matrix = matrix.tocsr()
    csr_matrix.indices = csr_matrix.indices.astype(np.int32)
    csr_matrix.indptr = csr_matrix.indptr.astype(np.int32)
    return csr_matrix


# Alias for backward compatibility with existing notebooks.
fix32 = downcast_sparse_int32


def shape_features(complaint_text_series: Any) -> np.ndarray:
    """
    Extract text shape features from complaint narratives.

    Args:
        complaint_text_series: pandas Series of text strings.

    Returns:
        np.ndarray of shape (n, 4) with columns:
            - log1p(text length)
            - proportion of uppercase letters
            - log1p(redaction count, i.e., "XX" occurrences)
            - indicator for presence of "$" (dollar sign).
    """
    text_series = complaint_text_series.fillna("")
    text_length = np.log1p(text_series.str.len().to_numpy())
    uppercase_prop = text_series.apply(
        lambda t: sum(c.isupper() for c in t) / max(len(t), 1)
    ).to_numpy()
    redaction_count = np.log1p(text_series.str.count("XX").to_numpy())
    has_dollar_sign = text_series.str.contains(r"\$", regex=True).astype(int).to_numpy()
    return np.vstack([text_length, uppercase_prop, redaction_count, has_dollar_sign]).T
