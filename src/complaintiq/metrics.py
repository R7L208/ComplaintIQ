"""Evaluation metrics for ranking and classification tasks in ComplaintIQ."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def top_k_lift(
    y_true: Any, scores: Any, k: float = 0.10, random_state: int = 42
) -> tuple[float, float]:
    """
    Precision and lift in the top-k fraction by score.

    Break score ties at random (not by row order) so a constant-score model gets
    a representative slice, yielding lift ~1x, instead of an artifact of row order.

    Args:
        y_true: Binary labels (0/1).
        scores: Predicted scores or probabilities.
        k: Fraction of the top rows to consider (e.g., 0.10 for top 10%).
        random_state: Seed for tie-breaking (ensures consistent ranking).

    Returns:
        tuple: (precision, lift) where lift = precision / base_rate.
    """
    n_top = int(len(y_true) * k)
    # Tie-break via random secondary sort to avoid row-order artifacts.
    tie_breaker = np.random.RandomState(random_state).random(len(scores))
    sorted_indices = np.lexsort((tie_breaker, -scores))
    top_indices = sorted_indices[:n_top]
    precision = y_true[top_indices].mean()
    base_rate = y_true.mean()
    lift = precision / base_rate if base_rate > 0 else 0.0
    return precision, lift


def report(
    name: str, y_true: Any, scores: Any, random_state: int = 42
) -> dict[str, float | str]:
    """
    Report ranking metrics: PR-AUC, ROC-AUC, Brier, and lift at multiple queue sizes.

    Lift is computed at the top 1%, 5%, and 10% of the scored population. Top-10%
    lift is capped at 10x by construction (lift = precision / base_rate, precision <= 1,
    so <= 1/k); smaller queues reveal ranking power the 10% slice cannot express.

    Args:
        name: Model name for display.
        y_true: Binary labels (0/1).
        scores: Predicted scores or probabilities.
        random_state: Seed for top_k_lift tie-breaking.

    Returns:
        dict: Dictionary of metrics including 'model', 'pr_auc', 'roc_auc', 'brier',
              'lift_1pct', 'lift_5pct', 'lift', 'top10_precision'.

    Prints:
        A formatted one-liner with the metrics.
    """
    pr_auc = average_precision_score(y_true, scores)

    # ROC-AUC is undefined when every score is identical (no ranking).
    # By convention, that is 0.5 - the "no discrimination" value.
    if len(np.unique(scores)) > 1:
        roc_auc = roc_auc_score(y_true, scores)
    else:
        roc_auc = 0.5
    brier = brier_score_loss(y_true, np.clip(scores, 0, 1))

    # Compute lift at 1%, 5%, 10%.
    lift_dict = {
        k: top_k_lift(y_true, scores, k=k, random_state=random_state)[1]
        for k in (0.01, 0.05, 0.10)
    }
    precision_at_10pct, _ = top_k_lift(
        y_true, scores, k=0.10, random_state=random_state
    )

    print(
        f"{name:<24} PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}  Brier={brier:.4f}  "
        f"lift@1%={lift_dict[0.01]:.1f}x  @5%={lift_dict[0.05]:.1f}x  @10%={lift_dict[0.10]:.1f}x"
    )

    return {
        "model": name,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier,
        "lift_1pct": lift_dict[0.01],
        "lift_5pct": lift_dict[0.05],
        "lift": lift_dict[0.10],
        "top10_precision": precision_at_10pct,
    }
