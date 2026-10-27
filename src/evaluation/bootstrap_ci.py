"""
Non-parametric bootstrap confidence interval estimation.
Computes 95% bootstrap confidence intervals (B=1000 resamples) with fixed random seed
for empirical rigor across evaluation metrics.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


def compute_bootstrap_ci(
    values: List[float],
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Computes empirical mean and bootstrap confidence interval for a list of metric observations.
    """
    if not values:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}

    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    if n <= 1:
        return {
            "mean": round(mean_val, 4),
            "ci_lower": round(mean_val, 4),
            "ci_upper": round(mean_val, 4),
            "std": round(std_val, 4),
        }

    rng = np.random.default_rng(seed)
    # Resample with replacement B times
    resample_indices = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = np.mean(arr[resample_indices], axis=1)

    alpha = (1.0 - ci_level) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    ci_lower = float(np.percentile(resampled_means, lower_pct))
    ci_upper = float(np.percentile(resampled_means, upper_pct))

    return {
        "mean": round(mean_val, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "std": round(std_val, 4),
    }

