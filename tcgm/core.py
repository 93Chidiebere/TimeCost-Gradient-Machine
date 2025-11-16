"""
Core utilities and a lightweight training engine for TCGM.

Purpose:
- Provide numerical helpers (sigmoid/logit)
- Implement adaptive learning-rate update logic
- Provide time-weighted sampling utilities
- Provide a simple drift detector (PSI-like)
- Provide a small TrainerEngine class that encapsulates the train loop pattern
  used in TimeCostGradientMachine (optional convenience wrapper).

Place this file at: tcgm/core.py
"""

from typing import List, Tuple, Optional, Callable, Dict
import numpy as np
import pandas as pd


# -----------------------
# numerical helpers
# -----------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid."""
    x = np.asarray(x, dtype=float)
    # clip to avoid overflow
    x = np.clip(x, -100.0, 100.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    """Stable logit (inverse sigmoid)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


# -----------------------
# adaptive learning rate
# -----------------------
def compute_volatility(loss_history: List[float], window: int = 3) -> float:
    """
    Compute volatility as std(mean-window) / mean(mean-window).
    Returns 0.0 if insufficient data or mean is near zero.
    """
    if len(loss_history) < window:
        return 0.0
    recent = np.array(loss_history[-window:], dtype=float)
    mean = recent.mean()
    if np.isclose(mean, 0.0):
        return 0.0
    vol = float(np.std(recent) / (mean + 1e-12))
    return vol


def adaptive_lr(base_lr: float, loss_history: List[float], window: int = 3, min_lr: float = 1e-6) -> float:
    """
    Compute an adjusted learning rate that decreases when volatility increases.
    lr = base_lr / (1 + volatility)
    Ensures lr >= min_lr.
    """
    vol = compute_volatility(loss_history, window=window)
    lr = float(base_lr * (1.0 / (1.0 + vol)))
    return max(lr, float(min_lr))


# -----------------------
# time-weighted sampling
# -----------------------
def time_weighted_sample_indices(df: pd.DataFrame,
                                 time_col: str,
                                 sample_rate: float = 0.7,
                                 by_group: Optional[str] = None,
                                 random_state: Optional[int] = None) -> np.ndarray:
    """
    Return an index array sampled by time bias (more recent observations relatively more likely).
    If by_group is provided, sampling is done by unique groups (group-level bootstrap).
    sample_rate is fraction of unique units (observations or groups) to sample (with replacement).
    """
    rng = np.random.RandomState(random_state)
    if by_group is not None and by_group in df.columns:
        groups = df[by_group].values
        unique_groups = np.unique(groups)
        n_sample = max(1, int(len(unique_groups) * float(sample_rate)))
        sampled_groups = rng.choice(unique_groups, size=n_sample, replace=True)
        idx = np.where(np.isin(groups, sampled_groups))[0]
        return idx
    else:
        n = len(df)
        n_sample = max(1, int(n * float(sample_rate)))
        # time bias: weight by recency (linear)
        times = pd.to_datetime(df[time_col])
        # normalize to [0,1]
        rel = (times - times.min()).dt.total_seconds()
        rel = np.asarray(rel, dtype=float)
        if rel.max() > 0:
            weights = 0.5 + 0.5 * (rel / rel.max())  # older 0.5, newest 1.0
        else:
            weights = np.ones_like(rel, dtype=float)
        probs = weights / weights.sum()
        idx = rng.choice(np.arange(n), size=n_sample, replace=True, p=probs)
        return idx


# -----------------------
# simple drift detection
# -----------------------
def population_stability_index(expected: np.ndarray,
                               actual: np.ndarray,
                               buckets: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) between expected and actual arrays.
    Higher PSI indicates more change.
    """
    expected = np.asarray(expected).ravel()
    actual = np.asarray(actual).ravel()
    # create bins from expected
    try:
        quantiles = np.nanpercentile(expected, np.linspace(0, 100, buckets + 1))
    except Exception:
        # fallback to min-max bins
        quantiles = np.linspace(expected.min(), expected.max(), buckets + 1)
    # ensure unique bin edges
    quantiles = np.unique(quantiles)
    if len(quantiles) < 2:
        return 0.0
    e_counts, _ = np.histogram(expected, bins=quantiles)
    a_counts, _ = np.histogram(actual, bins=quantiles)
    e_perc = np.clip(e_counts.astype(float) / (e_counts.sum() + 1e-12), 1e-6, 1.0)
    a_perc = np.clip(a_counts.astype(float) / (a_counts.sum() + 1e-12), 1e-6, 1.0)
    psi = np.sum((e_perc - a_perc) * np.log(e_perc / a_perc))
    return float(psi)


def detect_drift_series(train_series: pd.Series,
                        new_series: pd.Series,
                        psi_threshold: float = 0.2,
                        buckets: int = 10) -> Dict[str, float]:
    """
    Return a dict with PSI value and boolean flag if drift detected.
    psi < 0.1: no significant change
    0.1 <= psi < 0.25: moderate shift
    psi >= 0.25: major shift
    """
    psi = population_stability_index(train_series.values, new_series.values, buckets=buckets) # type: ignore
    return {"psi": psi, "drift": bool(psi >= psi_threshold)}


# -----------------------
# TrainerEngine (lightweight)
# -----------------------
class TrainerEngine:
    """
    A small orchestration helper to run TCGM-style training loops.
    It does not replace your models implementation but provides a standard pattern:
      - generate time-based splits
      - compute time-weighted sampling indices
      - compute adaptive learning rate updates
    This class is intentionally minimal and import-friendly.
    """

    def __init__(self,
                 time_col: str = "timestamp",
                 sample_rate: float = 0.7,
                 base_lr: float = 0.05,
                 lr_window: int = 3,
                 min_lr: float = 1e-6,
                 random_state: Optional[int] = None):
        self.time_col = time_col
        self.sample_rate = float(sample_rate)
        self.base_lr = float(base_lr)
        self.lr_window = int(lr_window)
        self.min_lr = float(min_lr)
        self.random_state = random_state
        self.loss_history: List[float] = []
        self.current_lr = float(base_lr)

    def next_sample(self, df: pd.DataFrame, by_group: Optional[str] = None) -> np.ndarray:
        """Return sampled indices for training based on time-weighted policy."""
        return time_weighted_sample_indices(df, time_col=self.time_col,
                                            sample_rate=self.sample_rate,
                                            by_group=by_group,
                                            random_state=self.random_state)

    def update_learning_rate(self):
        """Update and return the adaptive learning rate using loss history."""
        self.current_lr = adaptive_lr(self.base_lr, self.loss_history, window=self.lr_window, min_lr=self.min_lr)
        return self.current_lr

    def record_loss(self, loss: float):
        """Append a loss to history and update LR."""
        self.loss_history.append(float(loss))
        return self.update_learning_rate()

    def reset(self):
        """Reset history and lr."""
        self.loss_history = []
        self.current_lr = float(self.base_lr)