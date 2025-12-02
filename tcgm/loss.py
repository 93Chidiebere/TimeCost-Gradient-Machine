# tcgm/loss.py
"""
Financial loss functions and gradients used by the TimeCost Gradient Machine (TCGM).

Contains:
 - financial_loss: per-sample expected monetary loss (probabilistic)
 - grad_financial: gradient dL/dp (w.r.t. predicted probability)
 - grad_logit_safe: maps dL/dp -> dL/d(logit) (numerically stable)
"""

from typing import Union, Sequence
import numpy as np

ArrayLike = Union[np.ndarray, Sequence]


def financial_loss(y_true: ArrayLike, y_prob: ArrayLike, cost_fp: float = 1.0, cost_fn: float = 5.0) -> np.ndarray:
    """
    Probabilistic expected loss per sample:
        L = cost_fp * (1 - y) * p + cost_fn * y * (1 - p)

    Parameters
    ----------
    y_true : array-like (0/1)
    y_prob : array-like floats [0,1]
    cost_fp : float
    cost_fn : float

    Returns
    -------
    numpy.ndarray : per-sample loss (float)
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.clip(np.asarray(y_prob).astype(float), 0.0, 1.0)
    fp_component = cost_fp * (1.0 - y_true) * y_prob
    fn_component = cost_fn * y_true * (1.0 - y_prob)
    return fp_component + fn_component


def grad_financial(y_true: ArrayLike, y_prob: ArrayLike, cost_fp: float = 1.0, cost_fn: float = 5.0) -> np.ndarray:
    """
    Gradient of the probabilistic loss with respect to probability p:
        dL/dp = cost_fp * (1 - y) - cost_fn * y

    This is constant per class (w.r.t. p) because the chosen surrogate is linear in p.
    """
    y_true = np.asarray(y_true).astype(float)
    return cost_fp * (1.0 - y_true) - cost_fn * y_true


def grad_logit_safe(y_true: ArrayLike, y_prob: ArrayLike, cost_fp: float = 1.0, cost_fn: float = 5.0) -> np.ndarray:
    """
    Convert gradient dL/dp into dL/d(logit) using dp/dlogit = p*(1-p).
    Safer when operating in logit space (our boosting loop).
    """
    grad_p = grad_financial(y_true, y_prob, cost_fp=cost_fp, cost_fn=cost_fn)
    p = np.clip(np.asarray(y_prob).astype(float), 1e-12, 1.0 - 1e-12)
    return grad_p * (p * (1.0 - p))