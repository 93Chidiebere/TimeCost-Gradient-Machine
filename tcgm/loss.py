# # tcgm/loss.py
# """
# Financial loss functions and gradients used by the TimeCost Gradient Machine (TCGM).

# Contains:
#  - financial_loss: per-sample expected monetary loss (probabilistic)
#  - grad_financial: gradient dL/dp (w.r.t. predicted probability)
#  - grad_logit_safe: maps dL/dp -> dL/d(logit) (numerically stable)
# """

# from typing import Union, Sequence
# import numpy as np

# ArrayLike = Union[np.ndarray, Sequence]


# def financial_loss(y_true: ArrayLike, y_prob: ArrayLike, cost_fp: float = 1.0, cost_fn: float = 5.0) -> np.ndarray:
#     """
#     Probabilistic expected loss per sample:
#         L = cost_fp * (1 - y) * p + cost_fn * y * (1 - p)

#     Parameters
#     ----------
#     y_true : array-like (0/1)
#     y_prob : array-like floats [0,1]
#     cost_fp : float
#     cost_fn : float

#     Returns
#     -------
#     numpy.ndarray : per-sample loss (float)
#     """
#     y_true = np.asarray(y_true).astype(float)
#     y_prob = np.clip(np.asarray(y_prob).astype(float), 0.0, 1.0)
#     fp_component = cost_fp * (1.0 - y_true) * y_prob
#     fn_component = cost_fn * y_true * (1.0 - y_prob)
#     return fp_component + fn_component


# def grad_financial(y_true: ArrayLike, y_prob: ArrayLike, cost_fp: float = 1.0, cost_fn: float = 5.0) -> np.ndarray:
#     """
#     Gradient of the probabilistic loss with respect to probability p:
#         dL/dp = cost_fp * (1 - y) - cost_fn * y

#     This is constant per class (w.r.t. p) because the chosen surrogate is linear in p.
#     """
#     y_true = np.asarray(y_true).astype(float)
#     return cost_fp * (1.0 - y_true) - cost_fn * y_true


# def grad_logit_safe(y_true: ArrayLike, y_prob: ArrayLike, cost_fp: float = 1.0, cost_fn: float = 5.0) -> np.ndarray:
#     """
#     Convert gradient dL/dp into dL/d(logit) using dp/dlogit = p*(1-p).
#     Safer when operating in logit space (our boosting loop).
#     """
#     grad_p = grad_financial(y_true, y_prob, cost_fp=cost_fp, cost_fn=cost_fn)
#     p = np.clip(np.asarray(y_prob).astype(float), 1e-12, 1.0 - 1e-12)
#     return grad_p * (p * (1.0 - p))

# tcgm/loss.py
"""
Loss functions used by TCGM.

Contains:
 - financial_loss: probabilistic asymmetric loss for classification (monetary)
 - grad_financial: gradient dL/dp (w.r.t predicted probability)
 - grad_logit_safe: converts dL/dp -> dL/d(logit) via p*(1-p)
 - asymmetric_mae_loss: asymmetric absolute error (regression)
 - grad_asymmetric_mae: piecewise gradient of asymmetric MAE (w.r.t prediction)
"""

from typing import Sequence, Union
import numpy as np

ArrayLike = Union[np.ndarray, Sequence]


def financial_loss(y_true: ArrayLike, y_prob: ArrayLike, cost_fp: float = 1.0, cost_fn: float = 5.0) -> np.ndarray:
    """
    Probabilistic expected loss per sample for classification:
      L = cost_fp * (1-y) * p + cost_fn * y * (1-p)
    y_true: 0/1
    y_prob: predicted P(y=1)
    """
    y = np.asarray(y_true).astype(float)
    p = np.clip(np.asarray(y_prob).astype(float), 0.0, 1.0)
    fp = cost_fp * (1.0 - y) * p
    fn = cost_fn * y * (1.0 - p)
    return fp + fn


def grad_financial(y_true: ArrayLike, y_prob: ArrayLike, cost_fp: float = 1.0, cost_fn: float = 5.0) -> np.ndarray:
    """
    dL/dp for financial_loss:
      dL/dp = cost_fp * (1 - y) - cost_fn * y
    """
    y = np.asarray(y_true).astype(float)
    return cost_fp * (1.0 - y) - cost_fn * y


def grad_logit_safe(y_true: ArrayLike, y_prob: ArrayLike, cost_fp: float = 1.0, cost_fn: float = 5.0) -> np.ndarray:
    """
    Convert dL/dp -> dL/d(logit) using dp/dlogit = p*(1-p).
    """
    grad_p = grad_financial(y_true, y_prob, cost_fp=cost_fp, cost_fn=cost_fn)
    p = np.clip(np.asarray(y_prob).astype(float), 1e-12, 1.0 - 1e-12)
    return grad_p * (p * (1.0 - p))


# -------------------------
# Regression asymmetric MAE (piecewise)
# -------------------------
def asymmetric_mae_loss(y_true: ArrayLike, y_pred: ArrayLike, c_over: float = 1.0, c_under: float = 1.0) -> np.ndarray:
    """
    L = c_over * [ŷ - y]_+ + c_under * [y - ŷ]_+
    This is asymmetric absolute error: overestimation and underestimation costs differ.
    """
    y = np.asarray(y_true).astype(float)
    yhat = np.asarray(y_pred).astype(float)
    diff = yhat - y
    loss = c_over * np.maximum(diff, 0.0) + c_under * np.maximum(-diff, 0.0)
    return loss


def grad_asymmetric_mae(y_true: ArrayLike, y_pred: ArrayLike, c_over: float = 1.0, c_under: float = 1.0) -> np.ndarray:
    """
    Piecewise gradient dL/dŷ:
      if ŷ > y: dL/dŷ = c_over
      if ŷ < y: dL/dŷ = -c_under
      if ŷ == y: gradient 0 (subgradient could be any between [-c_under, c_over])
    For boosting we return a deterministic choice: 0 when equal.
    """
    y = np.asarray(y_true).astype(float)
    yhat = np.asarray(y_pred).astype(float)
    grad = np.zeros_like(yhat, dtype=float)
    grad[yhat > y] = float(c_over)
    grad[yhat < y] = -float(c_under)
    # equal stays 0
    return grad


def convex_cs_logloss(y_true: ArrayLike, y_prob: ArrayLike, cost_fp: float = 1.0, cost_fn: float = 5.0) -> np.ndarray:
    """
    Convex cost-sensitive log-loss (cross-entropy formulation):
      L = - [cost_fn * y * log(p) + cost_fp * (1 - y) * log(1 - p)]
    """
    y = np.asarray(y_true).astype(float)
    p = np.clip(np.asarray(y_prob).astype(float), 1e-12, 1.0 - 1e-12)
    return - (cost_fn * y * np.log(p) + cost_fp * (1.0 - y) * np.log(1.0 - p))


def grad_convex_cs_logloss(y_true: ArrayLike, y_prob: ArrayLike, cost_fp: float = 1.0, cost_fn: float = 5.0) -> np.ndarray:
    """
    Gradient dL/dlogit for the convex cost-sensitive log-loss:
      dL/dlogit = cost_fp * (1 - y) * p - cost_fn * y * (1 - p)
    This gradient scales with prediction error and avoids vanishing gradient.
    """
    y = np.asarray(y_true).astype(float)
    p = np.clip(np.asarray(y_prob).astype(float), 1e-12, 1.0 - 1e-12)
    return cost_fp * (1.0 - y) * p - cost_fn * y * (1.0 - p)


# -------------------------
# Regression asymmetric MSE
# -------------------------
def asymmetric_mse_loss(y_true: ArrayLike, y_pred: ArrayLike, c_over: float = 1.0, c_under: float = 1.0) -> np.ndarray:
    """
    Asymmetric Mean Squared Error loss:
      L = c_over * (y_pred - y)^2 if y_pred > y else c_under * (y_pred - y)^2
    """
    y = np.asarray(y_true).astype(float)
    yhat = np.asarray(y_pred).astype(float)
    diff = yhat - y
    return np.where(diff > 0, c_over * (diff ** 2), c_under * (diff ** 2))


def grad_asymmetric_mse(y_true: ArrayLike, y_pred: ArrayLike, c_over: float = 1.0, c_under: float = 1.0) -> np.ndarray:
    """
    Gradient dL/dyhat of Asymmetric MSE:
      dL/dyhat = 2 * c_over * (y_pred - y) if y_pred > y else 2 * c_under * (y_pred - y)
    """
    y = np.asarray(y_true).astype(float)
    yhat = np.asarray(y_pred).astype(float)
    diff = yhat - y
    return np.where(diff > 0, 2.0 * c_over * diff, 2.0 * c_under * diff)