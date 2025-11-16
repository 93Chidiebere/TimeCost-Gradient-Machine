"""
Financial objective functions for TCGM.
"""

import numpy as np

def financial_loss(y_true, y_pred, cost_fp=1.0, cost_fn=5.0):
    """
    Expected financial loss per sample.
    y_true: {0,1}
    y_pred: probability in [0,1]
    cost_fp: cost when we predict positive but true is 0 (per FP)
    cost_fn: cost when we fail to predict positive but true is 1 (per FN)
    Returns: array of per-sample expected loss
    """
    y_true = np.asarray(y_true)
    y_pred = np.clip(np.asarray(y_pred), 0.0, 1.0)
    # If we decide by threshold, decision-related costs are non-differentiable.
    # Here we return expected loss in probabilistic form:
    # expected loss = P(pred=1) * cost_fp * I(y=0) + (1 - P(pred=1)) * cost_fn * I(y=1)
    loss = np.where(y_true == 1, cost_fn * (1.0 - y_pred), cost_fp * y_pred)
    return loss

def grad_financial(y_true, y_pred, cost_fp=1.0, cost_fn=5.0):
    """
    Gradient of financial_loss with respect to y_pred (prediction probability).
    For y=1: d/dp [cost_fn * (1-p)] = -cost_fn
    For y=0: d/dp [cost_fp * p] = cost_fp
    We return the gradient array (shape = n_samples).
    """
    y_true = np.asarray(y_true)
    y_pred = np.clip(np.asarray(y_pred), 1e-12, 1.0 - 1e-12)
    grad = np.zeros_like(y_pred, dtype=float)
    grad[y_true == 1] = -cost_fn  # pushing probability up reduces FN loss
    grad[y_true == 0] = cost_fp   # pushing probability down reduces FP loss
    # Note: this is a simple gradient for expected-loss surrogate; you may design smooth surrogates too.
    return grad