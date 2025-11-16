"""
Financial cost utilities for the TimeCost Gradient Machine (TCGM).

This module contains the compute_expected_monetary_loss() function,
used for evaluating model predictions in monetary terms (FP cost & FN cost).

It is purposely separate from loss.py (which contains gradients for training).
"""

import numpy as np

def compute_expected_monetary_loss(y_true, y_prob, cost_fp=1.0, cost_fn=5.0):
    """
    Compute expected monetary loss from probability predictions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels.
    y_prob : array-like of shape (n_samples,)
        Predicted probability of the positive class.
    cost_fp : float
        Monetary penalty for false positives.
    cost_fn : float
        Monetary penalty for false negatives.

    Returns
    -------
    float
        Expected monetary loss.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)

    # Expected loss per sample:
    #   FN loss = P(y=1) * P(pred=0) * cost_fn
    #   FP loss = P(y=0) * P(pred=1) * cost_fp
    fn_loss = y_true * (1 - y_prob) * cost_fn
    fp_loss = (1 - y_true) * y_prob * cost_fp

    return float(np.mean(fn_loss + fp_loss))