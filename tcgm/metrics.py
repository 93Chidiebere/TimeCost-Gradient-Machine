"""
Evaluation metrics and helpers for financial models.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from .loss import financial_loss

def evaluate_financial_performance(y_true, y_pred, cost_fp=1.0, cost_fn=5.0):
    """
    Compute AUC, Brier, and Expected Loss.
    y_pred should be probabilities in [0,1].
    Returns dict with metrics (numbers).
    """
    y_true = np.asarray(y_true)
    y_pred = np.clip(np.asarray(y_pred), 0.0, 1.0)
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
    brier = brier_score_loss(y_true, y_pred)
    exp_loss = float(np.mean(financial_loss(y_true, y_pred, cost_fp, cost_fn)))
    return {"AUC": auc, "Brier": brier, "Expected_Loss": exp_loss}

def compute_expected_monetary_loss(y_true, y_prob, exposure, lgd=0.6, cost_fp=50.0, thresholds=None):
    """
    Compute expected monetary loss across thresholds and return best threshold & loss.
    This function treats decisions as thresholding probabilities.
    y_true: binary labels
    y_prob: predicted probabilities
    exposure: per-sample exposure (monetary)
    lgd: loss given default fraction (applies when y=1 and not predicted)
    cost_fp: monetary cost per false positive (e.g., operational cost)
    thresholds: iterable of thresholds to evaluate; if None defaults to np.linspace(0,1,101)
    """
    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob), 0.0, 1.0)
    exposure = np.asarray(exposure)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    losses = []
    for t in thresholds:
        decisions = (y_prob >= t).astype(int)
        # false positives: decision=1 but y=0 -> cost_fp each
        fp_costs = np.where((decisions == 1) & (y_true == 0), cost_fp, 0.0)
        # missed positives: y=1 and decision=0 -> exposure * lgd
        miss_costs = np.where((decisions == 0) & (y_true == 1), exposure * lgd, 0.0)
        total_loss = float(np.sum(fp_costs) + np.sum(miss_costs))
        losses.append(total_loss)
    losses = np.asarray(losses)
    best_idx = int(np.argmin(losses))
    return {
        "thresholds": thresholds,
        "losses": losses,
        "best_threshold": float(thresholds[best_idx]),
        "best_loss": float(losses[best_idx])
    }