"""
Financial loss functions and gradients used by the TimeCostGradientMachine (TCGM).
The design enforces asymmetric penalties for FP/FN — critical for fraud, credit
risk, AML, chargebacks, and operational risk modeling.
"""

import numpy as np


# -------------------------------------------------------------------------
# 1) FINANCIAL LOSS FUNCTION
# -------------------------------------------------------------------------
def financial_loss(y_true, y_prob, cost_fp=1.0, cost_fn=5.0):
    """
    Per-sample asymmetric financial loss.

    y_true : {0,1}
    y_prob : predicted probability of y=1

    cost_fp: cost when model predicts fraud but it's legitimate
    cost_fn: cost when model misses actual fraud

    Loss is continuous in probability space:
      L = cost_fp * (1-y) * p + cost_fn * y * (1-p)
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.clip(np.asarray(y_prob).astype(float), 1e-6, 1 - 1e-6)

    fp_component = cost_fp * (1 - y_true) * y_prob
    fn_component = cost_fn * y_true * (1 - y_prob)

    return fp_component + fn_component


# -------------------------------------------------------------------------
# 2) GRADIENT OF FINANCIAL LOSS (w.r.t probability)
# -------------------------------------------------------------------------
def grad_financial(y_true, y_prob, cost_fp=1.0, cost_fn=5.0):
    """
    Gradient dL/dp for the asymmetric financial loss.
    TCGM uses this inside the boosting loop.

        dL/dp = cost_fp * (1 - y) - cost_fn * y

    For fraud (y=1), gradient pushes probability upward.
    For good behavior (y=0), gradient pushes downward.
    """
    y_true = np.asarray(y_true).astype(float)
    return cost_fp * (1 - y_true) - cost_fn * y_true


# -------------------------------------------------------------------------
# 3) LOGIT-SAFE GRADIENT (optional)
# -------------------------------------------------------------------------
def grad_logit_safe(y_true, y_prob, cost_fp=1.0, cost_fn=5.0):
    """
    Converts dL/dp → dL/d(logit).
    Safe for extreme probabilities (0 or 1).

    Used only if the model needs defensive stability.
    """
    grad_p = grad_financial(y_true, y_prob, cost_fp, cost_fn)
    p = np.clip(y_prob, 1e-6, 1 - 1e-6)
    return grad_p * (p * (1 - p))