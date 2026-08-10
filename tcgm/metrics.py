# # """
# # Financial evaluation metrics for TCGM.
# # Provides:
# #     - AUC
# #     - Brier Score
# #     - Expected Financial Loss
# #     - Expected Monetary Loss (threshold optimizer)
# # """

# # import numpy as np
# # from sklearn.metrics import roc_auc_score, brier_score_loss
# # from .loss import financial_loss


# # # -------------------------------------------------------------------------
# # # AUC calculation (safe when only one class exists)
# # # -------------------------------------------------------------------------
# # def _safe_auc(y_true, y_prob):
# #     y_true = np.asarray(y_true)
# #     y_prob = np.asarray(y_prob)

# #     # AUC is undefined if only one class exists
# #     if len(np.unique(y_true)) < 2:
# #         return float("nan")

# #     return float(roc_auc_score(y_true, y_prob))


# # # -------------------------------------------------------------------------
# # # Brier Score
# # # -------------------------------------------------------------------------
# # def _brier(y_true, y_prob):
# #     return float(brier_score_loss(y_true, y_prob))


# # # -------------------------------------------------------------------------
# # # Financial Expected Loss (model-implied asymmetric loss)
# # # -------------------------------------------------------------------------
# # def evaluate_financial_performance(y_true, y_prob, cost_fp=1.0, cost_fn=5.0):
# #     """
# #     Computes:
# #         - AUC
# #         - Brier Score
# #         - Expected Loss using the financial asymmetric loss function
    
# #     y_true: array-like of 0/1 labels
# #     y_prob: predicted probabilities
# #     """
# #     y_true = np.asarray(y_true).astype(int)
# #     y_prob = np.clip(np.asarray(y_prob), 0.0, 1.0)

# #     auc = _safe_auc(y_true, y_prob)
# #     brier = _brier(y_true, y_prob)

# #     # Mean asymmetric loss across all samples
# #     exp_loss = float(np.mean(financial_loss(y_true, y_prob, cost_fp, cost_fn)))

# #     return {
# #         "AUC": auc,
# #         "Brier": brier,
# #         "Expected_Loss": exp_loss,
# #     }


# # # -------------------------------------------------------------------------
# # # Expected Monetary Loss (threshold sweep for decision optimization)
# # # -------------------------------------------------------------------------
# # def compute_expected_monetary_loss(
# #     y_true,
# #     y_prob,
# #     exposure,
# #     lgd=0.6,
# #     cost_fp=50.0,
# #     thresholds=None
# # ):
# #     """
# #     Computes threshold-dependent monetary loss:
# #         FP → operational cost
# #         FN → exposure × LGD

# #     Returns:
# #         {
# #             "best_threshold": float,
# #             "best_loss": float,
# #             "thresholds": array,
# #             "loss_curve": array
# #         }

# #     y_true: binary ground truth (0/1)
# #     y_prob: model predicted probabilities
# #     exposure: monetary exposure per transaction
# #     lgd: loss given default (fraction of exposure lost)
# #     cost_fp: cost of wrongly blocking a legitimate transaction
# #     thresholds: optional iterable of thresholds
# #     """

# #     y_true = np.asarray(y_true).astype(int)
# #     y_prob = np.clip(np.asarray(y_prob), 0.0, 1.0)
# #     exposure = np.asarray(exposure).astype(float)

# #     if thresholds is None:
# #         thresholds = np.linspace(0.01, 0.99, 99)

# #     loss_values = []

# #     best_loss = np.inf
# #     best_threshold = None

# #     for t in thresholds:
# #         decisions = (y_prob >= t).astype(int)

# #         # False Positives → blocking legitimate customers
# #         fp_mask = (decisions == 1) & (y_true == 0)
# #         fp_loss = fp_mask.sum() * cost_fp

# #         # False Negatives → missed fraud → lose exposure × LGD
# #         fn_mask = (decisions == 0) & (y_true == 1)
# #         fn_loss = np.sum(fn_mask * exposure * lgd)

# #         total_loss = fp_loss + fn_loss
# #         loss_values.append(total_loss)

# #         if total_loss < best_loss:
# #             best_loss = total_loss
# #             best_threshold = t

# #     return {
# #         "best_threshold": float(best_threshold),
# #         "best_loss": float(best_loss),
# #         "thresholds": np.array(thresholds),
# #         "loss_curve": np.array(loss_values)
# #     }


# """
# Evaluation metrics for TCGM:
#  - AUC (safe when one class present)
#  - Brier score
#  - evaluate_financial_performance (AUC, Brier, Expected_Loss)
#  - compute_expected_monetary_loss (threshold sweep using exposure & LGD)
# """

# from typing import Optional, Dict, Any, Sequence
# import numpy as np
# from sklearn.metrics import roc_auc_score, brier_score_loss
# from .loss import financial_loss

# ArrayLike = Sequence


# def _safe_auc(y_true: ArrayLike, y_prob: ArrayLike) -> float:
#     y_true = np.asarray(y_true)
#     y_prob = np.asarray(y_prob)
#     if len(np.unique(y_true)) < 2:
#         return float("nan")
#     return float(roc_auc_score(y_true, y_prob))


# def _brier(y_true: ArrayLike, y_prob: ArrayLike) -> float:
#     return float(brier_score_loss(y_true, y_prob))


# def evaluate_financial_performance(y_true: ArrayLike, y_prob: ArrayLike, cost_fp: float = 1.0, cost_fn: float = 5.0) -> Dict[str, float]:
#     """
#     Returns a dict with:
#       - AUC
#       - Brier
#       - Expected_Loss (mean of financial_loss)
#     """
#     y_true = np.asarray(y_true).astype(int)
#     y_prob = np.clip(np.asarray(y_prob).astype(float), 0.0, 1.0)
#     auc = _safe_auc(y_true, y_prob)
#     brier = _brier(y_true, y_prob)
#     exp_loss = float(np.mean(financial_loss(y_true, y_prob, cost_fp=cost_fp, cost_fn=cost_fn)))
#     return {"AUC": auc, "Brier": brier, "Expected_Loss": exp_loss}


# def compute_expected_monetary_loss(
#     y_true: ArrayLike,
#     y_prob: ArrayLike,
#     exposure: ArrayLike,
#     lgd: float = 0.6,
#     cost_fp: float = 50.0,
#     thresholds: Optional[ArrayLike] = None,
# ) -> Dict[str, Any]:
#     """
#     Sweep thresholds and compute monetary loss:
#       - FP cost = cost_fp per false positive
#       - FN cost = exposure * lgd for each missed positive

#     Returns dictionary:
#       { "best_threshold", "best_loss", "thresholds", "loss_curve" }
#     """
#     y_true = np.asarray(y_true).astype(int)
#     y_prob = np.clip(np.asarray(y_prob).astype(float), 0.0, 1.0)
#     exposure = np.asarray(exposure).astype(float)

#     if thresholds is None:
#         thresholds = np.linspace(0.01, 0.99, 99)

#     loss_values = []
#     best_loss = float("inf")
#     best_t = None

#     for t in thresholds:
#         preds = (y_prob >= t).astype(int)
#         # FP: predicted 1 while true 0
#         fp_count = int(((preds == 1) & (y_true == 0)).sum())
#         fp_cost = fp_count * cost_fp
#         # FN: predicted 0 while true 1 -> exposure * lgd for those samples
#         fn_mask = (preds == 0) & (y_true == 1)
#         fn_cost = float(np.sum(exposure[fn_mask] * lgd)) if np.any(fn_mask) else 0.0
#         total = fp_cost + fn_cost
#         loss_values.append(total)
#         if total < best_loss:
#             best_loss = total
#             best_t = float(t)

#     return {
#         "best_threshold": best_t,
#         "best_loss": float(best_loss),
#         "thresholds": np.asarray(thresholds, dtype=float),
#         "loss_curve": np.asarray(loss_values, dtype=float),
#     }

# tcgm/metrics.py
"""
Evaluation metrics for TCGM.

Provides:
 - evaluate_financial_performance (AUC, Brier, Expected Loss)
 - compute_expected_monetary_loss (threshold sweep using exposure & LGD)
 - evaluate_regression_cost (asymmetric regression metrics)
"""

from typing import Sequence, Optional, Dict, Any
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from .loss import financial_loss, asymmetric_mae_loss, asymmetric_mse_loss

ArrayLike = Sequence


def _safe_auc(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    y = np.asarray(y_true)
    p = np.asarray(y_prob)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _brier(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    return float(brier_score_loss(y_true, y_prob))


def evaluate_financial_performance(y_true: ArrayLike, y_prob: ArrayLike, cost_fp: float = 1.0, cost_fn: float = 5.0) -> Dict[str, float]:
    """
    Return AUC, Brier, and Expected_Loss (mean of financial_loss).
    """
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(y_prob).astype(float), 0.0, 1.0)
    return {
        "AUC": _safe_auc(y, p),
        "Brier": _brier(y, p),
        "Expected_Loss": float(np.mean(financial_loss(y, p, cost_fp=cost_fp, cost_fn=cost_fn)))
    }


def compute_expected_monetary_loss(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    exposure: ArrayLike,
    lgd: float = 0.6,
    cost_fp: float = 50.0,
    thresholds: Optional[ArrayLike] = None,
) -> Dict[str, Any]:
    """
    Sweep thresholds and compute monetary loss:
      FP: cost_fp per false positive
      FN: exposure * lgd per missed positive
    Returns dict with best threshold and loss curve.
    """
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(y_prob).astype(float), 0.0, 1.0)
    exposure = np.asarray(exposure).astype(float)

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    loss_curve = []
    best_loss = float("inf")
    best_t = None

    for t in thresholds:
        preds = (p >= t).astype(int)
        fp_count = int(((preds == 1) & (y == 0)).sum())
        fp_cost = fp_count * float(cost_fp)
        fn_mask = (preds == 0) & (y == 1)
        fn_cost = float(np.sum(exposure[fn_mask] * lgd)) if np.any(fn_mask) else 0.0
        total = fp_cost + fn_cost
        loss_curve.append(total)
        if total < best_loss:
            best_loss = total
            best_t = float(t)

    return {
        "best_threshold": best_t,
        "best_loss": float(best_loss),
        "thresholds": np.asarray(thresholds, dtype=float),
        "loss_curve": np.asarray(loss_curve, dtype=float),
    }


# -------------------------
# Regression evaluation (asymmetric cost-weighted metrics)
# -------------------------
def evaluate_regression_cost(y_true: ArrayLike, y_pred: ArrayLike, c_over: float = 1.0, c_under: float = 1.0) -> Dict[str, float]:
    """
    Compute asymmetric MAE, asymmetric MSE, and standard MAE/RMSE.
    Returns:
      {"Asymmetric_MAE": ..., "Asymmetric_MSE": ..., "MAE": ..., "RMSE": ...}
    """
    y = np.asarray(y_true).astype(float)
    yhat = np.asarray(y_pred).astype(float)
    asym_mae = float(np.mean(asymmetric_mae_loss(y, yhat, c_over=c_over, c_under=c_under)))
    asym_mse = float(np.mean(asymmetric_mse_loss(y, yhat, c_over=c_over, c_under=c_under)))
    mae = float(np.mean(np.abs(y - yhat)))
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return {"Asymmetric_MAE": asym_mae, "Asymmetric_MSE": asym_mse, "MAE": mae, "RMSE": rmse}