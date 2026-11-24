# tcgm/__init__.py
"""
Public API for TimeCost Gradient Machine (TCGM).
Exposes classifier, regressor, forecaster, loss/Hessian helpers, and financial metrics.
"""

__version__ = "0.2.0"

from .models import TCGMClassifier, TCGMRegressor, TimeCostGradientMachine
from .ts import TCGMForecaster
from .loss import (
    financial_loss,
    grad_financial,
    grad_logit_safe,
    asymmetric_mae_loss,
    grad_asymmetric_mae,
    convex_cs_logloss,
    grad_convex_cs_logloss,
    hess_convex_cs_logloss,
    asymmetric_mse_loss,
    grad_asymmetric_mse,
    hess_asymmetric_mse,
    compute_expectile,
    compute_weighted_quantile,
)
from .metrics import (
    evaluate_financial_performance,
    compute_expected_monetary_loss,
    evaluate_regression_cost,
)

__all__ = [
    "__version__",
    "TCGMClassifier",
    "TCGMRegressor",
    "TCGMForecaster",
    "TimeCostGradientMachine",
    "financial_loss",
    "grad_financial",
    "grad_logit_safe",
    "asymmetric_mae_loss",
    "grad_asymmetric_mae",
    "convex_cs_logloss",
    "grad_convex_cs_logloss",
    "hess_convex_cs_logloss",
    "asymmetric_mse_loss",
    "grad_asymmetric_mse",
    "hess_asymmetric_mse",
    "compute_expectile",
    "compute_weighted_quantile",
    "evaluate_financial_performance",
    "compute_expected_monetary_loss",
    "evaluate_regression_cost",
]
