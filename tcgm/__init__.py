# """
# Public API for TimeCost Gradient Machine (TCGM).
# Expose the main model, loss helpers and metrics.
# """

# from .models import TimeCostGradientMachine
# from .loss import financial_loss, grad_financial, grad_logit_safe
# from .metrics import evaluate_financial_performance, compute_expected_monetary_loss

# __all__ = [
#     "TimeCostGradientMachine",
#     "financial_loss",
#     "grad_financial",
#     "grad_logit_safe",
#     "evaluate_financial_performance",
#     "compute_expected_monetary_loss",
# ]

# __version__ = "0.1.3"

# tcgm/__init__.py
"""
Public API for TimeCost Gradient Machine (TCGM).
Expose classifier + regressor + forecaster, loss helpers and metrics.
"""
from .models import TCGMClassifier, TCGMRegressor
from .ts import TCGMForecaster
from .loss import (
    financial_loss,
    grad_financial,
    grad_logit_safe,
    asymmetric_mae_loss,
    grad_asymmetric_mae,
    convex_cs_logloss,
    grad_convex_cs_logloss,
    asymmetric_mse_loss,
    grad_asymmetric_mse
)
from .metrics import (
    evaluate_financial_performance,
    compute_expected_monetary_loss,
    evaluate_regression_cost
)

__all__ = [
    "TCGMClassifier",
    "TCGMRegressor",
    "TCGMForecaster",
    "financial_loss",
    "grad_financial",
    "grad_logit_safe",
    "asymmetric_mae_loss",
    "grad_asymmetric_mae",
    "convex_cs_logloss",
    "grad_convex_cs_logloss",
    "asymmetric_mse_loss",
    "grad_asymmetric_mse",
    "evaluate_financial_performance",
    "compute_expected_monetary_loss",
    "evaluate_regression_cost",
]

__version__ = "0.1.5"