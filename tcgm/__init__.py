# from .models import TimeCostGradientMachine

# # Loss functions
# from .loss import (
#     financial_loss,
#     grad_financial,
#     grad_logit_safe
# )

# # Metrics
# from .metrics import (
#     evaluate_financial_performance,
#     compute_expected_monetary_loss,
# )

# __all__ = [
#     "TimeCostGradientMachine",
#     "financial_loss",
#     "grad_financial",
#     "grad_logit_safe",
#     "evaluate_financial_performance",
#     "compute_expected_monetary_loss",
# ]

# __version__ = "0.1.1"


"""
Public API for TimeCost Gradient Machine (TCGM).
Expose the main model, loss helpers and metrics.
"""

from .models import TimeCostGradientMachine
from .loss import financial_loss, grad_financial, grad_logit_safe
from .metrics import evaluate_financial_performance, compute_expected_monetary_loss

__all__ = [
    "TimeCostGradientMachine",
    "financial_loss",
    "grad_financial",
    "grad_logit_safe",
    "evaluate_financial_performance",
    "compute_expected_monetary_loss",
]

__version__ = "0.1.2"