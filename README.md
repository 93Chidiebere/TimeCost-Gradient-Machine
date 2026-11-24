# TimeCost Gradient Machine (`tcgm`)

**A Financial Cost-Sensitive and Time-Aware Gradient Boosting Framework for Credit Risk, Fraud Detection, Loss Severity (LGD), and Liquidity Forecasting.**

[![PyPI version](https://img.shields.io/pypi/v/tcgm.svg)](https://pypi.org/project/tcgm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

Standard gradient boosted decision trees optimize symmetric loss functions (Binary Cross-Entropy or Mean Squared Error) that treat False Positives and False Negatives—or under-predictions and over-predictions—as equally costly. In financial risk management, error consequences are strongly asymmetric and data streams drift over time.

**`tcgm` (v0.2.0)** embeds asymmetric financial cost matrices (scalar or instance-dependent) and exponential in-loop temporal recency weighting directly into **second-order gradient boosting backends** (`xgboost` and `lightgbm` custom objectives), while exposing a clean scikit-learn compatible API:

- **`TCGMClassifier`**: Asymmetric cost-sensitive binary classification with exact first-order gradients and second-order Hessians, cost-weighted prior initialization, validation early stopping on cost-sensitive log-loss, temporal recency decay, and built-in **inverse-Elkan probability recalibration** (`predict_proba_calibrated`) for IFRS 9 / Basel III PD pricing.
- **`TCGMRegressor`**: Asymmetric Mean Squared Error (expectile regression) and Asymmetric MAE regression with exact piecewise-constant Hessians, analytical $\tau$-expectile initialization ($\tau = \frac{c_{\text{under}}}{c_{\text{under}} + c_{\text{over}}}$), and early stopping on validation asymmetric loss.
- **`TCGMForecaster`**: Recursive multi-step time-series forecaster wrapping `TCGMRegressor` with leak-free lag features, shifted rolling window statistics, and calendar day-of-week (`dow`) encoding.

---

## Installation

```bash
pip install --upgrade tcgm
```

Or install from source:

```bash
git clone https://github.com/93Chidiebere/TimeCost-Gradient-Machine.git
cd TimeCost-Gradient-Machine
pip install -e .
```

---

## Quick Start Examples

### 1. Credit Default Classification (`TCGMClassifier` with Scalar Costs & Temporal Decay)

```python
import pandas as pd
from tcgm import TCGMClassifier

# Assume X_train, X_val, X_test are DataFrames containing a timestamp column 'issue_dt'
model = TCGMClassifier(
    n_estimators=200,
    max_depth=4,
    num_leaves=15,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    cost_fp=50.0,               # Cost of rejecting a good borrower ($)
    cost_fn=500.0,              # Cost of approving a default ($) -> 10:1 ratio
    time_col="issue_dt",        # Automatically used for recency decay and dropped from features
    recency_weighting=True,     # Exponentially down-weight older records
    recency_lambda=0.001,       # Daily decay rate
    backend="lightgbm",         # "xgboost", "lightgbm", or "sklearn"
    early_stopping_rounds=25,   # Early-stops on validation cost-sensitive log-loss
    random_state=42,
)

model.fit(X_train, y_train, eval_set=(X_val, y_val))

# 1. Cost-warped probabilities (optimal decision boundary sits naturally near tau = 0.50)
p_warped = model.predict_proba(X_test)[:, 1]
decisions = model.predict(X_test, threshold=0.50)

# 2. Inverse-Elkan recalibrated probabilities (for IFRS 9 / Basel III PD pricing & Brier score)
p_calibrated = model.predict_proba_calibrated(X_test)[:, 1]
```

---

### 2. Instance-Dependent Payment Fraud Detection (`cost_fn_vec` & `cost_fp_vec`)

When every transaction has a different monetary exposure (e.g., missed fraud costs `TransactionAmt + $15`, while a false alert costs `$10` to investigate), pass per-instance cost arrays directly to `fit()`:

```python
import numpy as np
from tcgm import TCGMClassifier

cfn_train = train_df["TransactionAmt"].to_numpy() + 15.0
cfp_train = np.full(len(train_df), 10.0)
cfn_val = val_df["TransactionAmt"].to_numpy() + 15.0
cfp_val = np.full(len(val_df), 10.0)

fraud_model = TCGMClassifier(
    n_estimators=200,
    max_depth=4,
    num_leaves=15,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.80,
    time_col="event_time",
    recency_weighting=True,
    recency_lambda=0.003,
    backend="lightgbm",
    early_stopping_rounds=25,
    random_state=42,
)

fraud_model.fit(
    X_train, y_train,
    cost_fp_vec=cfp_train,
    cost_fn_vec=cfn_train,
    eval_set=(X_val, y_val),
    eval_cost_fp_vec=cfp_val,
    eval_cost_fn_vec=cfn_val,
)

fraud_scores = fraud_model.predict_proba(X_test)[:, 1]
```

---

### 3. Credit Loss Severity / LGD Regression (`TCGMRegressor`)

For continuous targets where under-provisioning is penalized more heavily than over-provisioning (e.g., 8:1 penalty corresponding to the $\tau = 8/9$ expectile):

```python
from tcgm import TCGMRegressor

lgd_model = TCGMRegressor(
    n_estimators=200,
    max_depth=4,
    num_leaves=15,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    c_over=1.0,                 # Penalty weight for over-predicting loss
    c_under=8.0,                # Penalty weight for under-predicting loss (capital shortfall)
    objective="mse",            # "mse" (Asymmetric MSE / expectile) or "mae" (Asymmetric MAE)
    backend="lightgbm",         # "xgboost", "lightgbm", or "sklearn"
    time_col="issue_dt",
    early_stopping_rounds=25,   # Early-stops on validation Asymmetric MSE
    random_state=42,
)

lgd_model.fit(X_train, y_lgd_train, eval_set=(X_val, y_lgd_val))
predicted_loss = lgd_model.predict(X_test)
```

---

### 4. Multi-Step Asymmetric Liquidity Forecasting (`TCGMForecaster`)

```python
from tcgm import TCGMForecaster, TCGMRegressor

forecaster = TCGMForecaster(
    estimator=TCGMRegressor(
        n_estimators=90,
        max_depth=3,
        num_leaves=10,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        c_over=1.0,             # Idle cash holding penalty
        c_under=8.0,            # Cash stockout penalty
        objective="mse",
        backend="lightgbm",     # or "xgboost"
        random_state=42,
    ),
    lags=[1, 2, 7, 14, 28],
    rolling_windows=[(7, "mean"), (28, "mean")],
    include_dayofweek=True,     # Adds calendar day-of-week feature automatically
)

forecaster.fit(daily_cash_series)
next_56_days = forecaster.predict(steps=56)
```

---

## Parameter Reference

### `TCGMClassifier` & `TCGMRegressor` Parameters

| Parameter | Default | Description & Tuning Guidance |
|---|---|---|
| `backend` | `"xgboost"` | Boosting engine: `"xgboost"` (second-order C++ hist trees), `"lightgbm"` (leaf-wise second-order C++ trees), or `"sklearn"` (legacy first-order Python loop). |
| `n_estimators` | `160` | Maximum number of boosting rounds. Set higher (e.g., `200`–`500`) when using `early_stopping_rounds`. |
| `learning_rate` | `0.05` | Shrinkage step size per tree. Lower values (`0.03`–`0.05`) improve generalization under heavy cost asymmetry. |
| `max_depth` | `4` | Maximum tree depth. Controls interaction order and overfitting. |
| `num_leaves` | `15` | Maximum leaves per tree (used by `backend="lightgbm"`). Keep $\le 2^{\text{max\_depth}} - 1$. |
| `min_samples_leaf` | `30` / `20` | Minimum observations per leaf (`min_child_samples` in LightGBM, `min_samples_leaf` in sklearn). |
| `subsample` | `0.85` | Row subsampling fraction per boosting round for variance reduction. |
| `colsample_bytree` | `0.85` | Feature subsampling fraction per tree. |
| `reg_lambda` | `1.0` | L2 regularization penalty on leaf weights ($w_j^* = -\frac{\sum g_i}{\sum h_i + \lambda}$). |
| `min_child_weight` | `1.0` | Minimum sum of instance Hessians needed in a child leaf (XGBoost). Stabilizes extreme probability leaves. |
| `cost_fp`, `cost_fn` | `10.0`, `100.0` | *(Classifier)* Scalar False Positive and False Negative monetary costs. Can be overridden per row via `cost_fp_vec` / `cost_fn_vec` in `fit()`. |
| `c_over`, `c_under` | `1.0`, `8.0` | *(Regressor)* Over-prediction and under-prediction penalty weights. Defines target expectile $\tau = \frac{c_{\text{under}}}{c_{\text{under}} + c_{\text{over}}}$. |
| `objective` | `"convex"` / `"mse"` | Classifier: `"convex"` (weighted log-loss). Regressor: `"mse"` (Asymmetric MSE / expectile) or `"mae"` (Asymmetric MAE). |
| `time_col` | `None` | Name of datetime column in `X`. Used to compute elapsed days $\Delta t_i$ for recency weighting and automatically excluded from model features. |
| `recency_weighting` | `False` | If `True`, scales sample gradients and Hessians by mean-normalized exponential decay $w_i = \exp(-\lambda \Delta t_i)$. |
| `recency_lambda` | `0.001` | Daily exponential decay rate $\lambda$. Tune on rolling validation splits (e.g., `[0.0, 0.0005, 0.001, 0.003]`). |
| `early_stopping_rounds` | `None` | Stops boosting if validation asymmetric loss does not improve for this many rounds (requires `eval_set=(X_val, y_val)` in `fit()`). |
| `n_jobs` | `4` | Number of CPU threads for XGBoost/LightGBM backends. |
| `random_state` | `42` | Random seed for full reproducibility. |

### `TCGMForecaster` Parameters

| Parameter | Default | Description |
|---|---|---|
| `estimator` | `TCGMRegressor(...)` | Underlying `TCGMRegressor` instance used for recursive forecasting. |
| `lags` | `[1, 2, 3]` | List of autoregressive lag offsets (e.g., `[1, 2, 7, 14, 28]` for daily series). |
| `rolling_windows` | `[(7, "mean")]` | List of `(window_size, stat)` tuples (`"mean"`, `"std"`, `"max"`, `"min"`), always shifted by 1 step to prevent leakage. |
| `include_dayofweek` | `True` | Automatically extracts calendar day-of-week (`0`–`6`) from `DatetimeIndex` or integer step index and propagates it across the forecast horizon. |

---

## Understanding the Evaluation Metrics

| Metric | Formula / Definition | Operational Meaning |
|---|---|---|
| **Expected Monetary Loss (EML)** | $\frac{1}{N}\sum \left[c_{fp,i}\mathbb{I}(\hat{y}_i=1, y_i=0) + c_{fn,i}\mathbb{I}(\hat{y}_i=0, y_i=1)\right]$ | Average financial loss per loan or transaction. Lower is directly better for the institution's bottom line. |
| **Test EML @ 0.50** | EML evaluated at the default threshold $\tau = 0.50$ | Measures whether the model internalizes cost asymmetry directly without brittle post-hoc threshold tuning. |
| **ROC-AUC & PR-AUC** | Area under ROC and Precision-Recall curves | Threshold-invariant ranking quality. PR-AUC is especially informative under severe fraud/default imbalance. |
| **Raw vs. Recalibrated Brier Score** | $\frac{1}{N}\sum (p_i - y_i)^2$ | Raw Brier score is higher for cost-warped models by construction (Elkan shift). Use `predict_proba_calibrated()` to verify unweighted probability calibration for IFRS 9 / Basel III PD pricing. |
| **Asymmetric MSE (Expectile Loss)** | $\frac{1}{N}\sum \left[c_{\text{under}}(y_i-\hat{y}_i)_+^2 + c_{\text{over}}(\hat{y}_i-y_i)_+^2\right]$ | Penalizes severe under-provisioning quadratically relative to over-provisioning. |
| **Asymmetric MAE (Pinball Cost)** | $\frac{1}{N}\sum \left[c_{\text{under}}(y_i-\hat{y}_i)_+ + c_{\text{over}}(\hat{y}_i-y_i)_+\right]$ | Linear operational penalty per day/loan (e.g., 8:1 ATM stockout vs. idle cash holding cost). |
| **Severe Shortfall Rate (> $5k)** | $\frac{1}{N}\sum \mathbb{I}(y_i - \hat{y}_i > 5000)$ | Share of defaulted loans where realized loss exceeded provisioned reserves by more than $5,000. |
| **Stockout Day Rate** | $\frac{1}{N}\sum \mathbb{I}(\hat{y}_t < y_t)$ | Share of forecast days where predicted cash fell below actual ATM demand. |

---

## Citation

If you use `tcgm` in academic or industrial research, please cite:

```bibtex
@software{christopher2026tcgm,
  author  = {Christopher, Chidiebere V.},
  title   = {TimeCost Gradient Machine (TCGM): A Financial Cost-Sensitive and Time-Aware Gradient Boosting Framework},
  year    = {2026},
  version = {0.2.0},
  url     = {https://pypi.org/project/tcgm/}
}
```

## License

MIT License. Copyright (c) 2025–2026 Chidiebere V. Christopher.
