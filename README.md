# TimeCost Gradient Machine (TCGM)

**TCGM** is a domain-specific gradient-learning framework built for financial prediction and decision optimization, where time, money, asymmetry, and tail-risk are more important than abstract statistical accuracy.

Developed by **Chidiebere V. Christopher**, TCGM is designed specifically for:
* **Banking & Payments**: Fraud detection, chargeback prevention, and network analysis.
* **Lending & Credit Risk**: Cost-sensitive credit scoring, probability of default (PD) estimation, and Loss Given Default (LGD) optimization.
* **Operations & Supply Chain**: Cash flow forecasting, cash point replenishment (ATMs), inventory demand optimization, and SLA risk forecasting.

Unlike general-purpose ML libraries, TCGM embeds financial logic directly into the model’s optimization loop.

---

## Core Components

### 1. `TCGMClassifier` (Cost-Sensitive Classification)
Traditional classifiers optimize for log-loss or entropy, treating False Positives (FP) and False Negatives (FN) symmetrically. `TCGMClassifier` optimizes the Expected Monetary Loss (EML) directly during boosting.
* **Linear Objective (`objective="linear"`)**: Optimizes the exact financial loss:
  $$L(y, p) = c_{fp}(1 - y)p + c_{fn}y(1 - p)$$
* **Convex Objective (`objective="convex"`)**: Avoids the gradient vanishing problem of linear logit models for extreme misclassifications by optimizing a convex cost-sensitive log-loss with the logit gradient:
  $$\frac{\partial L}{\partial F} = c_{fp}(1 - y)p - c_{fn}y(1 - p)$$

### 2. `TCGMRegressor` (Asymmetric Regression)
For forecasting magnitudes (e.g. fraud amount, credit losses, cash reserves) where underestimating costs vastly different than overestimating.
* **MAE Objective (`objective="mae"`)**: Optimizes asymmetric absolute error:
  $$L_{\text{MAE}}(y, \hat{y}) = c_{\text{over}} \max(\hat{y} - y, 0) + c_{\text{under}} \max(y - \hat{y}, 0)$$
* **MSE Objective (`objective="mse"`)**: Continuous, error-scaled asymmetric squared loss that avoids step-gradient convergence oscillations:
  $$L_{\text{MSE}}(y, \hat{y}) = c_{\text{over}} \max(\hat{y} - y, 0)^2 + c_{\text{under}} \max(y - \hat{y}, 0)^2$$

### 3. `TCGMForecaster` (Asymmetric Time-Series Forecasting)
A recursive multi-step forecaster wrapper module (`tcgm.ts`) that:
* Automates time-series feature engineering by creating lag variables and rolling statistics without data leakage.
* Recursively forecasts multiple steps into the future, using asymmetric regression to skew predictions safely away from high-cost errors (e.g. avoiding stockouts or cashout events).

### 4. `Time-Aware Gradient Flow` (Temporal Decay)
Financial patterns drift. TCGM applies a true time-distance weighting to discount stale historical records during boosting:
  $$w_i = \exp(-\lambda \cdot (t_{\text{max}} - t_i))$$
where $t_{\text{max}}$ is the latest timestamp in the dataset, and $\lambda$ is the recency decay rate.

---

## Installation

Install the stable release via PyPI:
```bash
pip install tcgm==0.1.5
```

Or install in development mode from source:
```bash
git clone https://github.com/93Chidiebere/TimeCost-Gradient-Machine.git
cd timecost-gradient-machine
pip install -e .
```

---

## Quick Start Guide

### 1. Cost-Sensitive Classification (`TCGMClassifier`)
Optimizing classification thresholds and boosting steps for investigation cost vs. fraud loss.

```python
import pandas as pd
from tcgm import TCGMClassifier
from tcgm.metrics import evaluate_financial_performance

# 1. Instantiate classifier
model = TCGMClassifier(
    n_estimators=60,
    learning_rate=0.1,
    max_depth=4,
    cost_fp=50.0,          # Cost of a False Positive ($50 audit cost)
    cost_fn=200.0,         # Cost of a False Negative ($200 fraud loss)
    objective="convex",    # "convex" (recommended) or "linear"
    time_col="timestamp",  # Automatically handle time-distance decay
    recency_weighting=True,
    recency_lambda=0.01
)

# 2. Fit the model
model.fit(X_train, y_train)

# 3. Predict class probabilities
probs = model.predict_proba(X_test)[:, 1]

# 4. Evaluate financial impact
metrics = evaluate_financial_performance(y_test, probs, cost_fp=50.0, cost_fn=200.0)
print("Expected Monetary Loss per sample:", metrics["Expected_Loss"])
```

### 2. Asymmetric Regression (`TCGMRegressor`)
Optimizing regression lines for asymmetric business penalties.

```python
from tcgm import TCGMRegressor
from tcgm.metrics import evaluate_regression_cost

# 1. Instantiate regressor
model = TCGMRegressor(
    n_estimators=50,
    learning_rate=0.05,
    c_over=1.0,            # Low cost for overestimating cash reserves
    c_under=10.0,          # Very high penalty for running out of cash
    objective="mse",       # "mse" (smooth gradients) or "mae" (quantiles)
    time_col="timestamp",
    recency_weighting=True
)

# 2. Fit
model.fit(X_train, y_train)

# 3. Predict magnitudes
predictions = model.predict(X_test)

# 4. Evaluate asymmetric costs
costs = evaluate_regression_cost(y_test, predictions, c_over=1.0, c_under=10.0)
print("Asymmetric Mean Squared Error:", costs["Asymmetric_MSE"])
```

### 3. Recursive Time-Series Forecasting (`TCGMForecaster`)
Performing recursive multi-step forecasting skewed by cost profiles.

```python
from tcgm import TCGMRegressor, TCGMForecaster

# 1. Define base asymmetric regressor
base_model = TCGMRegressor(
    c_over=1.0,            # Low cost of over-forecasting inventory
    c_under=8.0,           # High cost of under-forecasting (stockout)
    n_estimators=40,
    objective="mse"
)

# 2. Setup forecaster with lag features and rolling means
forecaster = TCGMForecaster(
    estimator=base_model,
    lags=[1, 2, 3, 7],
    rolling_windows=[(7, "mean"), (14, "std")]
)

# 3. Train on raw time-series values
forecaster.fit(historical_series)

# 4. Recursively forecast 14 steps into the future
forecasts = forecaster.predict(steps=14)
print("14-Day recursive forecast:", forecasts)
```

### 4. Online Incremental Boosting (`partial_fit`)
Updating models incrementally as new streaming batches of transactions arrive.

```python
# Train initially on batch 1
model = TCGMClassifier(n_estimators=10)
model.fit(X_batch1, y_batch1)

# Append new trees to the existing ensemble using batch 2
model.partial_fit(X_batch2, y_batch2)
print("Total trees in ensemble:", len(model.base_models_)) # Output: 20
```

---

## Comparison with Standard GBDTs (XGBoost, LightGBM)
Standard GBDTs target symmetric statistical metrics (e.g. log-loss or mean squared error). While sample weighting can help, it is decoupled from decision thresholding and temporal distances. TCGM bridges this gap by combining:
1. **Explicit Business Alignment**: Loss gradients and tree splits are driven directly by operational costs ($c_{fp}$, $c_{fn}$, $c_{over}$, $c_{under}$).
2. **Built-in Auto-thresholding**: Sweeps Expected Monetary Loss curves to automatically select the optimal classification threshold.
3. **Chronological Importance**: Automatically decays historical data weights based on actual days/hours passed.

---

## 👤 Author

**Chidiebere V. Christopher**
* **LinkedIn**: [Chidiebere Christopher](https://www.linkedin.com/in/chidiebere-christopher/)
* **GitHub**: [93Chidiebere](https://github.com/93Chidiebere)
* **Email**: vchidiebere.vc@gmail.com
