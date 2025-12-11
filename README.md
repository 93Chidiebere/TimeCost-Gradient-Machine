# 🕒💰 TimeCost Gradient Machine (TCGM)

**TCGM** is a domain-specific gradient-learning framework built for financial prediction, where
time, money, asymmetry, and tail-risk are more important than plain accuracy.

Developed by **Chidiebere V. Christopher**, TCGM delivers native financial intelligence inside the model’s optimization loop,
making it the first ML framework designed specifically for:

- Banking

- Payments

- Lending / Credit Risk

- Fraud & AML

- Insurance

- Demand & Revenue Forecasting

Unlike general-purpose ML algorithms, TCGM does not rely on bagging, boosting, or stacking to behave intelligently,
the financial logic is built into the gradients themselves.

## 🚀 Key Features
1. **TCGMClassifier**: Cost-Sensitive Financial Classification

A gradient model that minimizes **Expected Monetary Loss** (EML) instead of raw classification error.

✔ Cost-sensitive FP/FN
✔ Time-aware gradients
✔ Tail-risk focus
✔ Regulator-penalty integration
✔ Leakage-safe temporal encoding

2. **TCGMRegressor**: Asymmetric Financial Regression

A regression engine for forecasting financial magnitudes (fraud amount, loan loss, demand, revenue, cashflow).

Uses Asymmetric MAE:
~~~ java
∂L/∂ŷ = c_over  if ŷ > y
       -c_under if ŷ < y
pseudo = -∂L/∂ŷ
~~~

Handles:

. Underestimation risk (credit loss, liquidity)

. Overestimation risk (inventory, cash forecasting)

. Heavy-tailed distributions common in finance

. Time-dependent weights & exposure scaling


3. **Time-Aware Gradient Flow**

Financial outcomes drift over time.
TCGM applies chronological penalties + temporal learning rates to fit data in motion.

4. **Asymmetric & Tail-Risk Optimization**

Most losses in finance occur in the tails, not the average.

TCGM integrates:

asymmetric costs

exposure-weighted errors

recency weighting

tail quantile penalties

```bash
pip install tcgm==0.1.4
```

Expected Monetary Loss (EML)

Tail-Weighted False Negative Cost

Time-Discounted Cost

Risk-Adjusted AUC

Precision-at-Loss-Threshold (PaLT)

Net Financial Benefit (NFB)

**Regressor Metrics**

Asymmetric MAE (primary loss)

Asymmetric RMSE

Expected Monetary Deviation (EMD)

Tail-Quantile Loss (TQL)

Mean Financial Error (MFE)

Financial Coverage Ratio (FCR)

These metrics reflect real financial risk, not abstract statistical error.

## 📦 Installation

~~~bash
pip install tcgm==0.1.4
~~~

Or development mode:
~~~
git clone https://github.com/93Chidiebere/TimeCost-Gradient-Machine.git
cd timecost-gradient-machine
pip install -e .
~~~

## 🗂 Project Structure
~~~text
tcgm/
│
├── __init__.py
├── classifier.py        # TCGMClassifier
├── regressor.py         # TCGMRegressor
├── core.py
├── loss.py              # classification + asymmetric regression loss
├── metrics.py           # expanded financial metrics suite
├── encoders.py
└── utils.py
~~~

## 🧠 What You Can Do With TCGM

**Banking**

. Fraud detection

. Transaction risk scoring

. Credit scoring & PD modeling

. Loan loss forecasting (EAD, LGD, expected loss)

. ATM cash forecasting

. Customer churn & retention modeling

. BNPL approval / repayment risk

**Insurance**

. Claim occurrence prediction

. Claim cost forecasting

**FinTech & Payments**

. Chargeback risk

. Exposure-adjusted customer lifetime value

. Merchant-level transaction anomaly detection

**Forecasting**

. Demand forecasting

. Inventory optimization

. Revenue prediction

. Operational SLA risk

**Where time + money + risk matter, TCGM outperforms standard ML**

## 🧪 Quick Start (Classification)
~~~python
from tcgm import TCGMClassifier

model = TCGMClassifier(
    n_estimators=80,
    learning_rate=0.08,
    max_depth=4,
    cost_fp=50,
    cost_fn=250,
    time_col="timestamp"
)

model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]

~~~


Metric:
~~~python
from tcgm.metrics import compute_expected_monetary_loss

eml = compute_expected_monetary_loss(
    y_true=y_test,
    y_prob=probs,
    exposure=X_test["amount"],
    lgd=0.6,
    cost_fp=50
)
~~~

## 📉 Quick Start (Regression)
~~~python
from tcgm import TCGMRegressor

model = TCGMRegressor(
    n_estimators=80,
    learning_rate=0.1,
    max_depth=3,
    c_over=2.0,
    c_under=6.0,     # penalize underestimation more
    time_col="date"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
~~~

Evaluate financial performance:
~~~python
from tcgm.metrics import asymmetric_mae, financial_coverage_ratio

print("Asymmetric MAE:", asymmetric_mae(y_test, y_pred, c_over=2, c_under=6))
print("Coverage:", financial_coverage_ratio(y_test, y_pred, X_test["exposure"]))
~~~

## 🧩 Why TCGM Over GPML (XGBoost, LightGBM, CatBoost)?

Traditional ML algorithms target accuracy, not financial cost.
They require ensemble tricks (bagging, boosting, stacking) to approximate risk behavior.

**TCGM** removes this gap by optimizing actual money loss directly:

. built-in cost asymmetry

. time-aware learning

. automatic handling of class imbalance

. exposure-weighted gradients

. financial penalty logic integrated in loss

**TCGM** = algorithm that understands money the way banks and insurers do.

## 🌍 Designed for African banks & Emerging Market Systems

**TCGM** was developed to solve real operational challenges in:

. banking

. mobile payments

. ATM networks

. switching infrastructure

. microfinance

. digital lending platforms

. agency banking

. P2P payment rails

It integrates practical constraints like:

. low signal data

. noisy high-volume transactions

. rapidly shifting fraud patterns

. limited labels

. regulatory penalties

## 👤 Author

**Chidiebere V. Christopher**
Data Scientist, ML Researcher

LinkedIn: https://www.linkedin.com/in/chidiebere-christopher/










