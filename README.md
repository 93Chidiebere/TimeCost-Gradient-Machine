# 🕒💰 TimeCost Gradient Machine (TCGM)

**TCGM** is a domain-specific gradient learning algorithm developed by Chidiebere V. Christopher for **financial prediction tasks** where  
time, cost, risk asymmetry, and regulatory constraints all matter.  

Unlike general-purpose ML algorithms that require bagging, boosting, and stacking to improve performance,  
**TCGM builds financial intelligence into its core optimization loop.**

---

## 🚀 Key Features

### **1. Time-Aware Gradient Flow**
Handles chronological dependencies, financial drift, and delayed outcomes.

### **2. Cost-Sensitive Optimization**
Directly minimizes expected monetary loss, not raw classification error.

### **3. Asymmetric Risk Handling**
Treats false positives and false negatives according to business impact.

### **4. Regulator-Penalty Integration**
Allows monetary penalties or risk-weight adjustments tied to policy rules.

### **5. Leakage-Safe Target Encoding**
Prevents future data from leaking into the model — essential for banking timelines.

### **6. Boosting-Like Dynamics (Built In)**
TCGM integrates:
- cost-weighted residual learning  
- time-step penalties  
- financial correction cycles  

…so **it behaves like a boosting algorithm natively** without needing XGBoost/LightGBM wrappers.

---

## 🔧 Installation

Install in editable mode during development:

```bash
pip install tcgm==0.1.3
```

Or install in your virtual environment

```bash
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows

pip install tcgm==0.1.3
```
## 📦 Project Structure
```ardiuno
timecost-gradient-machine/
│
├── tcgm/
│   ├── __init__.py
│   ├── core.py
│   ├── tcgm_model.py
│   ├── models.py
│   ├── loss.py
│   ├── metrics.py
│   ├── encoders.py
│   └── utils.py
│
├── examples/
│   └── churn_prediction.ipynb
│
├── tests/
│   └── test_tcgm.py
│
├── setup.py
├── pyproject.toml
├──setup.cfg
├── README.md
└── requirements.txt
```
## 🧠 What Can You Use TCGM For?
TCGM works exceptionally well for financial prediction tasks that involve uneven cost of mistakes:

✔ Banking Models

Credit Scoring | Loan Default / Repayment | Fraud Detection

Transaction Risk | BNPL Approval Models | ATM Cash Forecasting

Income Stability Risk | Customer Churn | Staff Exit / HR Risk Scoring

Operational SLA Risk Models | Liquidity Exposure Forecasting

✔ Insurance

Claims probability | Cost per claim forecasting

✔ FinTech / Payment Systems

Transaction fraud | Customer lifetime value | Chargeback prediction

## Anywhere time + money + risk form the core of decision-making, TCGM excels.

## 🧪 Quick Example

```python
from tcgm import TimeCostGradientMachine
from tcgn.metrics import evaluate_financial_performance, compute_expected_monetary_loss

model = TimeCostGradientMachine(
    n_estimators=60,
    learning_rate=0.1,
    max_depth=4,
    min_samples_leaf=20,
    cost_fp=50.0,
    cost_fn=200.0
)

start = time.time()

model.fit(X_train, y_train)
print(f"Training time: {time.time() - start:.2f} seconds")

# Predict probabilities
probs = model.predict_proba(X_test)[:, 1]
probs[:10]

# Evaluate Model
report = evaluate_financial_performance(
    y_test,
    probs,
    cost_fp=50,
    cost_fn=200
)
report

# Compute Expected Monetary Loss
exposure = X_test["amount"].values

eml = compute_expected_monetary_loss(
    y_true = y_test,
    y_prob = probs,
    exposure = exposure,
    lgd = 0.6,
    cost_fp= 50
)

eml

# Plot full loss curve
plt.plot(eml["thresholds"], eml["loss_curve"], label="Loss Curve")

# Mark best threshold
plt.axvline(eml["best_threshold"], linestyle="--", label="Best Threshold")

plt.title("TCGM Expected Monetary Loss Curve")
plt.xlabel("Threshold")
plt.ylabel("Loss (#)")
plt.grid(True)
plt.legend()
plt.show()

# Save the model
joblib.dump(model, "model.pkl")

```

## 🧩 Why TCGM Instead of GPML Algorithms?

General-purpose ML models require:

bagging → to stabilize

boosting → to correct errors

stacking → to compensate weaknesses

## TCGM eliminates those needs because it directly optimizes financial loss, not generic error.

## ⭐ Acknowledgment

TCGM is designed for real-world Nigerian and African banking environments, including:

NIP
agency banking
P2P payment rails
transaction monitoring
credit/loan lifecycle modeling

## Developed to solve challenges faced by analysts, data scientists, and ML engineers in financial institutions.

## Author
Chidiebere V. Christopher |
| LinkedIn: https://www.linkedin.com/in/chidiebere-christopher/
