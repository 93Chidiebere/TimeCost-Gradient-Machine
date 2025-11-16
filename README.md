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
pip install -e .
```

Or install in your virtual environment

```bash
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows

pip install -e .
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
│   ├── cost.py
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
from sklearn.tree import DecisionTreeClassifier

model = TimeCostGradientMachine(
    base_model=DecisionTreeClassifier(max_depth=4),
    time_cost_weight=1.0,
    regulator_penalty=0.2,
    economic_sensitivity=1.3,
)

model.fit(X_train, y_train, cost=cost_vector, timestamps=ts)
preds = model.predict_proba(X_test)

score = model.financial_score(X_test, y_test, cost=test_cost_vector)
print("Expected Monetary Gain:", score)

```

## 🧩 Why TCGM Instead of GPML Algorithms?

General-purpose ML models require:

bagging → to stabilize

boosting → to correct errors

stacking → to compensate weaknesses

## TCGM eliminates those needs because it directly optimizes financial loss, not generic error.

## ⭐ Acknowledgment

TCGM is designed for real-world Nigerian banking environments, including:

NIP
agency banking
P2P payment rails
transaction monitoring
credit/loan lifecycle modeling

## Developed to solve challenges faced by analysts, data scientists, and ML engineers in financial institutions.

## Author
Chidiebere V. Christopher |
| LinkedIn: https://www.linkedin.com/in/chidiebere-christopher/
