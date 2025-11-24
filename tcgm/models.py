import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from .loss import grad_financial, financial_loss
from .metrics import evaluate_financial_performance


class TimeCostGradientMachine:
    """
    Time-Cost Gradient Machine (TCGM)
    --------------------------------
    A custom asymmetric-loss gradient boosting model designed for:
        • financial fraud detection
        • exposure-aware risk scoring
        • cost-sensitive classification
        • time-ordered patterns (drift, seasonality)
    
    Features:
        ✔ Numeric-only auto-filtering
        ✔ Drops time column automatically
        ✔ Boosting using financial asymmetric gradient
        ✔ SKLearn-like API (fit / predict / predict_proba / evaluate)
        ✔ Recency-weight ready (hooks inside fit)
    """

    def __init__(
        self,
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=30,
        cost_fp=10.0,
        cost_fn=100.0,
        time_col=None,
        recency_weighting=False,
        recency_alpha=0.90,
        random_state=42
    ):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.cost_fp = float(cost_fp)
        self.cost_fn = float(cost_fn)
        self.time_col = time_col
        self.recency_weighting = recency_weighting
        self.recency_alpha = float(recency_alpha)
        self.random_state = int(random_state)

        self.base_models = []
        self.init_pred = 0.0
        self.feature_columns_ = None

    # -------------------------------------------------------------------------
    # INTERNAL: clean & convert input matrix
    # -------------------------------------------------------------------------
    def _prepare_X(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()

            # drop time column if available
            if self.time_col and self.time_col in df.columns:
                df = df.drop(columns=[self.time_col])

            # keep numeric columns only
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] == 0:
                # attempt conversion
                converted = df.apply(pd.to_numeric, errors='coerce')
                numeric_df = converted.select_dtypes(include=[np.number])

                if numeric_df.shape[1] == 0:
                    raise ValueError(
                        "No numeric features available. "
                        "Please convert categorical/time features before training."
                    )

            self.feature_columns_ = list(numeric_df.columns)
            return numeric_df.astype(float).to_numpy(copy=False)

        if isinstance(X, np.ndarray):
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return X.astype(float)

        try:
            df = pd.DataFrame(X)
            return self._prepare_X(df)
        except:
            raise ValueError("Unable to convert X into numeric matrix.")

    # -------------------------------------------------------------------------
    # INTERNAL: compute boosted logit
    # -------------------------------------------------------------------------
    def _predict_logit(self, X):
        X_arr = self._prepare_X(X)
        logit = np.full(X_arr.shape[0], self.init_pred, dtype=float)

        for tree in self.base_models:
            logit += self.learning_rate * tree.predict(X_arr)

        return logit

    # -------------------------------------------------------------------------
    # FIT
    # -------------------------------------------------------------------------
    def fit(self, X, y):
        X_arr = self._prepare_X(X)
        y_arr = np.asarray(y).astype(int)

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        # initial prediction = log odds of fraud frequency
        p0 = np.clip(np.mean(y_arr), 1e-6, 1 - 1e-6)
        self.init_pred = float(np.log(p0 / (1 - p0)))

        pred = np.full_like(y_arr, fill_value=self.init_pred, dtype=float)
        self.base_models = []

        # optional recency weights
        if self.recency_weighting:
            w = np.power(self.recency_alpha, np.arange(len(y_arr))[::-1])
            w = w / w.mean()
        else:
            w = np.ones_like(y_arr)

        # boosting loop
        for t in range(self.n_estimators):

            prob = 1.0 / (1.0 + np.exp(-pred))  # convert logit → probability

            # gradient wrt probability
            grad_p = grad_financial(
                y_arr, prob,
                cost_fp=self.cost_fp,
                cost_fn=self.cost_fn
            )

            # convert gradient: dL/dlogit = dL/dp * p(1-p)
            pseudo_residual = -grad_p * (prob * (1 - prob))

            # fit weak learner
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + t,
            )

            tree.fit(X_arr, pseudo_residual, sample_weight=w)
            self.base_models.append(tree)

            # update logits
            pred += self.learning_rate * tree.predict(X_arr)

        return self

    # -------------------------------------------------------------------------
    # PREDICT PROBA
    # -------------------------------------------------------------------------
    def predict_proba(self, X):
        logit = self._predict_logit(X)
        prob = 1.0 / (1.0 + np.exp(-logit))
        return np.vstack([1 - prob, prob]).T

    # -------------------------------------------------------------------------
    # PREDICT LABELS
    # -------------------------------------------------------------------------
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    # -------------------------------------------------------------------------
    # EVALUATION WRAPPER
    # -------------------------------------------------------------------------
    def evaluate(self, X, y):
        """
        Returns a dictionary:
            {
                "AUC": float,
                "Brier": float,
                "Expected_Loss": float
            }
        """
        probs = self.predict_proba(X)[:, 1]
        return evaluate_financial_performance(
            y,
            probs,
            cost_fp=self.cost_fp,
            cost_fn=self.cost_fn
        )