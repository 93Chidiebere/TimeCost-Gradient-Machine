# """
# High-level API for TimeCost Gradient Machine (TCGM)
# A clean sklearn-style wrapper around a model + financial gradient updates.
# """

# import numpy as np

# from .core import (
#     sigmoid,
#     logit,
#     TrainerEngine,
#     detect_drift_series
# )

# from .loss import financial_loss, grad_financial
# from .metrics import evaluate_financial_performance
# from .encoders import LeakageSafeTargetEncoder
# from .cost import compute_expected_monetary_loss

# __all__ = [
#     "TimeCostGradientMachine",
#     "LeakageSafeTargetEncoder",
#     "compute_expected_monetary_loss",
#     "evaluate_financial_performance",
#     "TrainerEngine",
#     "detect_drift_series",
#     "grad_financial",
#     "financial_loss",
# ]


# class TimeCostGradientMachine:
#     """
#     A simple gradient-based learner that adjusts model parameters using
#     financial gradients and time-aware learning rate from TrainerEngine.
#     """

#     def __init__(
#         self,
#         base_model,
#         learning_rate=0.05,
#         n_epochs=50,
#         cost_fp=1.0,
#         cost_fn=5.0
#     ):
#         self.base_model = base_model
#         self.learning_rate = float(learning_rate)
#         self.n_epochs = int(n_epochs)
#         self.cost_fp = float(cost_fp)
#         self.cost_fn = float(cost_fn)

#         # Correct TrainerEngine call
#         self.engine = TrainerEngine(
#             base_lr=self.learning_rate,
#             sample_rate=0.7,
#             lr_window=3,
#             random_state=42
#         )

#     # -------------------------------------------------------
#     # Fit: repeatedly update model using custom financial gradient
#     # -------------------------------------------------------
#     def fit(self, X, y, cost=None, timestamps=None):
#         X = np.asarray(X)
#         y = np.asarray(y).reshape(-1)

#         for epoch in range(self.n_epochs):

#             # 1. predict current probabilities
#             preds = self.base_model.predict_proba(X)[:, 1]

#             # 2. compute loss & record LR update
#             loss_vals = financial_loss(y, preds, self.cost_fp, self.cost_fn)
#             mean_loss = float(np.mean(loss_vals))
#             lr = self.engine.record_loss(mean_loss)

#             # 3. compute gradient of loss wrt prediction
#             grad = grad_financial(y, preds, self.cost_fp, self.cost_fn)

#             # 4. update model using pseudo-gradient step (works for linear models or trees supporting partial_fit)
#             if hasattr(self.base_model, "partial_fit"):
#                 self.base_model.partial_fit(X, y)
#             else:
#                 # fallback: adjust predicted logits manually
#                 logits = logit(preds)
#                 updated_logits = logits - lr * grad
#                 new_probs = sigmoid(updated_logits)

#                 # require base model to support fit()
#                 self.base_model.fit(X, new_probs)

#         return self

#     # -------------------------------------------------------
#     def predict_proba(self, X):
#         return self.base_model.predict_proba(X)

#     def predict(self, X, threshold=0.5):
#         return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

#     # -------------------------------------------------------
#     def financial_score(self, X, y):
#         preds = self.predict_proba(X)[:, 1]
#         return evaluate_financial_performance(
#             y_true=y,
#             y_pred=preds,
#             cost_fp=self.cost_fp,
#             cost_fn=self.cost_fn
#         )

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit

from .loss import grad_financial, financial_loss
from .metrics import evaluate_financial_performance


class TimeCostGradientMachine:
    """
    Boosted financial gradient machine with custom asymmetric loss
    for fraud detection / cost-sensitive banking problems.
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
        n_splits=5,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.time_col = time_col
        self.n_splits = n_splits
        self.random_state = random_state

        self.base_models = []
        self.init_pred = None

    # -------------------------------------------------------
    # INTERNAL PREDICTION LAYER
    # -------------------------------------------------------
    def _predict_boosted(self, X):
        pred = np.full(X.shape[0], self.init_pred)
        for model in self.base_models:
            pred += self.learning_rate * model.predict(X)
        return pred

    # -------------------------------------------------------
    # FITTING THE BOOSTING MODEL
    # -------------------------------------------------------
    def fit(self, X, y):
        """
        Gradient boosting with asymmetric fraud cost loss.
        """

        # Initial prediction = log-odds of fraud rate
        p = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        self.init_pred = np.log(p / (1 - p))

        pred = np.full(len(y), self.init_pred)

        for t in range(self.n_estimators):

            # Compute negative gradient using COST-SENSITIVE loss
            grad = grad_financial(
                y_true=y,
                y_pred=pred,
                cost_fp=self.cost_fp,
                cost_fn=self.cost_fn,
            )

            # Fit regression tree to gradient
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + t,
            )

            tree.fit(X, grad)
            self.base_models.append(tree)

            # Update boosted prediction
            pred = pred + self.learning_rate * tree.predict(X)

        return self

    # -------------------------------------------------------
    # PREDICT PROBABILITY
    # -------------------------------------------------------
    def predict_proba(self, X):
        logit = self._predict_boosted(X)
        return 1 / (1 + np.exp(-logit))

    # -------------------------------------------------------
    # CLASS PRED
    # -------------------------------------------------------
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    # -------------------------------------------------------
    # EVALUATE (WITH TIME SERIES CV)
    # -------------------------------------------------------
    def evaluate(self, df, feature_cols, target_col):
        """
        Time-aware evaluation using TimeSeriesSplit.
        """

        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        all_metrics = []

        for train_idx, test_idx in tscv.split(df):

            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            X_train = train_df[feature_cols].values
            y_train = train_df[target_col].values

            X_test = test_df[feature_cols].values
            y_test = test_df[target_col].values

            model = TimeCostGradientMachine(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                cost_fp=self.cost_fp,
                cost_fn=self.cost_fn,
                time_col=self.time_col,
                n_splits=self.n_splits,
                random_state=self.random_state,
            )

            model.fit(X_train, y_train)

            proba = model.predict_proba(X_test)
            pred = (proba >= 0.5).astype(int)

            metrics = evaluate_financial_performance(
                y_test, pred, proba, self.cost_fp, self.cost_fn
            )

            all_metrics.append(metrics)

        return all_metrics