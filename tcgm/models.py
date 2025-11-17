# import numpy as np
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import TimeSeriesSplit

# from .loss import grad_financial, financial_loss
# from .metrics import evaluate_financial_performance


# class TimeCostGradientMachine:
#     """
#     Boosted financial gradient machine with custom asymmetric loss
#     for fraud detection / cost-sensitive banking problems.
#     """

#     def __init__(
#         self,
#         n_estimators=50,
#         learning_rate=0.1,
#         max_depth=3,
#         min_samples_leaf=30,
#         cost_fp=10.0,
#         cost_fn=100.0,
#         time_col=None,
#         n_splits=5,
#         random_state=42,
#     ):
#         self.n_estimators = n_estimators
#         self.learning_rate = learning_rate
#         self.max_depth = max_depth
#         self.min_samples_leaf = min_samples_leaf
#         self.cost_fp = cost_fp
#         self.cost_fn = cost_fn
#         self.time_col = time_col
#         self.n_splits = n_splits
#         self.random_state = random_state

#         self.base_models = []
#         self.init_pred = None

#     # -------------------------------------------------------
#     # INTERNAL PREDICTION LAYER
#     # -------------------------------------------------------
#     def _predict_boosted(self, X):
#         pred = np.full(X.shape[0], self.init_pred)
#         for model in self.base_models:
#             pred += self.learning_rate * model.predict(X)
#         return pred

#     # -------------------------------------------------------
#     # FITTING THE BOOSTING MODEL
#     # -------------------------------------------------------
#     def fit(self, X, y):
#         """
#         Gradient boosting with asymmetric fraud cost loss.
#         """

#         # Initial prediction = log-odds of fraud rate
#         p = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
#         self.init_pred = np.log(p / (1 - p))

#         pred = np.full(len(y), self.init_pred)

#         for t in range(self.n_estimators):

#             # Compute negative gradient using COST-SENSITIVE loss
#             grad = grad_financial(
#                 y_true=y,
#                 y_pred=pred,
#                 cost_fp=self.cost_fp,
#                 cost_fn=self.cost_fn,
#             )

#             # Fit regression tree to gradient
#             tree = DecisionTreeRegressor(
#                 max_depth=self.max_depth,
#                 min_samples_leaf=self.min_samples_leaf,
#                 random_state=self.random_state + t,
#             )

#             tree.fit(X, grad)
#             self.base_models.append(tree)

#             # Update boosted prediction
#             pred = pred + self.learning_rate * tree.predict(X)

#         return self

#     # -------------------------------------------------------
#     # PREDICT PROBABILITY
#     # -------------------------------------------------------
#     def predict_proba(self, X):
#         logit = self._predict_boosted(X)
#         return 1 / (1 + np.exp(-logit))

#     # -------------------------------------------------------
#     # CLASS PRED
#     # -------------------------------------------------------
#     def predict(self, X, threshold=0.5):
#         return (self.predict_proba(X) >= threshold).astype(int)

#     # -------------------------------------------------------
#     # EVALUATE (WITH TIME SERIES CV)
#     # -------------------------------------------------------
#     def evaluate(self, df, feature_cols, target_col):
#         """
#         Time-aware evaluation using TimeSeriesSplit.
#         """

#         tscv = TimeSeriesSplit(n_splits=self.n_splits)

#         all_metrics = []

#         for train_idx, test_idx in tscv.split(df):

#             train_df = df.iloc[train_idx]
#             test_df = df.iloc[test_idx]

#             X_train = train_df[feature_cols].values
#             y_train = train_df[target_col].values

#             X_test = test_df[feature_cols].values
#             y_test = test_df[target_col].values

#             model = TimeCostGradientMachine(
#                 n_estimators=self.n_estimators,
#                 learning_rate=self.learning_rate,
#                 max_depth=self.max_depth,
#                 min_samples_leaf=self.min_samples_leaf,
#                 cost_fp=self.cost_fp,
#                 cost_fn=self.cost_fn,
#                 time_col=self.time_col,
#                 n_splits=self.n_splits,
#                 random_state=self.random_state,
#             )

#             model.fit(X_train, y_train)

#             proba = model.predict_proba(X_test)
#             pred = (proba >= 0.5).astype(int)

#             metrics = evaluate_financial_performance(
#                 y_test, pred, proba, self.cost_fp, self.cost_fn
#             )

#             all_metrics.append(metrics)

#         return all_metrics

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from .loss import grad_financial, financial_loss
from .metrics import evaluate_financial_performance

class TimeCostGradientMachine:
    """
    Boosted financial gradient machine (robust feature handling).

    Prepares input automatically:
      - If X is a DataFrame and contains `time_col`, that column is removed.
      - Only numeric columns are used for training by default.
      - Raises informative error if no numeric features remain.
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
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.cost_fp = float(cost_fp)
        self.cost_fn = float(cost_fn)
        self.time_col = time_col
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)

        self.base_models = []
        self.init_pred = 0.0
        self.feature_columns_ = None  # if DataFrame provided, store numeric column names

    # ---------------------
    # Helper: prepare X -> numeric numpy array
    # ---------------------
    def _prepare_X(self, X):
        """
        Accepts pandas DataFrame or numpy array.
        If DataFrame: drop time_col (if present) and non-numeric cols,
        store numeric column names in self.feature_columns_.
        Returns a 2D numpy array of floats.
        """
        if isinstance(X, pd.DataFrame):
            df = X.copy()

            # Drop time column if present
            if self.time_col is not None and self.time_col in df.columns:
                df = df.drop(columns=[self.time_col])

            # Select numeric columns only
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] == 0:
                # No numeric columns — attempt to convert common types (bool -> int)
                converted = df.apply(pd.to_numeric, errors="coerce")
                numeric_df = converted.select_dtypes(include=[np.number])
                if numeric_df.shape[1] == 0:
                    raise ValueError(
                        "No numeric features found in X after dropping time_col. "
                        "Please provide numeric features or preprocess categorical/date features."
                    )

            # store columns used (helpful for predict)
            self.feature_columns_ = list(numeric_df.columns)
            arr = numeric_df.to_numpy(dtype=float, copy=False)
            return arr

        # If numpy array, ensure it's 2D numeric
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            # ensure float dtype
            try:
                return X.astype(float, copy=False)
            except Exception as e:
                raise ValueError("Input numpy array X cannot be converted to float: " + str(e))

        # Any other input: try to coerce with pandas
        try:
            df = pd.DataFrame(X)
            return self._prepare_X(df)
        except Exception as e:
            raise ValueError("Unable to prepare feature matrix X: " + str(e))

    # ---------------------
    # Internal prediction sum of trees (logit space)
    # ---------------------
    def _predict_logit(self, X):
        X_arr = self._prepare_X(X)
        logit = np.full(shape=(X_arr.shape[0],), fill_value=self.init_pred, dtype=float)
        for model in self.base_models:
            logit += self.learning_rate * model.predict(X_arr)
        return logit

    # ---------------------
    # Fit (boosting loop)
    # ---------------------
    def fit(self, X, y):
        """
        X: pandas DataFrame or numpy array. If DataFrame, time_col is removed automatically.
        y: array-like (0/1)
        """
        # prepare arrays
        X_arr = self._prepare_X(X)
        y_arr = np.asarray(y).reshape(-1)

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"X and y must have same number of rows: X={X_arr.shape[0]}, y={y_arr.shape[0]}")

        # initialize log-odds
        p = np.clip(np.mean(y_arr), 1e-6, 1 - 1e-6)
        self.init_pred = float(np.log(p / (1 - p)))

        pred = np.full(shape=y_arr.shape, fill_value=self.init_pred, dtype=float)

        self.base_models = []

        for t in range(self.n_estimators):
            # compute (negative) gradient; grad_financial returns gradient of loss w.r.t probability.
            # Our boosting loop expects target residuals: we convert probability-space grad to logit-space pseudo-residuals.
            prob = 1.0 / (1.0 + np.exp(-pred))  # convert logit->prob
            grad_prob = grad_financial(y_arr, prob, cost_fp=self.cost_fp, cost_fn=self.cost_fn)
            # Convert dL/dp to pseudo-response for regression tree (approximate): dL/dlogit = dL/dp * dp/dlogit, dp/dlogit = p*(1-p)
            pseudo_response = - (grad_prob * (prob * (1 - prob)))  # negative gradient (we'll fit tree to this)
            # Fit tree to pseudo-response
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + t,
            )
            tree.fit(X_arr, pseudo_response)
            self.base_models.append(tree)
            # Update preds (logit space)
            pred += self.learning_rate * tree.predict(X_arr)

        return self

    # ---------------------
    # Probabilities / Predictions
    # ---------------------
    def predict_proba(self, X):
        logit = self._predict_logit(X)
        prob = 1.0 / (1.0 + np.exp(-logit))
        # return shape (n_samples, 2) like sklearn
        return np.vstack([1.0 - prob, prob]).T

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    # ---------------------
    # Evaluate helper (optional)
    # ---------------------
    def evaluate(self, X, y):
        """
        Return the evaluate_financial_performance dict (AUC, Brier, Expected_Loss)
        """
        probs = self.predict_proba(X)[:, 1]
        return evaluate_financial_performance(y, probs, cost_fp=self.cost_fp, cost_fn=self.cost_fn)
    


    def compute_expected_monetary_loss(
        y_true,
        probs,
        exposure,
        lgd=0.6,
        cost_fp=50.0,
        thresholds=None
    ):
        """
        Computes the expected monetary loss across thresholds and selects the best one.
        
        y_true: array of 0/1 labels
        probs: predicted probability of fraud
        exposure: Naira exposure or transaction amount at risk
        lgd: loss given default (fraction of exposure lost if fraud occurs)
        cost_fp: cost of wrongly blocking a good customer
        """

        y_true = np.array(y_true)
        probs = np.array(probs)
        exposure = np.array(exposure)

        # define thresholds if not provided
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 99)

        results = []
        for t in thresholds:
            preds = (probs >= t).astype(int)

            # False Positives (blocked good customers)
            fp_loss = ((preds == 1) & (y_true == 0)).sum() * cost_fp

            # False Negatives (missed fraud) → exposure × LGD
            fn_loss = (exposure * lgd * ((preds == 0) & (y_true == 1))).sum()

            total_loss = fp_loss + fn_loss
            results.append((t, total_loss))

        # pick best threshold
        best_threshold, best_loss = min(results, key=lambda x: x[1])

        return {
            "best_threshold": best_threshold,
            "best_loss": best_loss,
            "curve": results
        }