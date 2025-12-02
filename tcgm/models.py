# tcgm/models.py
"""
TimeCost Gradient Machine (TCGM) - standalone classifier.

Features:
 - Tree-based gradient boosting using asymmetric financial gradient.
 - Time-awareness (time_col hooks) & recency weighting.
 - Exposure/sample weighting support.
 - Leakage-safe target encoder (included as a small helper class).
 - sklearn-like API: fit, predict_proba, predict, evaluate.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import joblib

from .loss import grad_financial, grad_logit_safe
from .metrics import evaluate_financial_performance, compute_expected_monetary_loss
from .core import TrainerEngine

# -----------------------
# Small leakage-safe encoder (kept inside models.py by request)
# -----------------------
class LeakageSafeTargetEncoder:
    """
    Build mapping on training partition only, then transform validation/test safely.
    Usage:
      enc = LeakageSafeTargetEncoder(cols=["merchant_id"], prior=0.05)
      enc.fit_on_train(train_df, target_col="isFraud")
      X_train_te = enc.transform(train_df)
      X_val_te   = enc.transform(val_df)
    """

    def __init__(self, cols: Optional[List[str]] = None, prior: float = 0.05):
        self.cols = cols or []
        self.prior = float(prior)
        self.maps_: Dict[str, Dict[Any, float]] = {}
        self.global_mean_: Optional[float] = None

    def fit_on_train(self, train_df: pd.DataFrame, target_col: str = "isFraud"):
        df = train_df.copy()
        self.global_mean_ = float(df[target_col].mean())
        self.maps_ = {}
        k = 1.0 / max(self.prior, 1e-9)
        for c in self.cols:
            agg = df.groupby(c)[target_col].agg(count="count", mean="mean").reset_index()
            agg["smooth"] = (agg["count"] * agg["mean"] + k * self.global_mean_) / (agg["count"] + k)
            self.maps_[c] = dict(zip(agg[c], agg["smooth"]))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in self.cols:
            mapping = self.maps_.get(c, {})
            out[f"{c}_te"] = df[c].map(mapping).fillna(self.global_mean_)
        return out

# -----------------------
# TCGM Class
# -----------------------
class TimeCostGradientMachine(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 30,
        cost_fp: float = 10.0,
        cost_fn: float = 100.0,
        time_col: Optional[str] = None,
        exposure_col: Optional[str] = None,
        recency_weighting: bool = False,
        recency_alpha: float = 0.9,
        base_lr: float = 0.05,
        use_logit_safe_grad: bool = True,
        random_state: int = 42,
    ):
        # hyperparameters
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.cost_fp = float(cost_fp)
        self.cost_fn = float(cost_fn)
        self.time_col = time_col
        self.exposure_col = exposure_col
        self.recency_weighting = bool(recency_weighting)
        self.recency_alpha = float(recency_alpha)
        self.base_lr = float(base_lr)
        self.use_logit_safe_grad = bool(use_logit_safe_grad)
        self.random_state = int(random_state)

        # internals
        self.base_models_: List[DecisionTreeRegressor] = []
        self.init_logit_: float = 0.0
        self.feature_columns_: Optional[List[str]] = None
        self.loss_history_: List[float] = []
        self.lr_history_: List[float] = []
        self.trainer_ = TrainerEngine(time_col=self.time_col, base_lr=self.base_lr, random_state=self.random_state)

    # -------------------------
    # helper: prepare X (DataFrame -> numeric matrix) and store feature names
    # -------------------------
    def _prepare_X(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            if self.time_col and self.time_col in df.columns:
                df = df.drop(columns=[self.time_col])
            numeric_df = df.select_dtypes(include=[np.number]).copy()
            if numeric_df.shape[1] == 0:
                converted = df.apply(pd.to_numeric, errors="coerce")
                numeric_df = converted.select_dtypes(include=[np.number])
                if numeric_df.shape[1] == 0:
                    raise ValueError("No numeric features found after dropping time column.")
            self.feature_columns_ = list(numeric_df.columns)
            return numeric_df.to_numpy(dtype=float, copy=False)
        elif isinstance(X, np.ndarray):
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            return arr.astype(float)
        else:
            df = pd.DataFrame(X)
            return self._prepare_X(df)

    # -------------------------
    # fit: main boosting loop
    # -------------------------
    def fit(self, X: Any, y: Any, exposure: Optional[Any] = None, sample_weight: Optional[Any] = None):
        """
        Fit TCGM.

        Parameters
        ----------
        X : DataFrame or array-like
        y : array-like (0/1)
        exposure : optional per-sample monetary exposure (float array)
        sample_weight : optional additional sample weights
        """
        X_arr = self._prepare_X(X)
        y_arr = np.asarray(y).reshape(-1).astype(int)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have same number of rows")

        n = X_arr.shape[0]

        # initial logit (log-odds)
        p0 = np.clip(np.mean(y_arr), 1e-6, 1 - 1e-6)
        self.init_logit_ = float(np.log(p0 / (1.0 - p0)))
        F = np.full(shape=y_arr.shape, fill_value=self.init_logit_, dtype=float)

        # compute sample weights (exposure and optional weight)
        sw = np.ones(n, dtype=float)
        if sample_weight is not None:
            sw = sw * np.asarray(sample_weight, dtype=float).reshape(-1)
        if exposure is not None:
            exp = np.asarray(exposure, dtype=float).reshape(-1)
            # rescale exposures to avoid numerical issues
            exp_norm = exp / (np.mean(exp) + 1e-12)
            sw = sw * exp_norm
        if self.recency_weighting:
            rec = np.power(self.recency_alpha, np.arange(n)[::-1])
            rec = rec / (np.mean(rec) + 1e-12)
            sw = sw * rec

        self.base_models_ = []
        self.loss_history_ = []
        self.lr_history_ = []

        for t in range(self.n_estimators):
            prob = 1.0 / (1.0 + np.exp(-F))
            # gradient wrt prob
            grad_p = grad_financial(y_arr, prob, cost_fp=self.cost_fp, cost_fn=self.cost_fn)
            # convert to pseudo-response (dL/dlogit)
            if self.use_logit_safe_grad:
                pseudo = -grad_p * (prob * (1.0 - prob))
            else:
                pseudo = -grad_p

            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, random_state=self.random_state + t)
            try:
                tree.fit(X_arr, pseudo, sample_weight=sw)
            except Exception:
                tree.fit(X_arr, pseudo)
            self.base_models_.append(tree)

            # update predictions (logit space)
            F += self.learning_rate * tree.predict(X_arr)

            # monitoring
            prob_now = 1.0 / (1.0 + np.exp(-F))
            perf = evaluate_financial_performance(y_arr, prob_now, cost_fp=self.cost_fp, cost_fn=self.cost_fn)
            self.loss_history_.append(perf["Expected_Loss"])
            lr = self.trainer_.record_loss(perf["Expected_Loss"])
            self.lr_history_.append(lr)

        return self

    # -------------------------
    # predictions
    # -------------------------
    def _predict_logit(self, X: Any) -> np.ndarray:
        X_arr = self._prepare_X(X)
        logit = np.full(shape=(X_arr.shape[0],), fill_value=self.init_logit_, dtype=float)
        for m in self.base_models_:
            logit += self.learning_rate * m.predict(X_arr)
        return logit

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "base_models_")
        logits = self._predict_logit(X)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.vstack([1.0 - probs, probs]).T

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    # -------------------------
    # evaluation helpers
    # -------------------------
    def evaluate(self, X: Any, y: Any) -> Dict[str, float]:
        probs = self.predict_proba(X)[:, 1]
        return evaluate_financial_performance(y, probs, cost_fp=self.cost_fp, cost_fn=self.cost_fn)

    def compute_expected_monetary_loss(self, X: Any, y: Any, exposure: Any, lgd: float = 0.6, cost_fp: float = 50.0, thresholds: Optional[Any] = None) -> Dict[str, Any]:
        probs = self.predict_proba(X)[:, 1]
        return compute_expected_monetary_loss(y, probs, exposure, lgd=lgd, cost_fp=cost_fp, thresholds=thresholds)

    # -------------------------
    # persistence
    # -------------------------
    def save(self, path: str):
        joblib.dump({"model": self, "feature_columns": self.feature_columns_}, path)

    @classmethod
    def load(cls, path: str):
        loaded = joblib.load(path)
        return loaded["model"]