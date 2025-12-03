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
        """
        Convert input to numpy array and handle feature validation.
        
        During training: stores feature schema
        During prediction: validates and aligns features
        """
        # -------------------------
        # If input is a DataFrame
        # -------------------------
        if isinstance(X, pd.DataFrame):
            df = X.copy()

            # Drop time column if configured
            if self.time_col and self.time_col in df.columns:
                df = df.drop(columns=[self.time_col])

            # Select numeric columns (FIXED: added .copy())
            numeric_df = df.select_dtypes(include=[np.number]).copy()

            # Attempt conversion if none are numeric
            if numeric_df.shape[1] == 0:
                df_numeric = df.apply(pd.to_numeric, errors="coerce")
                numeric_df = df_numeric.select_dtypes(include=[np.number]).copy()
                if numeric_df.shape[1] == 0:
                    raise ValueError("No numeric features found after processing.")

            # -------------------------------------
            # TRAINING mode — store feature schema
            # -------------------------------------
            if self.feature_columns_ is None:
                self.feature_columns_ = list(numeric_df.columns)

            # -------------------------------------
            # PREDICTION mode — validate schema
            # -------------------------------------
            else:
                # Missing columns
                missing = set(self.feature_columns_) - set(numeric_df.columns)
                if missing:
                    raise ValueError(
                        f"Missing features during prediction: {missing}\n"
                        f"Expected features: {self.feature_columns_}"
                    )

                # Extra columns (warning only)
                extra = set(numeric_df.columns) - set(self.feature_columns_)
                if extra:
                    import warnings
                    warnings.warn(f"Extra features will be ignored: {extra}")

                # Reorder to match training
                numeric_df = numeric_df[self.feature_columns_]

            return numeric_df.to_numpy(dtype=float, copy=False)

        # -----------------------------------------
        # If input is a NumPy array
        # -----------------------------------------
        elif isinstance(X, np.ndarray):
            arr = np.asarray(X)

            # reshape 1D arrays
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)

            # Validate number of features
            if self.feature_columns_ is not None:
                expected = len(self.feature_columns_)
                if arr.shape[1] != expected:
                    raise ValueError(
                        f"X has {arr.shape[1]} features, but model expects {expected}."
                    )

            return arr.astype(float)

        # -----------------------------------------
        # If list-like → convert to DataFrame
        # -----------------------------------------
        else:
            return self._prepare_X(pd.DataFrame(X))

    # -------------------------
    # fit: main boosting loop
    # -------------------------
    def fit(self, X: Any, y: Any, exposure: Optional[Any] = None, sample_weight: Optional[Any] = None):
        """
        Fit TCGM.

        Parameters
        ----------
        X : DataFrame or array-like
            Feature matrix
        y : array-like (0/1)
            Binary target variable
        exposure : optional per-sample monetary exposure (float array)
            Per-sample exposure amounts
        sample_weight : optional additional sample weights
            Additional sample weights to apply
        
        Returns
        -------
        self : TimeCostGradientMachine
            Fitted estimator
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

        # Main boosting loop
        for t in range(self.n_estimators):
            # Current probabilities
            prob = 1.0 / (1.0 + np.exp(-F))
            
            # Compute gradient with respect to probability
            grad_p = grad_financial(y_arr, prob, cost_fp=self.cost_fp, cost_fn=self.cost_fn)
            
            # Convert to pseudo-response (dL/dlogit)
            if self.use_logit_safe_grad:
                pseudo = -grad_p * (prob * (1.0 - prob))
            else:
                pseudo = -grad_p

            # Fit tree to pseudo-residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + t
            )
            
            try:
                tree.fit(X_arr, pseudo, sample_weight=sw)
            except TypeError:
                # Fallback if sample_weight not supported
                tree.fit(X_arr, pseudo)
            
            self.base_models_.append(tree)

            # Update predictions in logit space
            F += self.learning_rate * tree.predict(X_arr)

            # Monitor performance
            prob_now = 1.0 / (1.0 + np.exp(-F))
            perf = evaluate_financial_performance(
                y_arr, prob_now, 
                cost_fp=self.cost_fp, 
                cost_fn=self.cost_fn
            )
            self.loss_history_.append(perf["Expected_Loss"])
            
            # Adaptive learning rate (via TrainerEngine)
            lr = self.trainer_.record_loss(perf["Expected_Loss"])
            self.lr_history_.append(lr)

        return self

    # -------------------------
    # predictions
    # -------------------------
    def _predict_logit(self, X: Any) -> np.ndarray:
        """Predict log-odds (logits) for input X."""
        X_arr = self._prepare_X(X)
        logit = np.full(shape=(X_arr.shape[0],), fill_value=self.init_logit_, dtype=float)
        
        for m in self.base_models_:
            logit += self.learning_rate * m.predict(X_arr)
        
        return logit

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Input features
        
        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Class probabilities [P(y=0), P(y=1)]
        """
        check_is_fitted(self, "base_models_")
        logits = self._predict_logit(X)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.vstack([1.0 - probs, probs]).T

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary class labels for X.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Input features
        threshold : float, default=0.5
            Classification threshold
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    # -------------------------
    # evaluation helpers
    # -------------------------
    def evaluate(self, X: Any, y: Any) -> Dict[str, float]:
        """
        Evaluate model performance with financial metrics.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Input features
        y : array-like
            True binary labels
        
        Returns
        -------
        metrics : dict
            Dictionary of financial performance metrics
        """
        probs = self.predict_proba(X)[:, 1]
        return evaluate_financial_performance(
            y, probs, 
            cost_fp=self.cost_fp, 
            cost_fn=self.cost_fn
        )

    def compute_expected_monetary_loss(
        self, 
        X: Any, 
        y: Any, 
        exposure: Any, 
        lgd: float = 0.6, 
        cost_fp: float = 50.0, 
        thresholds: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Compute expected monetary loss across different thresholds.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Input features
        y : array-like
            True binary labels
        exposure : array-like
            Per-sample exposure amounts
        lgd : float, default=0.6
            Loss given default rate
        cost_fp : float, default=50.0
            Cost of false positive
        thresholds : array-like, optional
            Thresholds to evaluate
        
        Returns
        -------
        results : dict
            Dictionary containing loss metrics across thresholds
        """
        probs = self.predict_proba(X)[:, 1]
        return compute_expected_monetary_loss(
            y, probs, exposure, 
            lgd=lgd, 
            cost_fp=cost_fp, 
            thresholds=thresholds
        )

    # -------------------------
    # persistence
    # -------------------------
    def save(self, path: str):
        """
        Save model to disk.
        
        Parameters
        ----------
        path : str
            Path to save the model
        """
        joblib.dump({
            "model": self, 
            "feature_columns": self.feature_columns_
        }, path)

    @classmethod
    def load(cls, path: str):
        """
        Load model from disk.
        
        Parameters
        ----------
        path : str
            Path to the saved model
        
        Returns
        -------
        model : TimeCostGradientMachine
            Loaded model instance
        """
        loaded = joblib.load(path)
        return loaded["model"]