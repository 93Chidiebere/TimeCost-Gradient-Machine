# tcgm/models.py
"""
TCGM models: TCGMClassifier and TCGMRegressor.

Supports production second-order gradient boosting backends ("xgboost", "lightgbm")
via custom objective callbacks returning exact first-order gradients and second-order
Hessians, as well as a standalone "sklearn" first-order fallback.

Key design choices:
 - Cost-consistent initialization: weighted prior logit for classification, exact
   tau-expectile for asymmetric MSE regression, and weighted tau-quantile for asymmetric MAE.
 - Supports scalar or instance-dependent costs, exposure weighting, and optional
   chronological recency weighting (time_col).
 - Supports validation early stopping directly on the asymmetric objective.
 - Feature schema is captured at fit time and validated on predict.
"""

from typing import Any, Optional, List, Tuple, Union
import warnings
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted
import joblib

from .loss import (
    grad_financial,
    grad_asymmetric_mae,
    convex_cs_logloss,
    grad_convex_cs_logloss,
    hess_convex_cs_logloss,
    asymmetric_mse_loss,
    grad_asymmetric_mse,
    hess_asymmetric_mse,
    asymmetric_mae_loss,
    compute_expectile,
    compute_weighted_quantile,
)
from .metrics import evaluate_financial_performance, evaluate_regression_cost, compute_expected_monetary_loss


# -----------------------
# Helper: feature preparation & alignment
# -----------------------
def _prepare_dataframe_for_model(df: pd.DataFrame, time_col: Optional[str], feature_columns: Optional[List[str]] = None):
    """
    If feature_columns is None -> training mode: return numeric_df, columns.
    Else -> prediction mode: validate that required columns are present, reorder to match training.
    """
    d = df.copy()
    if time_col and time_col in d.columns:
        d = d.drop(columns=[time_col])

    numeric_df = d.select_dtypes(include=[np.number]).copy()
    if numeric_df.shape[1] == 0:
        converted = d.apply(pd.to_numeric, errors="coerce")
        numeric_df = converted.select_dtypes(include=[np.number]).copy()
        if numeric_df.shape[1] == 0:
            raise ValueError("No numeric features found after processing input DataFrame.")

    if feature_columns is None:
        return numeric_df, list(numeric_df.columns)
    else:
        missing = set(feature_columns) - set(numeric_df.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}. Expected: {feature_columns}")
        numeric_df = numeric_df[feature_columns]
        return numeric_df, feature_columns


def _compute_recency_weights(
    X: Any,
    n: int,
    time_col: Optional[str],
    recency_lambda: float,
    recency_alpha: float,
) -> np.ndarray:
    """Compute mean-normalized exponential recency decay weights."""
    if time_col and isinstance(X, pd.DataFrame) and time_col in X.columns:
        times = pd.to_datetime(X[time_col])
        delta_t = (times.max() - times).dt.total_seconds() / (24.0 * 3600.0)
        rec = np.exp(-recency_lambda * delta_t.to_numpy(dtype=float))
    else:
        warnings.warn(
            "recency_weighting is True but time_col was not found in input DataFrame. "
            "Falling back to index-based decay."
        )
        rec = np.power(recency_alpha, np.arange(n)[::-1])
    return rec / (float(np.mean(rec)) + 1e-12)


# -----------------------
# Classifier
# -----------------------
class TCGMClassifier(BaseEstimator, ClassifierMixin):
    """
    TCGMClassifier: asymmetric-cost, time-aware gradient boosting classifier.

    Supports production second-order backends ("xgboost", "lightgbm") using exact
    gradients and Hessians of the convex cost-sensitive log-loss, plus a legacy
    first-order "sklearn" backend.
    """
    def __init__(
        self,
        n_estimators: int = 160,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        min_samples_leaf: int = 30,
        num_leaves: int = 15,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        reg_lambda: float = 1.0,
        min_child_weight: float = 1.0,
        cost_fp: float = 10.0,
        cost_fn: float = 100.0,
        time_col: Optional[str] = None,
        recency_weighting: bool = False,
        recency_alpha: float = 0.9,
        recency_lambda: float = 0.001,
        objective: str = "convex",
        backend: str = "xgboost",
        early_stopping_rounds: Optional[int] = None,
        n_jobs: int = 4,
        use_logit_safe_grad: bool = True,
        random_state: int = 42,
    ):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.num_leaves = int(num_leaves)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.reg_lambda = float(reg_lambda)
        self.min_child_weight = float(min_child_weight)
        self.cost_fp = float(cost_fp)
        self.cost_fn = float(cost_fn)
        self.time_col = time_col
        self.recency_weighting = bool(recency_weighting)
        self.recency_alpha = float(recency_alpha)
        self.recency_lambda = float(recency_lambda)
        self.objective = str(objective)
        self.backend = str(backend).lower()
        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = int(n_jobs)
        self.use_logit_safe_grad = bool(use_logit_safe_grad)
        self.random_state = int(random_state)

        self.base_models_: List[Any] = []
        self.booster_: Any = None
        self.init_logit_: float = 0.0
        self.feature_columns_: Optional[List[str]] = None
        self.loss_history_: List[float] = []
        self.best_iteration_: Optional[int] = None

    def _prepare_X(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            df, cols = _prepare_dataframe_for_model(df=X, time_col=self.time_col, feature_columns=self.feature_columns_)
            if self.feature_columns_ is None:
                self.feature_columns_ = cols
            return df.to_numpy(dtype=float, copy=False)
        else:
            df = pd.DataFrame(X)
            return self._prepare_X(df)

    def _compute_cost_consistent_init_logit(
        self,
        y_arr: np.ndarray,
        cfp: Union[float, np.ndarray],
        cfn: Union[float, np.ndarray],
        sw: np.ndarray,
    ) -> float:
        """
        Analytical constant minimizer of weighted cost-sensitive log-loss:
          F0 = log( sum(sw * c_fn * y) / sum(sw * c_fp * (1 - y)) )
        """
        pos_mass = float(np.sum(sw * np.asarray(cfn, dtype=float) * (y_arr == 1)))
        neg_mass = float(np.sum(sw * np.asarray(cfp, dtype=float) * (y_arr == 0)))
        ratio = max(pos_mass, 1e-12) / max(neg_mass, 1e-12)
        return float(np.clip(np.log(ratio), -12.0, 12.0))

    def fit(
        self,
        X: Any,
        y: Any,
        exposure: Optional[Any] = None,
        sample_weight: Optional[Any] = None,
        cost_fp_vec: Optional[Any] = None,
        cost_fn_vec: Optional[Any] = None,
        eval_set: Optional[Tuple[Any, Any]] = None,
        eval_cost_fp_vec: Optional[Any] = None,
        eval_cost_fn_vec: Optional[Any] = None,
    ):
        self.feature_columns_ = None
        X_arr = self._prepare_X(X)
        y_arr = np.asarray(y).reshape(-1).astype(int)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have same number of rows.")
        n = X_arr.shape[0]

        cfp = np.asarray(cost_fp_vec, dtype=float).reshape(-1) if cost_fp_vec is not None else self.cost_fp
        cfn = np.asarray(cost_fn_vec, dtype=float).reshape(-1) if cost_fn_vec is not None else self.cost_fn

        # Sample weights (exposure + optional temporal recency)
        sw = np.ones(n, dtype=float)
        if sample_weight is not None:
            sw = sw * np.asarray(sample_weight, dtype=float).reshape(-1)
        if exposure is not None:
            exp = np.asarray(exposure, dtype=float).reshape(-1)
            sw = sw * (exp / (float(np.mean(exp)) + 1e-12))
        if self.recency_weighting:
            sw = sw * _compute_recency_weights(X, n, self.time_col, self.recency_lambda, self.recency_alpha)

        # Normalize effective cost scale so L2 regularization lambda behaves consistently
        eff_w = sw * np.where(y_arr == 1, np.asarray(cfn, dtype=float), np.asarray(cfp, dtype=float))
        scale = float(np.mean(eff_w)) + 1e-12
        cfp_scaled = np.asarray(cfp, dtype=float) / scale
        cfn_scaled = np.asarray(cfn, dtype=float) / scale

        # Cost-consistent initial logit (analytical optimum of training objective)
        self.init_logit_ = self._compute_cost_consistent_init_logit(y_arr, cfp, cfn, sw)
        self.base_models_ = []
        self.booster_ = None
        self.loss_history_ = []
        self.best_iteration_ = None

        if self.backend == "xgboost":
            import xgboost as xgb

            dtrain = xgb.DMatrix(X_arr, label=y_arr)
            dtrain.set_base_margin(np.full(n, self.init_logit_, dtype=float))

            def _xgb_obj(preds: np.ndarray, _dtrain):
                p = 1.0 / (1.0 + np.exp(-np.clip(preds, -30.0, 30.0)))
                grad = sw * grad_convex_cs_logloss(y_arr, p, cost_fp=cfp_scaled, cost_fn=cfn_scaled)
                hess = sw * hess_convex_cs_logloss(y_arr, p, cost_fp=cfp_scaled, cost_fn=cfn_scaled)
                return grad, hess

            params = {
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "reg_lambda": self.reg_lambda,
                "min_child_weight": self.min_child_weight,
                "tree_method": "hist",
                "nthread": self.n_jobs,
                "seed": self.random_state,
                "disable_default_eval_metric": 1,
            }

            evals = []
            custom_metric = None
            if eval_set is not None:
                X_va, y_va = eval_set
                X_va_arr = self._prepare_X(X_va)
                y_va_arr = np.asarray(y_va).reshape(-1).astype(int)
                dval = xgb.DMatrix(X_va_arr, label=y_va_arr)
                dval.set_base_margin(np.full(len(y_va_arr), self.init_logit_, dtype=float))
                evals = [(dval, "val")]
                va_cfp = np.asarray(eval_cost_fp_vec, dtype=float).reshape(-1) if eval_cost_fp_vec is not None else self.cost_fp
                va_cfn = np.asarray(eval_cost_fn_vec, dtype=float).reshape(-1) if eval_cost_fn_vec is not None else self.cost_fn

                def _xgb_eval(preds: np.ndarray, _dval):
                    p = 1.0 / (1.0 + np.exp(-np.clip(preds, -30.0, 30.0)))
                    loss = float(np.mean(convex_cs_logloss(y_va_arr, p, cost_fp=va_cfp, cost_fn=va_cfn)))
                    return "tcgm_cs_logloss", loss

                custom_metric = _xgb_eval

            es = self.early_stopping_rounds if (eval_set is not None and self.early_stopping_rounds) else None
            self.booster_ = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=self.n_estimators,
                obj=_xgb_obj,
                custom_metric=custom_metric,
                evals=evals,
                early_stopping_rounds=es,
                verbose_eval=False,
            )
            self.best_iteration_ = getattr(self.booster_, "best_iteration", self.n_estimators - 1)
            self.base_models_ = [self.booster_]

        elif self.backend == "lightgbm":
            import lightgbm as lgb

            init_tr = np.full(n, self.init_logit_, dtype=float)
            dtrain = lgb.Dataset(X_arr, label=y_arr, init_score=init_tr, free_raw_data=False)

            def _lgb_obj(preds: np.ndarray, _dataset):
                p = 1.0 / (1.0 + np.exp(-np.clip(preds, -30.0, 30.0)))
                grad = sw * grad_convex_cs_logloss(y_arr, p, cost_fp=cfp_scaled, cost_fn=cfn_scaled)
                hess = sw * hess_convex_cs_logloss(y_arr, p, cost_fp=cfp_scaled, cost_fn=cfn_scaled)
                return grad, hess

            params = {
                "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves,
                "max_depth": self.max_depth if self.max_depth > 0 else -1,
                "min_child_samples": self.min_samples_leaf,
                "subsample": self.subsample,
                "subsample_freq": 1 if self.subsample < 1.0 else 0,
                "colsample_bytree": self.colsample_bytree,
                "reg_lambda": self.reg_lambda,
                "n_jobs": self.n_jobs,
                "random_state": self.random_state,
                "verbosity": -1,
                "objective": _lgb_obj,
            }

            valid_sets = None
            feval = None
            callbacks = []
            if eval_set is not None:
                X_va, y_va = eval_set
                X_va_arr = self._prepare_X(X_va)
                y_va_arr = np.asarray(y_va).reshape(-1).astype(int)
                dval = lgb.Dataset(
                    X_va_arr,
                    label=y_va_arr,
                    init_score=np.full(len(y_va_arr), self.init_logit_, dtype=float),
                    reference=dtrain,
                    free_raw_data=False,
                )
                valid_sets = [dval]
                va_cfp = np.asarray(eval_cost_fp_vec, dtype=float).reshape(-1) if eval_cost_fp_vec is not None else self.cost_fp
                va_cfn = np.asarray(eval_cost_fn_vec, dtype=float).reshape(-1) if eval_cost_fn_vec is not None else self.cost_fn

                def _lgb_eval(preds: np.ndarray, _dataset):
                    p = 1.0 / (1.0 + np.exp(-np.clip(preds, -30.0, 30.0)))
                    loss = float(np.mean(convex_cs_logloss(y_va_arr, p, cost_fp=va_cfp, cost_fn=va_cfn)))
                    return "tcgm_cs_logloss", loss, False

                feval = _lgb_eval
                if self.early_stopping_rounds:
                    callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

            self.booster_ = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=self.n_estimators,
                valid_sets=valid_sets,
                feval=feval,
                callbacks=callbacks if callbacks else None,
            )
            self.best_iteration_ = getattr(self.booster_, "best_iteration", self.n_estimators)
            self.base_models_ = [self.booster_]

        else:
            # Legacy first-order scikit-learn tree loop (with cost-consistent init and no dead LR)
            F = np.full(shape=y_arr.shape, fill_value=self.init_logit_, dtype=float)
            for t in range(self.n_estimators):
                prob = 1.0 / (1.0 + np.exp(-np.clip(F, -30.0, 30.0)))
                if self.objective == "convex":
                    grad_logit = grad_convex_cs_logloss(y_arr, prob, cost_fp=cfp_scaled, cost_fn=cfn_scaled)
                    pseudo = -grad_logit
                else:
                    grad_p = grad_financial(y_arr, prob, cost_fp=self.cost_fp, cost_fn=self.cost_fn)
                    pseudo = -grad_p * (prob * (1.0 - prob)) if self.use_logit_safe_grad else -grad_p

                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state + t,
                )
                tree.fit(X_arr, pseudo, sample_weight=sw)
                self.base_models_.append(tree)
                F += self.learning_rate * tree.predict(X_arr)

        return self

    def _predict_logit(self, X: Any) -> np.ndarray:
        X_arr = self._prepare_X(X)
        if self.backend == "xgboost":
            import xgboost as xgb
            dtest = xgb.DMatrix(X_arr)
            dtest.set_base_margin(np.full(X_arr.shape[0], self.init_logit_, dtype=float))
            if self.early_stopping_rounds and self.best_iteration_ is not None:
                return self.booster_.predict(dtest, output_margin=True, iteration_range=(0, int(self.best_iteration_) + 1))
            return self.booster_.predict(dtest, output_margin=True)
        elif self.backend == "lightgbm":
            num_it = self.best_iteration_ if (self.early_stopping_rounds and self.best_iteration_) else None
            return self.init_logit_ + self.booster_.predict(X_arr, raw_score=True, num_iteration=num_it)
        else:
            logit = np.full(shape=(X_arr.shape[0],), fill_value=self.init_logit_, dtype=float)
            for m in self.base_models_:
                logit += self.learning_rate * m.predict(X_arr)
            return logit

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "base_models_")
        logits = self._predict_logit(X)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        return np.vstack([1.0 - probs, probs]).T

    def predict_proba_calibrated(
        self,
        X: Any,
        cost_fp: Optional[Union[float, np.ndarray]] = None,
        cost_fn: Optional[Union[float, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Apply the exact inverse Elkan transformation to recover unweighted event
        probabilities from cost-warped predictions (useful for IFRS 9 / Basel III PD pricing).
        """
        p_star = self.predict_proba(X)[:, 1]
        cfp = np.asarray(self.cost_fp if cost_fp is None else cost_fp, dtype=float)
        cfn = np.asarray(self.cost_fn if cost_fn is None else cost_fn, dtype=float)
        num = cfp * p_star
        den = num + cfn * (1.0 - p_star)
        p_cal = np.clip(num / np.maximum(den, 1e-15), 0.0, 1.0)
        return np.vstack([1.0 - p_cal, p_cal]).T

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def evaluate(self, X: Any, y: Any):
        probs = self.predict_proba(X)[:, 1]
        return evaluate_financial_performance(y, probs, cost_fp=self.cost_fp, cost_fn=self.cost_fn)

    def compute_expected_monetary_loss(self, X: Any, y: Any, exposure: Any, lgd: float = 0.6, cost_fp: float = 50.0, thresholds: Optional[Any] = None):
        probs = self.predict_proba(X)[:, 1]
        return compute_expected_monetary_loss(y, probs, exposure, lgd=lgd, cost_fp=cost_fp, thresholds=thresholds)

    def save(self, path: str):
        joblib.dump({"model": self, "feature_columns": self.feature_columns_}, path)

    @classmethod
    def load(cls, path: str):
        loaded = joblib.load(path)
        return loaded["model"]

    @property
    def init_pred(self):
        return getattr(self, "init_logit_", 0.0)

    @property
    def base_models(self):
        return getattr(self, "base_models_", [])

    @property
    def trees(self):
        return getattr(self, "base_models_", [])


# -----------------------
# Regressor
# -----------------------
class TCGMRegressor(BaseEstimator, RegressorMixin):
    """
    TCGMRegressor: gradient boosting regressor with asymmetric MSE (expectile) or
    asymmetric MAE (quantile) objective.

    Supports production second-order backends ("xgboost", "lightgbm") with exact
    gradients and piecewise-constant Hessians, cost-consistent expectile/quantile
    initialization, and validation early stopping on the true asymmetric loss.
    """
    def __init__(
        self,
        n_estimators: int = 160,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        min_samples_leaf: int = 20,
        num_leaves: int = 15,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        reg_lambda: float = 1.0,
        min_child_weight: float = 1.0,
        c_over: float = 1.0,
        c_under: float = 8.0,
        time_col: Optional[str] = None,
        recency_weighting: bool = False,
        recency_alpha: float = 0.9,
        recency_lambda: float = 0.001,
        objective: str = "mse",
        backend: str = "xgboost",
        early_stopping_rounds: Optional[int] = None,
        n_jobs: int = 4,
        random_state: int = 42,
    ):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.num_leaves = int(num_leaves)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.reg_lambda = float(reg_lambda)
        self.min_child_weight = float(min_child_weight)
        self.c_over = float(c_over)
        self.c_under = float(c_under)
        self.time_col = time_col
        self.recency_weighting = bool(recency_weighting)
        self.recency_alpha = float(recency_alpha)
        self.recency_lambda = float(recency_lambda)
        self.objective = str(objective).lower()
        self.backend = str(backend).lower()
        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = int(n_jobs)
        self.random_state = int(random_state)

        self.base_models_: List[Any] = []
        self.booster_: Any = None
        self.init_pred_: float = 0.0
        self.feature_columns_: Optional[List[str]] = None
        self.best_iteration_: Optional[int] = None

    def _prepare_X(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            df, cols = _prepare_dataframe_for_model(df=X, time_col=self.time_col, feature_columns=self.feature_columns_)
            if self.feature_columns_ is None:
                self.feature_columns_ = cols
            return df.to_numpy(dtype=float, copy=False)
        else:
            df = pd.DataFrame(X)
            return self._prepare_X(df)

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
        eval_set: Optional[Tuple[Any, Any]] = None,
    ):
        self.feature_columns_ = None
        X_arr = self._prepare_X(X)
        y_arr = np.asarray(y).reshape(-1).astype(float)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have same number of rows.")
        n = X_arr.shape[0]

        sw = np.ones(n, dtype=float)
        if sample_weight is not None:
            sw = sw * np.asarray(sample_weight, dtype=float).reshape(-1)
        if self.recency_weighting:
            sw = sw * _compute_recency_weights(X, n, self.time_col, self.recency_lambda, self.recency_alpha)

        # Cost-consistent initialization: exact tau-expectile for Asym-MSE, tau-quantile for Asym-MAE
        tau = self.c_under / (self.c_under + self.c_over)
        if self.objective == "mse":
            self.init_pred_ = compute_expectile(y_arr, tau=tau, sample_weight=sw)
        else:
            self.init_pred_ = compute_weighted_quantile(y_arr, tau=tau, sample_weight=sw)

        # Normalize cost weights so Hessian scale matches standard MSE (h ~ 2)
        c_scale = 0.5 * (self.c_over + self.c_under)
        c_over_s = self.c_over / c_scale
        c_under_s = self.c_under / c_scale

        self.base_models_ = []
        self.booster_ = None
        self.best_iteration_ = None

        if self.backend == "xgboost":
            import xgboost as xgb

            dtrain = xgb.DMatrix(X_arr, label=y_arr)
            dtrain.set_base_margin(np.full(n, self.init_pred_, dtype=float))

            def _xgb_reg_obj(preds: np.ndarray, _dtrain):
                if self.objective == "mse":
                    grad = sw * grad_asymmetric_mse(y_arr, preds, c_over=c_over_s, c_under=c_under_s)
                    hess = sw * hess_asymmetric_mse(y_arr, preds, c_over=c_over_s, c_under=c_under_s)
                else:
                    # Smooth Huber-like second-order approximation for Asym-MAE
                    grad = sw * grad_asymmetric_mae(y_arr, preds, c_over=c_over_s, c_under=c_under_s)
                    hess = sw * np.ones_like(preds, dtype=float)
                return grad, hess

            params = {
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "reg_lambda": self.reg_lambda,
                "min_child_weight": self.min_child_weight,
                "tree_method": "hist",
                "nthread": self.n_jobs,
                "seed": self.random_state,
                "disable_default_eval_metric": 1,
            }

            evals = []
            custom_metric = None
            if eval_set is not None:
                X_va, y_va = eval_set
                X_va_arr = self._prepare_X(X_va)
                y_va_arr = np.asarray(y_va).reshape(-1).astype(float)
                dval = xgb.DMatrix(X_va_arr, label=y_va_arr)
                dval.set_base_margin(np.full(len(y_va_arr), self.init_pred_, dtype=float))
                evals = [(dval, "val")]

                def _xgb_reg_eval(preds: np.ndarray, _dval):
                    if self.objective == "mse":
                        loss = float(np.mean(asymmetric_mse_loss(y_va_arr, preds, c_over=self.c_over, c_under=self.c_under)))
                        return "asym_mse", loss
                    else:
                        loss = float(np.mean(asymmetric_mae_loss(y_va_arr, preds, c_over=self.c_over, c_under=self.c_under)))
                        return "asym_mae", loss

                custom_metric = _xgb_reg_eval

            es = self.early_stopping_rounds if (eval_set is not None and self.early_stopping_rounds) else None
            self.booster_ = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=self.n_estimators,
                obj=_xgb_reg_obj,
                custom_metric=custom_metric,
                evals=evals,
                early_stopping_rounds=es,
                verbose_eval=False,
            )
            self.best_iteration_ = getattr(self.booster_, "best_iteration", self.n_estimators - 1)
            self.base_models_ = [self.booster_]

        elif self.backend == "lightgbm":
            import lightgbm as lgb

            init_tr = np.full(n, self.init_pred_, dtype=float)
            dtrain = lgb.Dataset(X_arr, label=y_arr, init_score=init_tr, free_raw_data=False)

            def _lgb_reg_obj(preds: np.ndarray, _dataset):
                if self.objective == "mse":
                    grad = sw * grad_asymmetric_mse(y_arr, preds, c_over=c_over_s, c_under=c_under_s)
                    hess = sw * hess_asymmetric_mse(y_arr, preds, c_over=c_over_s, c_under=c_under_s)
                else:
                    grad = sw * grad_asymmetric_mae(y_arr, preds, c_over=c_over_s, c_under=c_under_s)
                    hess = sw * np.ones_like(preds, dtype=float)
                return grad, hess

            params = {
                "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves,
                "max_depth": self.max_depth if self.max_depth > 0 else -1,
                "min_child_samples": self.min_samples_leaf,
                "subsample": self.subsample,
                "subsample_freq": 1 if self.subsample < 1.0 else 0,
                "colsample_bytree": self.colsample_bytree,
                "reg_lambda": self.reg_lambda,
                "n_jobs": self.n_jobs,
                "random_state": self.random_state,
                "verbosity": -1,
                "objective": _lgb_reg_obj,
            }

            valid_sets = None
            feval = None
            callbacks = []
            if eval_set is not None:
                X_va, y_va = eval_set
                X_va_arr = self._prepare_X(X_va)
                y_va_arr = np.asarray(y_va).reshape(-1).astype(float)
                dval = lgb.Dataset(
                    X_va_arr,
                    label=y_va_arr,
                    init_score=np.full(len(y_va_arr), self.init_pred_, dtype=float),
                    reference=dtrain,
                    free_raw_data=False,
                )
                valid_sets = [dval]

                def _lgb_reg_eval(preds: np.ndarray, _dataset):
                    if self.objective == "mse":
                        loss = float(np.mean(asymmetric_mse_loss(y_va_arr, preds, c_over=self.c_over, c_under=self.c_under)))
                        return "asym_mse", loss, False
                    else:
                        loss = float(np.mean(asymmetric_mae_loss(y_va_arr, preds, c_over=self.c_over, c_under=self.c_under)))
                        return "asym_mae", loss, False

                feval = _lgb_reg_eval
                if self.early_stopping_rounds:
                    callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

            self.booster_ = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=self.n_estimators,
                valid_sets=valid_sets,
                feval=feval,
                callbacks=callbacks if callbacks else None,
            )
            self.best_iteration_ = getattr(self.booster_, "best_iteration", self.n_estimators)
            self.base_models_ = [self.booster_]

        else:
            # Legacy first-order scikit-learn tree loop (with cost-consistent expectile/quantile init)
            F = np.full(shape=y_arr.shape, fill_value=self.init_pred_, dtype=float)
            for t in range(self.n_estimators):
                if self.objective == "mse":
                    grad = grad_asymmetric_mse(y_arr, F, c_over=c_over_s, c_under=c_under_s)
                else:
                    grad = grad_asymmetric_mae(y_arr, F, c_over=c_over_s, c_under=c_under_s)
                pseudo = -grad
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state + t,
                )
                tree.fit(X_arr, pseudo, sample_weight=sw)
                self.base_models_.append(tree)
                F += self.learning_rate * tree.predict(X_arr)

        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "base_models_")
        X_arr = self._prepare_X(X)
        if self.backend == "xgboost":
            import xgboost as xgb
            dtest = xgb.DMatrix(X_arr)
            dtest.set_base_margin(np.full(X_arr.shape[0], self.init_pred_, dtype=float))
            if self.early_stopping_rounds and self.best_iteration_ is not None:
                return self.booster_.predict(dtest, iteration_range=(0, int(self.best_iteration_) + 1))
            return self.booster_.predict(dtest)
        elif self.backend == "lightgbm":
            num_it = self.best_iteration_ if (self.early_stopping_rounds and self.best_iteration_) else None
            return self.init_pred_ + self.booster_.predict(X_arr, num_iteration=num_it)
        else:
            pred = np.full(shape=(X_arr.shape[0],), fill_value=self.init_pred_, dtype=float)
            for m in self.base_models_:
                pred += self.learning_rate * m.predict(X_arr)
            return pred

    def evaluate(self, X: Any, y: Any):
        preds = self.predict(X)
        return evaluate_regression_cost(y, preds, c_over=self.c_over, c_under=self.c_under)

    def save(self, path: str):
        joblib.dump({"model": self, "feature_columns": self.feature_columns_}, path)

    @classmethod
    def load(cls, path: str):
        loaded = joblib.load(path)
        return loaded["model"]

    @property
    def init_pred(self):
        return getattr(self, "init_pred_", 0.0)

    @property
    def base_models(self):
        return getattr(self, "base_models_", [])

    @property
    def trees(self):
        return getattr(self, "base_models_", [])


# Backwards-compatible alias
TimeCostGradientMachine = TCGMClassifier
