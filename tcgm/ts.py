"""
Time-series forecasting module for TimeCost Gradient Machine.

Provides:
 - TCGMForecaster: wrapper for recursive multi-step forecasting using TCGMRegressor.
"""

from typing import List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from .models import TCGMRegressor


class TCGMForecaster(BaseEstimator, RegressorMixin):
    """
    TCGMForecaster: Cost-sensitive, time-series forecasting wrapper.

    Automatically creates lag, rolling window, and calendar (day-of-week) features
    and performs recursive multi-step forecasting using TCGMRegressor.
    """
    def __init__(
        self,
        estimator: Optional[TCGMRegressor] = None,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[Tuple[int, str]]] = None,
        include_dayofweek: bool = True,
    ):
        self.estimator = estimator
        self.lags = lags or [1, 2, 3]
        # List of (window_size, stat_type) where stat_type is "mean", "std", "max", "min"
        self.rolling_windows = rolling_windows or [(7, "mean")]
        self.include_dayofweek = bool(include_dayofweek)

        # Instantiate default regressor if none provided
        self.estimator_ = estimator if estimator is not None else TCGMRegressor(n_estimators=90, learning_rate=0.05)

    def _create_features(self, series: pd.Series) -> pd.DataFrame:
        """
        Create lag, rolling, and optional calendar (day-of-week) features from raw series.
        Uses shift(1) for rolling stats to prevent contemporaneous data leakage.
        """
        df = pd.DataFrame(index=series.index)
        df["target"] = series.values

        # Generate Lags
        for lag in self.lags:
            df[f"lag_{lag}"] = series.shift(lag)

        # Generate Rolling Statistics (shifted by 1 to prevent leakage)
        shifted_series = series.shift(1)
        for window, stat in self.rolling_windows:
            if stat == "mean":
                df[f"rolling_mean_{window}"] = shifted_series.rolling(window=window).mean()
            elif stat == "std":
                df[f"rolling_std_{window}"] = shifted_series.rolling(window=window).std()
            elif stat == "max":
                df[f"rolling_max_{window}"] = shifted_series.rolling(window=window).max()
            elif stat == "min":
                df[f"rolling_min_{window}"] = shifted_series.rolling(window=window).min()
            else:
                raise ValueError(f"Unknown rolling statistic type: {stat}")

        # Calendar feature: day-of-week (matches standard GBDT forecasting baselines)
        if self.include_dayofweek:
            if isinstance(series.index, pd.DatetimeIndex):
                df["dow"] = series.index.dayofweek.to_numpy(dtype=float)
            else:
                df["dow"] = (np.arange(len(series)) % 7).astype(float)

        return df

    def fit(self, series: Any, sample_weight: Optional[Any] = None, eval_fraction: float = 0.0):
        """
        Fit the forecaster on the historical series.
        """
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        self.history_ = series.copy()
        self.is_datetime_index_ = isinstance(series.index, pd.DatetimeIndex)
        if self.is_datetime_index_:
            self.last_timestamp_ = series.index[-1]
        self.train_len_ = len(series)

        # Build features dataframe
        df_feat = self._create_features(series)

        # Drop initial rows with NaNs from shifts/rolling windows
        valid_mask = df_feat.notna().all(axis=1)
        df_feat = df_feat.loc[valid_mask]

        if len(df_feat) == 0:
            raise ValueError(
                f"No training samples remaining after generating lags {self.lags} "
                f"and rolling windows {self.rolling_windows}. Series length might be too short."
            )

        X = df_feat.drop(columns=["target"])
        y = df_feat["target"]

        # Align sample weights with dropped rows
        if sample_weight is not None:
            sw = np.asarray(sample_weight)[valid_mask.to_numpy()]
        else:
            sw = None

        if eval_fraction > 0.0 and len(X) > 40:
            cut = int(len(X) * (1.0 - eval_fraction))
            X_tr, y_tr = X.iloc[:cut], y.iloc[:cut]
            X_va, y_va = X.iloc[cut:], y.iloc[cut:]
            sw_tr = sw[:cut] if sw is not None else None
            self.estimator_.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=(X_va, y_va))
        else:
            self.estimator_.fit(X, y, sample_weight=sw)
        self.feature_columns_ = list(X.columns)
        return self

    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Recursive multi-step forecast for a given number of steps into the future.
        Properly computes next-step lags from history[-lag], rolling windows from history[-w:],
        and calendar day-of-week for the exact forecast horizon index.
        """
        if not hasattr(self, "history_"):
            raise ValueError("Forecaster is not fitted yet. Call fit() first.")

        history = list(self.history_.values.astype(float))
        predictions = []

        for step in range(steps):
            row = {}
            for lag in self.lags:
                row[f"lag_{lag}"] = history[-lag]

            for window, stat in self.rolling_windows:
                win_vals = np.asarray(history[-window:], dtype=float)
                if stat == "mean":
                    row[f"rolling_mean_{window}"] = float(np.mean(win_vals))
                elif stat == "std":
                    row[f"rolling_std_{window}"] = float(np.std(win_vals, ddof=1)) if len(win_vals) > 1 else 0.0
                elif stat == "max":
                    row[f"rolling_max_{window}"] = float(np.max(win_vals))
                elif stat == "min":
                    row[f"rolling_min_{window}"] = float(np.min(win_vals))

            if self.include_dayofweek:
                if self.is_datetime_index_:
                    next_ts = self.last_timestamp_ + pd.Timedelta(days=step + 1)
                    row["dow"] = float(next_ts.dayofweek)
                else:
                    row["dow"] = float((self.train_len_ + step) % 7)

            X_next = pd.DataFrame([row], columns=self.feature_columns_)
            pred = float(self.estimator_.predict(X_next)[0])
            predictions.append(pred)
            history.append(pred)

        return np.array(predictions, dtype=float)
