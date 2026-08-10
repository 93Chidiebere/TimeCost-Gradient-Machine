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
    
    Automatically creates lag and rolling window features and performs
    recursive multi-step forecasting using TCGMRegressor.
    """
    def __init__(
        self,
        estimator: Optional[TCGMRegressor] = None,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[Tuple[int, str]]] = None,
    ):
        self.estimator = estimator
        self.lags = lags or [1, 2, 3]
        # List of (window_size, stat_type) where stat_type is "mean", "std", "max", "min"
        self.rolling_windows = rolling_windows or [(7, "mean")]
        
        # Instantiate default regressor if none provided
        self.estimator_ = estimator if estimator is not None else TCGMRegressor(n_estimators=30, learning_rate=0.1)

    def _create_features(self, series: pd.Series) -> pd.DataFrame:
        """
        Create lag and rolling features from raw series.
        Uses shift(1) for rolling stats to prevent data leakage.
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

        return df

    def fit(self, series: Any, sample_weight: Optional[Any] = None):
        """
        Fit the forecaster on the historical series.
        """
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        self.history_ = series.copy()
        
        # Build features dataframe
        df_feat = self._create_features(series)
        
        # Drop initial rows with NaNs from shifts/rolling windows
        df_feat = df_feat.dropna()
        
        if len(df_feat) == 0:
            raise ValueError(
                f"No training samples remaining after generating lags {self.lags} "
                f"and rolling windows {self.rolling_windows}. Series length might be too short."
            )
            
        X = df_feat.drop(columns=["target"])
        y = df_feat["target"]

        # Align sample weights with dropped index rows
        if sample_weight is not None:
            sw = np.asarray(sample_weight)[df_feat.index]
        else:
            sw = None

        self.estimator_.fit(X, y, sample_weight=sw)
        self.feature_columns_ = list(X.columns)
        return self

    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Recursive multi-step forecast for a given number of steps into the future.
        """
        if not hasattr(self, "history_"):
            raise ValueError("Forecaster is not fitted yet. Call fit() first.")

        history = list(self.history_.values)
        predictions = []

        for _ in range(steps):
            curr_series = pd.Series(history)
            df_feat = self._create_features(curr_series)
            
            # Predict the next step using the last row of features
            last_feat = df_feat.iloc[[-1]].drop(columns=["target"])
            pred = float(self.estimator_.predict(last_feat)[0])
            
            predictions.append(pred)
            # Append forecast to history for the next iteration's lags/rolling calculations
            history.append(pred)

        return np.array(predictions)
