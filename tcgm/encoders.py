"""
Leakage-safe target encoder for high-cardinality categoricals.
Computes smoothed target means using only training data.
"""

import pandas as pd
import numpy as np

class LeakageSafeTargetEncoder:
    """
    For each fold, call fit_on_train(train_df, target_col) then transform(df).
    Smoothing uses prior.
    """
    def __init__(self, cols=None, prior=0.05):
        self.cols = cols or []
        self.prior = float(prior)
        self.maps_ = {}
        self.global_mean_ = None

    def fit_on_train(self, df_train, target_col="label"):
        """
        Build mapping for each categorical column using only train partition.
        """
        df_train = df_train.copy()
        self.global_mean_ = float(df_train[target_col].mean())
        self.maps_ = {}
        for c in self.cols:
            agg = df_train.groupby(c)[target_col].agg(count="count", mean="mean").reset_index()
            # smoothing parameter k derived from prior: larger prior -> more shrinkage
            k = 1.0 / max(self.prior, 1e-9)
            agg["smooth"] = (agg["count"] * agg["mean"] + k * self.global_mean_) / (agg["count"] + k)
            self.maps_[c] = dict(zip(agg[c], agg["smooth"]))

    def transform(self, df):
        """
        Return DataFrame with new columns: <col>_te
        """
        out = pd.DataFrame(index=df.index)
        for c in self.cols:
            mapping = self.maps_.get(c, {})
            out[c + "_te"] = df[c].map(mapping).fillna(self.global_mean_)
        return out