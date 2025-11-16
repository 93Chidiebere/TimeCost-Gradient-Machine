"""
Utility functions: time-based splits and helpers.
"""

import pandas as pd
import numpy as np

def time_walk_forward_splits(df, time_col="timestamp", min_train_days=180, test_days=30, n_splits=5):
    """
    Create walk-forward train/test index pairs (list of (train_idx, test_idx))
    df: DataFrame with a time column
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    times = pd.to_datetime(df[time_col])
    start = times.min() + pd.Timedelta(days=min_train_days)
    splits = []
    for i in range(n_splits):
        train_end = start + pd.Timedelta(days=i * test_days)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=test_days - 1)
        train_idx = df.index[times <= train_end].tolist()
        test_idx = df.index[(times >= test_start) & (times <= test_end)].tolist()
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits