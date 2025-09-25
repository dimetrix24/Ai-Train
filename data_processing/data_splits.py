import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ==============================================================
# Imbalance handling functions
# ==============================================================
def smart_undersample(X, y, logger=None):
    counts = y.value_counts()
    if len(counts) < 2:
        return X, y
    
    minority_class = counts.idxmin()
    minority_size = counts.min()
    
    balanced_idx = (
        y[y == minority_class].index.tolist() +
        y[y != minority_class].sample(minority_size, random_state=42).index.tolist()
    )
    
    if logger:
        logger.info(
            f"Smart undersample: before={len(y)}, after={len(balanced_idx)}, "
            f"class distribution={y.loc[balanced_idx].value_counts().to_dict()}"
        )
    return X.loc[balanced_idx], y.loc[balanced_idx]


def random_undersample(X, y, logger=None):
    counts = y.value_counts()
    min_size = counts.min()
    sampled_idx = []
    
    for cls in counts.index:
        sampled_idx.extend(y[y == cls].sample(min_size, random_state=42).index.tolist())
    
    if logger:
        logger.info(f"Random undersample: target size per class={min_size}")
    
    return X.loc[sampled_idx], y.loc[sampled_idx]


def random_oversample(X, y, logger=None):
    counts = y.value_counts()
    max_size = counts.max()
    sampled_idx = []
    
    for cls in counts.index:
        idx = y[y == cls].index
        if len(idx) < max_size:
            idx = np.random.choice(idx, max_size, replace=True)
        sampled_idx.extend(idx)
    
    if logger:
        logger.info(f"Random oversample: target size per class={max_size}")
    
    return X.loc[sampled_idx], y.loc[sampled_idx]


# ==============================================================
# Data splitting functions
# ==============================================================
def split_by_pct(df, test_size=0.2, valid_size=0.1):
    n = len(df)
    test_idx = int(n * (1 - test_size))
    valid_idx = int(test_idx * (1 - valid_size))

    train_df = df.iloc[:valid_idx]
    valid_df = df.iloc[valid_idx:test_idx]
    test_df = df.iloc[test_idx:]
    
    return train_df, valid_df, test_df


def expanding_window(df, train_size=0.6, valid_size=0.2, test_size=0.2):
    n = len(df)
    train_end = int(n * train_size)
    valid_end = train_end + int(n * valid_size)
    test_end = valid_end + int(n * test_size)

    return df.iloc[:train_end], df.iloc[train_end:valid_end], df.iloc[valid_end:test_end]


def rolling_window(df, window_size=1000, step=500):
    n = len(df)
    for start in range(0, n - window_size, step):
        end = start + window_size
        yield df.iloc[start:end]


# ==============================================================
# DataSplitter Class (patched anti-data-leak)
# ==============================================================
class DataSplitter:
    def __init__(self, method="timeseries", imbalance_method="smart_undersample", logger=None):
        self.method = method
        self.imbalance_method = imbalance_method
        self.logger = logger or logging.getLogger(__name__)

    def split(self, df, target_col=None):
        """
        Split dataframe menjadi train, valid, test.

        - Jika target_col=None → raw split (train_df, valid_df, test_df, None, None, None)
        - Jika target_col ada → supervised split (X_train, X_valid, X_test, y_train, y_valid, y_test)
        """
        # === Case 1: Raw split (anti-leakage) ===
        if target_col is None:
            if self.method == "timeseries":
                train_df, valid_df, test_df = split_by_pct(df)
            elif self.method == "random":
                train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
                train_df, valid_df = train_test_split(train_df, test_size=0.1, random_state=42, shuffle=True)
            elif self.method == "rolling":
                windows = list(rolling_window(df))
                if len(windows) < 3:
                    raise ValueError("Not enough windows for rolling split")
                train_df = pd.concat(windows[:-2])
                valid_df = windows[-2]
                test_df = windows[-1]
            elif self.method == "none":
                train_df, valid_df, test_df = df, df.iloc[0:0], df.iloc[0:0]
            else:
                raise ValueError(f"Unknown split method: {self.method}")

            self.logger.info(f"Raw split completed: train={train_df.shape}, valid={valid_df.shape}, test={test_df.shape}")
            return train_df, valid_df, test_df, None, None, None

        # === Case 2: Supervised split ===
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in dataframe")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        if self.method == "timeseries":
            train_df, valid_df, test_df = split_by_pct(df)
            X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
            X_valid, y_valid = valid_df.drop(columns=[target_col]), valid_df[target_col]
            X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]

        elif self.method == "random":
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=True
            )
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_temp, y_temp, test_size=0.1, random_state=42, shuffle=True
            )

        elif self.method == "rolling":
            windows = list(rolling_window(df))
            if len(windows) < 3:
                raise ValueError("Not enough windows for rolling split")
            train_df = pd.concat(windows[:-2])
            valid_df = windows[-2]
            test_df = windows[-1]
            X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
            X_valid, y_valid = valid_df.drop(columns=[target_col]), valid_df[target_col]
            X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]

        elif self.method == "none":
            X_train, y_train = X, y
            X_valid, y_valid = X.iloc[0:0], y.iloc[0:0]
            X_test, y_test = X.iloc[0:0], y.iloc[0:0]

        else:
            raise ValueError(f"Unknown split method: {self.method}")

        # balancing hanya training
        X_train, y_train = self._apply_balance(X_train, y_train)

        self.logger.info(
            f"Supervised split completed: X_train={X_train.shape}, X_valid={X_valid.shape}, X_test={X_test.shape}, "
            f"imbalance_method={self.imbalance_method}, split_method={self.method}"
        )
        
        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def _apply_balance(self, X, y):
        if self.imbalance_method == "smart_undersample":
            return smart_undersample(X, y, self.logger)
        elif self.imbalance_method == "undersample":
            return random_undersample(X, y, self.logger)
        elif self.imbalance_method == "oversample":
            return random_oversample(X, y, self.logger)
        elif self.imbalance_method == "none":
            self.logger.info("No imbalance handling applied")
            return X, y
        else:
            raise ValueError(f"Unknown imbalance method: {self.imbalance_method}")