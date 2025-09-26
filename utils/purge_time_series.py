import numpy as np
import pandas as pd
from typing import Union


class PurgedTimeSeriesSplit:
    def __init__(self, n_splits: int = 5, gap_ratio: float = 0.05):
        self.n_splits = int(n_splits)
        self.gap_ratio = float(gap_ratio)

    def split(self, X: Union[pd.DataFrame, np.ndarray]):
        n_samples = len(X)
        gap_size = int(n_samples * self.gap_ratio)
        fold_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            train_indices = np.arange(0, train_end)

            gap_end = min(train_end + gap_size, n_samples)
            val_start = gap_end
            val_end = min(val_start + fold_size, n_samples)

            if val_start >= n_samples:
                break

            val_indices = np.arange(val_start, val_end)

            if len(val_indices) > 0:
                yield train_indices, val_indices