import numpy as np
import pandas as pd
from config.settings import Config

class OutlierDetector:
    def __init__(self, logger):
        self.logger = logger

    def detect_outliers_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and remove outliers column by column using Z-score method.
        This version is memory-efficient (slower but safer on low-RAM systems).
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.logger.info(f"Detecting outliers for {len(numeric_cols)} numeric columns...")

        # Proses per kolom, bukan semua sekaligus
        clean_df = df.copy()
        for col in numeric_cols:
            col_mean = clean_df[col].mean()
            col_std = clean_df[col].std()
            if col_std == 0 or np.isnan(col_std):
                continue  # skip kalau std = 0 atau NaN

            z_scores = (clean_df[col] - col_mean) / col_std
            before_len = len(clean_df)
            clean_df = clean_df[z_scores.abs() < Config.Z_SCORE_THRESHOLD]
            after_len = len(clean_df)

            removed = before_len - after_len
            if removed > 0:
                self.logger.debug(f"Removed {removed} outliers from column {col}")

        self.logger.info(f"Outlier detection completed. Final rows: {len(clean_df)}")
        return clean_df