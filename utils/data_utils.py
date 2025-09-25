import logging
import pandas as pd

def drop_residual_leakage(datasets, logger: logging.Logger = None):
    """
    Drop kolom yang tidak boleh bocor antar split dataset (contoh: 'future_return').
    """
    logger = logger or logging.getLogger(__name__)
    clean_sets = []
    for i, df in enumerate(datasets):
        if "future_return" in df.columns:
            df = df.drop(columns=["future_return"])
            logger.info(f"🧹 future_return dropped dari dataset {i}")
        clean_sets.append(df)
    return clean_sets


def check_and_drop_high_corr(datasets, logger: logging.Logger = None, threshold: float = 0.99):
    """
    Cek korelasi fitur dan drop kolom yang terlalu berkorelasi.
    """
    logger = logger or logging.getLogger(__name__)
    df_train = datasets[0]

    corr_matrix = df_train.corr().abs()
    upper = corr_matrix.where(
        pd.np.triu(pd.np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    if to_drop:
        logger.info(f"⚠️ Drop highly correlated features: {to_drop}")
        clean_sets = [df.drop(columns=to_drop, errors="ignore") for df in datasets]
        return clean_sets
    else:
        logger.info("✅ Tidak ada fitur dengan korelasi tinggi")
        return datasets