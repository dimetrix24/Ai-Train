import logging
import numpy as np
import pandas as pd

def detect_market_regime_series(
    df: pd.DataFrame,
    atr_col: str = "atr",
    bb_width_col: str = "bb_width",
    volume_col: str = "volume",
    ratio_col: str = "price_volume_ratio",
    window: int = 252,
    q: float = 0.7
) -> pd.Series:
    """
    Adaptif market regime detection pakai rolling quantile.
      1 -> high_volatility
     -1 -> low_liquidity
      0 -> normal
    """
    regime = pd.Series(0, index=df.index, dtype="int8")
    if df.empty:
        return regime

    atr = df.get(atr_col, pd.Series(np.nan, index=df.index))
    bb  = df.get(bb_width_col, pd.Series(np.nan, index=df.index))
    vol = df.get(volume_col, pd.Series(np.nan, index=df.index))
    ratio = df.get(ratio_col, pd.Series(np.nan, index=df.index))

    atr_q = atr.rolling(window, min_periods=50).quantile(q)
    bb_q  = bb.rolling(window, min_periods=50).quantile(q)
    vol_q = vol.rolling(window, min_periods=50).quantile(1 - q)   # low liquidity = bawah quantile
    ratio_q = ratio.rolling(window, min_periods=50).quantile(1 - q)

    high_vol = (atr > atr_q) | (bb > bb_q)
    low_liq  = (vol < vol_q) | (ratio < ratio_q)

    regime[high_vol] = 1
    regime[low_liq] = -1
    return regime