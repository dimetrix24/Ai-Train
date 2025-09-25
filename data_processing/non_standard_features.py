import logging
import numpy as np
import polars as pl
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)

class NonStandardFeatures:
    """
    Fitur Non-Standar untuk ML (versi Polars Hybrid):
    - Price/Volume ratio
    - Divergence berbasis RSI (pakai EWM pandas fallback)
    - Heikin-Ashi
    - Ichimoku Cloud (pakai rolling high/low)
    - Fibonacci Levels (pakai rolling high/low)
    """

    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        self.use_ichimoku = self.params.get("use_ichimoku", True)
        self.enable_heikin = self.params.get("enable_heikin", True)
        self.use_fibonacci = self.params.get("use_fibonacci", True)
        self.use_divergence = self.params.get("use_divergence", False)

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline: input pandas → convert ke Polars → features → balik ke pandas"""
        pl_df = pl.from_pandas(df)
        logger.info("=== Generating Non-Standard Features (Polars Hybrid) ===")

        pl_df = self.price_volume_ratio(pl_df)

        if self.use_divergence:
            pl_df = self.detect_divergence(pl_df)

        if self.enable_heikin:
            pl_df = self.heikin_features(pl_df)

        if self.use_ichimoku:
            pl_df = self.ichimoku_features(pl_df)

        if self.use_fibonacci:
            pl_df = self.fibonacci_features(pl_df)

        final_df = pl_df.to_pandas()
        logger.info(f"Non-standard features added. Shape: {final_df.shape}")
        return final_df

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper agar kompatibel dengan feature_engineering.py"""
        return self.generate(df)

    # --- Base Features ---
    def price_volume_ratio(self, df: pl.DataFrame) -> pl.DataFrame:
        if all(col in df.columns for col in ["close", "volume"]):
            df = df.with_columns((df["close"] / (df["volume"] + 1e-9)).alias("price_volume_ratio"))
        return df

    # --- RSI Divergence (pakai EWM pandas fallback) ---
    def rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        avg_gain = pd.Series(gain, index=series.index).ewm(span=window, adjust=False).mean()
        avg_loss = pd.Series(loss, index=series.index).ewm(span=window, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def detect_divergence(self, df: pl.DataFrame) -> pl.DataFrame:
        if "close" not in df.columns:
            return df.with_columns(pl.lit(0).alias("divergence_signal"))

        pdf = df.to_pandas()
        window = self.params.get("divergence_window", 14)
        pdf["RSI"] = self.rsi(pdf["close"], window)

        pdf["divergence_signal"] = 0
        pdf.loc[
            (pdf["close"] < pdf["close"].shift(window)) & 
            (pdf["RSI"] > pdf["RSI"].shift(window)), "divergence_signal"
        ] = 1
        pdf.loc[
            (pdf["close"] > pdf["close"].shift(window)) & 
            (pdf["RSI"] < pdf["RSI"].shift(window)), "divergence_signal"
        ] = -1

        return pl.from_pandas(pdf)

    # --- Heikin Ashi ---
    def heikin_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if not all(c in df.columns for c in ["open", "high", "low", "close"]):
            return df

        pdf = df.to_pandas()
        ha_df = pdf.copy()

        ha_df["HA_Close"] = (pdf["open"] + pdf["high"] + pdf["low"] + pdf["close"]) / 4
        ha_df["HA_Open"] = (pdf["open"].iloc[0] + pdf["close"].iloc[0]) / 2
        for i in range(1, len(pdf)):
            ha_df.loc[i, "HA_Open"] = (ha_df["HA_Open"].iloc[i-1] + ha_df["HA_Close"].iloc[i-1]) / 2

        ha_df["HA_High"] = ha_df[["high", "HA_Open", "HA_Close"]].max(axis=1)
        ha_df["HA_Low"] = ha_df[["low", "HA_Open", "HA_Close"]].min(axis=1)
        ha_df["HA_trend"] = np.where(
            ha_df["HA_Close"] > ha_df["HA_Open"], 1,
            np.where(ha_df["HA_Close"] < ha_df["HA_Open"], -1, 0)
        )

        merged = pd.concat([pdf, ha_df[["HA_Open", "HA_High", "HA_Low", "HA_Close", "HA_trend"]]], axis=1)
        return pl.from_pandas(merged)

    # --- Ichimoku (pakai rolling) ---
    def ichimoku_features(self, df: pl.DataFrame) -> pl.DataFrame:
        conv = self.params.get("ichimoku_conversion", 9)
        base = self.params.get("ichimoku_base", 26)
        span_b = self.params.get("ichimoku_leading_span_b", 52)

        if not all(c in df.columns for c in ["high", "low"]):
            return df

        pdf = df.to_pandas()
        high, low = pdf["high"], pdf["low"]

        pdf["ichimoku_conversion"] = (high.rolling(conv).max() + low.rolling(conv).min()) / 2
        pdf["ichimoku_base"] = (high.rolling(base).max() + low.rolling(base).min()) / 2
        pdf["ichimoku_span_a"] = (pdf["ichimoku_conversion"] + pdf["ichimoku_base"]) / 2
        pdf["ichimoku_span_b"] = (high.rolling(span_b).max() + low.rolling(span_b).min()) / 2

        return pl.from_pandas(pdf)

    # --- Fibonacci (pakai rolling high/low) ---
    def fibonacci_features(self, df: pl.DataFrame) -> pl.DataFrame:
        lookback = self.params.get("fibonacci_lookback", 100)
        if not all(c in df.columns for c in ["high", "low"]):
            return df

        pdf = df.to_pandas()
        pdf["fib_high"] = pdf["high"].rolling(lookback).max()
        pdf["fib_low"] = pdf["low"].rolling(lookback).min()
        rng = pdf["fib_high"] - pdf["fib_low"]

        for lvl, ratio in [("236", 0.236), ("382", 0.382), ("500", 0.5),
                           ("618", 0.618), ("786", 0.786)]:
            pdf[f"fib_{lvl}"] = pdf["fib_high"] - rng * ratio

        return pl.from_pandas(pdf)