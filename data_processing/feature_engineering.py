import numpy as np
import pandas as pd
import logging
import warnings

from data_processing.settings import DataProcessingConfig as Config
from .non_standard_features import NonStandardFeatures

# Try TA libraries
try:
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.trend import MACD
    from ta.volatility import BollingerBands, AverageTrueRange
    _TA_AVAILABLE = True
except Exception:
    _TA_AVAILABLE = False


class FeatureEngineering:
    def __init__(self, fe_params: dict = None, logger: logging.Logger = None):
        self.fe_params = fe_params or {}
        self.logger = logger or logging.getLogger(__name__)
        self.config = Config()
        self.nonstd = NonStandardFeatures(params=self.fe_params.get("nonstd_params", {}))

    # ==================== TIMEFRAME HELPERS ====================
    def _timeframe_to_minutes(self, tf: str) -> int:
        tf = tf.upper().strip()
        if tf.endswith("T") or tf.endswith("MIN"):
            return int(tf[:-1]) if tf.endswith("T") else int(tf[:-3])
        elif tf.endswith("H"):
            return int(tf[:-1]) * 60
        elif tf.endswith("D"):
            return int(tf[:-1]) * 24 * 60
        elif tf.endswith("W"):
            return int(tf[:-1]) * 7 * 24 * 60
        else:
            try:
                return int(tf)
            except:
                return 60

    # ==================== MARKET REGIME DETECTION ====================
    def _detect_market_regime_series(
        self,
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
        bb = df.get(bb_width_col, pd.Series(np.nan, index=df.index))
        vol = df.get(volume_col, pd.Series(np.nan, index=df.index))
        ratio = df.get(ratio_col, pd.Series(np.nan, index=df.index))

        # Use rolling quantiles to adapt to market conditions
        atr_q = atr.rolling(window, min_periods=50).quantile(q)
        bb_q = bb.rolling(window, min_periods=50).quantile(q)
        vol_q = vol.rolling(window, min_periods=50).quantile(1 - q)   # low liquidity = bawah quantile
        ratio_q = ratio.rolling(window, min_periods=50).quantile(1 - q)

        # Calculate regime conditions
        high_vol = (atr > atr_q) | (bb > bb_q)
        low_liq = (vol < vol_q) | (ratio < ratio_q)

        regime[high_vol] = 1
        regime[low_liq] = -1
        return regime

    # ==================== CLEANUP ====================
    def _remove_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hapus semua kolom bertipe datetime supaya tidak bocor ke model ML"""
        if df is None or len(df) == 0:
            return df

        dt_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
        if dt_cols:
            self.logger.warning(f"[FE] Removing datetime columns: {dt_cols}")
            df = df.drop(columns=dt_cols)

        return df

    def _ensure_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hapus kolom non-numerik *kecuali* kolom inti OHLCV — OHLCV dipaksa konversi ke numeric."""
        if df is None or len(df) == 0:
            return df

        essential_cols = ["open", "high", "low", "close", "volume"]

        # 1) Coerce kolom inti ke numeric (errors='coerce' -> invalid -> NaN)
        for col in essential_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception as e:
                    self.logger.warning(f"[FE] Failed coercing column '{col}' to numeric: {e}")

        # 2) Temukan kolom non-numeric selain essential dan drop mereka
        non_numeric_cols = []
        for col in df.columns:
            if col in essential_cols:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_cols.append(col)

        if non_numeric_cols:
            self.logger.warning(f"[FE] Removing non-numeric columns (non-essential): {non_numeric_cols}")
            df = df.drop(columns=non_numeric_cols)

        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return df

        before_shape = df.shape

        # Drop kolom yang terlalu banyak NaN
        threshold = getattr(self.config, "nan_col_drop_threshold", 0.9)
        nan_ratio = df.isna().sum() / len(df)
        cols_to_drop = nan_ratio[nan_ratio > threshold].index.tolist()
        if cols_to_drop:
            self.logger.info(f"Dropping columns with >{threshold:.0%} NaN: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

        # Forward fill untuk indikator rolling dsb.
        df = df.ffill()

        # Drop bar kalau data harga inti hilang
        essential = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        if essential:
            df = df.dropna(subset=essential)
        else:
            df = df.dropna(how="all")

        after_shape = df.shape

        self.logger.info(f"[HandleMissing] Rows before={before_shape[0]}, after={after_shape[0]} (dropped={before_shape[0]-after_shape[0]})")
        nan_summary = df.isna().sum().sort_values(ascending=False)
        if not nan_summary.empty:
            self.logger.debug(f"[HandleMissing] Top NaN columns:\n{nan_summary.head(10)}")

        return df

    # ==================== BASE TIMEFRAME INDICATORS ====================
    def _technical_indicators(self, df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
        if not _TA_AVAILABLE:
            self.logger.warning("TA library not available, skipping technical indicators")
            return df
        try:
            df = df.copy()
            
            # SMA/EMA + Price-to-SMA/EMA (FIXED: No data leakage)
            for window in [10, 20, 50, 200]:
                df[f"sma_{window}{suffix}"] = df["close"].shift(1).rolling(window).mean()
                df[f"ema_{window}{suffix}"] = df["close"].shift(1).ewm(span=window, adjust=False).mean()
                df[f"price_to_sma_{window}{suffix}"] = df["close"] / df[f"sma_{window}{suffix}"]
                df[f"price_to_ema_{window}{suffix}"] = df["close"] / df[f"ema_{window}{suffix}"]

            # RSI (FIXED: No data leakage)
            df[f"rsi_14{suffix}"] = RSIIndicator(close=df["close"].shift(1), window=14).rsi()

            # MACD (FIXED: No data leakage)
            macd = MACD(close=df["close"].shift(1), window_slow=26, window_fast=12, window_sign=9)
            df[f"macd{suffix}"] = macd.macd()
            df[f"macd_signal{suffix}"] = macd.macd_signal()
            df[f"macd_diff{suffix}"] = macd.macd_diff()

            # Bollinger Bands (FIXED: No data leakage)
            bb = BollingerBands(close=df["close"].shift(1), window=20, window_dev=2)
            df[f"bb_high{suffix}"] = bb.bollinger_hband()
            df[f"bb_low{suffix}"] = bb.bollinger_lband()
            df[f"bb_mavg{suffix}"] = bb.bollinger_mavg()
            df[f"bb_width{suffix}"] = bb.bollinger_wband()

            # ATR (FIXED: No data leakage)
            atr = AverageTrueRange(high=df["high"].shift(1), low=df["low"].shift(1), close=df["close"].shift(1), window=14)
            df[f"atr{suffix}"] = atr.average_true_range()

            # Stochastic Oscillator (FIXED: No data leakage)
            stoch = StochasticOscillator(high=df["high"].shift(1), low=df["low"].shift(1), close=df["close"].shift(1), window=14, smooth_window=3)
            df[f"stoch_k{suffix}"] = stoch.stoch()
            df[f"stoch_d{suffix}"] = stoch.stoch_signal()

            return df
        except Exception as e:
            self.logger.error(f"Base indicator calculation failed: {e}")
            return df

    # ==================== HIGH TIMEFRAME (SCALED) ====================
    def _calculate_scaled_high_tf_indicators(self, base_df: pd.DataFrame, high_tf: str) -> pd.DataFrame:
        base_minutes = self._timeframe_to_minutes(self.config.main_timeframe)
        high_minutes = self._timeframe_to_minutes(high_tf)
        if high_minutes <= base_minutes:
            self.logger.warning(f"Skipping {high_tf}, not higher than base {self.config.main_timeframe}")
            return base_df
        ratio = max(1, high_minutes // base_minutes)
        suffix = f"_{high_tf}"
        df = base_df.copy()
        try:
            for window in [10, 20, 50]:
                df[f"sma_{window*ratio}{suffix}"] = df["close"].shift(1).rolling(window*ratio).mean()
                df[f"ema_{window*ratio}{suffix}"] = df["close"].shift(1).ewm(span=window*ratio, adjust=False).mean()
            
            # RSI with scaled window (FIXED: No data leakage)
            df[f"rsi_{14*ratio}{suffix}"] = RSIIndicator(close=df["close"].shift(1), window=14*ratio).rsi()
            
            # MACD with scaled windows (FIXED: No data leakage)
            macd = MACD(close=df["close"].shift(1), window_slow=26*ratio, window_fast=12*ratio, window_sign=9*ratio)
            df[f"macd{suffix}"] = macd.macd()
            df[f"macd_signal{suffix}"] = macd.macd_signal()
            df[f"macd_diff{suffix}"] = macd.macd_diff()
            
            # Bollinger Bands with scaled window (FIXED: No data leakage)
            bb = BollingerBands(close=df["close"].shift(1), window=20*ratio, window_dev=2)
            df[f"bb_high{suffix}"] = bb.bollinger_hband()
            df[f"bb_low{suffix}"] = bb.bollinger_lband()
            df[f"bb_mavg{suffix}"] = bb.bollinger_mavg()
            df[f"bb_width{suffix}"] = bb.bollinger_wband()
            
            # ATR with scaled window (FIXED: No data leakage)
            atr = AverageTrueRange(high=df["high"].shift(1), low=df["low"].shift(1), close=df["close"].shift(1), window=14*ratio)
            df[f"atr{suffix}"] = atr.average_true_range()
            
            # Stochastic with scaled window (FIXED: No data leakage)
            stoch = StochasticOscillator(high=df["high"].shift(1), low=df["low"].shift(1), close=df["close"].shift(1), window=14*ratio, smooth_window=3)
            df[f"stoch_k{suffix}"] = stoch.stoch()
            df[f"stoch_d{suffix}"] = stoch.stoch_signal()

            self.logger.info(f"✅ Added scaled indicators for {high_tf} (ratio={ratio})")
            return df
        except Exception as e:
            self.logger.error(f"High TF scaled indicator calc failed for {high_tf}: {e}")
            return df

    # ==================== MAIN ENTRY ====================
    def create_features(self, df: pd.DataFrame, high_timeframes=None) -> pd.DataFrame:
        if df is None or len(df) == 0:
            self.logger.error("Empty dataframe for feature creation")
            return df
        df = df.copy()
        self.logger.info(f"[FE] start create_features: rows={len(df)}, cols={len(df.columns)}")

        # Add base technical indicators
        df = self._technical_indicators(df)
        
        # Add high timeframe scaled indicators if configured
        if high_timeframes and getattr(self.config, "use_scaled_high_tf", True):
            for tf in high_timeframes:
                df = self._calculate_scaled_high_tf_indicators(df, tf)

        # Add non-standard features
        try:
            df_nonstd = self.nonstd.add_features(df)
            if df_nonstd is not None:
                df = df_nonstd
        except Exception as e:
            self.logger.warning(f"Non-standard features failed: {e}")

        # Add market regime detection
        try:
            regime_series = self._detect_market_regime_series_quantile(
                df,
                atr_col="atr",
                bb_width_col="bb_width",
                volume_col="volume",
                ratio_col="price_volume_ratio",
                window=getattr(self.config, "regime_window", 252),
                q=getattr(self.config, "regime_quantile", 0.7),
            )
            df["market_regime"] = regime_series.astype("int8")
        except Exception as e:
            self.logger.warning(f"Market regime detection failed: {e}")
            df["market_regime"] = 0

        # Final cleanup
        df = self._ensure_numeric_columns(df)
        df = self._handle_missing(df)
        df = self._remove_datetime_columns(df)

        self.logger.info(f"[FE] final features: rows={len(df)}, cols={len(df.columns)}")
        return df