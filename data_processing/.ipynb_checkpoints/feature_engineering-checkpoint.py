import pandas as pd
import numpy as np
import ta
from typing import Optional
import logging

class FeatureEngineer:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def calculate_atr_manual(self, high: pd.Series, low: pd.Series, 
                           close: pd.Series, window: int = 14) -> pd.Series:
        """Manual ATR calculation"""
        high = pd.to_numeric(high, errors='coerce')
        low = pd.to_numeric(low, errors='coerce')
        close = pd.to_numeric(close, errors='coerce')

        prev_close = close.shift(1)
        
        hl = (high - low).to_numpy(dtype=np.float64)
        hc = (high - prev_close).abs().to_numpy(dtype=np.float64)
        lc = (low - prev_close).abs().to_numpy(dtype=np.float64)

        tr_arr = np.maximum(np.maximum(hl, hc), lc)
        tr = pd.Series(tr_arr, index=high.index)
        tr.replace([np.inf, -np.inf], np.nan, inplace=True)

        atr = tr.rolling(window=window, min_periods=1).mean()
        return atr
    
    def add_technical_indicators(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Add comprehensive technical indicators"""
        df = data.copy()
        
        if len(df) < 50:
            self.logger.warning(f"Insufficient data ({len(df)} rows) for reliable indicators")
            return None

        # Momentum Indicators
        self.logger.info("Adding momentum indicators...")
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        # Trend Indicators
        self.logger.info("Adding trend indicators...")
        df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
        df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        df['EMA_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
        df['EMA_26'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()
        
        # Volatility Indicators
        self.logger.info("Adding volatility indicators...")
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_high'] = bb.bollinger_hband()
        df['BB_low'] = bb.bollinger_lband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_width'] = (df['BB_high'] - df['BB_low']) / df['BB_middle'].replace(0, np.nan)
        
        # MACD
        self.logger.info("Adding MACD...")
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_hist'] = macd.macd_diff()
        
        # Stochastic
        self.logger.info("Adding Stochastic...")
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # ATR and Price Action
        self.logger.info("Adding ATR and price action features...")
        df['ATR'] = self.calculate_atr_manual(df['High'], df['Low'], df['Close'], window=14)
        df['Price_change'] = df['Close'].pct_change()
        df['High_Low_ratio'] = (df['High'] - df['Low']) / df['Close'].replace(0, np.nan)
        
        # Additional features
        self.logger.info("Adding additional features...")
        df['volatility_5'] = df['Close'].pct_change().rolling(5).std()
        df['volatility_20'] = df['Close'].pct_change().rolling(20).std()
        df['RSI_MA'] = df['RSI'].rolling(5).mean()
        df['MACD_slope'] = df['MACD'].diff(3)
        df['body_ratio'] = (df['Close'] - df['Open']).abs() / (df['High'] - df['Low']).replace(0, 0.001)
        df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, 0.001)
        df['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['volume_zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std().replace(0, 0.001)
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Drop NaN values
        initial_count = len(df)
        df = df.dropna()
        dropped_count = initial_count - len(df)
        
        if dropped_count > 0:
            self.logger.info(f"Dropped {dropped_count} rows with NaN values")
        
        self.logger.info(f"Final dataset with indicators: {df.shape}")
        return df