import os
import json
import joblib
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_processing.data_loader import DataLoader
from feature_engineering import FeatureEngineering
from session_filter import SessionFilter
from config.settings import Config # Assuming config.settings contains TB parameters


class Backtester:
    def __init__(
        self,
        model_path=None,
        fe_params_path=None,
        tb_params_path=None,
        data_dir=None,
        output_dir="outputs",
        logger=None,
        timeframe="M1",
        sessions=None,
    ):
        """
        Backtester inti untuk load model, data, apply feature engineering, dan evaluasi.
        """
        self.model_path = model_path
        self.fe_params_path = fe_params_path
        self.tb_params_path = tb_params_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger("Backtester")
        self.timeframe = timeframe
        self.sessions = sessions

        os.makedirs(self.output_dir, exist_ok=True)

        self.model = None
        self.feature_engineer = None
        self.tb_params = {}

        self.logger.info("Backtester initialized")
        self.logger.info(f"Model path: {self.model_path}")
        self.logger.info(f"FE params path: {self.fe_params_path}")
        self.logger.info(f"TB params path: {self.tb_params_path}")
        self.logger.info(f"Data dir: {self.data_dir}")
        self.logger.info(f"Output dir: {self.output_dir}")
        self.logger.info(f"Timeframe: {self.timeframe}")
        self.logger.info(f"Sessions: {self.sessions}")

        # Auto load model & FE params & TB params
        if self.model_path and os.path.exists(self.model_path):
            self._load_model()
        if self.fe_params_path and os.path.exists(self.fe_params_path):
            self._load_fe_params()
        if self.tb_params_path and os.path.exists(self.tb_params_path):
            self._load_tb_params()
        else:
            # Fallback to config.settings if tb_params_path not provided or file doesn't exist
            self._load_tb_from_Config()

    # =====================================================
    # Loaders
    # =====================================================
    def _load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            self.logger.info(f"✅ Model loaded dari {self.model_path}")
        except Exception as e:
            self.logger.error(f"❌ Gagal load model {self.model_path}: {e}")
            raise

    def _load_fe_params(self):
        try:
            with open(self.fe_params_path, "r") as f:
                params = json.load(f)
            self.feature_engineer = FeatureEngineering(params=params, logger=self.logger)
            self.logger.info(f"✅ FE params loaded dari {self.fe_params_path}")
        except Exception as e:
            self.logger.error(f"❌ Gagal load FE params {self.fe_params_path}: {e}")
            raise

    def _load_tb_params(self):
        try:
            with open(self.tb_params_path, "r") as f:
                self.tb_params = json.load(f)
            self.logger.info(f"✅ TB params loaded dari {self.tb_params_path}")
        except Exception as e:
            self.logger.error(f"❌ Gagal load TB params {self.tb_params_path}: {e}")
            raise

    def _load_tb_from_Config(self):
        """Load triple barrier parameters from config.settings as fallback."""
        try:
            self.tb_params = {
                'vertical_barrier': getattr(Config, 'VERTICAL_BARRIER', 10),
                'profit_take': getattr(Config, 'PROFIT_TAKE', 0.02),
                'stop_loss': getattr(Config, 'STOP_LOSS', -0.01),
            }
            self.logger.info(f"✅ TB params loaded dari config.settings: {self.tb_params}")
        except Exception as e:
            self.logger.error(f"❌ Gagal load TB params dari config.settings: {e}")
            # Use hard-coded defaults if settings also fails
            self.tb_params = {'vertical_barrier': 10, 'profit_take': 0.02, 'stop_loss': -0.01}
            self.logger.warning("Using hard-coded defaults for TB params")

    def load_data(self):
        self.logger.info("Menggunakan fallback data loading...")
        data = self._load_data_fallback()
        
        if data is None or data.empty:
            self.logger.error("❌ Data tidak valid atau kosong setelah loading")
            return None

        if self.sessions:
            sf = SessionFilter(sessions=self.sessions, logger=self.logger)
            data = sf.apply(data)

        return data

    # =====================================================
    # Feature Engineering
    # =====================================================
    def apply_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_engineer:
            self.feature_engineer = FeatureEngineering(logger=self.logger)
        features = self.feature_engineer.create_features(data)
        return features

    # =====================================================
    # Triple Barrier Labeling
    # =====================================================
    def apply_triple_barrier(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply triple barrier labeling to the data.
        Labels: 1 (profit barrier hit), -1 (stop loss barrier hit), 0 (time barrier hit first).
        Parameters from tb_params (loaded from JSON or config.settings).
        """
        params = self.tb_params
        if not params:
            self.logger.warning("No TB params available, using defaults")
            params = {'vertical_barrier': 10, 'profit_take': 0.02, 'stop_loss': -0.01}

        vertical_barrier = params.get('vertical_barrier', 10)  # Number of periods for time barrier
        profit_take = params.get('profit_take', 0.02)  # Profit target (e.g., 2%)
        stop_loss = params.get('stop_loss', -0.01)  # Stop loss (e.g., -1%)

        self.logger.info(f"Applying triple barrier with params: vertical_barrier={vertical_barrier}, "
                         f"profit_take={profit_take}, stop_loss={stop_loss}")

        if 'close' not in data.columns or 'high' not in data.columns or 'low' not in data.columns:
            self.logger.error("Data must have 'close', 'high', 'low' columns for triple barrier")
            raise ValueError("Missing required OHLC columns")

        labels = pd.Series(index=data.index, dtype=int)
        n = len(data)

        for i in range(n):
            entry_price = data['close'].iloc[i]
            upper = entry_price * (1 + profit_take)
            lower = entry_price * (1 + stop_loss)

            hit_upper = False
            hit_lower = False

            end_idx = min(i + vertical_barrier, n)
            for j in range(i + 1, end_idx):
                high_j = data['high'].iloc[j]
                low_j = data['low'].iloc[j]

                if high_j >= upper:
                    labels.iloc[i] = 1
                    hit_upper = True
                    break
                if low_j <= lower:
                    labels.iloc[i] = -1
                    hit_lower = True
                    break

            if not hit_upper and not hit_lower:
                labels.iloc[i] = 0

        data = data.copy()
        data['label'] = labels
        self.logger.info(f"Triple barrier labels applied: {labels.value_counts().to_dict()}")
        return data

    # =====================================================
    # Backtest
    # =====================================================
    def run_backtest(self, data: pd.DataFrame, signals: pd.Series):
        """Jalankan backtest dengan equity tracking + DD + Sharpe"""
        if not data.index.equals(signals.index):
            self.logger.warning("Data and signals index don't match, aligning...")
            common_index = data.index.intersection(signals.index)
            data = data.loc[common_index]
            signals = signals.loc[common_index]
        
        if len(data) == 0 or len(signals) == 0:
            self.logger.error("No common data points between data and signals")
            return {"error": "No common data points"}

        equity = 10000
        balance_curve = [equity]
        position = 0
        returns = []

        for i in range(1, len(signals)):
            sig = signals.iloc[i]
            price_prev = data["close"].iloc[i - 1]
            price_now = data["close"].iloc[i]

            if sig == 1:  # BUY
                position = 1
            elif sig == -1:  # SELL
                position = -1
            elif sig == 0:  # FLAT
                position = 0

            pnl = (price_now - price_prev) * position * 10000
            equity += pnl
            balance_curve.append(equity)
            returns.append(pnl / equity if equity != 0 else 0)

        balance_curve = np.array(balance_curve)

        running_max = np.maximum.accumulate(balance_curve)
        drawdowns = (balance_curve - running_max) / running_max
        max_drawdown = drawdowns.min() if len(drawdowns) > 0 else 0

        sharpe_ratio = (
            np.mean(returns) / np.std(returns) * np.sqrt(252)
            if len(returns) > 1 and np.std(returns) > 0
            else 0
        )

        results = {
            "final_equity": float(balance_curve[-1]),
            "total_pnl": float(balance_curve[-1] - 10000),
            "max_equity": float(balance_curve.max()),
            "min_equity": float(balance_curve.min()),
            "num_trades": int((signals != 0).sum()),
            "max_drawdown": float(max_drawdown),
            "sharpe_ratio": float(sharpe_ratio),
        }
        return results

    # =====================================================
    # Saving Results
    # =====================================================
    def save_results(self, results: dict):
        if "error" in results:
            self.logger.error(f"Tidak bisa simpan hasil backtest: {results['error']}")
            return

        results_path = os.path.join(self.output_dir, "backtest_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        eq_path = os.path.join(self.output_dir, "equity_curve.csv")
        pd.DataFrame({"equity": [results["final_equity"]]}).to_csv(eq_path, index=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot([results["final_equity"]], marker="o", markersize=8)
        ax.set_title("Final Equity Value")
        ax.set_ylabel("Equity")
        ax.grid(True)
        fig.savefig(os.path.join(self.output_dir, "equity_curve.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

        self.logger.info(f"📊 Hasil backtest disimpan di {self.output_dir}")

    # =====================================================
    # Fallback Data Loading
    # =====================================================
    def _load_data_fallback(self):
        """Fallback method untuk load data jika DataLoader gagal"""
        try:
            csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            if not csv_files:
                self.logger.error("No CSV files found")
                return None

            dfs = []
            for csv_file in csv_files:
                filepath = os.path.join(self.data_dir, csv_file)
                try:
                    df = pd.read_csv(filepath)

                    if df.shape[1] == 1:
                        df = pd.read_csv(filepath, sep="\t")

                    if not any(col.lower() in ["datetime", "date", "time"] for col in df.columns):
                        self.logger.warning(f"File {csv_file} tidak punya header standar, paksa header=None")
                        df = pd.read_csv(
                            filepath,
                            header=None,
                            names=["datetime", "open", "high", "low", "close", "volume"],
                        )

                    self.logger.info(f"Loaded {csv_file} with columns: {df.columns.tolist()}")

                    datetime_col = None
                    if "datetime" in df.columns:
                        datetime_col = "datetime"
                    elif "date" in df.columns and "time" in df.columns:
                        df["datetime"] = pd.to_datetime(
                            df["date"].astype(str) + " " + df["time"].astype(str),
                            errors="coerce",
                        )
                        datetime_col = "datetime"

                    if datetime_col:
                        df["datetime_parsed"] = pd.to_datetime(df[datetime_col], errors="coerce")
                        df = df[df["datetime_parsed"].notna()]
                        df = df.set_index("datetime_parsed")
                        df = df.drop(columns=[datetime_col], errors="ignore")
                    else:
                        self.logger.error(f"Tidak ada kolom datetime pada {csv_file}, skip file ini")
                        continue

                    dfs.append(df)
                    self.logger.info(f"Processed {csv_file} dengan shape {df.shape}")

                except Exception as e:
                    self.logger.error(f"Failed to load {csv_file}: {e}")

            if not dfs:
                return None

            data = pd.concat(dfs)

            if isinstance(data.index, pd.DatetimeIndex):
                data = data.sort_index()
                self.logger.info("Data sorted by datetime index")

            self.logger.info(f"Combined data shape: {data.shape}")
            self.logger.info(f"Data columns: {data.columns.tolist()}")
            self.logger.info(f"Data index type: {type(data.index)}")

            return data

        except Exception as e:
            self.logger.error(f"Fallback loading failed: {e}")
            return None