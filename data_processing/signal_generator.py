import pandas as pd
import numpy as np
import logging
import os
import json
from typing import Optional, Dict, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import optuna
from data_processing.settings import DataProcessingConfig

# Tentukan root project = folder tempat main_train_multi.py berada
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")

class SignalGenerator:
	    
    PARAMS_FILE = os.path.join(PARAMS_DIR, "best_signal_params.json")
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        mode: str = None,
        future_bars: int = None,
        threshold: float = None,
        dynamic_threshold: Optional[bool] = None,
        use_triple_barrier: Optional[bool] = None,
        barrier_params: Optional[dict] = None,
        enable_tuning: bool = False,
        n_trials: int = 50,
        label_only_base_timeframe: bool = False,
        base_timeframe_column: str = "is_base_timeframe"
    ):
        # Config instance
        self.config = DataProcessingConfig()

        # Logging
        self.logger = logger or logging.getLogger(__name__)

        # Labeling mode
        self.mode = mode or self.config.label_mode

        # Params
        self.future_bars = future_bars or self.config.label_future_bars
        self.threshold = float(threshold) if threshold is not None else float(self.config.signal_threshold)
        self.dynamic_threshold = dynamic_threshold if dynamic_threshold is not None else self.config.signal_dynamic
        self.enable_tuning = enable_tuning
        self.n_trials = n_trials
        self.best_params = {}

        # Triple barrier settings
        if use_triple_barrier is not None:
            self.use_triple_barrier = use_triple_barrier
        else:
            self.use_triple_barrier = self.config.signal_triple_barrier

        default_barrier = self.config.triple_barrier_params or {}
        self.barrier_params = {**default_barrier, **(barrier_params or {})}
        self.barrier_params.setdefault("use_atr", False)
        self.barrier_params.setdefault("tp", 0.0005)
        self.barrier_params.setdefault("sl", 0.0005)
        self.barrier_params.setdefault("tp_atr_mult", 1.0)
        self.barrier_params.setdefault("sl_atr_mult", 1.0)
        self.barrier_params.setdefault("min_spread", 0.0)

        # Base timeframe labeling option
        self.label_only_base_timeframe = label_only_base_timeframe
        self.base_timeframe_column = base_timeframe_column

    def _save_best_params(self, data: Dict[str, Any]):
        """Simpan parameter terbaik + skor ke file JSON"""
        try:
            with open(self.PARAMS_FILE, "w") as f:
                json.dump(data, f, indent=4)
            self.logger.info(f"💾 Best params updated: {self.PARAMS_FILE}")
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan best params: {e}")

    def _load_best_params(self) -> Dict[str, Any]:
        """Load best params dari JSON (jika ada)"""
        try:
            with open(self.PARAMS_FILE, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal load best params: {e}")
            return {}

    def tune_parameters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Tuning parameter + simpan kalau lebih baik dari sebelumnya"""
        self.logger.info("🔧 Mulai parameter tuning...")

        if self.use_triple_barrier:
            best_params = self._tune_triple_barrier_parameters(df)
        else:
            best_params = self._tune_threshold_parameters(df)

        # Ambil skor terbaru dari optuna
        best_score = None
        if hasattr(self, "last_study") and self.last_study is not None:
            best_score = self.last_study.best_value

        # Ambil skor lama
        old_data = self._load_best_params()
        old_score = old_data.get("best_score", -1)

        if best_score is None:
            self.logger.warning("⚠️ Tidak ada best_score dari tuning, pakai default params.")
            self.best_params = best_params
            return best_params

        # Bandingkan skor
        if best_score > old_score:
            save_data = {
                "best_score": best_score,
                "params": best_params
            }
            self._save_best_params(save_data)
            self.best_params = best_params
            self.logger.info(f"✅ New best params disimpan (score {best_score:.4f} > {old_score:.4f})")
        else:
            self.logger.info(f"⚠️ Skor baru {best_score:.4f} <= lama {old_score:.4f}, keep old params")
            self.best_params = old_data.get("params", best_params)

        return self.best_params

    # ==========================================================
    # ENHANCED: Data Validation
    # ==========================================================
    def _validate_input_data(self, df: pd.DataFrame) -> bool:
        """Validate input dataframe for signal generation"""
        if df is None:
            self.logger.error("Input dataframe is None")
            return False
            
        if len(df) == 0:
            self.logger.error("Input dataframe is empty")
            return False
            
        # Check required columns
        required_cols = ["close"]
        if self.use_triple_barrier:
            required_cols.extend(["high", "low"])
            if self.barrier_params.get("use_atr", False):
                required_cols.append("atr")
                
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return False
            
        # Check for sufficient data
        min_required = max(self.future_bars + 1, 20)  # Minimum data points needed
        if len(df) < min_required:
            self.logger.warning(f"Insufficient data: {len(df)} rows, need at least {min_required}")
            return False
            
        # Check for valid price data
        if df["close"].isna().all():
            self.logger.error("All close prices are NaN")
            return False
            
        # Check for non-positive prices (invalid for financial data)
        invalid_prices = (df["close"] <= 0).sum()
        if invalid_prices > 0:
            self.logger.warning(f"Found {invalid_prices} non-positive prices, will be handled")
            
        return True

    # ==========================================================
    # ENHANCED: Signal Quality Evaluation
    # ==========================================================
    def _evaluate_signal_quality(self, df: pd.DataFrame, signal_col: str = "signal") -> Dict[str, float]:
        """Enhanced signal quality evaluation with more metrics"""
        if signal_col not in df.columns:
            return {"score": 0.0, "error": "signal_column_missing"}
            
        signals = df[signal_col]
        signal_counts = signals.value_counts()
        n_signals = len(signals)
        
        if n_signals == 0:
            return {"score": 0.0, "error": "no_signals"}

        # Balance score (how balanced are the signal classes)
        balance_score = 1.0
        if len(signal_counts) > 1:
            min_count = signal_counts.min()
            max_count = signal_counts.max()
            balance_score = min_count / max_count if max_count > 0 else 0.0
        elif len(signal_counts) == 1:
            # Only one class - not ideal
            balance_score = 0.1

        # Effectiveness score (based on future returns if available)
        effectiveness_score = 0.5  # Default neutral
        if "future_return" in df.columns:
            future_returns = df["future_return"]
            returns_by_signal = {}
            
            for signal_val in signals.unique():
                mask = signals == signal_val
                if mask.sum() > 0:
                    avg_return = future_returns[mask].mean()
                    if not np.isnan(avg_return):
                        returns_by_signal[signal_val] = avg_return
            
            if len(returns_by_signal) >= 2:
                return_diff = max(returns_by_signal.values()) - min(returns_by_signal.values())
                effectiveness_score = min(abs(return_diff) * 100, 1.0)
            
            # Check signal-return correlation
            try:
                correlation = np.corrcoef(signals.astype(float), future_returns)[0, 1]
                if not np.isnan(correlation):
                    correlation_score = abs(correlation)
                else:
                    correlation_score = 0.0
            except:
                correlation_score = 0.0
            
            effectiveness_score = max(effectiveness_score, correlation_score)

        # Coverage score (what percentage of data gets labeled)
        total_possible = len(df) - self.future_bars  # Can't label last N bars
        coverage_score = n_signals / total_possible if total_possible > 0 else 0.0
        coverage_score = min(coverage_score, 1.0)  # Cap at 1.0

        # Combined score with weights
        score = (balance_score * 0.3) + (effectiveness_score * 0.5) + (coverage_score * 0.2)
        
        return {
            "score": score,
            "balance_score": balance_score,
            "effectiveness_score": effectiveness_score,
            "coverage_score": coverage_score,
            "signal_distribution": signal_counts.to_dict(),
            "n_signals": n_signals,
            "coverage_ratio": coverage_score
        }

    # ==========================================================
    # ENHANCED: Hyperparameter Tuning with Better Error Handling
    # ==========================================================
    def _tune_threshold_parameters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced threshold parameter tuning with better error handling"""
        def objective(trial):
            try:
                future_bars = trial.suggest_int("future_bars", 3, min(20, len(df) // 10))
                threshold = trial.suggest_float("threshold", 0.0001, 0.01, log=True)
                dynamic_threshold = trial.suggest_categorical("dynamic_threshold", [True, False])
                
                # Create temporary generator for this trial
                temp_generator = SignalGenerator(
                    logger=self.logger,
                    mode=self.mode,
                    future_bars=future_bars,
                    threshold=threshold,
                    dynamic_threshold=dynamic_threshold,
                    use_triple_barrier=False,
                    enable_tuning=False,
                    label_only_base_timeframe=self.label_only_base_timeframe,
                    base_timeframe_column=self.base_timeframe_column
                )
                
                # Generate signals
                df_signals = temp_generator.create_trading_signals(df.copy())
                
                if df_signals is None or len(df_signals) == 0:
                    return 0.0
                    
                # Evaluate quality
                quality = self._evaluate_signal_quality(df_signals)
                score = quality.get("score", 0.0)
                
                # Penalize if too few signals
                if quality.get("n_signals", 0) < max(10, len(df) * 0.01):
                    score *= 0.5
                    
                # Bonus for good coverage
                coverage = quality.get("coverage_ratio", 0.0)
                if coverage > 0.1:  # At least 10% coverage
                    score *= (1 + coverage * 0.2)
                
                return score
                
            except Exception as e:
                self.logger.debug(f"Tuning trial failed: {e}")
                return 0.0

        # Create study with better configuration
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner()
        )
        
        # Run optimization with timeout protection
        try:
            study.optimize(objective, n_trials=self.n_trials, timeout=300)  # 5-minute timeout
        except Exception as e:
            self.logger.warning(f"Tuning optimization failed: {e}")
            return self._get_default_threshold_params()
            
        if study.best_value <= 0:
            self.logger.warning("No good parameters found during tuning, using defaults")
            return self._get_default_threshold_params()
            
        return study.best_params

    def _tune_triple_barrier_parameters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced triple barrier parameter tuning"""
        def objective(trial):
            try:
                future_bars = trial.suggest_int("future_bars", 3, min(20, len(df) // 10))
                use_atr = trial.suggest_categorical("use_atr", [True, False])
                
                if use_atr and "atr" in df.columns:
                    tp_mult = trial.suggest_float("tp_atr_mult", 0.5, 3.0)
                    sl_mult = trial.suggest_float("sl_atr_mult", 0.5, 3.0)
                    barrier_params = {
                        "use_atr": True,
                        "tp_atr_mult": tp_mult,
                        "sl_atr_mult": sl_mult,
                        "min_spread": trial.suggest_float("min_spread", 0.0, 0.001)
                    }
                else:
                    tp = trial.suggest_float("tp", 0.0001, 0.01, log=True)
                    sl = trial.suggest_float("sl", 0.0001, 0.01, log=True)
                    barrier_params = {
                        "use_atr": False,
                        "tp": tp,
                        "sl": sl,
                        "min_spread": trial.suggest_float("min_spread", 0.0, 0.001)
                    }

                temp_generator = SignalGenerator(
                    logger=self.logger,
                    mode=self.mode,
                    future_bars=future_bars,
                    use_triple_barrier=True,
                    barrier_params=barrier_params,
                    enable_tuning=False,
                    label_only_base_timeframe=self.label_only_base_timeframe,
                    base_timeframe_column=self.base_timeframe_column
                )
                
                df_signals = temp_generator.create_trading_signals(df.copy())
                
                if df_signals is None or len(df_signals) == 0:
                    return 0.0
                    
                quality = self._evaluate_signal_quality(df_signals)
                return quality.get("score", 0.0)
                
            except Exception as e:
                self.logger.debug(f"Triple barrier tuning trial failed: {e}")
                return 0.0

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        try:
            study.optimize(objective, n_trials=self.n_trials, timeout=300)
        except Exception as e:
            self.logger.warning(f"Triple barrier tuning failed: {e}")
            return self._get_default_barrier_params()
            
        if study.best_value <= 0:
            return self._get_default_barrier_params()
            
        return study.best_params

    def _get_default_threshold_params(self) -> Dict[str, Any]:
        """Get default threshold parameters as fallback"""
        return {
            "future_bars": self.future_bars,
            "threshold": self.threshold,
            "dynamic_threshold": self.dynamic_threshold
        }

    def _get_default_barrier_params(self) -> Dict[str, Any]:
        """Get default barrier parameters as fallback"""
        return {
            "future_bars": self.future_bars,
            **self.barrier_params
        }

    def tune_parameters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced parameter tuning with validation"""
        if not self._validate_input_data(df):
            self.logger.error("Input validation failed, using default parameters")
            if self.use_triple_barrier:
                return self._get_default_barrier_params()
            else:
                return self._get_default_threshold_params()
        
        self.logger.info("Starting enhanced signal parameter tuning...")
        
        if self.use_triple_barrier:
            best_params = self._tune_triple_barrier_parameters(df)
        else:
            best_params = self._tune_threshold_parameters(df)
            
        self.logger.info(f"Best parameters found: {best_params}")
        self.best_params = best_params
        self._save_best_params(best_params)

        # Update instance parameters
        if "future_bars" in best_params:
            self.future_bars = best_params["future_bars"]
            
        if not self.use_triple_barrier:
            if "threshold" in best_params:
                self.threshold = best_params["threshold"]
            if "dynamic_threshold" in best_params:
                self.dynamic_threshold = best_params["dynamic_threshold"]
        else:
            # Update barrier parameters
            for param in ["use_atr", "tp", "sl", "tp_atr_mult", "sl_atr_mult", "min_spread"]:
                if param in best_params:
                    self.barrier_params[param] = best_params[param]
                    
        return best_params

    # ==========================================================
    # ENHANCED: Triple Barrier Method with Better Error Handling
    # ==========================================================
    def _apply_triple_barrier(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced triple barrier implementation with better error handling"""
        n = len(df)
        if n == 0:
            return df
            
        # Validate required columns
        if not all(col in df.columns for col in ["close", "high", "low"]):
            self.logger.error("Triple barrier requires close, high, low columns")
            return df
            
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)

        # Handle invalid values
        close = np.nan_to_num(close, nan=0.0, posinf=0.0, neginf=0.0)
        high = np.nan_to_num(high, nan=0.0, posinf=0.0, neginf=0.0)
        low = np.nan_to_num(low, nan=0.0, posinf=0.0, neginf=0.0)

        tp_arr = np.full(n, np.nan, dtype=np.float64)
        sl_arr = np.full(n, np.nan, dtype=np.float64)

        use_atr = bool(self.barrier_params.get("use_atr", False))
        min_spread = float(self.barrier_params.get("min_spread", 0.0))

        # ATR-based or fixed barriers - ENHANCED with fallback
        use_atr = bool(self.barrier_params.get("use_atr", False))
        if use_atr:
            if "atr" in df.columns:
                # Use ATR-based barriers
                atr = df["atr"].to_numpy(dtype=np.float64)
                atr = np.nan_to_num(atr, nan=np.nanmean(close) * 0.01)
                tp_mult = float(self.barrier_params.get("tp_atr_mult", 1.0))
                sl_mult = float(self.barrier_params.get("sl_atr_mult", 1.0))
                tp_arr = atr * tp_mult
                sl_arr = atr * sl_mult
                self.logger.info("Using ATR-based barriers")
            else:
                # FALLBACK: Use percentage-based if ATR not available
                self.logger.warning("ATR column not found, falling back to percentage barriers")
                tp_pct = float(self.barrier_params.get("tp", 0.0005))
                sl_pct = float(self.barrier_params.get("sl", 0.0005))
                tp_arr = close * tp_pct
                sl_arr = close * sl_pct
        else:
            # Percentage-based barriers
            tp_pct = float(self.barrier_params.get("tp", 0.0005))
            sl_pct = float(self.barrier_params.get("sl", 0.0005))
            tp_arr = close * tp_pct
            sl_arr = close * sl_pct

        # Calculate barrier levels
        up_level = close + tp_arr
        down_level = close - sl_arr
        labels = np.zeros(n, dtype=np.int8)
        max_horizon = int(self.future_bars)

        # Apply labeling logic
        for i in range(n - max_horizon):  # Don't label last future_bars rows
            try:
                # Skip if price data is invalid
                if close[i] <= 0 or np.isnan(close[i]):
                    continue
                    
                # Skip labeling if only base timeframe should be labeled
                if (self.label_only_base_timeframe and 
                    self.base_timeframe_column in df.columns and 
                    not df.iloc[i][self.base_timeframe_column]):
                    labels[i] = 0
                    continue
                
                end = min(n, i + max_horizon + 1)
                future_high = high[i + 1:end] if (i + 1) < end else np.array([])
                future_low = low[i + 1:end] if (i + 1) < end else np.array([])
                
                if len(future_high) == 0 or len(future_low) == 0:
                    continue
                    
                up_thresh = up_level[i]
                down_thresh = down_level[i]
                
                # Check barriers
                tp_hit = False
                sl_hit = False
                
                for j in range(len(future_high)):
                    if (future_high[j] >= up_thresh and 
                        (future_high[j] - close[i]) >= min_spread):
                        tp_hit = True
                        break
                    if (future_low[j] <= down_thresh and 
                        (close[i] - future_low[j]) >= min_spread):
                        sl_hit = True
                        break
                
                # Assign labels
                if tp_hit:
                    labels[i] = 1
                elif sl_hit:
                    labels[i] = -1
                else:
                    labels[i] = 0
                    
            except (IndexError, ValueError) as e:
                self.logger.debug(f"Barrier calculation error at index {i}: {e}")
                labels[i] = 0
                continue

        # Create output dataframe
        df_out = df.copy()
        df_out["signal"] = labels

        # Set encoded signals based on mode
        if self.mode == "binary":
            # For binary mode, only keep non-zero signals
            df_out["signal_binary"] = np.where(
                df_out["signal"] == 1, 1,
                np.where(df_out["signal"] == -1, 0, np.nan)
            )
            df_out["signal_encoded"] = df_out["signal_binary"]
        elif self.mode == "regression":
            df_out["signal_encoded"] = df_out.get("future_return", 0)
        else:  # multiclass
            df_out["signal_encoded"] = df_out["signal"]

        return df_out

    # ==========================================================
    # ENHANCED: Main Signal Generation with Better Error Handling
    # ==========================================================
    def create_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced signal generation with comprehensive error handling"""
        
        # Input validation
        if not self._validate_input_data(df):
            self.logger.error("Input validation failed for signal generation")
            return pd.DataFrame()

        # Parameter handling
        if self.enable_tuning:
            self.logger.info("Hyperparameter tuning enabled for signal generation")
            try:
                self.tune_parameters(df)
            except Exception as e:
                self.logger.error(f"Parameter tuning failed: {e}, using defaults")
        else:
            # Load params dari file kalau ada
            best_params = self._load_best_params()
            if best_params:
                self.best_params = best_params
                self.logger.info(f"Using best params from JSON: {best_params}")
                
                # Update instance params
                if "future_bars" in best_params:
                    self.future_bars = best_params["future_bars"]
                    
                if not self.use_triple_barrier:
                    self.threshold = best_params.get("threshold", self.threshold)
                    self.dynamic_threshold = best_params.get("dynamic_threshold", self.dynamic_threshold)
                else:
                    self.barrier_params.update(best_params)
            else:
                self.logger.info("No JSON params found, using config defaults")

        # Create working copy
        df_work = df.copy()
        
        # Calculate future returns (handle edge cases)
        try:
            close_prices = df_work["close"]
            future_close = close_prices.shift(-self.future_bars)
            
            # Handle division by zero and inf values
            with np.errstate(divide='ignore', invalid='ignore'):
                future_returns = future_close / close_prices - 1
                future_returns = future_returns.replace([np.inf, -np.inf], np.nan)
                
            df_work["future_return"] = future_returns
            
        except Exception as e:
            self.logger.error(f"Failed to calculate future returns: {e}")
            return pd.DataFrame()

        # Apply appropriate labeling method
        if self.use_triple_barrier:
            self.logger.info(f"Using TRIPLE BARRIER labeling: params={self.barrier_params}")
            try:
                df_labeled = self._apply_triple_barrier(df_work)
            except Exception as e:
                self.logger.error(f"Triple barrier labeling failed: {e}")
                return pd.DataFrame()
        else:
            self.logger.info(f"Using threshold labeling: threshold={self.threshold}, dynamic={self.dynamic_threshold}")
            try:
                df_labeled = self._apply_threshold_labeling(df_work)
            except Exception as e:
                self.logger.error(f"Threshold labeling failed: {e}")
                return pd.DataFrame()

        # Post-processing and validation
        try:
            df_final = self._post_process_signals(df_labeled)
        except Exception as e:
            self.logger.error(f"Signal post-processing failed: {e}")
            return pd.DataFrame()

        # Quality evaluation
        quality = self._evaluate_signal_quality(df_final)
        self.logger.info(f"Signal quality: score={quality['score']:.3f}, "
                        f"distribution={quality['signal_distribution']}")

        return df_final

    def _apply_threshold_labeling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced threshold-based labeling"""
        df_work = df.copy()
        
        # Calculate dynamic threshold if needed
        if self.dynamic_threshold:
            try:
                returns = df_work["close"].pct_change()
                vol = returns.rolling(
                    window=max(20, int(self.future_bars * 3)), 
                    min_periods=1
                ).std()
                threshold_values = vol.fillna(vol.median()).replace(0, 1e-6)
                df_work["_threshold"] = threshold_values
            except Exception as e:
                self.logger.warning(f"Dynamic threshold calculation failed: {e}, using static")
                df_work["_threshold"] = self.threshold
        else:
            df_work["_threshold"] = self.threshold

        # Generate signals based on mode
        if self.mode == "binary":
            df_work["signal"] = 0
            df_work.loc[df_work["future_return"] > df_work["_threshold"], "signal"] = 1
            df_work.loc[df_work["future_return"] < -df_work["_threshold"], "signal"] = -1
            
            # For binary mode, only keep non-zero signals
            mask = df_work["signal"] != 0
            df_work = df_work[mask].copy()
            df_work["signal_binary"] = (df_work["signal"] > 0).astype(int)
            df_work["signal_encoded"] = df_work["signal_binary"]

        elif self.mode == "regression":
            df_work["signal"] = df_work["future_return"]
            df_work["signal_encoded"] = df_work["future_return"]

        elif self.mode == "multiclass":
            df_work["signal"] = 0
            df_work.loc[df_work["future_return"] > df_work["_threshold"], "signal"] = 1
            df_work.loc[df_work["future_return"] < -df_work["_threshold"], "signal"] = -1
            df_work["signal_encoded"] = df_work["signal"]

        # Apply base timeframe filtering if needed
        if (self.label_only_base_timeframe and 
            self.base_timeframe_column in df_work.columns):
            base_mask = df_work[self.base_timeframe_column]
            non_base_mask = ~base_mask
            
            df_work.loc[non_base_mask, "signal"] = 0
            df_work.loc[non_base_mask, "signal_encoded"] = 0
            if "signal_binary" in df_work.columns:
                df_work.loc[non_base_mask, "signal_binary"] = 0
            
            # For binary mode, remove filtered rows
            if self.mode == "binary":
                df_work = df_work[df_work["signal"] != 0].copy()

        return df_work

    def _post_process_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced post-processing of generated signals"""
        if df is None or len(df) == 0:
            return df
            
        df_final = df.copy()
        
        # Remove temporary columns
        temp_cols = ["_threshold", "_thr"]
        df_final = df_final.drop(columns=[col for col in temp_cols if col in df_final.columns], 
                                errors="ignore")
        
        # Clean up future-related columns (avoid data leakage)
        # Hapus SEMUA kolom yang berhubungan dengan future data
        future_related_patterns = [
            "future", "forward", "shift", "next", "ahead", 
            "tomorrow", "later", "subsequent"
        ]

        future_cols = []
        for col in df_final.columns:
            col_lower = col.lower()
        # Skip target signal columns
            if col in ["signal", "signal_binary", "signal_encoded"]:
                continue
            # Check for future-related patterns
            if any(pattern in col_lower for pattern in future_related_patterns):
                future_cols.append(col)
            # Also check for time-shifted columns (like close_shift_5 etc)
            elif any(keyword in col_lower for keyword in ["_shift", "_lag", "_lead"]):
                if not col_lower.startswith(("past_", "hist_", "prev_")):  # Allow past lags
                    future_cols.append(col)

        if future_cols:
            self.logger.warning(f"🚨 Removing FUTURE LEAKAGE columns: {future_cols}")
            df_final = df_final.drop(columns=future_cols, errors="ignore")
        
        # Drop rows with NaN signals (but preserve valid zeros)
        if "signal_encoded" in df_final.columns:
            initial_count = len(df_final)
            df_final = df_final.dropna(subset=["signal_encoded"])
            dropped_count = initial_count - len(df_final)
            if dropped_count > 0:
                self.logger.info(f"Dropped {dropped_count} rows with NaN signals")
        
        # Validate signal ranges based on mode
        if self.mode == "binary" and "signal_binary" in df_final.columns:
            # Binary signals should be 0 or 1
            invalid_binary = ~df_final["signal_binary"].isin([0, 1])
            if invalid_binary.any():
                self.logger.warning(f"Found {invalid_binary.sum()} invalid binary signals, fixing...")
                df_final.loc[invalid_binary, "signal_binary"] = 0
                
        elif self.mode == "multiclass" and "signal_encoded" in df_final.columns:
            # Multiclass signals should be -1, 0, or 1
            invalid_multi = ~df_final["signal_encoded"].isin([-1, 0, 1])
            if invalid_multi.any():
                self.logger.warning(f"Found {invalid_multi.sum()} invalid multiclass signals, fixing...")
                df_final.loc[invalid_multi, "signal_encoded"] = 0
                
        # Log final statistics
        mode_info = "dynamic volatility" if getattr(self, "dynamic_threshold", False) else f"static={self.threshold}"
        base_timeframe_info = " (base timeframe only)" if self.label_only_base_timeframe else ""
        
        self.logger.info(
            f"Signals generated using future {self.future_bars} bars "
            f"(mode={self.mode}, threshold={mode_info}{base_timeframe_info}), "
            f"final shape={df_final.shape}"
        )
        
        # Log signal distribution
        try:
            if "signal" in df_final.columns:
                dist = df_final["signal"].value_counts().to_dict()
                self.logger.info(f"Signal distribution: {dist}")
        except Exception as e:
            self.logger.debug(f"Could not log signal distribution: {e}")
            
        return df_final