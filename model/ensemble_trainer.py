import logging
from typing import Optional, Dict, Any, List, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from lightgbm import LGBMClassifier
import optuna
import joblib
import os
import json
import time
from collections import deque
from utils.purge_time_series import PurgedTimeSeriesSplit

from utils.logger import get_logger, setup_logger
from data_processing.feature_engineering import FeatureEngineering
from config.settings import Config
from data_processing.market_regime import detect_market_regime_series




class EnsembleTrainer:
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        config: Optional[dict] = None,
        df_features_all: Optional[pd.DataFrame] = None,
        **kwargs
    ):
        cfg = config or Config.ENSEMBLE_CONFIG
        self.min_samples_per_fold = cfg.get("min_samples_per_fold", 100)
        self.objective_metric = cfg.get("objective_metric", "profit_weighted_accuracy")
        self.logger = logger or get_logger("EnhancedScalpingAI")
        self.model: Optional[VotingClassifier] = None
        self.lgbm_device = "cpu"

        self.tune = bool(kwargs.get("tune", cfg.get("tune", True)))
        self.n_trials = int(kwargs.get("n_trials", cfg.get("n_trials", 30)))
        self.model_path = kwargs.get("model_path", cfg.get("model_path"))
        self.confidence_threshold = float(kwargs.get("confidence_threshold", cfg.get("confidence_threshold", 0.6)))
        self.performance_window = int(kwargs.get("performance_window", cfg.get("performance_window", 100)))

        self.best_params: Dict[str, Any] = {}
        self.task_type: str = "multiclass"
        self.fe = kwargs.get("fe", None)
        self.df_features_all = df_features_all

        self.label_mapping: Optional[Dict[int, int]] = None
        self.inverse_mapping: Optional[Dict[int, int]] = None

        self.recent_performance = deque(maxlen=self.performance_window)
        self.feature_importance_history: Dict[float, Dict[str, np.ndarray]] = {}
        self.training_metrics: Dict[str, Any] = {}

    # -------------------------
    # BUILD BASE MODELS
    # -------------------------
    def _build_lgbm(self, params: Optional[Dict] = None) -> LGBMClassifier:
        try:
            test_model = LGBMClassifier(device_type="gpu", n_estimators=1, random_state=42)
            test_model.fit(np.zeros((2, 2)), [0, 1])
            self.lgbm_device = "gpu"
            self.logger.info("LightGBM: GPU available → using GPU")
        except Exception:
            self.lgbm_device = "cpu"
            self.logger.info("LightGBM: GPU not available → using CPU")

        params = params or {}
        return LGBMClassifier(
            n_estimators=int(params.get("n_estimators", 400)),
            learning_rate=float(params.get("learning_rate", 0.01)),
            max_depth=int(params.get("max_depth", 6)),
            num_leaves=int(params.get("num_leaves", 31)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            n_jobs=-1,
            random_state=42,
            objective=("multiclass" if self.task_type == "multiclass" else "binary"),
            device_type=self.lgbm_device,
        )

    def _build_catboost(self, params: Optional[Dict] = None):
        from model.catboost_trainer import CatBoostTrainer  # local import to avoid circular
        cb_trainer = CatBoostTrainer(
            logger=self.logger,
            tune=False,
            n_trials=self.n_trials,
            objective_metric=self.objective_metric,
        )
        cb_trainer.task_type = self.task_type
        return cb_trainer._build_catboost(params)

    # -------------------------
    # TUNING
    # -------------------------
    def _tune_base_model(self, objective_func, n_trials: int) -> Dict[str, Any]:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5)
        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.optimize(objective_func, n_trials=n_trials)
        self.logger.info(f"Optuna best_value={study.best_value:.4f}, best_params={study.best_params}")
        return study.best_params

    def tune_lgbm(self, X_train, y_train):
        return {}

    def tune_catboost(self, X_train, y_train):
        from model.catboost_trainer import CatBoostTrainer
        cb_trainer = CatBoostTrainer(
            logger=self.logger,
            tune=True,
            n_trials=self.n_trials,
            objective_metric=self.objective_metric,
        )
        cb_trainer.task_type = self.task_type
        return cb_trainer._tune_catboost(X_train, y_train, self.df_features_all)

    def tune_ensemble_weights(self, X_train, y_train, lgbm_params, catboost_params):
        def objective(trial):
            w_lgbm = trial.suggest_float("w_lgbm", 0.0, 1.0)
            w_cat = 1.0 - w_lgbm
            model = VotingClassifier(
                estimators=[
                    ("lgbm", self._build_lgbm(lgbm_params)),
                    ("catboost", self._build_catboost(catboost_params)),
                ],
                voting="soft",
                weights=[w_lgbm, w_cat],
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_train)
            return accuracy_score(y_train, preds)

        best = self._tune_base_model(objective, self.n_trials)
        w = best.get("w_lgbm", 0.5)
        return [w, 1.0 - w]

    # -------------------------
    # ENSEMBLE
    # -------------------------
    def build_model(self, lgbm_params=None, catboost_params=None, weights=None):
        lgbm = self._build_lgbm(lgbm_params)
        cat_model = self._build_catboost(catboost_params)
        return VotingClassifier(
            estimators=[("lgbm", lgbm), ("catboost", cat_model)],
            voting="soft",
            weights=weights or [1, 1],
            n_jobs=-1,
        )

    def train(self, X, y, task_type: str = "multiclass"):
        self.task_type = task_type

        if self.tune:
            lgbm_params = self.tune_lgbm(X, y)
            catboost_params = self.tune_catboost(X, y)
            weights = self.tune_ensemble_weights(X, y, lgbm_params, catboost_params)
            self.best_params = {"lgbm": lgbm_params, "catboost": catboost_params, "weights": weights}
        else:
            lgbm_params = catboost_params = None
            weights = [1, 1]

        self.model = self.build_model(lgbm_params, catboost_params, weights)
        self.logger.info("🏋️ Training ensemble (LightGBM + CatBoost)...")
        self.model.fit(X, y)
        self.logger.info("✅ Ensemble training selesai")
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


if __name__ == "__main__":
    logger = setup_logger("EnsembleTrainer")
    X = pd.DataFrame(np.random.rand(200, 10))
    y = np.random.choice([0, 1, 2], size=200)
    trainer = EnsembleTrainer(logger=logger, tune=False)
    model = trainer.train(X, y, task_type="multiclass")
    preds = trainer.predict(X)
    logger.info(f"Prediksi: {preds[:10]}")    "ignore",
    message=".*XGBoost is not compiled with CUDA support.*",
    category=UserWarning
)


def check_xgboost_cuda_support() -> bool:
    """Detect whether XGBoost was built with CUDA support."""
    try:
        info = xgb.build_info()
        return info.get("USE_CUDA", False) or info.get("USE_NCCL", False)
    except Exception:
        return False


class PurgedTimeSeriesSplit:
    """
    Purged time-series splitter to reduce leakage. Simple forward CV with purge gap.
    """
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

            gap_start = train_end
            gap_end = min(train_end + gap_size, n_samples)

            val_start = gap_end
            val_end = min(val_start + fold_size, n_samples)

            if val_start >= n_samples:
                break

            val_indices = np.arange(val_start, val_end)

            if len(val_indices) > 0:
                yield train_indices, val_indices


class EnsembleTrainer:
    """
    Full-featured trainer dengan metric trading yang lebih tepat:
      - Profit-Weighted Accuracy: Accuracy dengan bobot potential profit
      - Balanced Accuracy: Handle class imbalance
      - High-Confidence Accuracy: Accuracy hanya pada prediction confident
      - F1 Macro: Untuk multiclass classification
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        config: Optional[dict] = None,
        df_features_all: Optional[pd.DataFrame] = None,
        **kwargs
    ):
        cfg = config or Config.ENSEMBLE_CONFIG
        self.min_samples_per_fold = cfg.get("min_samples_per_fold", 100)
        self.objective_metric = cfg.get("objective_metric", "profit_weighted_accuracy")  # DEFAULT CHANGED
        self.logger = logger or get_logger("EnhancedScalpingAI")
        self.model: Optional[VotingClassifier] = None
        self.lgbm_device = "cpu"
        self.xgb_device = "cpu"

        # Get from config, can be overridden via kwargs
        self.tune = bool(kwargs.get("tune", cfg.get("tune", True)))
        self.n_trials = int(kwargs.get("n_trials", cfg.get("n_trials", 30)))
        self.model_path = kwargs.get("model_path", cfg.get("model_path"))
        self.confidence_threshold = float(kwargs.get("confidence_threshold", cfg.get("confidence_threshold", 0.6)))
        self.performance_window = int(kwargs.get("performance_window", cfg.get("performance_window", 100)))

        self.best_params: Dict[str, Any] = {}
        self.task_type: str = "multiclass"
        self.fe = kwargs.get("fe", None)
        self.df_features_all = df_features_all

        # Label mapping for multiclass (-1,0,1) -> (0,1,2)
        self.label_mapping: Optional[Dict[int, int]] = None
        self.inverse_mapping: Optional[Dict[int, int]] = None

        # Monitoring
        self.recent_performance = deque(maxlen=self.performance_window)
        self.feature_importance_history: Dict[float, Dict[str, np.ndarray]] = {}
        self.training_metrics: Dict[str, Any] = {}

    # -------------------------
    # IMPROVED METRIC METHODS
    # -------------------------
    
    def profit_weighted_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               df_features: Optional[pd.DataFrame] = None) -> float:
        """
        Accuracy yang diberi bobot berdasarkan potential profit/loss.
        Benar predict arah → high weight, salah predict → penalty weight.
        """
        try:
            if df_features is None or 'close' not in df_features.columns:
                # Fallback ke accuracy biasa jika harga tidak ada
                return float(accuracy_score(y_true, y_pred))
            
            close_prices = df_features['close'].values
            
            # Pastikan length match dan ada cukup data untuk returns
            if len(close_prices) < 2 or len(y_true) < 2:
                return float(accuracy_score(y_true, y_pred))
            
            # Simple returns (percent change)
            returns = np.diff(close_prices) / close_prices[:-1]
            
            # Pastikan length match dengan target (y_true dan y_pred offset oleh 1 bar)
            min_len = min(len(y_true)-1, len(returns), len(y_pred)-1)
            if min_len <= 0:
                return float(accuracy_score(y_true, y_pred))
                
            y_true_eval = y_true[1:min_len+1]
            y_pred_eval = y_pred[1:min_len+1]
            returns_eval = returns[:min_len]
            
            # Weight berdasarkan absolute return (semakin besar return, semakin penting predictionnya)
            weights = np.abs(returns_eval)
            if np.max(weights) > 0:
                weights = weights / np.max(weights)  # Normalize 0-1
            
            correct_predictions = (y_true_eval == y_pred_eval).astype(float)
            
            # Jika semua weights 0, gunakan accuracy biasa
            if np.sum(weights) == 0:
                return float(accuracy_score(y_true_eval, y_pred_eval))
                
            weighted_accuracy = np.average(correct_predictions, weights=weights)
            return float(weighted_accuracy)
            
        except Exception as e:
            self.logger.warning(f"Profit-weighted accuracy calculation failed: {e}")
            return float(accuracy_score(y_true, y_pred))

    def directional_accuracy_with_confidence(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                           y_proba: Optional[np.ndarray] = None, 
                                           threshold: float = 0.6) -> float:
        """
        Accuracy hanya pada prediction dengan confidence tinggi.
        Mengabaikan prediction yang tidak confident.
        """
        if y_proba is None or len(y_proba) == 0:
            return float(accuracy_score(y_true, y_pred))
        
        try:
            confidences = np.max(y_proba, axis=1)
            high_conf_mask = confidences >= threshold
            
            if np.sum(high_conf_mask) == 0:
                return 0.0
            
            return float(accuracy_score(y_true[high_conf_mask], y_pred[high_conf_mask]))
        except Exception as e:
            self.logger.warning(f"High-confidence accuracy calculation failed: {e}")
            return float(accuracy_score(y_true, y_pred))

    def calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns series."""
        try:
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / (running_max + 1e-12)
            return float(np.min(drawdown))
        except Exception:
            return 0.0

    def _calculate_trading_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        df_features: Optional[pd.DataFrame] = None,
        spread_cost: float = 0.0001
    ) -> Dict[str, float]:
        """
        IMPROVED: Comprehensive trading metrics suite dengan fallback yang robust.
        """
        metrics = {}
        
        # 1. BASIC ACCURACY METRICS
        try:
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        except Exception:
            metrics["accuracy"] = metrics["balanced_accuracy"] = 0.0

        # 2. PROFIT-AWARE METRICS
        metrics["profit_weighted_accuracy"] = self.profit_weighted_accuracy(y_true, y_pred, df_features)
        
        # 3. CONFIDENCE-AWARE METRICS
        metrics["high_conf_accuracy"] = self.directional_accuracy_with_confidence(
            y_true, y_pred, y_proba, threshold=0.6
        )
        
        # 4. CONFIDENCE METRICS
        try:
            metrics["avg_confidence"] = float(np.max(y_proba, axis=1).mean()) if y_proba is not None else 0.0
        except Exception:
            metrics["avg_confidence"] = 0.0

        # 5. CLASSIFICATION METRICS
        if self.task_type == "multiclass":
            try:
                metrics["f1_macro"] = float(f1_score(y_true, y_pred, average='macro'))
                metrics["precision_macro"] = float(precision_score(y_true, y_pred, average='macro')) 
                metrics["recall_macro"] = float(recall_score(y_true, y_pred, average='macro'))
            except Exception:
                metrics["f1_macro"] = metrics["precision_macro"] = metrics["recall_macro"] = 0.0
        else:
            try:
                metrics["f1"] = float(f1_score(y_true, y_pred))
                metrics["precision"] = float(precision_score(y_true, y_pred))
                metrics["recall"] = float(recall_score(y_true, y_pred))
                if y_proba is not None and y_proba.shape[1] > 1:
                    metrics["auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    metrics["auc"] = 0.0
            except Exception:
                metrics["f1"] = metrics["precision"] = metrics["recall"] = metrics["auc"] = 0.0
        
        # 6. TRADING-LIKE METRICS dengan fallback robust
        try:
            # Simplified profit factor calculation
            correct_trades = (y_true == y_pred)
            win_rate = np.mean(correct_trades)
            
            # Simulated returns based on correctness
            simulated_returns = np.where(correct_trades, 0.001, -0.001)
            total_profit = np.sum(simulated_returns[simulated_returns > 0])
            total_loss = np.abs(np.sum(simulated_returns[simulated_returns < 0]))
            
            metrics["win_rate"] = float(win_rate)
            metrics["simulated_profit_factor"] = float(total_profit / total_loss) if total_loss > 0 else float('inf')
            metrics["max_drawdown"] = self.calculate_max_drawdown(simulated_returns)
            
        except Exception:
            metrics["win_rate"] = metrics["accuracy"]
            metrics["simulated_profit_factor"] = 1.0
            metrics["max_drawdown"] = 0.0

        # 7. REAL TRADING METRICS (jika harga tersedia)
        if df_features is not None and 'close' in df_features.columns:
            try:
                close_prices = df_features['close'].values
                if len(close_prices) > 1:
                    returns = np.diff(close_prices) / close_prices[:-1]
                    min_len = min(len(returns), len(y_pred)-1)
                    
                    if min_len > 0:
                        pred_direction = np.sign(y_pred[1:min_len+1])  # Assuming predictions are directional
                        realized_returns = pred_direction * returns[:min_len]
                        
                        metrics["real_avg_return"] = float(np.mean(realized_returns))
                        metrics["real_volatility"] = float(np.std(realized_returns))
                        metrics["real_sharpe_ratio"] = float(metrics["real_avg_return"] / (metrics["real_volatility"] + 1e-12))
                        metrics["real_max_drawdown"] = self.calculate_max_drawdown(realized_returns)
            except Exception as e:
                self.logger.debug(f"Real trading metrics calculation skipped: {e}")

        return metrics

    # -------------------------
    # Helper Methods
    # -------------------------
    @staticmethod
    def _safe_index(X: Union[pd.DataFrame, pd.Series, np.ndarray],
                    idx: Union[List[int], np.ndarray]) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        """Index DataFrame/Series or numpy array safely."""
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.iloc[idx]
        return X[idx]

    def _param_or_default(self, params: Optional[Dict], key: str, default: Any) -> Any:
        """Get parameter value or return default if not present."""
        if params and key in params:
            return params[key]
        return default
        
    def export_onnx(self, X_sample, prefix="ensemble"):
        """
        Export model ensemble (LightGBM + XGBoost) ke ONNX + wrapper.
        """
        if export_to_onnx is None:
            self.logger.error("onnx_utils belum tersedia. Install dulu dependency skl2onnx, onnxmltools, onnxruntime.")
            return
        if self.model is None:
            self.logger.error("Model belum di-train, tidak bisa export.")
            return

        try:
            export_to_onnx(self, X_sample, prefix=prefix)
            self.logger.info(f"ONNX export selesai, prefix={prefix}")
        except Exception as e:
            self.logger.error(f"ONNX export gagal: {e}")

    # -------------------------
    # Persistence Methods
    # -------------------------
    def _best_params_path(self) -> Optional[str]:
        """Get path for best parameters JSON file."""
        if not self.model_path:
            return None
        return self.model_path.replace(".pkl", "_best_params.json")

    def _save_best_params(self, params: Dict[str, Any]) -> None:
        """Save (merge) best params to JSON alongside model_path."""
        path = self._best_params_path()
        if not path:
            self.logger.warning("model_path not set → best_params not saved")
            return

        existing = {}
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}

        merged = {**existing, **params}
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump(merged, f, indent=4)
            self.logger.info(f"Saved best_params to {path}")
        except Exception as e:
            self.logger.error(f"Failed saving best_params: {e}")

    def _load_best_params(self) -> Dict[str, Any]:
        """Load best parameters from JSON file."""
        path = self._best_params_path()
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                params = json.load(f)
            self.logger.info(f"Loaded best_params from {path}")
            return params
        except Exception as e:
            self.logger.error(f"Failed loading best_params: {e}")
            return {}

    # -------------------------
    # Model Building Methods
    # -------------------------
    def _build_lgbm(self, params: Optional[Dict] = None) -> LGBMClassifier:
        """Construct LGBMClassifier dengan safe param access dan GPU autodetect."""
        # Detect GPU availability
        try:
            test_model = LGBMClassifier(device_type="gpu", n_estimators=1, random_state=42)
            test_model.fit(np.zeros((2, 2)), [0, 1])
            self.lgbm_device = "gpu"
            self.logger.info("LightGBM: GPU available → using GPU")
        except Exception:
            self.lgbm_device = "cpu"
            self.logger.info("LightGBM: GPU not available → using CPU")

        max_depth = int(self._param_or_default(params, "max_depth", 6))
        max_num_leaves = 2 ** max_depth

        metric = "multi_logloss" if self.task_type == "multiclass" else "binary_logloss"
        objective = "multiclass" if self.task_type == "multiclass" else "binary"

        lgbm_params = {
            "n_estimators": int(self._param_or_default(params, "n_estimators", 400)),
            "num_leaves": int(self._param_or_default(params, "num_leaves", min(31, max_num_leaves))),
            "colsample_bytree": float(self._param_or_default(params, "colsample_bytree", 0.8)),
            "learning_rate": float(self._param_or_default(params, "learning_rate", 0.001)),
            "max_depth": max_depth,
            "n_jobs": int(self._param_or_default(params, "n_jobs", -1)),
            "verbosity": -1,
            "feature_fraction": 0.9,
            "importance_type": "gain",
            "min_data_in_bin": int(self._param_or_default(params, "min_data_in_bin", 1)),
            "min_data_in_leaf": int(self._param_or_default(params, "min_data_in_leaf", 5)),
            "metric": self._param_or_default(params, "metric", metric),
            "subsample": float(self._param_or_default(params, "subsample", 0.8)),
            "bagging_freq": int(self._param_or_default(params, "bagging_freq", 1)),
            "min_child_samples": int(self._param_or_default(params, "min_child_samples", 20)),
            "reg_alpha": float(self._param_or_default(params, "reg_alpha", 0.1)),
            "reg_lambda": float(self._param_or_default(params, "reg_lambda", 1.0)),
            "random_state": 42,
            "device_type": self.lgbm_device,
            "objective": objective,
        }

        if self.lgbm_device == "gpu":
            lgbm_params.update({"gpu_platform_id": 0, "gpu_device_id": 0})

        return LGBMClassifier(**lgbm_params)

    def _build_xgb(self, params: Optional[Dict] = None) -> XGBClassifier:
        """Construct XGBClassifier dengan safe param access dan CUDA detection."""
        has_cuda = check_xgboost_cuda_support()
        self.xgb_device = "cuda" if has_cuda else "cpu"

        if has_cuda:
            self.logger.info("XGBoost: CUDA available → using GPU")
        else:
            self.logger.info("XGBoost: GPU not available → using CPU")

        if self.task_type == "multiclass":
            objective = "multi:softprob"
            eval_metric = "mlogloss"
        else:
            objective = "binary:logistic"
            eval_metric = "logloss"

        xgb_params = {
            "n_estimators": int(self._param_or_default(params, "n_estimators", 500)),
            "learning_rate": float(self._param_or_default(params, "learning_rate", 0.002)),
            "max_depth": int(self._param_or_default(params, "max_depth", 6)),
            "min_child_weight": int(self._param_or_default(params, "min_child_weight", 3)),
            "subsample": float(self._param_or_default(params, "subsample", 0.8)),
            "colsample_bytree": float(self._param_or_default(params, "colsample_bytree", 0.8)),
            "gamma": float(self._param_or_default(params, "gamma", 0.1)),
            "reg_alpha": float(self._param_or_default(params, "reg_alpha", 0.1)),
            "reg_lambda": float(self._param_or_default(params, "reg_lambda", 1.0)),
            "random_state": 42,
            "n_jobs": -1,
            "device": self.xgb_device,
            "eval_metric": eval_metric,
            "objective": objective,
        }

        return XGBClassifier(**xgb_params)

    # -------------------------
    # Tuning Methods (Optuna) - UPDATED METRICS
    # -------------------------
    def _tune_base_model(self, objective_func, n_trials: int) -> Dict[str, Any]:
        """Run Optuna study dengan MedianPruner."""
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5)
        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.optimize(objective_func, n_trials=n_trials)
        self.logger.info(f"Optuna best_value={study.best_value:.4f}, best_params={study.best_params}")
        return study.best_params

    def tune_lgbm(self, X_train: Union[pd.DataFrame, np.ndarray],
                  y_train: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Tune LightGBM hyperparameters menggunakan Profit-Weighted Accuracy sebagai objective."""
        self.logger.info("Starting LightGBM tuning...")

        def objective(trial):
            max_depth = trial.suggest_int("max_depth", 3, 12)
            num_leaves_high = max(31, 2 ** max_depth)

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 700),
                "num_leaves": trial.suggest_int("num_leaves", 31, num_leaves_high),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 0.5, log=True),
                "max_depth": max_depth,
                "n_jobs": -1,
                "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                "device_type": "gpu",
            }

            params["min_data_in_bin"] = trial.suggest_int("min_data_in_bin", 1, 255)
            params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 1, 100)

            model = self._build_lgbm(params)
            tscv = PurgedTimeSeriesSplit(n_splits=2, gap_ratio=0.02)

            self.logger.info(f"🎯 Using {self.objective_metric} as objective metric for LightGBM tuning")

            scores = []
            for tr_idx, val_idx in tscv.split(X_train):
                if len(tr_idx) < self.min_samples_per_fold or len(val_idx) < self.min_samples_per_fold:
                    self.logger.warning(f"⚠️ Skipping fold: too few samples (train={len(tr_idx)}, val={len(val_idx)})")
                    continue

                X_tr = self._safe_index(X_train, tr_idx)
                X_val = self._safe_index(X_train, val_idx)
                y_tr = self._safe_index(y_train, tr_idx)
                y_val = self._safe_index(y_train, val_idx)

                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[])
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)
                
                # Pilih df_features untuk evaluasi metrik
                df_val_features = None
                if self.df_features_all is not None:
                    try:
                        df_val_features = self._safe_index(self.df_features_all, val_idx)
                    except Exception:
                        df_val_features = self._safe_index(X_train, val_idx) if isinstance(X_train, pd.DataFrame) else None
                else:
                    df_val_features = self._safe_index(X_train, val_idx) if isinstance(X_train, pd.DataFrame) else None

                metrics = self._calculate_trading_metrics(
                    y_val, y_pred, y_proba, df_features=df_val_features
                )

                score = metrics.get(self.objective_metric, metrics.get("accuracy", 0.0))
                scores.append(score)

            return float(np.mean(scores)) if scores else -1e6

        best = self._tune_base_model(objective, self.n_trials)
        self._save_best_params({"lgbm": best})
        return best

    def tune_xgb(self, X_train: Union[pd.DataFrame, np.ndarray],
                 y_train: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Tune XGBoost hyperparameters menggunakan Profit-Weighted Accuracy."""
        self.logger.info("Starting XGBoost tuning...")

        def objective(trial):
            max_depth = trial.suggest_int("max_depth", 3, 12)

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 0.3, log=True),
                "max_depth": max_depth,
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            }

            params["min_split_loss"] = trial.suggest_float("min_split_loss", 0, 10.0)
            params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 0.5, 2.0)

            model = self._build_xgb(params)
            tscv = PurgedTimeSeriesSplit(n_splits=2, gap_ratio=0.02)

            self.logger.info(f"🎯 Using {self.objective_metric} as objective metric for XGBoost tuning")

            scores = []
            for tr_idx, val_idx in tscv.split(X_train):
                if len(tr_idx) < self.min_samples_per_fold or len(val_idx) < self.min_samples_per_fold:
                    self.logger.warning(f"⚠️ Skipping fold: too few samples (train={len(tr_idx)}, val={len(val_idx)})")
                    continue

                X_tr = self._safe_index(X_train, tr_idx)
                X_val = self._safe_index(X_train, val_idx)
                y_tr = self._safe_index(y_train, tr_idx)
                y_val = self._safe_index(y_train, val_idx)

                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)
                
                df_val_features = None
                if self.df_features_all is not None:
                    try:
                        df_val_features = self._safe_index(self.df_features_all, val_idx)
                    except Exception:
                        df_val_features = self._safe_index(X_train, val_idx) if isinstance(X_train, pd.DataFrame) else None
                else:
                    df_val_features = self._safe_index(X_train, val_idx) if isinstance(X_train, pd.DataFrame) else None

                metrics = self._calculate_trading_metrics(
                    y_val, y_pred, y_proba, df_features=df_val_features
                )

                score = metrics.get(self.objective_metric, metrics.get("accuracy", 0.0))
                scores.append(score)

            return float(np.mean(scores)) if scores else -1e6

        best = self._tune_base_model(objective, self.n_trials)
        self._save_best_params({"xgb": best})
        return best

    def tune_ensemble_weights(self, X_train: Union[pd.DataFrame, np.ndarray],
                              y_train: Union[pd.Series, np.ndarray],
                              lgbm_params: Dict[str, Any],
                              xgb_params: Dict[str, Any]) -> List[float]:
        """Tune ensemble weights menggunakan Profit-Weighted Accuracy."""
        self.logger.info("Starting ensemble weights tuning...")

        def objective(trial):
            w1 = trial.suggest_float("w1", 0.1, 5.0, log=True)
            w2 = trial.suggest_float("w2", 0.1, 5.0, log=True)

            model = VotingClassifier(
                estimators=[
                    ("lgbm", self._build_lgbm(lgbm_params)),
                    ("xgb", self._build_xgb(xgb_params)),
                ],
                voting="soft",
                weights=[w1, w2],
                n_jobs=-1,
            )

            tscv = PurgedTimeSeriesSplit(n_splits=2, gap_ratio=0.02)
            scores = []

            self.logger.info(f"🎯 Using {self.objective_metric} as objective metric for ensemble weights tuning")

            for tr_idx, val_idx in tscv.split(X_train):
                if len(tr_idx) < self.min_samples_per_fold or len(val_idx) < self.min_samples_per_fold:
                    self.logger.warning(f"⚠️ Skipping fold: too few samples (train={len(tr_idx)}, val={len(val_idx)})")
                    continue

                X_tr = self._safe_index(X_train, tr_idx)
                X_val = self._safe_index(X_train, val_idx)
                y_tr = self._safe_index(y_train, tr_idx)
                y_val = self._safe_index(y_train, val_idx)

                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)
                
                df_val_features = None
                if self.df_features_all is not None:
                    try:
                        df_val_features = self._safe_index(self.df_features_all, val_idx)
                    except Exception:
                        df_val_features = self._safe_index(X_train, val_idx) if isinstance(X_train, pd.DataFrame) else None
                else:
                    df_val_features = self._safe_index(X_train, val_idx) if isinstance(X_train, pd.DataFrame) else None

                metrics = self._calculate_trading_metrics(
                    y_val, y_pred, y_proba, df_features=df_val_features
                )

                score = metrics.get(self.objective_metric, metrics.get("accuracy", 0.0))
                scores.append(score)

            return float(np.mean(scores)) if scores else -1e6

        best = self._tune_base_model(objective, self.n_trials)
        best_weights = [best.get("w1", 1.0), best.get("w2", 1.0)]
        self.logger.info(f"Best ensemble weights found: {best_weights}")
        return best_weights

    # -------------------------
    # Model Save/Manage Methods
    # -------------------------
    def _save_model_if_improved(self, score: float, metric_name: str = "accuracy",
                               X: Optional[pd.DataFrame] = None) -> None:
        """Save model + FE bundle + score JSON hanya jika score lebih baik."""
        if not self.model_path:
            self.logger.warning("model_path not set ⇒ model not saved")
            return

        score_path = self.model_path.replace(".pkl", "_score.json")
        best_old = None
        if os.path.exists(score_path):
            try:
                with open(score_path, "r") as f:
                    best_old = json.load(f).get("best_score", None)
            except Exception:
                best_old = None

        save = False
        if best_old is None or score > float(best_old):
            save = True

        if save:
            try:
                os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
                joblib.dump(self.model, self.model_path)
                self.logger.info(f"Model saved to {self.model_path} (metric {metric_name}={score:.4f})")
            except Exception as e:
                self.logger.error(f"Error saving model file: {e}")
                return

            # FE bundle
            fe_bundle_path = self.model_path.replace(".pkl", "_fe.json")
            fe_bundle = {
                "fe_params": self.fe.params if (self.fe is not None and hasattr(self.fe, "params")) else {},
                "features": [],
                "label_mapping": self.label_mapping,
                "task_type": self.task_type,
            }

            try:
                lgbm_est = self.model.named_estimators_.get("lgbm")
                if lgbm_est is not None and hasattr(lgbm_est, "feature_name_"):
                    fe_bundle["features"] = list(getattr(lgbm_est, "feature_name_", []))
            except Exception:
                fe_bundle["features"] = []

            if not fe_bundle["features"] and X is not None and isinstance(X, pd.DataFrame):
                fe_bundle["features"] = list(X.columns)

            try:
                with open(fe_bundle_path, "w") as f:
                    json.dump(fe_bundle, f, indent=4)
                self.logger.info(f"FE bundle saved to {fe_bundle_path}")
            except Exception as e:
                self.logger.error(f"Error saving FE bundle: {e}")

            try:
                with open(score_path, "w") as f:
                    json.dump({"best_score": float(score), "metric": metric_name}, f, indent=4)
                self.logger.info(f"Score saved to {score_path}")
            except Exception as e:
                self.logger.error(f"Error saving score file: {e}")
        else:
            self.logger.info(f"New score {score:.4f} not better than existing best {best_old}, skipping save.")

    # -------------------------
    # Monitoring Methods
    # -------------------------
    def monitor_feature_importance(self) -> Dict[str, np.ndarray]:
        """Collect feature importance dari LGBM dan XGB."""
        if self.model is None:
            return {}

        importance: Dict[str, np.ndarray] = {}

        try:
            lgbm_est = getattr(self.model, "named_estimators_", {}).get("lgbm")
            if lgbm_est is not None and hasattr(lgbm_est, "feature_importances_"):
                importance["lgbm"] = np.array(lgbm_est.feature_importances_)
        except Exception:
            pass

        try:
            xgb_est = getattr(self.model, "named_estimators_", {}).get("xgb")
            if xgb_est is not None and hasattr(xgb_est, "feature_importances_"):
                importance["xgb"] = np.array(xgb_est.feature_importances_)
        except Exception:
            pass

        ts = time.time()
        self.feature_importance_history[ts] = importance

        if len(self.feature_importance_history) > 10:
            oldest = min(self.feature_importance_history.keys())
            del self.feature_importance_history[oldest]

        return importance

    # -------------------------
    # Prediction Methods
    # -------------------------
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict_proba(X)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict dan decode labels untuk multiclass tasks."""
        if self.model is None:
            raise ValueError("Model not trained.")

        y_pred_encoded = self.model.predict(X)

        if self.task_type == "multiclass" and self.inverse_mapping:
            return np.array([self.inverse_mapping[int(i)] for i in y_pred_encoded])

        return y_pred_encoded

    def predict_with_confidence(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        market_regime: str = "normal"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict dengan confidence threshold dan regime adjustment."""
        if self.model is None:
            raise ValueError("Model not trained.")

        probs = self.model.predict_proba(X)
        raw_pred_encoded = self.model.predict(X)
        conf_scores = np.max(probs, axis=1)
        adjusted = probs.copy()

        if market_regime == "high_volatility":
            adjusted *= 0.85
            threshold = self.confidence_threshold * 1.2
        elif market_regime == "low_liquidity":
            adjusted *= 0.7
            threshold = self.confidence_threshold * 1.5
        else:
            threshold = self.confidence_threshold

        preds_encoded = raw_pred_encoded.copy().astype(int)
        if self.task_type == "multiclass":
            neutral = 1
            preds_encoded[conf_scores < threshold] = neutral
        else:
            preds_encoded[conf_scores < threshold] = 0

        if self.task_type == "multiclass" and self.inverse_mapping:
            preds = np.array([self.inverse_mapping[int(i)] for i in preds_encoded])
        else:
            preds = preds_encoded

        counts = int(np.sum(conf_scores >= threshold))
        self.logger.info(f"Predictions with confidence >= {threshold:.2f}: {counts}/{len(preds)}")

        return preds, adjusted, conf_scores

    # -------------------------
    # Build Ensemble & Train - UPDATED WITH COMPREHENSIVE METRICS
    # -------------------------
    def build_model(self, lgbm_params: Optional[Dict] = None,
                    xgb_params: Optional[Dict] = None,
                    weights: Optional[List[float]] = None) -> VotingClassifier:
        """Build the ensemble model."""
        lgbm = self._build_lgbm(lgbm_params)
        xgb_model = self._build_xgb(xgb_params)

        ensemble = VotingClassifier(
            estimators=[("lgbm", lgbm), ("xgb", xgb_model)],
            voting="soft",
            weights=weights or [1, 1],
            n_jobs=-1,
        )
        return ensemble

    def train(self, X: Union[pd.DataFrame, np.ndarray],
              y: Union[pd.Series, np.ndarray],
              task_type: str = "multiclass") -> VotingClassifier:
        """Main training loop dengan comprehensive metrics reporting."""
        self.task_type = task_type
        unique = np.unique(y)

        # LOGGING: Training data overview
        self.logger.info("🎯 TRAINING DATA OVERVIEW:")
        self.logger.info(f"📊 Samples: {len(X)}")
        self.logger.info(f"📊 Features: {X.shape[1]}")
        self.logger.info(f"📊 Task type: {task_type}")
        self.logger.info(f"📊 Original labels: {unique.tolist()}")
        self.logger.info(f"📊 Label distribution: {pd.Series(y).value_counts().to_dict()}")

        # Feature statistics
        if isinstance(X, pd.DataFrame):
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            self.logger.info(f"📊 Numeric features: {len(numeric_cols)}")

        # Label encoding untuk multiclass
        if task_type == "multiclass":
            valid = {-1, 0, 1}
            if not set(unique).issubset(valid):
                raise ValueError(f"Multiclass labels must be subset of {-1,0,1}, found {unique}")

            self.label_mapping = {-1: 0, 0: 1, 1: 2}
            self.inverse_mapping = {0: -1, 1: 0, 2: 1}
            y_encoded = np.array([self.label_mapping[int(v)] for v in y], dtype=int)
        else:
            self.label_mapping = None
            self.inverse_mapping = None
            y_encoded = np.array(y, dtype=int)

        # Tuning atau load best params
        best_from_file = self._load_best_params()

        if self.tune:
            self.logger.info("🎛️ TUNING CONFIGURATION:")
            self.logger.info(f"   - Tuning enabled: {self.tune}")
            self.logger.info(f"   - Number of trials: {self.n_trials}")
            self.logger.info(f"   - Objective metric: {self.objective_metric}")
            
            self.logger.info("🔍 Starting LightGBM tuning...")
            lgbm_params = self.tune_lgbm(X, y_encoded)
            
            self.logger.info("🔍 Starting XGBoost tuning...")
            xgb_params = self.tune_xgb(X, y_encoded)
            
            self.logger.info("🔍 Tuning ensemble weights...")
            weights = self.tune_ensemble_weights(X, y_encoded, lgbm_params, xgb_params)
            
            self.best_params = {"lgbm": lgbm_params, "xgb": xgb_params, "weights": weights}
            self._save_best_params(self.best_params)
        elif best_from_file:
            self.best_params = best_from_file
            lgbm_params = self.best_params.get("lgbm")
            xgb_params = self.best_params.get("xgb")
            weights = self.best_params.get("weights", [1, 1])
            self.logger.info("✅ Loaded best parameters from file")
        else:
            lgbm_params = xgb_params = None
            weights = [1, 1]
            self.logger.info("ℹ️ Using default parameters")

        # Build & fit model
        self.model = self.build_model(lgbm_params, xgb_params, weights)
        self.logger.info("🏋️ Training started...")

        start_time = time.time()
        self.model.fit(X, y_encoded)
        training_time = time.time() - start_time

        self.logger.info(f"✅ Training finished in {training_time:.2f} seconds")
        self.logger.info("📈 Running comprehensive in-sample evaluation...")

        # COMPREHENSIVE EVALUATION
        try:
            y_pred_encoded = self.model.predict(X)
            y_pred = (
                np.array([self.inverse_mapping[int(i)] for i in y_pred_encoded])
                if self.task_type == "multiclass" and self.inverse_mapping
                else y_pred_encoded
            )
            y_proba = self.model.predict_proba(X)
            
            # Get features for metric calculation
            df_features_eval = self.df_features_all if self.df_features_all is not None else (
                X if isinstance(X, pd.DataFrame) else None
            )

            # Calculate comprehensive metrics
            metrics = self._calculate_trading_metrics(
                y_true=y,
                y_pred=y_pred,
                y_proba=y_proba,
                df_features=df_features_eval
            )

            # COMPREHENSIVE SUMMARY REPORT
            self.logger.info("🎯 COMPREHENSIVE TRAINING RESULTS:")
            self.logger.info("📊 CORE ACCURACY METRICS:")
            self.logger.info(f"   - Accuracy: {metrics.get('accuracy', 0):.4f}")
            self.logger.info(f"   - Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
            
            self.logger.info("📊 TRADING-AWARE METRICS:")
            self.logger.info(f"   - Profit-Weighted Accuracy: {metrics.get('profit_weighted_accuracy', 0):.4f}")
            self.logger.info(f"   - High-Confidence Accuracy: {metrics.get('high_conf_accuracy', 0):.4f}")
            self.logger.info(f"   - Win Rate: {metrics.get('win_rate', 0):.4f}")
            
            self.logger.info("📊 CLASSIFICATION METRICS:")
            if task_type == "multiclass":
                self.logger.info(f"   - F1 Macro: {metrics.get('f1_macro', 0):.4f}")
                self.logger.info(f"   - Precision Macro: {metrics.get('precision_macro', 0):.4f}")
                self.logger.info(f"   - Recall Macro: {metrics.get('recall_macro', 0):.4f}")
            else:
                self.logger.info(f"   - F1: {metrics.get('f1', 0):.4f}")
                self.logger.info(f"   - Precision: {metrics.get('precision', 0):.4f}")
                self.logger.info(f"   - Recall: {metrics.get('recall', 0):.4f}")
                self.logger.info(f"   - AUC: {metrics.get('auc', 0):.4f}")
            
            self.logger.info("📊 RISK METRICS:")
            self.logger.info(f"   - Max Drawdown: {metrics.get('max_drawdown', 0):.4f}")
            self.logger.info(f"   - Simulated Profit Factor: {metrics.get('simulated_profit_factor', 0):.2f}")
            self.logger.info(f"   - Average Confidence: {metrics.get('avg_confidence', 0):.4f}")
            
            # Real trading metrics jika available
            if 'real_sharpe_ratio' in metrics:
                self.logger.info("📊 REAL TRADING METRICS:")
                self.logger.info(f"   - Real Sharpe Ratio: {metrics.get('real_sharpe_ratio', 0):.4f}")
                self.logger.info(f"   - Real Max Drawdown: {metrics.get('real_max_drawdown', 0):.4f}")

            # Store semua metrics
            metrics["training_time"] = training_time
            unique_pred, counts = np.unique(y_pred, return_counts=True)
            metrics["class_counts"] = dict(zip([int(u) for u in unique_pred], [int(c) for c in counts]))
            
            self.training_metrics = metrics

            # Classification report detail
            try:
                report = classification_report(y, y_pred, digits=4, output_dict=False)
                self.logger.info("📊 DETAILED CLASSIFICATION REPORT:")
                self.logger.info(report)
            except Exception as e:
                self.logger.debug(f"Detailed classification report skipped: {e}")

        except Exception as e:
            self.logger.error(f"❌ Error during comprehensive evaluation: {e}")
            self.training_metrics = {}

        # Feature importance logging
        try:
            fi = self.monitor_feature_importance()
            if 'lgbm' in fi and len(fi['lgbm']) > 0 and isinstance(X, pd.DataFrame):
                top_features_idx = np.argsort(fi['lgbm'])[-10:]
                top_features = X.columns[top_features_idx].tolist()
                top_importance = fi['lgbm'][top_features_idx]
                self.logger.info("🏆 TOP 10 FEATURES (LightGBM):")
                for feat, imp in zip(top_features, top_importance):
                    self.logger.info(f"   - {feat}: {imp:.4f}")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not extract feature importance: {e}")

        # Save model berdasarkan objective metric
        main_metric = self.objective_metric
        score = float(self.training_metrics.get(main_metric, 0.0))

        if score == 0.0 and main_metric not in self.training_metrics:
            self.logger.warning(f"⚠️ Objective metric '{main_metric}' not found, fallback to accuracy")
            main_metric = "accuracy"
            score = float(self.training_metrics.get(main_metric, 0.0))

        self.logger.info(f"💾 Model improvement check based on {main_metric}={score:.4f}")
        try:
            self._save_model_if_improved(score, main_metric, X if isinstance(X, pd.DataFrame) else None)
        except Exception as e:
            self.logger.error(f"❌ Error while trying to save model: {e}")

        return self.model


# -------------------------
# Example / quick test
# -------------------------
if __name__ == "__main__":
    logger = setup_logger("EnhancedScalpingAI")
    logger.info("=== EnsembleTrainer example run ===")

    # Dummy data dengan kolom close untuk testing
    X = pd.DataFrame(np.random.rand(200, 20), columns=[f"f{i}" for i in range(20)])
    X['close'] = np.cumprod(1 + np.random.normal(0, 0.01, 200))  # Simulated price series
    y = pd.Series(np.random.choice([-1, 0, 1], size=200))

    # Feature engineering
    fe = FeatureEngineering(logger=logger)

    # Use config from settings.py
    trainer = EnsembleTrainer(
        logger=logger,
        config=Config.ENSEMBLE_CONFIG,
        fe=fe,
        df_features_all=X  # Provide features with close price
    )

    model = trainer.train(X, y, task_type="multiclass")
    preds = trainer.predict(X)
    logger.info(f"Example predictions (first 10): {preds[:10]}")
