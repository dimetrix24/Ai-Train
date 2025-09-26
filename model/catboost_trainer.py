import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
import optuna
from collections import deque
from utils.logger import get_logger, setup_logger
from data_processing.feature_engineering import FeatureEngineering
from utils.purge_time_series import PurgedTimeSeriesSplit
from data_processing.market_regime import detect_market_regime_series
try:
    from utils.onnx_utils import export_to_onnx
except ImportError:
    export_to_onnx = None
    


class CatBoostTrainer:
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        tune: bool = False,
        n_trials: int = 50,
        model_path: Optional[str] = None,
        fe: FeatureEngineering = None,
        objective_metric: str = "profit_weighted_accuracy",
        min_samples_per_fold: int = 50,
        gap_ratio: float = 0.02,
        n_splits: int = 5,
    ):
        self.logger = logger or get_logger("CatBoostTrainer")
        self.model: Optional[CatBoostClassifier] = None
        self.device = "cpu"
        self.tune = tune
        self.n_trials = n_trials
        self.best_params: Dict[str, Any] = {}
        self.task_type: str = "multiclass"
        self.model_path = model_path
        self.label_mapping: Dict[int, int] = None
        self.inverse_mapping: Dict[int, int] = None
        self.fe = fe
        self.objective_metric = objective_metric
        self.min_samples_per_fold = min_samples_per_fold
        self.gap_ratio = gap_ratio
        self.n_splits = n_splits

    # =============================
    # METRIK TRADING
    # =============================
    def profit_weighted_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               df_features: Optional[pd.DataFrame] = None) -> float:
        try:
            if df_features is None or 'close' not in df_features.columns:
                return float(accuracy_score(y_true, y_pred))
            close_prices = df_features['close'].values
            if len(close_prices) < 2 or len(y_true) < 2:
                return float(accuracy_score(y_true, y_pred))
            returns = np.diff(close_prices) / close_prices[:-1]
            min_len = min(len(y_true)-1, len(returns), len(y_pred)-1)
            if min_len <= 0:
                return float(accuracy_score(y_true, y_pred))
            y_true_eval = y_true[1:min_len+1]
            y_pred_eval = y_pred[1:min_len+1]
            returns_eval = returns[:min_len]
            weights = np.abs(returns_eval)
            if np.max(weights) > 0:
                weights = weights / np.max(weights)
            correct_predictions = (y_true_eval == y_pred_eval).astype(float)
            if np.sum(weights) == 0:
                return float(accuracy_score(y_true_eval, y_pred_eval))
            return float(np.average(correct_predictions, weights=weights))
        except Exception as e:
            self.logger.warning(f"Profit-weighted accuracy failed: {e}")
            return float(accuracy_score(y_true, y_pred))

    def directional_accuracy_with_confidence(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                           y_proba: Optional[np.ndarray] = None, 
                                           threshold: float = 0.6) -> float:
        if y_proba is None or len(y_proba) == 0:
            return float(accuracy_score(y_true, y_pred))
        try:
            confidences = np.max(y_proba, axis=1)
            high_conf_mask = confidences >= threshold
            if np.sum(high_conf_mask) == 0:
                return 0.0
            return float(accuracy_score(y_true[high_conf_mask], y_pred[high_conf_mask]))
        except Exception as e:
            self.logger.warning(f"High-confidence accuracy failed: {e}")
            return float(accuracy_score(y_true, y_pred))

    def calculate_max_drawdown(self, returns: np.ndarray) -> float:
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
        metrics = {}
        try:
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        except Exception:
            metrics["accuracy"] = metrics["balanced_accuracy"] = 0.0

        metrics["profit_weighted_accuracy"] = self.profit_weighted_accuracy(y_true, y_pred, df_features)
        metrics["high_conf_accuracy"] = self.directional_accuracy_with_confidence(
            y_true, y_pred, y_proba, threshold=0.6
        )
        try:
            metrics["avg_confidence"] = float(np.max(y_proba, axis=1).mean()) if y_proba is not None else 0.0
        except Exception:
            metrics["avg_confidence"] = 0.0

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

        try:
            correct_trades = (y_true == y_pred)
            win_rate = np.mean(correct_trades)
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

        if df_features is not None and 'close' in df_features.columns:
            try:
                close_prices = df_features['close'].values
                if len(close_prices) > 1:
                    returns = np.diff(close_prices) / close_prices[:-1]
                    min_len = min(len(returns), len(y_pred)-1)
                    if min_len > 0:
                        pred_direction = np.sign(y_pred[1:min_len+1])
                        realized_returns = pred_direction * returns[:min_len]
                        metrics["real_avg_return"] = float(np.mean(realized_returns))
                        metrics["real_volatility"] = float(np.std(realized_returns))
                        metrics["real_sharpe_ratio"] = float(metrics["real_avg_return"] / (metrics["real_volatility"] + 1e-12))
                        metrics["real_max_drawdown"] = self.calculate_max_drawdown(realized_returns)
            except Exception as e:
                self.logger.debug(f"Real trading metrics skipped: {e}")

        return metrics

    # =============================
    # Build Model
    # =============================
    def _build_catboost(self, params: Optional[Dict] = None):
        # Cek GPU
        try:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(task_type="GPU", iterations=1, verbose=True)
            model.fit([[1.0]], [0])
            self.device = "GPU"
        except Exception:
            self.device = "CPU"
        self.logger.info(f"CatBoost device: {self.device}")

        cb_params = {
            "iterations": params.get("iterations", 200) if params else 200,
            "learning_rate": params.get("learning_rate", 0.02) if params else 0.02,
            "depth": params.get("depth", 6) if params else 6,
            "l2_leaf_reg": params.get("l2_leaf_reg", 3.0) if params else 3.0,
            "random_strength": params.get("random_strength", 1.0) if params else 1.0,
            "subsample": params.get("subsample", 0.8) if params else 0.8,
            "rsm": params.get("rsm", 0.8) if params else 0.8,
            "min_data_in_leaf": params.get("min_data_in_leaf", 100) if params else 100,
            "grow_policy": "SymmetricTree",
            "bootstrap_type": "Bernoulli",
            "eval_metric": "MultiClass" if self.task_type == "multiclass" else "Logloss",
            "loss_function": "MultiClass" if self.task_type == "multiclass" else "Logloss",
            "task_type": self.device,
            "random_seed": 42,
            "verbose": False,
        }

        if self.task_type == "multiclass":
            cb_params["class_names"] = [0, 1, 2]

        return CatBoostClassifier(**cb_params)

    # =============================
    # Hyperparameter Tuning
    # =============================
    def _tune_catboost(self, X_train, y_train, df_features_all: Optional[pd.DataFrame] = None):
        self.logger.info(f"🔧 Tuning CatBoost dengan objective: {self.objective_metric}...")

        def objective(trial):
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.0005, 0.1, log=True),
                "depth": trial.suggest_int("depth", 2, 16),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0),
                "random_strength": trial.suggest_float("random_strength", 1.0, 10.0),
                "subsample": trial.suggest_float("subsample", 0.7, 0.95),
                "rsm": trial.suggest_float("rsm", 0.6, 0.9),
            }
            model = self._build_catboost(params)

            tscv = PurgedTimeSeriesSplit(n_splits=self.n_splits, gap_ratio=self.gap_ratio)
            scores = []

            for tr_idx, val_idx in tscv.split(X_train):
                if len(tr_idx) < self.min_samples_per_fold or len(val_idx) < self.min_samples_per_fold:
                    continue

                X_tr = X_train.iloc[tr_idx] if isinstance(X_train, pd.DataFrame) else X_train[tr_idx]
                X_val = X_train.iloc[val_idx] if isinstance(X_train, pd.DataFrame) else X_train[val_idx]
                y_tr = y_train.iloc[tr_idx] if isinstance(y_train, pd.Series) else y_train[tr_idx]
                y_val = y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]

                try:
                    model.fit(
                        X_tr, y_tr,
                        eval_set=(X_val, y_val),
                        use_best_model=True,
                        early_stopping_rounds=20,
                        verbose=False
                    )
                except Exception as e:
                    return -1e6

                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)

                df_val_features = None
                if df_features_all is not None:
                    try:
                        df_val_features = df_features_all.iloc[val_idx]
                    except Exception:
                        pass

                metrics = self._calculate_trading_metrics(
                    y_val, y_pred, y_proba, df_features=df_val_features
                )
                score = metrics.get(self.objective_metric, metrics.get("accuracy", 0.0))
                scores.append(score)

            return float(np.mean(scores)) if scores else -1e6

        pruner = optuna.pruners.MedianPruner(n_startup_trials=5)
        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.optimize(objective, n_trials=self.n_trials)
        self.logger.info(f"🎯 Best {self.objective_metric}: {study.best_value:.4f}")
        self.best_params = study.best_params
        return study.best_params

    # =============================
    # Training
    # =============================
    def train(self, X, y, task_type: str = "multiclass", df_features_all: Optional[pd.DataFrame] = None):
        self.task_type = task_type
        unique_classes = np.unique(y)
        self.logger.info(f"📊 Training CatBoost: {X.shape}, task={task_type}, classes={unique_classes}")

        if task_type == "multiclass":
            valid_labels = {-1, 0, 1}
            if not set(unique_classes).issubset(valid_labels):
                raise ValueError(f"Label tidak valid: {unique_classes}, harus {-1,0,1}")
            self.label_mapping = {-1: 0, 0: 1, 1: 2}
            self.inverse_mapping = {0: -1, 1: 0, 2: 1}
            y_encoded = np.array([self.label_mapping[label] for label in y])
        else:
            y_encoded = y
            self.label_mapping = None
            self.inverse_mapping = None

        if self.tune:
            cb_params = self._tune_catboost(X, pd.Series(y_encoded), df_features_all)
        else:
            cb_params = None

        self.model = self._build_catboost(cb_params)
        self.model.fit(X, y_encoded, verbose=False)
        self.logger.info("✅ Training selesai")

        # Evaluasi akhir
        y_pred_final = self.model.predict(X)
        y_proba_final = self.model.predict_proba(X)
        final_metrics = self._calculate_trading_metrics(
            y_encoded if task_type == "multiclass" else y,
            y_pred_final,
            y_proba_final,
            df_features=df_features_all
        )
        main_score = final_metrics.get(self.objective_metric, final_metrics["accuracy"])
        self.logger.info(f"📊 Final {self.objective_metric}: {main_score:.4f}")

        # Simpan model (.pkl)
        if self.model_path:
            score_file = self.model_path.replace(".pkl", "_score.json")
            best_old = None
            if os.path.exists(score_file):
                with open(score_file, "r") as f:
                    best_old = json.load(f).get("best_score", None)

            if best_old is None or main_score > best_old:
                joblib.dump(self.model, self.model_path)
                self.logger.info(f"💾 Model disimpan ke: {self.model_path}")

                if self.fe is not None:
                    fe_params_path = self.model_path.replace(".pkl", "_fe_params.json")
                    self.fe.save_params(fe_params_path)

                with open(score_file, "w") as f:
                    json.dump({
                        "best_score": main_score,
                        "metric": self.objective_metric,
                        "params": self.best_params if self.tune else "default"
                    }, f, indent=4)
            else:
                self.logger.warning(f"⚠️ Skor {main_score:.4f} ≤ best {best_old:.4f}, tidak simpan.")

        return self.model

    # =============================
    # Prediction
    # =============================
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model belum dilatih.")
        return self.model.predict_proba(X)

    def predict(self, X, decode: bool = True):
        if self.model is None:
            raise ValueError("Model belum dilatih.")
        y_pred = self.model.predict(X)
        if decode and self.task_type == "multiclass" and self.inverse_mapping is not None:
            return np.array([self.inverse_mapping[int(label)] for label in y_pred])
        return y_pred

    def predict_with_confidence(
        self,
        X,
        threshold: float = 0.6,
        decode: bool = True,
        df_features: Optional[pd.DataFrame] = None
        ):
        """
        Prediksi dengan confidence threshold + deteksi market regime.
        Return dict berisi preds, proba, confidences, regime.
        """
        if self.model is None:
            raise ValueError("Model belum dilatih.")

        # Probabilitas
        y_proba = self.model.predict_proba(X)
        confidences = np.max(y_proba, axis=1)
        raw_preds = np.argmax(y_proba, axis=1)

        # Decode label
        if decode and self.task_type == "multiclass" and self.inverse_mapping is not None:
            y_pred = np.array([self.inverse_mapping[int(label)] for label in raw_preds])
        else:
            y_pred = raw_preds

        # Filter by confidence
        y_pred_conf = []
        for p, c in zip(y_pred, confidences):
            if c >= threshold:
                y_pred_conf.append(p)
            else:
                y_pred_conf.append(None)  # bisa fallback ke 0 kalau mau
        y_pred_conf = np.array(y_pred_conf, dtype=object)

        # Market regime (opsional)
        regime = None
        if df_features is not None:
            try:
                regime = detect_market_regime_series(df_features)
            except Exception as e:
                self.logger.warning(f"Market regime detection gagal: {e}")

        return {
            "preds": y_pred_conf,
            "proba": y_proba,
            "confidences": confidences,
            "regime": regime
        }

    # =============================
    # ONNX Export (FIXED)
    # =============================
    def export_to_onnx(self, X_sample, prefix="catboost"):
        """
        Ekspor model CatBoost ke ONNX + simpan metadata fitur & mapping.
        Lebih aman pakai save_model(format="onnx") daripada convert_to_onnx.
        """
        if self.model is None:
            raise ValueError("Model belum dilatih.")

        onnx_path = f"{prefix}.onnx"

        try:
            # ✅ Export langsung dari CatBoost
            self.model.save_model(onnx_path, format="onnx")
        except Exception as e:
            raise RuntimeError(f"ONNX export gagal via CatBoost.save_model: {e}")

        # Tentukan fitur asli
        if hasattr(X_sample, "columns"):
            original_cols = list(X_sample.columns)
        else:
            original_cols = [f"f{i}" for i in range(X_sample.shape[1])]

        feature_map = {f"f{i}": col for i, col in enumerate(original_cols)}

        # Simpan wrapper info
        wrapper_info = {
            "features": original_cols,
            "task_type": self.task_type,
            "label_mapping": self.label_mapping,
            "inverse_mapping": self.inverse_mapping,
            "feature_map_file": f"{prefix}_feature_map.json",
            "model_type": "catboost"
        }
        joblib.dump(wrapper_info, f"{prefix}_wrapper.pkl")

        # Simpan feature map JSON
        with open(f"{prefix}_feature_map.json", "w") as f:
            json.dump(feature_map, f, indent=4)

        self.logger.info(f"✅ ONNX export sukses: {onnx_path}, {prefix}_wrapper.pkl")


# =============================
# Fungsi Inference Eksternal
# =============================
def catboost_predict(X, prefix="catboost"):
    """Prediksi menggunakan model CatBoost dalam format ONNX."""
    import onnxruntime as rt
    import numpy as np

    # Load wrapper info
    wrapper_info = joblib.load(f"{prefix}_wrapper.pkl")
    with open(wrapper_info["feature_map_file"], "r") as f:
        feature_map = json.load(f)

    # Urutkan fitur sesuai training
    ordered_features = [feature_map[f"f{i}"] for i in range(len(feature_map))]
    if hasattr(X, "loc"):
        X = X[ordered_features].to_numpy(dtype=np.float32)
    else:
        X = np.array(X, dtype=np.float32)

    # Jalankan inference
    sess = rt.InferenceSession(f"{prefix}.onnx", providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: X})

    # ✅ Handle output fleksibel
    if len(outputs) == 2:
        labels, probs = outputs
    else:
        probs = outputs[0]
        labels = np.argmax(probs, axis=1)

    # Mapping balik label kalau ada
    if wrapper_info["inverse_mapping"]:
        labels = np.array([wrapper_info["inverse_mapping"].get(int(p), int(p)) for p in labels])

    return labels, probs

def catboost_predict_with_confidence(
    X,
    prefix="catboost",
    threshold: float = 0.6,
    df_features: Optional[pd.DataFrame] = None
    ):
    """Prediksi ONNX dengan confidence threshold + market regime."""
    labels, probs = catboost_predict(X, prefix=prefix)
    confidences = np.max(probs, axis=1)

    preds_conf = []
    for l, c in zip(labels, confidences):
        preds_conf.append(l if c >= threshold else None)
    preds_conf = np.array(preds_conf, dtype=object)

    regime = None
    if df_features is not None:
        try:
            regime = detect_market_regime_series(df_features)
        except Exception as e:
            print(f"⚠️ Market regime detection gagal: {e}")

    return {
        "preds": preds_conf,
        "proba": probs,
        "confidences": confidences,
        "regime": regime
    }

# =============================
# Contoh Penggunaan
# =============================
if __name__ == "__main__":
    logger = setup_logger("CatBoostTrainer")
    logger.info("=== Memulai CatBoostTrainer ===")

    # Data dummy
    X = pd.DataFrame(np.random.rand(500, 20), columns=[f"f{i}" for i in range(20)])
    y = pd.Series(np.random.choice([-1, 0, 1], size=500))
    df_features = pd.DataFrame({
        "close": np.cumprod(1 + np.random.randn(500) * 0.001) * 1900
    })

    fe = FeatureEngineering(logger=logger)
    trainer = CatBoostTrainer(
        logger=logger,
        tune=True,
        n_trials=10,
        model_path="outputs/models/catboost_scalping.pkl",
        fe=fe,
        objective_metric="profit_weighted_accuracy",
        n_splits=3,
        gap_ratio=0.05
    )

    model = trainer.train(X, y, task_type="multiclass", df_features_all=df_features)
    trainer.export_to_onnx(X.head(1), prefix="outputs/models/catboost_scalping")

    # Contoh prediksi
    preds, probs = catboost_predict(X.head(5), prefix="outputs/models/catboost_scalping")
    logger.info(f"Prediksi ONNX: {preds}")
    
    # Contoh prediksi dengan confidence
    result_conf = catboost_predict_with_confidence(
        X.head(5), 
        prefix="outputs/models/catboost_scalping",
        threshold=0.6,
        df_features=df_features.head(5)
    )
    logger.info(f"Prediksi dengan confidence: {result_conf['preds']}")
    logger.info(f"Market regime: {result_conf['regime']}")