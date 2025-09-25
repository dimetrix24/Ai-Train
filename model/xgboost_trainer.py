import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import xgboost as xgb
import optuna

from utils.logger import get_logger, setup_logger
from data_processing.feature_engineering import FeatureEngineering


def check_xgboost_cuda_support() -> bool:
    """Cek apakah XGBoost build memiliki dukungan CUDA / GPU."""
    info = xgb.build_info()
    return info.get("USE_CUDA", False)


class XGBoostTrainer:
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        tune: bool = False,
        n_trials: int = 50,
        model_path: Optional[str] = None,
        fe: FeatureEngineering = None,   # Tambahan: FE instance
    ):
        self.logger = logger or get_logger("XGBoostTrainer")
        self.model: Optional[XGBClassifier] = None
        self.device = "cpu"
        self.tune = tune
        self.n_trials = n_trials
        self.best_params: Dict[str, Any] = {}
        self.task_type: str = "multiclass"
        self.model_path = model_path
        self.label_mapping: Dict[int, int] = None
        self.inverse_mapping: Dict[int, int] = None
        self.fe = fe   # Simpan reference ke FeatureEngineering

    # =============================
    # Build Model
    # =============================
    def _build_xgb(self, params: Optional[Dict] = None):
        has_cuda = check_xgboost_cuda_support()
        self.device = "cuda" if has_cuda else "cpu"
        self.logger.info(f"XGBoost device: {self.device}")

        xgb_params = {
            "n_estimators": params.get("n_estimators", 500) if params else 500,
            "learning_rate": params.get("learning_rate", 0.01) if params else 0.05,
            "max_depth": params.get("max_depth", 6) if params else 6,
            "subsample": params.get("subsample", 0.8) if params else 0.8,
            "colsample_bytree": params.get("colsample_bytree", 0.8) if params else 0.8,
            "reg_alpha": params.get("reg_alpha", 0) if params else 0,
            "reg_lambda": params.get("reg_lambda", 0) if params else 0,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "device": self.device,
        }

        if self.task_type == "multiclass":
            xgb_params["objective"] = "multi:softprob"
            xgb_params["eval_metric"] = "mlogloss"
            xgb_params["num_class"] = 3
        else:
            xgb_params["objective"] = "binary:logistic"
            xgb_params["eval_metric"] = "logloss"

        return XGBClassifier(**xgb_params)

    # =============================
    # Hyperparameter tuning
    # =============================
    def _tune_xgb(self, X_train, y_train):
        self.logger.info("🔧 Mulai tuning XGBoost...")

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 800),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.04, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 2.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            }
            model = self._build_xgb(params)

            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                scores.append(model.score(X_val, y_val))
            return np.mean(scores)

        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.optimize(objective, n_trials=self.n_trials)
        self.logger.info(f"🎯 Best params: {study.best_params}, CV score: {study.best_value:.4f}")
        return study.best_params

    # =============================
    # Training
    # =============================
    def train(self, X, y, task_type: str = "multiclass"):
        self.task_type = task_type
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        self.logger.info(f"📊 Mulai training XGBoost: {X.shape}, task={task_type}, n_classes={n_classes}")

        # Label encoding untuk multiclass
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

        # Tuning opsional
        if self.tune:
            xgb_params = self._tune_xgb(X, pd.Series(y_encoded))
            self.best_params = xgb_params
        else:
            xgb_params = None

        # Build & train
        self.model = self._build_xgb(xgb_params)
        self.model.fit(X, y_encoded, verbose=False)
        self.logger.info("✅ Training selesai")

        # Evaluasi training score
        if task_type == "binary":
            y_proba = self.model.predict_proba(X)[:, 1]
            score = roc_auc_score(y_encoded, y_proba)
            metric_name = "AUC"
        else:
            y_pred = self.model.predict(X)
            score = accuracy_score(y_encoded, y_pred)
            metric_name = "Accuracy"
        self.logger.info(f"📊 Training {metric_name}: {score:.4f}")

        # Save model & FE params
        if self.model_path:
            score_file = self.model_path.replace(".pkl", "_score.json")

            best_old = None
            if os.path.exists(score_file):
                with open(score_file, "r") as f:
                    best_old = json.load(f).get("best_score", None)

            if best_old is None or score > best_old:
                joblib.dump(self.model, self.model_path)
                self.logger.info(f"💾 Model terbaik disimpan ke: {self.model_path}")

                if self.fe is not None:
                    fe_params_path = self.model_path.replace(".pkl", "_fe_params.json")
                    self.fe.save_params(fe_params_path)
                    self.logger.info(f"📝 FE params disimpan ke: {fe_params_path}")

                with open(score_file, "w") as f:
                    json.dump({"best_score": score, "metric": metric_name}, f, indent=4)
            else:
                self.logger.warning(f"⚠️ Skor {score:.4f} lebih rendah dari best {best_old:.4f}, tidak overwrite model.")

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
            return np.array([self.inverse_mapping[label] for label in y_pred])
        return y_pred


# =============================
# Contoh entry point
# =============================
if __name__ == "__main__":
    logger = setup_logger("XGBoostTrainer")
    logger.info("=== Memulai XGBoostTrainer ===")

    # Dummy data
    X = pd.DataFrame(np.random.rand(200, 20))
    y = pd.Series(np.random.choice([-1, 0, 1], size=200))

    fe = FeatureEngineering(logger=logger)
    trainer = XGBoostTrainer(
        logger=logger,
        tune=True,
        n_trials=10,
        model_path="outputs/models/xgb_multiclass.pkl",
        fe=fe
    )

    model = trainer.train(X, y, task_type="multiclass")
    preds = trainer.predict(X, decode=True)
    logger.info(f"Contoh prediksi: {preds[:10]}")