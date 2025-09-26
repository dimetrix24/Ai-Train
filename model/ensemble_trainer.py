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
    logger.info(f"Prediksi: {preds[:10]}")