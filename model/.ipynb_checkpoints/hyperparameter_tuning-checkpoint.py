import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from typing import Dict, Any
import logging
from config.settings import Config

class HyperparameterOptimizer:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def optimize(self, X_train, y_train) -> Dict[str, Any]:
        """Optimize hyperparameters using GridSearchCV"""
        self.logger.info("Starting hyperparameter optimization...")
        
        # Use smaller parameter grid for faster optimization
        param_grid = Config.PARAM_GRID
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=Config.CV_SPLITS)
        
        # Base model
        model = xgb.XGBClassifier(
            random_state=Config.RANDOM_STATE,
            eval_metric=Config.DEFAULT_MODEL_PARAMS['eval_metric'],
            use_label_encoder=False
        )
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=tscv,
            scoring=Config.CV_SCORING,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_