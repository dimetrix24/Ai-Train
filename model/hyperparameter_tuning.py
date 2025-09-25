# model/hyperparameter_tuning.py
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from typing import Dict, Any, Callable
import logging
from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    General hyperparameter optimizer using Optuna for models like LightGBM, XGBoost.
    Supports time-series CV for forex scalping data.
    """

    def __init__(
        self,
        logger: logging.Logger = logger,
        n_trials: int = 50,
        cv_splits: int = 3,
        scoring: str = "accuracy",
        direction: str = "maximize"
    ):
        self.logger = logger
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.scoring = scoring
        self.direction = direction
        self.study = None
        self.best_params = None

    def optimize(
        self,
        model_class: Callable,
        param_space: Dict[str, Dict[str, Any]],
        X: Any,
        y: Any,
        model_kwargs: Dict[str, Any] = None,
        fit_kwargs: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            model_class: Model constructor (e.g., LGBMClassifier).
            param_space: Dict of param names to suggest functions, e.g.,
                         {'n_estimators': {'type': 'int', 'low': 50, 'high': 300}}.
            X, y: Training data.
            model_kwargs: Fixed kwargs for model init.
            fit_kwargs: Fixed kwargs for fit.
        
        Returns:
            Best parameters.
        """
        def objective(trial):
            params = {}
            for param_name, space in param_space.items():
                if space['type'] == 'int':
                    params[param_name] = trial.suggest_int(param_name, space['low'], space['high'])
                elif space['type'] == 'float':
                    params[param_name] = trial.suggest_float(param_name, space['low'], space['high'])
                elif space['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, space['choices'])
            
            # Build model
            model = model_class(**params, **(model_kwargs or {}))
            
            # Time-series CV
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            scores = cross_val_score(
                model, X, y, cv=tscv, scoring=self.scoring, n_jobs=-1
            )
            return scores.mean()

        self.study = optuna.create_study(direction=self.direction)
        self.study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = self.study.best_params
        self.logger.info(f"Best params: {self.best_params}, Best score: {self.study.best_value:.4f}")
        return self.best_params

    def get_best_model(self, model_class: Callable, model_kwargs: Dict[str, Any] = None) -> BaseEstimator:
        """Build best model with optimized params."""
        if self.best_params is None:
            raise ValueError("Run optimize() first.")
        return model_class(**self.best_params, **(model_kwargs or {}))


# Example usage (untuk testing)
if __name__ == "__main__":
    # Stub data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    
    # Example for LGBM
    optimizer = HyperparameterOptimizer(n_trials=10)
    param_space_lgbm = {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 200},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.2},
        'max_depth': {'type': 'int', 'low': 3, 'high': 8}
    }
    best_params = optimizer.optimize(LGBMClassifier, param_space_lgbm, X, y)
    print(best_params)