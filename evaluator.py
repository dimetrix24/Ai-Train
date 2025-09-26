import logging
import numpy as np
import time
import json
import joblib
from collections import Counter
from typing import Optional, Dict, Any, Union
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score
)
import pandas as pd


class ModelEvaluator:
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    def _get_gpu_memory(self):
        """Check GPU memory usage (in MB)"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total = mem_info.total / 1024**2
            used = mem_info.used / 1024**2
            free = mem_info.free / 1024**2
            pynvml.nvmlShutdown()
            return {"total_MB": total, "used_MB": used, "free_MB": free}
        except Exception as e:
            self.logger.warning(f"Unable to read GPU memory: {e}")
            return None

    def _calculate_trading_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        df_features: Optional[pd.DataFrame] = None,
        model=None
    ) -> Dict[str, float]:
        """Calculate comprehensive trading metrics for scalping."""
        metrics = {}

        # 1. Basic trading stats
        try:
            correct = (y_true == y_pred)
            win_rate = np.mean(correct)
            simulated_returns = np.where(correct, 0.001, -0.001)
            total_profit = np.sum(simulated_returns[simulated_returns > 0])
            total_loss = np.abs(np.sum(simulated_returns[simulated_returns < 0]))
            metrics["win_rate"] = float(win_rate)
            metrics["simulated_profit_factor"] = float(total_profit / total_loss) if total_loss > 0 else float('inf')
        except Exception as e:
            metrics["win_rate"] = metrics["simulated_profit_factor"] = 0.0

        # 2. Max drawdown
        try:
            cumulative = np.cumsum(simulated_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / (running_max + 1e-12)
            metrics["max_drawdown"] = float(np.min(drawdown))
        except Exception:
            metrics["max_drawdown"] = 0.0

        # 3. Real trading metrics (jika harga tersedia)
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
                        metrics["real_sharpe_ratio"] = float(
                            metrics["real_avg_return"] / (metrics["real_volatility"] + 1e-12)
                        )
                        # Real max drawdown
                        cum_real = np.cumsum(realized_returns)
                        running_max_real = np.maximum.accumulate(cum_real)
                        dd_real = (cum_real - running_max_real) / (running_max_real + 1e-12)
                        metrics["real_max_drawdown"] = float(np.min(dd_real))
            except Exception as e:
                self.logger.debug(f"Real trading metrics skipped: {e}")

        # 4. Profit-weighted accuracy
        try:
            if df_features is not None and 'close' in df_features.columns:
                close_prices = df_features['close'].values
                if len(close_prices) >= 2 and len(y_true) >= 2:
                    returns = np.diff(close_prices) / close_prices[:-1]
                    min_len = min(len(y_true)-1, len(returns), len(y_pred)-1)
                    if min_len > 0:
                        y_true_eval = y_true[1:min_len+1]
                        y_pred_eval = y_pred[1:min_len+1]
                        returns_eval = returns[:min_len]
                        weights = np.abs(returns_eval)
                        if np.max(weights) > 0:
                            weights = weights / np.max(weights)
                        correct_pred = (y_true_eval == y_pred_eval).astype(float)
                        if np.sum(weights) > 0:
                            metrics["profit_weighted_accuracy"] = float(np.average(correct_pred, weights=weights))
                        else:
                            metrics["profit_weighted_accuracy"] = float(accuracy_score(y_true_eval, y_pred_eval))
                    else:
                        metrics["profit_weighted_accuracy"] = float(accuracy_score(y_true, y_pred))
                else:
                    metrics["profit_weighted_accuracy"] = float(accuracy_score(y_true, y_pred))
            else:
                metrics["profit_weighted_accuracy"] = float(accuracy_score(y_true, y_pred))
        except Exception as e:
            metrics["profit_weighted_accuracy"] = float(accuracy_score(y_true, y_pred))

        # 5. High-confidence accuracy
        try:
            if y_proba is not None:
                confidences = np.max(y_proba, axis=1)
                high_conf_mask = confidences >= 0.6
                if np.sum(high_conf_mask) > 0:
                    metrics["high_conf_accuracy"] = float(
                        accuracy_score(y_true[high_conf_mask], y_pred[high_conf_mask])
                    )
                else:
                    metrics["high_conf_accuracy"] = 0.0
            else:
                metrics["high_conf_accuracy"] = 0.0
        except Exception:
            metrics["high_conf_accuracy"] = 0.0

        return metrics

    def _predict_catboost_onnx(self, onnx_prefix: str, X: np.ndarray):
        """Predict using CatBoost ONNX model."""
        try:
            import onnxruntime as rt
            sess = rt.InferenceSession(f"{onnx_prefix}.onnx", providers=["CPUExecutionProvider"])
            input_name = sess.get_inputs()[0].name
            outputs = sess.run(None, {input_name: X.astype(np.float32)})
            probs = outputs[1] if len(outputs) == 2 else outputs[0]
            preds = np.argmax(probs, axis=1)
            return preds, probs
        except Exception as e:
            raise RuntimeError(f"ONNX prediction failed: {e}")

    def _decode_predictions(self, y_pred: np.ndarray, wrapper_info: Optional[Dict] = None) -> np.ndarray:
        """Decode predictions using inverse mapping from wrapper."""
        if wrapper_info and "inverse_mapping" in wrapper_info and wrapper_info["inverse_mapping"]:
            try:
                return np.array([wrapper_info["inverse_mapping"][int(p)] for p in y_pred])
            except Exception as e:
                self.logger.warning(f"Failed to decode predictions: {e}")
        return y_pred

    def _load_wrapper_info(self, model_path: str) -> Optional[Dict]:
        """Load wrapper info for ONNX models."""
        try:
            if model_path.endswith(".onnx"):
                prefix = model_path.replace(".onnx", "")
            else:
                prefix = model_path.replace(".pkl", "")
            wrapper_path = f"{prefix}_wrapper.pkl"
            if os.path.exists(wrapper_path):
                return joblib.load(wrapper_path)
        except Exception as e:
            self.logger.warning(f"Could not load wrapper info: {e}")
        return None

    def evaluate(
        self,
        model: Union[Any, str],
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: np.ndarray,
        df_features: Optional[pd.DataFrame] = None,
        task_type: str = "multiclass"
    ) -> Dict[str, Any]:
        """
        Evaluate model with comprehensive trading metrics.
        
        Args:
            model: Trained model object OR path to .pkl/.onnx file
            X_test: Test features
            y_test: True labels
            df_features: DataFrame with 'close' price for trading metrics
            task_type: "binary" or "multiclass"
        """
        start_time = time.time()
        self.logger.info("Starting model evaluation...")

        # Handle model loading
        if isinstance(model, str):
            model_path = model
            if model_path.endswith(".onnx"):
                # ONNX model
                wrapper_info = self._load_wrapper_info(model_path)
                X_np = X_test.values if hasattr(X_test, "values") else np.array(X_test)
                y_pred, y_proba = self._predict_catboost_onnx(model_path.replace(".onnx", ""), X_np)
                y_pred_decoded = self._decode_predictions(y_pred, wrapper_info)
                label_mapping = wrapper_info.get("label_mapping") if wrapper_info else None
            else:
                # Pickle model
                loaded_model = joblib.load(model_path)
                wrapper_info = self._load_wrapper_info(model_path)
                y_pred = loaded_model.predict(X_test)
                y_proba = loaded_model.predict_proba(X_test) if hasattr(loaded_model, "predict_proba") else None
                y_pred_decoded = self._decode_predictions(y_pred, wrapper_info)
                label_mapping = getattr(loaded_model, "label_mapping", None)
        else:
            # Model object
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            wrapper_info = None
            label_mapping = getattr(model, "label_mapping", None)
            y_pred_decoded = self._decode_predictions(y_pred, {"inverse_mapping": getattr(model, "inverse_mapping", None)})

        # Ensure numpy arrays
        y_test = np.asarray(y_test)
        y_pred_decoded = np.asarray(y_pred_decoded)
        y_proba = np.asarray(y_proba) if y_proba is not None else None

        # Basic metrics
        acc = accuracy_score(y_test, y_pred_decoded)
        balanced_acc = balanced_accuracy_score(y_test, y_pred_decoded)
        
        if task_type == "multiclass" or len(np.unique(y_test)) > 2:
            f1 = f1_score(y_test, y_pred_decoded, average="weighted", zero_division=0)
        else:
            f1 = f1_score(y_test, y_pred_decoded, zero_division=0)

        # AUC (only if proba available)
        auc_value = None
        if y_proba is not None:
            try:
                if task_type == "multiclass":
                    if label_mapping is not None:
                        y_test_encoded = np.array([label_mapping.get(label, label) for label in y_test])
                    else:
                        y_test_encoded = y_test
                    auc_value = roc_auc_score(y_test_encoded, y_proba, multi_class="ovr")
                else:
                    auc_value = roc_auc_score(y_test, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            except Exception as e:
                self.logger.warning(f"AUC calculation failed: {e}")

        # Trading metrics
        trading_metrics = self._calculate_trading_metrics(
            y_test, y_pred_decoded, y_proba, df_features, model
        )

        # Distributions
        class_dist = dict(Counter(y_test))
        pred_dist = dict(Counter(y_pred_decoded))
        cm = confusion_matrix(y_test, y_pred_decoded, labels=np.unique(y_test))

        # Log results
        eval_time = time.time() - start_time
        self.logger.info(f"Evaluation completed in {eval_time:.2f}s")
        self.logger.info(f"Class distribution: {class_dist}")
        self.logger.info(f"Prediction distribution: {pred_dist}")
        self.logger.info(f"Accuracy: {acc:.4f}, Balanced Acc: {balanced_acc:.4f}, F1: {f1:.4f}, AUC: {auc_value}")
        self.logger.info(f"Trading metrics: win_rate={trading_metrics.get('win_rate', 0):.4f}, "
                        f"profit_factor={trading_metrics.get('simulated_profit_factor', 0):.2f}, "
                        f"max_dd={trading_metrics.get('max_drawdown', 0):.4f}")

        return {
            "accuracy": acc,
            "balanced_accuracy": balanced_acc,
            "f1": f1,
            "auc": auc_value,
            "class_distribution": class_dist,
            "pred_distribution": pred_dist,
            "confusion_matrix": cm.tolist(),
            "n_classes": len(np.unique(y_test)),
            "task_type": task_type,
            "evaluation_time_sec": eval_time,
            **trading_metrics
        }


# =============================
# Helper function for ONNX evaluation
# =============================
def evaluate_onnx_model(
    onnx_path: str,
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: np.ndarray,
    df_features: Optional[pd.DataFrame] = None,
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """Convenience function to evaluate ONNX model directly."""
    evaluator = ModelEvaluator(logger=logger or logging.getLogger(__name__))
    return evaluator.evaluate(onnx_path, X_test, y_test, df_features, task_type="multiclass")


# =============================
# Example usage
# =============================
if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ModelEvaluator")

    # Dummy data
    X_test = pd.DataFrame(np.random.rand(100, 20))
    y_test = np.random.choice([-1, 0, 1], size=100)
    df_features = pd.DataFrame({"close": np.cumprod(1 + np.random.randn(100) * 0.001) * 1900})

    # Example 1: Evaluate pickle model
    # evaluator = ModelEvaluator(logger)
    # results = evaluator.evaluate("outputs/models/catboost_scalping.pkl", X_test, y_test, df_features)

    # Example 2: Evaluate ONNX model
    # results = evaluate_onnx_model(
    #     "outputs/models/catboost_scalping.onnx",
    #     X_test, y_test, df_features, logger
    # )

    logger.info("ModelEvaluator siap digunakan untuk evaluasi scalping.")