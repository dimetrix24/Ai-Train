import logging
import numpy as np
import time
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc
)


class ModelEvaluator:
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    def _get_gpu_memory(self):
        """Check GPU memory usage (in MB)"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU:0
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total = mem_info.total / 1024**2
            used = mem_info.used / 1024**2
            free = mem_info.free / 1024**2
            pynvml.nvmlShutdown()
            return {"total_MB": total, "used_MB": used, "free_MB": free}
        except Exception as e:
            self.logger.warning(f"Unable to read GPU memory: {e}")
            return None

    def find_optimal_threshold(self, y_true, y_proba):
        """Find optimal threshold for binary classification using F1 score"""
        thresholds = np.linspace(0.1, 0.9, 50)
        best_threshold, best_f1 = 0.5, 0

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_threshold = f1, threshold

        self.logger.info(f"Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.3f})")
        return best_threshold

    def _compute_per_class_auc(self, y_true, y_proba, n_classes):
        """Compute ROC-AUC per class for multiclass problems"""
        from sklearn.preprocessing import label_binarize

        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        aucs = {}

        for i in range(n_classes):
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                aucs[f"class_{i}"] = auc(fpr, tpr)
            except Exception:
                aucs[f"class_{i}"] = None

        return aucs

    def _get_model_mappings(self, model):
        """Get label mappings from model (support both attribute naming conventions)"""
        label_mapping = getattr(model, "label_mapping", None) or getattr(model, "label_mapping_", None)
        inverse_mapping = getattr(model, "inverse_mapping", None) or getattr(model, "inverse_mapping_", None)
        return label_mapping, inverse_mapping

    def _decode_predictions(self, y_pred, model, task_type):
        """Decode predictions back to original format if needed"""
        if task_type == "multiclass":
            _, inverse_mapping = self._get_model_mappings(model)
            if inverse_mapping is not None:
                try:
                    return np.array([inverse_mapping[int(label)] for label in y_pred])
                except Exception:
                    self.logger.warning("Failed to decode predictions using inverse mapping")
        return y_pred

    def _handle_single_class_evaluation(self, model, X_test, y_test, task_type):
        """Handle evaluation when test set contains only one class"""
        self.logger.warning(f"⚠️ Single-class detected in y_test: {np.unique(y_test)}. Skipping AUC/PR computations.")

        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

        # Decode predictions if model provides inverse mapping
        _, inverse_mapping = self._get_model_mappings(model)
        if inverse_mapping is not None:
            try:
                y_pred_decoded = np.array([inverse_mapping[int(i)] for i in y_pred])
            except Exception:
                y_pred_decoded = y_pred
        else:
            y_pred_decoded = y_pred

        # Calculate basic metrics
        acc = accuracy_score(y_test, y_pred_decoded)
        f1 = f1_score(y_test, y_pred_decoded, average="weighted", zero_division=0)
        class_dist = dict(Counter(y_test))
        pred_dist = dict(Counter(y_pred_decoded))
        cm = confusion_matrix(y_test, y_pred_decoded, labels=np.unique(y_test))

        # Log results
        self.logger.info(f"Class distribution: {class_dist}")
        self.logger.info(f"Prediction distribution: {pred_dist}")
        self.logger.info(f"Confusion matrix:\n{cm}")
        self.logger.info(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: None")

        return {
            "accuracy": acc,
            "f1": f1,
            "auc": None,
            "auc_per_class": None,
            "class_distribution": class_dist,
            "pred_distribution": pred_dist,
            "confusion_matrix": cm.tolist(),
            "n_classes": len(np.unique(y_test)),
            "task_type": task_type,
        }

    def _predict_with_fallback(self, model, X_test):
        """Make predictions with fallback for different model types"""
        y_pred, y_proba = None, None

        try:
            start = time.time()

            # Try to get predict_proba if available
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)

            # Always get predictions
            y_pred = model.predict(X_test)
            runtime = time.time() - start
            self.logger.info(f"Inference completed in {runtime:.3f} seconds")

        except Exception as e:
            # Fallback for XGBoost booster models
            self.logger.warning(f"Standard prediction failed: {e}. Trying XGBoost DMatrix fallback.")
            try:
                import xgboost as xgb
                dtest = xgb.DMatrix(X_test)
                y_pred = model.predict(dtest)
                y_proba = None  # Cannot reliably get probabilities from booster
            except Exception as e2:
                self.logger.error(f"Fallback prediction also failed: {e2}")
                raise

        return y_pred, y_proba

    def _calculate_auc_metrics(self, y_test, y_proba, model, n_classes):
        """Calculate AUC metrics for different scenarios"""
        if y_proba is None:
            return None, None

        try:
            # Handle binary classification
            if y_proba.ndim == 1 or (y_proba.ndim == 2 and y_proba.shape[1] == 1):
                auc_value = roc_auc_score(y_test, y_proba)
                return auc_value, None

            # Binary with two columns
            elif y_proba.ndim == 2 and y_proba.shape[1] == 2:
                auc_value = roc_auc_score(y_test, y_proba[:, 1])
                return auc_value, None

            # Multiclass
            else:
                label_mapping, _ = self._get_model_mappings(model)

                if label_mapping is not None:
                    try:
                        y_test_encoded = np.array([label_mapping[label] for label in y_test])
                        auc_value = roc_auc_score(y_test_encoded, y_proba, multi_class="ovr")
                        auc_per_class = self._compute_per_class_auc(y_test_encoded, y_proba, y_proba.shape[1])
                    except Exception:
                        auc_value = roc_auc_score(y_test, y_proba, multi_class="ovr")
                        auc_per_class = self._compute_per_class_auc(y_test, y_proba, y_proba.shape[1])
                else:
                    auc_value = roc_auc_score(y_test, y_proba, multi_class="ovr")
                    auc_per_class = self._compute_per_class_auc(y_test, y_proba, y_proba.shape[1])

                return auc_value, auc_per_class

        except Exception as e:
            self.logger.warning(f"Failed to calculate ROC-AUC: {e}")
            return None, None

    def _calculate_trading_metrics(self, y_true, y_pred, y_proba=None, model=None):
        """Calculate trading-oriented metrics: win_rate, profit_factor, sharpe_ratio, max_drawdown"""
        metrics = {}
        try:
            if model is not None and getattr(model, "task_type", None) == "multiclass" and getattr(model, "label_mapping", None):
                # Multiclass: original -1(sell), 0(neutral), 1(buy)
                buy = (y_pred == 1)
                sell = (y_pred == -1)
                actual_returns = np.zeros_like(y_true, dtype=float)
                actual_returns[np.array(y_true) == 1] = 0.001
                actual_returns[np.array(y_true) == -1] = -0.001
            else:
                # Binary default: 1 = buy, 0 = sell
                buy = (y_pred == 1)
                sell = (y_pred == 0)
                actual_returns = np.where(np.array(y_true) == 1, 0.001, -0.001)

            predicted_returns = np.zeros_like(actual_returns, dtype=float)
            predicted_returns[buy] = actual_returns[buy]
            predicted_returns[sell] = -actual_returns[sell]

            profitable = predicted_returns > 0
            losses = predicted_returns < 0
            n_profitable = np.sum(profitable)
            n_losses = np.sum(losses)

            metrics["win_rate"] = float(n_profitable / (np.sum(buy | sell) + 1e-12)) if (np.sum(buy | sell) > 0) else 0.0
            avg_win = float(np.mean(predicted_returns[profitable])) if n_profitable > 0 else 0.0
            avg_loss = float(np.mean(predicted_returns[losses])) if n_losses > 0 else 0.0
            metrics["profit_factor"] = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

            if predicted_returns.size > 1:
                std = np.std(predicted_returns)
                metrics["sharpe_ratio"] = float((np.mean(predicted_returns) / (std + 1e-12)) * np.sqrt(252)) if std > 0 else 0.0
            else:
                metrics["sharpe_ratio"] = 0.0

            cum = np.cumsum(predicted_returns)
            running_max = np.maximum.accumulate(cum)
            drawdown = cum - running_max
            metrics["max_drawdown"] = float(np.min(drawdown)) if drawdown.size > 0 else 0.0

        except Exception as e:
            self.logger.warning(f"Trading metrics calculation failed: {e}")
            metrics = {k: 0.0 for k in ["win_rate", "profit_factor", "sharpe_ratio", "max_drawdown"]}

        return metrics

    def evaluate(self, model, X_test=None, y_test=None, task_type="binary"):
        """Main evaluation method (extended with trading metrics)"""
        if X_test is None or y_test is None:
            raise ValueError("X_test and y_test must be provided for evaluation")

        unique_classes = np.unique(y_test)
        if len(unique_classes) < 2:
            return self._handle_single_class_evaluation(model, X_test, y_test, task_type)

        y_pred, y_proba = self._predict_with_fallback(model, X_test)
        if y_proba is not None:
            y_proba = np.asarray(y_proba)

        y_pred_decoded = self._decode_predictions(y_pred, model, task_type)

        acc = accuracy_score(y_test, y_pred_decoded)
        if task_type == "multiclass" or len(unique_classes) > 2:
            f1 = f1_score(y_test, y_pred_decoded, average="weighted", zero_division=0)
        else:
            f1 = f1_score(y_test, y_pred_decoded, zero_division=0)

        auc_value, auc_per_class = self._calculate_auc_metrics(y_test, y_proba, model, len(unique_classes))

        class_dist = dict(Counter(y_test))
        pred_dist = dict(Counter(y_pred_decoded))
        cm = confusion_matrix(y_test, y_pred_decoded, labels=unique_classes)

        trading_metrics = self._calculate_trading_metrics(y_test, y_pred_decoded, y_proba, model)

        self.logger.info(f"Class distribution: {class_dist}")
        self.logger.info(f"Prediction distribution: {pred_dist}")
        self.logger.info(f"Confusion matrix:\n{cm}")
        self.logger.info(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc_value}")
        self.logger.info(f"Trading metrics: {trading_metrics}")

        return {
            "accuracy": acc,
            "f1": f1,
            "auc": auc_value,
            "auc_per_class": auc_per_class,
            "class_distribution": class_dist,
            "pred_distribution": pred_dist,
            "confusion_matrix": cm.tolist(),
            "n_classes": len(unique_classes),
            "task_type": task_type,
            **trading_metrics
        }