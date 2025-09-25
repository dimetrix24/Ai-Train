import os
import logging
import joblib
import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from lightgbm import early_stopping
from config.settings import Config as LIGHTGBM_CONFIG


class LightGBMTrainer:
    def __init__(self, params=None, model_path="outputs/lightgbm_model.pkl", logger=None, task_type="binary", use_class_weight=True):
        """
        Args:
            params: dict LightGBM params
            model_path: path simpan model
            task_type: "binary" atau "multiclass"
            use_class_weight: bool, enable/disable class_weight
        """
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.model_path = model_path
        self.task_type = task_type
        self.class_to_idx = None
        self.idx_to_class = None
        self.use_class_weight = use_class_weight

        # Ambil params dari config kalau tidak diberikan
        if params is None:
            self.params = LIGHTGBM_CONFIG[self.task_type].copy()
        else:
            self.params = params

        # Tambahkan class_weight jika enable
        if self.use_class_weight:
            if self.task_type == "binary":
                self.params.setdefault("class_weight", "balanced")
            elif self.task_type == "multiclass":
                # LightGBM bisa menerima dict {class_idx: weight}
                self.params.setdefault("class_weight", "balanced")

    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        """Train LightGBM model (binary atau multiclass)"""
        self.logger.info(f"🚀 Training LightGBM model... (task_type={self.task_type}, use_class_weight={self.use_class_weight})")

        # --- pastikan numeric ---
        X_train = X_train.astype(float)
        y_train = np.ravel(y_train)
        if X_valid is not None and y_valid is not None:
            X_valid = X_valid.astype(float)
            y_valid = np.ravel(y_valid)

        # --- Mapping multiclass (agar LightGBM bisa handle -1,0,1) ---
        if self.task_type == "multiclass":
            classes = sorted(np.unique(y_train))  # contoh: [-1,0,1]
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

            y_train = np.array([self.class_to_idx[y] for y in y_train])
            if X_valid is not None and y_valid is not None:
                y_valid = np.array([self.class_to_idx[y] for y in y_valid])

            self.logger.info(f"📌 Label mapping: {self.class_to_idx}")

        # --- log distribusi kelas ---
        unique, counts = np.unique(y_train, return_counts=True)
        self.logger.info(f"📊 Training set: shape={X_train.shape}, dist={dict(zip(unique, counts))}")

        if X_valid is not None and y_valid is not None:
            unique_v, counts_v = np.unique(y_valid, return_counts=True)
            self.logger.info(f"📊 Validation set: shape={X_valid.shape}, dist={dict(zip(unique_v, counts_v))}")

            if X_valid.shape[1] != X_train.shape[1]:
                raise ValueError(
                    f"Mismatch jumlah fitur: train={X_train.shape[1]} vs valid={X_valid.shape[1]}"
                )

        # --- analisis korelasi fitur dengan target ---
        X_train_df = pd.DataFrame(X_train)
        correlations = X_train_df.corrwith(pd.Series(y_train))
        high_corr = correlations[abs(correlations) > 0.5]
        if not high_corr.empty:
            self.logger.warning(f"⚠️ High correlations with target: {high_corr}")

        # --- ambil parameter dari constructor / fallback ---
        params = self.params.copy()
        if self.task_type == "multiclass":
            params["num_class"] = len(np.unique(y_train))

        self.logger.info(f"⚙️ LightGBM params: {params}")

        # --- definisi model ---
        self.model = lgb.LGBMClassifier(**params)

        # --- training ---
        if X_valid is not None and y_valid is not None:
            self.logger.info("✅ Training dengan validation set + early stopping")
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric=params.get("metric", "multi_logloss"),
                callbacks=[early_stopping(stopping_rounds=100)]
            )
        else:
            self.logger.info("✅ Training tanpa validation set")
            self.model.fit(X_train, y_train)

        # --- simpan model ---
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        self.logger.info(f"📁 Model disimpan ke {self.model_path}")

        return self.model

    def predict(self, X):
        """Prediksi dengan decoding untuk multiclass"""
        if self.model is None:
            raise ValueError("Model belum dilatih")
        
        X = X.astype(float)
        y_pred = self.model.predict(X)
        
        # Decode kembali ke nilai asli untuk multiclass
        if self.task_type == "multiclass" and self.idx_to_class is not None:
            y_pred = np.array([self.idx_to_class[int(i)] for i in y_pred])
        
        return y_pred

    def predict_proba(self, X):
        """Prediksi probabilitas"""
        if self.model is None:
            raise ValueError("Model belum dilatih")
        
        X = X.astype(float)
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """Evaluasi model di test set"""
        if self.model is None:
            raise ValueError("Model belum dilatih")

        X_test = X_test.astype(float)
        y_test = np.ravel(y_test)

        # --- prediksi ---
        if self.task_type == "multiclass":
            # Prediksi dengan model (akan menghasilkan indeks 0,1,2)
            y_pred = self.model.predict(X_test)
            
            # Decode kembali ke nilai asli: 0 -> -1, 1 -> 0, 2 -> 1
            y_pred_decoded = np.array([self.idx_to_class[int(i)] for i in y_pred])
            
            # Gunakan y_test asli (-1,0,1) untuk evaluasi
            acc = accuracy_score(y_test, y_pred_decoded)
            f1 = f1_score(y_test, y_pred_decoded, average="macro")
            self.logger.info("\n" + classification_report(y_test, y_pred_decoded))
            results = {"accuracy": acc, "f1": f1, "auc": None}
        else:
            # Untuk binary classification, tidak perlu decoding
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            results = {"accuracy": acc, "f1": f1, "auc": auc}

        self.logger.info(f"📊 Evaluation results: {results}")
        return results

    def get_model_info(self):
        """Dapatkan informasi model"""
        if not self.model:
            return {"status": "Model not trained"}
        
        info = {
            "n_classes": len(self.idx_to_class) if self.idx_to_class else None,
            "classes": list(self.idx_to_class.values()) if self.idx_to_class else None,
            "class_mapping": self.class_to_idx,
            "model_type": self.task_type
        }
        
        return info

    def save(self, filename=None):
        """Simpan model dengan metadata"""
        if filename is None:
            filename = self.model_path
        
        model_data = {
            'model': self.model,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'task_type': self.task_type,
            'params': self.params
        }
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        joblib.dump(model_data, filename)
        self.logger.info(f"📁 Model disimpan ke {filename}")
        return filename

    def load(self, filepath):
        """Load model dengan metadata"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.class_to_idx = model_data['class_to_idx']
        self.idx_to_class = model_data['idx_to_class']
        self.task_type = model_data['task_type']
        self.params = model_data.get('params', {})
        
        self.logger.info(f"📁 Model loaded dari {filepath} (task_type={self.task_type})")
        return self.model