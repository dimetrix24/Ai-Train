import joblib
import onnxmltools
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
import numpy as np

def export_to_onnx(trainer, X_sample, prefix="ensemble"):
    """
    Export model hasil training EnsembleTrainer ke ONNX:
    - lgbm_model.onnx
    - xgb_model.onnx
    + wrapper info (weights, features)
    """
    # Ambil estimator
    lgbm_model = trainer.model.named_estimators_["lgbm"]
    xgb_model = trainer.model.named_estimators_["xgb"]
    weights = trainer.best_params.get("weights", [1.0, 1.0])

    # Tentukan jumlah fitur
    n_features = X_sample.shape[1]
    initial_type = [("input", FloatTensorType([None, n_features]))]

    # Export LightGBM
    onnx_lgbm = onnxmltools.convert_lightgbm(lgbm_model, initial_types=initial_type)
    with open(f"{prefix}_lgbm.onnx", "wb") as f:
        f.write(onnx_lgbm.SerializeToString())

    # Export XGBoost
    onnx_xgb = onnxmltools.convert_xgboost(xgb_model, initial_types=initial_type)
    with open(f"{prefix}_xgb.onnx", "wb") as f:
        f.write(onnx_xgb.SerializeToString())

    # Simpan info ensemble wrapper
    wrapper_info = {
        "weights": weights,
        "features": list(X_sample.columns) if hasattr(X_sample, "columns") else [f"f{i}" for i in range(n_features)],
        "task_type": trainer.task_type,
        "label_mapping": trainer.label_mapping,
        "inverse_mapping": trainer.inverse_mapping,
    }
    joblib.dump(wrapper_info, f"{prefix}_ensemble_wrapper.pkl")
    print(f"✅ Export selesai: {prefix}_lgbm.onnx, {prefix}_xgb.onnx, {prefix}_ensemble_wrapper.pkl")


def ensemble_predict(X, prefix="ensemble"):
    """
    Inference wrapper pakai ONNXRuntime:
    Load lgbm_model.onnx + xgb_model.onnx, gabungkan pakai weights.
    """
    # Load wrapper info
    wrapper_info = joblib.load(f"{prefix}_ensemble_wrapper.pkl")
    weights = wrapper_info["weights"]

    # Load ONNX sessions
    sess_lgbm = rt.InferenceSession(f"{prefix}_lgbm.onnx", providers=["CPUExecutionProvider"])
    sess_xgb = rt.InferenceSession(f"{prefix}_xgb.onnx", providers=["CPUExecutionProvider"])

    input_name_lgbm = sess_lgbm.get_inputs()[0].name
    input_name_xgb = sess_xgb.get_inputs()[0].name

    # Run inference
    probs_lgbm = sess_lgbm.run(None, {input_name_lgbm: X.astype(np.float32)})[1]  # [0]=label, [1]=prob
    probs_xgb = sess_xgb.run(None, {input_name_xgb: X.astype(np.float32)})[1]

    # Ensemble weighting (soft voting)
    probs = (weights[0] * probs_lgbm + weights[1] * probs_xgb) / (weights[0] + weights[1])
    preds = np.argmax(probs, axis=1)

    # Decode labels kalau multiclass
    if wrapper_info["inverse_mapping"]:
        preds = np.array([wrapper_info["inverse_mapping"][int(p)] for p in preds])

    return preds, probs