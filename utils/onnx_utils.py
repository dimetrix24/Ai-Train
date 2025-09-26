import joblib
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as rt
import numpy as np
import json

def export_to_onnx(trainer, X_sample, prefix="ensemble"):
    """
    Export model hasil training EnsembleTrainer ke ONNX:
    - lgbm_model.onnx
    - xgb_model.onnx
    + wrapper info (weights, features, mappings)
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

    # Buat mapping nama fitur -> fN
    if hasattr(X_sample, "columns"):
        original_cols = list(X_sample.columns)
    else:
        original_cols = [f"f{i}" for i in range(n_features)]

    feature_map = {f"f{i}": col for i, col in enumerate(original_cols)}

    # Simpan wrapper info
    wrapper_info = {
        "weights": weights,
        "features": original_cols,
        "task_type": trainer.task_type,
        "label_mapping": trainer.label_mapping,
        "inverse_mapping": trainer.inverse_mapping,
        "feature_map_file": f"{prefix}_feature_map.json"
    }
    joblib.dump(wrapper_info, f"{prefix}_ensemble_wrapper.pkl")

    # Simpan mapping ke JSON
    with open(f"{prefix}_feature_map.json", "w") as f:
        json.dump(feature_map, f, indent=4)

    print(f"✅ Export selesai: {prefix}_lgbm.onnx, {prefix}_xgb.onnx, "
          f"{prefix}_ensemble_wrapper.pkl, {prefix}_feature_map.json")


def ensemble_predict(X, prefix="ensemble"):
    """
    Inference wrapper pakai ONNXRuntime:
    Load lgbm_model.onnx + xgb_model.onnx, gabungkan pakai weights.
    """
    # Load wrapper info
    wrapper_info = joblib.load(f"{prefix}_ensemble_wrapper.pkl")
    weights = wrapper_info["weights"]

    # Load feature map
    with open(wrapper_info["feature_map_file"], "r") as f:
        feature_map = json.load(f)

    # Reorder X sesuai urutan fitur training
    ordered_features = [feature_map[f"f{i}"] for i in range(len(feature_map))]
    if hasattr(X, "loc"):
        X = X[ordered_features].to_numpy(dtype=np.float32)
    else:
        X = np.array(X, dtype=np.float32)

    # Load ONNX sessions
    sess_lgbm = rt.InferenceSession(f"{prefix}_lgbm.onnx", providers=["CPUExecutionProvider"])
    sess_xgb = rt.InferenceSession(f"{prefix}_xgb.onnx", providers=["CPUExecutionProvider"])

    input_name_lgbm = sess_lgbm.get_inputs()[0].name
    input_name_xgb = sess_xgb.get_inputs()[0].name

    # Run inference (handle 1-output atau 2-output)
    out_lgbm = sess_lgbm.run(None, {input_name_lgbm: X})
    out_xgb = sess_xgb.run(None, {input_name_xgb: X})

    probs_lgbm = out_lgbm[1] if len(out_lgbm) > 1 else out_lgbm[0]
    probs_xgb = out_xgb[1] if len(out_xgb) > 1 else out_xgb[0]

    # Ensemble weighting (soft voting)
    probs = (weights[0] * probs_lgbm + weights[1] * probs_xgb) / (weights[0] + weights[1])
    preds = np.argmax(probs, axis=1)

    # Decode labels kalau multiclass
    if wrapper_info["inverse_mapping"]:
        preds = np.array([wrapper_info["inverse_mapping"].get(int(p), int(p)) for p in preds])

    return preds, probs