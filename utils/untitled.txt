import joblib
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, Any
from config.settings import Config

def save_model(model, filename: str = None) -> str:
    """Save trained model to file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scalping_model_{timestamp}.pkl"
    
    filepath = os.path.join(Config.MODEL_DIR, filename)
    joblib.dump(model, filepath)
    
    return filepath

def load_model(filepath: str):
    """Load trained model from file"""
    return joblib.load(filepath)

def save_results(results: Dict[str, Any], filename: str = None) -> str:
    """Save training results to JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_results_{timestamp}.json"
    
    filepath = os.path.join(Config.OUTPUT_DIR, filename)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    results_serializable = convert_numpy_types(results)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    return filepath

def save_feature_importance(feature_importance: pd.DataFrame, filename: str = None) -> str:
    """Save feature importance to CSV file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feature_importance_{timestamp}.csv"
    
    filepath = os.path.join(Config.OUTPUT_DIR, filename)
    feature_importance.to_csv(filepath, index=False)
    
    return filepath