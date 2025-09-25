import os
from pathlib import Path


class Config:
    # ======================
    # Path settings
    # ======================
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

    # ======================
    # DataLoader
    # ======================
    DATA_FILE = "data.csv"
    SYMBOLS = ["EURUSD"]
    TIMEFRAME = "15T"                     # Pandas offset alias
    BAR_INTERVALS = ["15T", "1H"]
    SESSION_FILTER = "all"               # e.g. "asia", "us", "europe"

    # ======================
    # Feature Engineering
    # ======================
    DEFAULT_FE_PARAMS = {
        "rsi_period": 14,
        "sma_windows": [20, 50],
        "ema_windows": [12, 26],
        "bb_period": 20,
        "bb_std": 2.0,
        "stoch_period": 14,
        "atr_period": 14,
        "use_rsi": True,
        "use_sma": True,
        "use_ema": True,
        "use_bb": True,
        "use_macd": True,
        "use_stoch": True,
        "use_atr": True,
        "use_price_action": True,
        "use_additional": True,
        "use_volume": True,
        "use_time": True,
    }
    FE_PARAMS_JSON = os.path.join(OUTPUT_DIR, "fe_params.json")
    
    # Signal Generation Tuning Settings
    SIGNAL_TUNING_ENABLED = True
    SIGNAL_TUNING_TRIALS = 50
    SIGNAL_FUTURE_BARS_RANGE = (4, 12)  # Range untuk tuning
    SIGNAL_THRESHOLD_RANGE = (0.0005, 0.005)
    TRIPLE_BARRIER_TUNING = True
    
    # Default signal parameters
    DEFAULT_SIGNAL_PARAMS = {
        'future_bars': 5,
        'threshold': 0.0005,
        'dynamic_threshold': True,
        'use_triple_barrier': True,
    }


    # ======================
    # Split & balancing
    # ======================
    SPLIT_METHOD = "timeseries"         # "timeseries" | "pct" | "rolling" | "expanding"
    BALANCE_MODE = "none"  # "smart_undersample" | "undersample" | "oversample" | "none"

    # ======================
    # Class weight toggle
    # ======================
    USE_CLASS_WEIGHT = True             # True → LightGBM pakai class_weight, False → disable

    # ======================
    # Ensemble Trainer
    # ======================
    ENSEMBLE_CONFIG = {
        "tune": False,              # aktifkan Optuna tuning
        "n_trials": 30,            # jumlah percobaan tuning
        "model_path": os.path.join(MODEL_DIR, "ensemble_multiclass.pkl"),
        "confidence_threshold": 0.6,
        "performance_window": 100,
    }

    # ======================
    # Validation
    # ======================
    CV_SPLITS = 5
    CV_SCORING = "accuracy"
    USE_TIMESERIES_SPLIT = True
    MIN_ACCEPTABLE_ACCURACY = 0.55
    MIN_ACCEPTABLE_PRECISION = 0.50
    MAX_OVERFITTING_GAP = 0.05

    # ======================
    # Logging
    # ======================
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)-12s %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"