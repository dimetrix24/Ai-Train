# config/settings.py
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
    # Data settings
    # ======================
    BAR_INTERVAL = ["1min", "5min", "15min"]   # multi-timeframe (sinkron dengan DataLoader & backtest)
    BASE_TIMEFRAME = "5min"                    # timeframe utama untuk alias OHLCV
    TRAIN_END_DATE = None                      # contoh: "2024-12-31"
    TEST_SIZE = 0.2                            # fallback: 20% terakhir data untuk test
    VAL_SIZE = 0.1                             # 10% terakhir train untuk validasi
    RANDOM_STATE = 42

    # ======================
    # Trading parameters
    # ======================
    FUTURE_BARS = 9        # horizon label generator (≈ 45 menit jika 5min)
    YEARS = 3              # default backtest hanya 3 tahun terakhir
    TP_ATR = 1.1
    SL_ATR = 1.1
    LOT_SIZE = 1000
    STARTING_BALANCE = 10000

    # ======================
    # Mode barrier
    # ======================
    SIGNAL_MODE = "atr"        # "atr" atau "fixed"
    USE_ATR_FOR_BARRIERS = True

    # Kalau pakai ATR
    TP_MULT_ATR = 1.3
    SL_MULT_ATR = 1.3

    # Kalau pakai pip (fallback)
    PIP_SIZE = 0.0001
    TP_PIPS = 8
    SL_PIPS = 8

    # ======================
    # Model parameters
    # ======================
    DEFAULT_MODEL_PARAMS = {
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'n_jobs': 2
    }

    # ======================
    # Hyperparameter tuning
    # ======================
    USE_RANDOM_SEARCH = True
    N_ITER_RANDOM_SEARCH = 20

    PARAM_GRID = {
        'n_estimators': [200, 300, 400, 500],
        'learning_rate': [0.01, 0.02, 0.03, 0.05],
        'max_depth': [3, 4, 5],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 0.3, 0.5],
        'reg_lambda': [1, 1.5, 2, 3]
    }

    # ======================
    # Cross-validation
    # ======================
    CV_SPLITS = 3
    CV_SCORING = 'accuracy'
    USE_TIMESERIES_SPLIT = True

    # ======================
    # Feature selection
    # ======================
    FEATURE_IMPORTANCE_THRESHOLD = 0.01

    # ======================
    # Sampling strategy
    # ======================
    @staticmethod
    def compute_sampling_strategy(len_train, minority_classes=[0, 2], majority_class=1):
        cap = 30000
        target = min(cap, max(2000, int(len_train * 0.15)))
        smote_strategy = {cls: target for cls in minority_classes}
        undersample_strategy = {majority_class: target}
        return smote_strategy, undersample_strategy

    # ======================
    # Outlier detection
    # ======================
    Z_SCORE_THRESHOLD = 3.0

    # ======================
    # Logging
    # ======================
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    # ======================
    # Early stopping
    # ======================
    EARLY_STOPPING_ROUNDS = 30


# Pastikan direktori ada
os.makedirs(Config.DATA_DIR, exist_ok=True)
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.MODEL_DIR, exist_ok=True)