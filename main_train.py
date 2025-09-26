import os
import glob
import argparse
import logging
import pandas as pd
import numpy as np

from data_processing.data_loader import DataLoader
from data_processing.feature_engineering import FeatureEngineering
from data_processing.signal_generator import SignalGenerator
from session_filter import SessionFilter
from model.catboost_trainer import CatBoostTrainer
from model.lightgbm_trainer import LightGBMTrainer
from model.ensemble_trainer import EnsembleTrainer
from evaluator import ModelEvaluator
from collections import Counter

# Config imports
from config.settings import Config
from data_processing.settings import DataProcessingConfig
from utils.logger import setup_logger

logger = setup_logger()
config = DataProcessingConfig()


# ==================== NEW: HIGH TIMEFRAME RESOLUTION ====================
def resolve_high_timeframes(args, config):
    """Resolve high timeframes from arguments or config"""
    
    # Priority 1: Command line argument
    if args.high_tf and len(args.high_tf) > 0:
        high_tf = args.high_tf
        logger.info(f"Using high timeframes from CLI: {high_tf}")
        return high_tf
    
    # Priority 2: Profile from CLI
    if args.high_tf_profile:
        profile_options = getattr(config, "high_timeframe_options", {
            "scalping": ["15M", "30M"],
            "day_trading": ["1H", "4H"], 
            "swing": ["4H", "1D"],
            "all": ["15M", "30M", "1H", "4H", "1D"]
        })
        if args.high_tf_profile in profile_options:
            high_tf = profile_options[args.high_tf_profile]
            logger.info(f"Using high timeframe profile '{args.high_tf_profile}': {high_tf}")
            return high_tf
    
    # Priority 3: Profile from config
    if hasattr(config, "high_timeframe_profile"):
        profile = config.high_timeframe_profile
        profile_options = getattr(config, "high_timeframe_options", {})
        if profile in profile_options:
            high_tf = profile_options[profile]
            logger.info(f"Using high timeframe profile from config '{profile}': {high_tf}")
            return high_tf
    
    # Priority 4: Direct from config
    if hasattr(config, "high_timeframes") and config.high_timeframes:
        high_tf = config.high_timeframes
        logger.info(f"Using high timeframes from config: {high_tf}")
        return high_tf
    
    # Priority 5: Default fallback
    default_high_tf = ["1H", "4H"]
    logger.info(f"Using default high timeframes: {default_high_tf}")
    return default_high_tf


# ==================== NEW: SIMPLIFIED PROCESSING FUNCTION ====================
def process_split_with_new_approach(dict_split, name, fe, signal_generator, main_tf, signal_mode, high_timeframes, logger):
    """
    NEW: Simplified processing menggunakan approach feature engineering baru
    """
    if main_tf not in dict_split:
        logger.error(f"[{name}] Main timeframe {main_tf} not found")
        return None, None

    base_df = dict_split[main_tf].copy()
    if base_df is None or len(base_df) == 0:
        logger.warning(f"[{name}] Base dataframe is empty")
        return None, None

    logger.info(f"[{name}] Processing {len(base_df)} rows from {main_tf}")

    # STEP 1: Generate signals on base TF
    try:
        logger.info(f"[{name}] Generating trading signals...")
        df_signals = signal_generator.create_trading_signals(base_df)
        
        if df_signals is None or len(df_signals) == 0:
            logger.warning(f"[{name}] Signal generation returned empty")
            return None, None
            
        logger.info(f"[{name}] Signals generated: {len(df_signals)} rows")
        
    except Exception as e:
        logger.error(f"[{name}] Signal generation failed: {e}")
        return None, None

    # STEP 2: Create features dengan high TF indicators
    try:
        logger.info(f"[{name}] Creating features with high TFs: {high_timeframes}")
        
        # NEW: Gunakan approach feature engineering baru dengan high_timeframes parameter
        df_features = fe.create_features(base_df, high_timeframes=high_timeframes)
        
        if df_features is None or len(df_features) == 0:
            logger.error(f"[{name}] Feature engineering returned empty")
            return None, None
            
        logger.info(f"[{name}] Features created: {df_features.shape}")
        
    except Exception as e:
        logger.error(f"[{name}] Feature engineering failed: {e}")
        return None, None

    # If feature_engineering removed datetime columns (new FE does), reattach datetime from base_df
    try:
        if "datetime" not in df_features.columns:
            # Prefer explicit column named 'datetime' or any col containing 'datetime'
            dt_col = None
            base_dt_cols = [c for c in base_df.columns if "datetime" in c.lower()]
            if len(base_dt_cols) > 0:
                dt_col = base_dt_cols[0]
                logger.info(f"[{name}] Using base_df column '{dt_col}' to restore datetime in features")
                df_features["datetime"] = base_df[dt_col].values
            elif isinstance(base_df.index, pd.DatetimeIndex):
                logger.info(f"[{name}] Using base_df DatetimeIndex to restore datetime in features")
                df_features["datetime"] = base_df.index.to_series().values
            else:
                # Try any time-like column
                time_like = [c for c in base_df.columns if "time" in c.lower()]
                if len(time_like) > 0:
                    logger.info(f"[{name}] Using base_df time-like column '{time_like[0]}' to restore datetime in features")
                    df_features["datetime"] = base_df[time_like[0]].values
                else:
                    # fallback to integer index (still usable for index-align fallback)
                    logger.warning(f"[{name}] No datetime found in base_df; adding integer index as 'datetime' for fallback")
                    df_features["datetime"] = pd.RangeIndex(start=0, stop=len(df_features))
        # Ensure datetime dtype is datetime64 when possible
        try:
            df_features["datetime"] = pd.to_datetime(df_features["datetime"])
        except Exception:
            logger.debug(f"[{name}] Could not convert feature datetime to datetime64; leaving as-is")
        
    except Exception as e:
        logger.warning(f"[{name}] While restoring datetime to features: {e}")

    # STEP 3: IMPROVED ALIGNMENT - Align by datetime instead of index
    try:
        logger.info(f"[{name}] Aligning features and signals...")
    
        # Ensure signals have datetime column (or restore from base_df)
        if "datetime" not in df_signals.columns:
            base_dt_cols = [c for c in base_df.columns if "datetime" in c.lower()]
            if len(base_dt_cols) > 0:
                df_signals["datetime"] = base_df[base_dt_cols[0]].values
                logger.info(f"[{name}] Restored datetime in signals from base_df column '{base_dt_cols[0]}'")
            elif isinstance(base_df.index, pd.DatetimeIndex):
                df_signals["datetime"] = base_df.index.to_series().values
            else:
                df_signals["datetime"] = pd.RangeIndex(start=0, stop=len(df_signals))

        # Convert to datetime where possible
        try:
            df_signals["datetime"] = pd.to_datetime(df_signals["datetime"])
        except Exception:
            logger.debug(f"[{name}] Could not convert signal datetime to datetime64; leaving as-is")

        # ✅ Align by datetime using merge_asof when both are time-like
        can_merge_on_datetime = (
            "datetime" in df_features.columns and
            "datetime" in df_signals.columns and
            pd.api.types.is_datetime64_any_dtype(df_features["datetime"]) and
            pd.api.types.is_datetime64_any_dtype(df_signals["datetime"])
        )

        if can_merge_on_datetime:
            merged = pd.merge_asof(
                df_features.sort_values("datetime").reset_index(drop=True),
                df_signals.sort_values("datetime").reset_index(drop=True),
                on="datetime",
                direction="backward"
            )
        else:
            logger.warning(f"[{name}] Datetime not suitable for merge_asof; falling back to index-based alignment")
            min_len = min(len(df_features), len(df_signals))
            merged = pd.concat(
                [df_features.iloc[:min_len].reset_index(drop=True),
                 df_signals.iloc[:min_len].reset_index(drop=True)],
                axis=1
            )
    
        # ✅ IMPROVED VALIDATION dengan detailed logging
        validation_passed = True

        if 'signal_encoded' not in merged.columns:
            logger.error(f"❌ signal_encoded missing. Available columns: {merged.columns.tolist()}")
            validation_passed = False
        else:
            nan_count = merged['signal_encoded'].isna().sum()
            if nan_count == len(merged):
                logger.error(f"❌ All signal_encoded values are NaN ({nan_count}/{len(merged)})")
                validation_passed = False
            elif nan_count > 0:
                logger.warning(f"⚠️  Found {nan_count} NaN values in signal_encoded")

        if not validation_passed:
            logger.error(f"❌ Alignment validation failed for {name}")
            return None, None

        logger.info(f"✅ Alignment validation passed for {name}")    
        logger.info(f"[{name}] Alignment successful: {len(merged)} rows")
        
    except Exception as e:
        logger.error(f"[{name}] Alignment failed: {e}")
        return None, None

    # STEP 4: Post-processing
    if 'signal_encoded' not in merged.columns:
        logger.error(f"[{name}] signal_encoded column missing")
        return None, None
        
    # Remove rows with NaN signals
    initial_count = len(merged)
    merged = merged.dropna(subset=['signal_encoded'])
    final_count = len(merged)
    
    if final_count == 0:
        logger.error(f"[{name}] No valid signals after cleaning")
        return None, None
        
    logger.info(f"[{name}] Cleaned signals: {final_count}/{initial_count} rows retained")

    # STEP 5: Extract target and features
    X, y, target_col = get_target_and_data(merged, signal_mode, logger)
    if X is None or y is None:
        logger.error(f"[{name}] Failed to extract target and features")
        return None, None

    # Remove leakage columns
    X = drop_residual_leakage(X, name, logger)
    
    logger.info(f"[{name}] Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y


# =====================================================================
# Helper functions (keep existing ones)
# =====================================================================
def drop_residual_leakage(X, name, logger=None):
    """Remove columns that may contain leakage information from the feature matrix."""
    if X is None:
        return X
    
    leak_patterns = [
        "future", "future_return", "return_", "next_", 
        "signal_encoded", "target", "leak", "label"
    ]
    
    residual_cols = []
    for column in X.columns:
        column_name = column.lower()
        if column_name == "signal" or any(pattern in column_name for pattern in leak_patterns):
            residual_cols.append(column)
    
    if residual_cols:
        if logger:
            logger.warning(f"[{name}] ⚠️ Dropped leakage-like columns: {residual_cols}")
        X = X.drop(columns=residual_cols, errors="ignore")
    elif logger:
        logger.info(f"[{name}] ✅ No leakage-like columns found")
    
    if logger:
        logger.info(f"[{name}] Final features shape: {X.shape}")
    
    return X


def get_target_and_data(df_signals, signal_mode, logger):
    """Extract target and features from signal dataframe"""
    if df_signals is None or len(df_signals) == 0:
        logger.warning("Empty signal dataframe")
        return None, None, None

    target_cols = {
        "binary": "signal_binary",
        "multiclass": "signal_encoded",
        "regression": "signal_encoded"
    }

    target_col = target_cols.get(signal_mode)
    if target_col not in df_signals.columns:
        # ✅ fallback jika signal_binary hilang
        fallback_col = "signal_encoded" if "signal_encoded" in df_signals.columns else None
        if fallback_col:
            logger.warning(f"{signal_mode} mode expected {target_col}, using fallback {fallback_col}")
            target_col = fallback_col
        else:
            logger.error(f"{signal_mode} mode requires {target_col}, but not found")
            return None, None, None

    X = df_signals.drop(columns=[target_col])
    y = df_signals[target_col]
    
    logger.info(f"✅ Extracted features: {X.shape[1]} columns, {len(X)} rows")
    logger.info(f"✅ Target distribution: {pd.Series(y).value_counts().to_dict()}")
    
    return X, y, target_col


def validate_dataframe_consistency(df: pd.DataFrame, name: str, logger: logging.Logger) -> bool:
    """Validasi konsistensi dataframe"""
    if df is None:
        logger.error(f"[{name}] DataFrame is None")
        return False
        
    if len(df) == 0:
        logger.error(f"[{name}] DataFrame is empty")
        return False
        
    # Check for essential columns
    essential_cols = ['open', 'high', 'low', 'close']
    missing_essential = [col for col in essential_cols if col not in df.columns]
    if missing_essential:
        logger.error(f"[{name}] Missing essential columns: {missing_essential}")
        return False
    
    logger.info(f"[{name}] DataFrame validation passed: {len(df)} rows, {len(df.columns)} columns")
    return True


def ensure_test_has_classes(X_train, y_train, X_valid, y_valid, X_test, y_test, min_classes=2, logger=None):
    """Ensure test set has at least min_classes classes by expanding from validation/train sets"""
    def n_classes(y): 
        return len(np.unique(y)) if y is not None and len(y) > 0 else 0
    
    if n_classes(y_test) >= min_classes:
        return X_train, y_train, X_valid, y_valid, X_test, y_test
    
    if logger: 
        logger.warning("⚠️ Test set has <2 classes, attempting to expand window...")
    
    # Extend from validation
    try:
        while n_classes(y_test) < min_classes and y_valid is not None and len(y_valid) > 0:
            take = max(1, int(len(y_valid) * 0.1))
            if hasattr(y_valid, "iloc"):
                X_move, y_move = X_valid.iloc[-take:], y_valid.iloc[-take:]
                X_valid, y_valid = X_valid.iloc[:-take], y_valid.iloc[:-take]
                X_test = pd.concat([X_move, X_test]) if X_test is not None else X_move
                y_test = pd.concat([y_move, y_test]) if y_test is not None else y_move
            else:
                X_move, y_move = X_valid[-take:], y_valid[-take:]
                X_valid, y_valid = X_valid[:-take], y_valid[:-take]
                X_test = np.concatenate([X_move, X_test]) if X_test is not None else X_move
                y_test = np.concatenate([y_move, y_test]) if y_test is not None else y_move
            
            if logger: 
                logger.info(f"Expanded test by {take}, new classes: {n_classes(y_test)}")
            if len(y_valid) == 0: 
                break
    except Exception as e:
        if logger: 
            logger.warning(f"Auto-expand test from valid failed: {e}")
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def split_data_by_timeframe(data_dict, split_method="timeseries", test_size=0.2, valid_size=0.1):
    """Split multi-timeframe data consistently across all timeframes"""
    train_dict, valid_dict, test_dict = {}, {}, {}
    
    # Use base TF to determine split points
    base_tf = list(data_dict.keys())[0]
    base_df = data_dict[base_tf]
    n = len(base_df)
    
    if split_method == "timeseries":
        test_idx = int(n * (1 - test_size))
        valid_idx = int(test_idx * (1 - valid_size))
        
        for tf, df in data_dict.items():
            train_dict[tf] = df.iloc[:valid_idx].copy()
            valid_dict[tf] = df.iloc[valid_idx:test_idx].copy()
            test_dict[tf] = df.iloc[test_idx:].copy()
            
    elif split_method == "random":
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        test_cutoff = int(n * (1 - test_size))
        valid_cutoff = int(test_cutoff * (1 - valid_size))
        
        train_idx = indices[:valid_cutoff]
        valid_idx = indices[valid_cutoff:test_cutoff]
        test_idx = indices[test_cutoff:]
        
        for tf, df in data_dict.items():
            train_dict[tf] = df.iloc[train_idx].copy()
            valid_dict[tf] = df.iloc[valid_idx].copy()
            test_dict[tf] = df.iloc[test_idx:].copy()
            
    elif split_method == "none":
        for tf, df in data_dict.items():
            train_dict[tf] = df.copy()
            valid_dict[tf] = df.iloc[0:0].copy()
            test_dict[tf] = df.iloc[0:0].copy()
    
    return train_dict, valid_dict, test_dict


# Imbalance handling functions
def smart_undersample(X, y, logger=None):
    counts = y.value_counts()
    if len(counts) < 2:
        return X, y
    
    minority_class = counts.idxmin()
    minority_size = counts.min()
    
    balanced_idx = (
        y[y == minority_class].index.tolist() +
        y[y != minority_class].sample(minority_size, random_state=42).index.tolist()
    )
    
    if logger:
        logger.info(f"Smart undersample: before={len(y)}, after={len(balanced_idx)}")
    return X.loc[balanced_idx], y.loc[balanced_idx]


# =====================================================================
# MAIN PIPELINE - ENHANCED
# =====================================================================
def run(args):
    logger.info("=== Starting EnhancedScalpingAI Pipeline ===")
    logger.info(f"Arguments: {vars(args)}")
    
    # Resolve high timeframes
    high_timeframes = resolve_high_timeframes(args, config)
    logger.info(f"Resolved high timeframes: {high_timeframes}")
    
    # Resolve main timeframe
    main_tf = args.main_tf if args.main_tf else config.main_timeframe
    logger.info(f"Main timeframe: {main_tf}")

    # --- PHASE 1: Data loading (HANYA base timeframe) ---
    logger.info("PHASE 1: Data loading (base timeframe only)")
    
    loader = DataLoader(
        data_dir=args.data_dir or config.data_dir,
        timeframe=main_tf,  # Hanya load main timeframe
        logger=logger,
    )

    # Load hanya base timeframe data
    df_raw = loader.load_data(max_rows=args.max_rows, merge_multiple=True)
    data_dict = {main_tf: df_raw}  # Hanya base timeframe

    logger.info(f"Loaded base TF data: {main_tf}, {len(df_raw)} rows")
    logger.info(f"High timeframes for resampling: {high_timeframes}")

    # Validate loaded data
    if not validate_dataframe_consistency(df_raw, "Base_TF", logger):
        logger.error("Data validation failed, stopping pipeline")
        return

    # --- PHASE 2: Session filtering ---
    logger.info("PHASE 2: Session filtering")
    session_filter = SessionFilter(sessions=args.session or config.session_filter, logger=logger)
    
    original_count = len(data_dict[main_tf])
    data_dict[main_tf] = session_filter.apply(data_dict[main_tf])
    filtered_count = len(data_dict[main_tf])
    
    logger.info(f"📊 Session filtering: {original_count} → {filtered_count} rows "
               f"({filtered_count/original_count*100:.1f}% retained)")

    # --- PHASE 3: Raw data splitting ---
    logger.info("PHASE 3: Raw data splitting")
    
    train_dict, valid_dict, test_dict = split_data_by_timeframe(
        data_dict, 
        split_method=args.split_method or "timeseries",
        test_size=0.2,
        valid_size=0.1
    )
    
    logger.info(f"📊 Split results - Train: {len(train_dict.get(main_tf, []))}, "
               f"Valid: {len(valid_dict.get(main_tf, []))}, Test: {len(test_dict.get(main_tf, []))}")

    # --- PHASE 4: Feature engineering & signal generation ---
    logger.info("PHASE 4: Feature engineering & signal generation")

    # Initialize components dengan config dari data_processing.settings
    fe_params = getattr(config, 'default_fe_params', {
        'rsi_period': 14,
        'ema_periods': [20, 50],
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2
    })
    
    fe = FeatureEngineering(logger=logger, fe_params=fe_params)

    # Signal generator parameters dari config
    signal_generator_kwargs = {
        "logger": logger,
        "mode": args.signal_mode or config.signal_mode,
        "enable_tuning": False,
        "label_only_base_timeframe": True,
        "base_timeframe_column": "is_base_timeframe"
    }

    # Add optional parameters dari config dengan fallback ke default
    optional_params = {
        "future_bars": args.signal_future_bars or getattr(config, 'signal_future_bars', 5),
        "threshold": args.signal_threshold or getattr(config, 'signal_threshold', 0.002),
        "dynamic_threshold": args.signal_dynamic or getattr(config, 'signal_dynamic', False),
        "use_triple_barrier": args.use_triple_barrier or getattr(config, 'use_triple_barrier', False),
        "n_trials": args.signal_trials or getattr(config, 'signal_tuning_trials', 20)
    }

    for param_name, param_value in optional_params.items():
        signal_generator_kwargs[param_name] = param_value

    signal_generator = SignalGenerator(**signal_generator_kwargs)

    # Signal parameter tuning (if enabled)
    if args.tune_signal:
        logger.info("🔧 Tuning signal parameters on TRAINING DATA ONLY")
        
        # Create features for tuning
        df_train_features = fe.create_features(
            train_dict[main_tf], 
            high_timeframes=high_timeframes
        )
        
        if df_train_features is not None and len(df_train_features) > 0:
            signal_generator.enable_tuning = True
            best_params = signal_generator.tune_parameters(df_train_features)
            signal_generator.enable_tuning = False
            logger.info(f"✅ Best signal parameters: {best_params}")
        else:
            logger.warning("⚠️ Cannot tune on empty training features, using defaults")

    # --- PHASE 5: Process all splits dengan NEW approach ---
    logger.info("🔥 Processing splits with NEW feature engineering approach")

    X_train, y_train = process_split_with_new_approach(
        train_dict, "train", fe, signal_generator, main_tf, args.signal_mode, high_timeframes, logger)

    X_valid, y_valid = process_split_with_new_approach(
        valid_dict, "valid", fe, signal_generator, main_tf, args.signal_mode, high_timeframes, logger)

    X_test, y_test = process_split_with_new_approach(
        test_dict, "test", fe, signal_generator, main_tf, args.signal_mode, high_timeframes, logger)

    # Validate results
    if X_train is None or y_train is None or len(X_train) == 0:
        logger.error("CRITICAL: Training set is empty after processing")
        raise ValueError("Training set is empty")

    # Ensure test set has multiple classes
    X_train, y_train, X_valid, y_valid, X_test, y_test = ensure_test_has_classes(
        X_train, y_train, X_valid, y_valid, X_test, y_test, min_classes=2, logger=logger
    )
    
    # Dalam fungsi run(), sebelum PHASE 6: Training
    def ensure_numeric_dataframes(X_train, X_valid, X_test, logger):
        """Pastikan semua DataFrame hanya berisi kolom numerik"""
    
        def _clean_df(df, name):
            if df is None:
                return df
            # Hapus kolom non-numerik
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            non_numeric_cols = set(df.columns) - set(numeric_cols)
        
            if non_numeric_cols:
                logger.warning(f"[{name}] Removing non-numeric columns: {list(non_numeric_cols)}")
                df = df[numeric_cols]
        
            return df
    
        X_train = _clean_df(X_train, "train")
        X_valid = _clean_df(X_valid, "valid") 
        X_test = _clean_df(X_test, "test")
    
        return X_train, X_valid, X_test


    
    logger.info(f"🎯 FINAL DATA SUMMARY:")
    logger.info(f"📊 Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    logger.info(f"📊 Validation: {X_valid.shape[0] if X_valid is not None else 0} samples")
    logger.info(f"📊 Test: {X_test.shape[0] if X_test is not None else 0} samples")
    
    # Target distribution
    if y_train is not None:
        train_dist = pd.Series(y_train).value_counts()
        logger.info(f"🎯 Training target distribution: {train_dist.to_dict()}")

    # --- PHASE 6: Training ---
    logger.info("PHASE 6: Training")
    task_type = "binary" if args.signal_mode == "binary" else args.signal_mode
    # Simpan salinan fitur sebelum pembersihan numeric (dipertahankan untuk metrik)
    X_train_raw = X_train.copy() if X_train is not None else None
    X_valid_raw = X_valid.copy() if X_valid is not None else None
    X_test_raw = X_test.copy() if X_test is not None else None
    X_train, X_valid, X_test = ensure_numeric_dataframes(X_train, X_valid, X_test, logger)
    
    # Handle training based on trainer type
    if args.trainer == "lightgbm":
        params = Config.LIGHTGBM_CONFIG[task_type].copy()
        if task_type == "multiclass":
            params["num_class"] = len(np.unique(y_train))
        model_path = os.path.join(config.model_dir, f"lightgbm_{task_type}.pkl")
        trainer = LightGBMTrainer(params=params, model_path=model_path, logger=logger, task_type=task_type)
        model = trainer.train(X_train, y_train, X_valid, y_valid)
        
    elif args.trainer == "ensemble":
        model_path = os.path.join(config.model_dir, f"ensemble_{task_type}.pkl")
        trainer = EnsembleTrainer(
            logger=logger,
            tune=args.tune_ensemble,
            n_trials=args.n_trials,
            model_path=model_path,
            fe=fe,
            df_features_all=X_train_raw
        )
        model = trainer.train(X_train, y_train, task_type=task_type)

        # 🔎 Tambahan: evaluasi trading metrics lengkap
        logger.info("PHASE 6B: EnsembleTrainer trading metrics evaluation")
        try:
            trainer.evaluate(X_test, y_test, df_features=X_test_raw)
        except Exception as e:
            logger.warning(f"Trading metrics evaluation skipped: {e}")

        # ✅ Export ke ONNX setelah training selesai
        try:
            from utils.onnx_utils import export_ensemble_to_onnx
            sample_input = X_train.iloc[:1] if isinstance(X_train, pd.DataFrame) else X_train[:1]
            export_ensemble_to_onnx(trainer, sample_input, prefix=os.path.join(config.model_dir, f"ensemble_{task_type}"))
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
    else:  # CatBoost
        model_filename = f"{args.trainer}_{task_type}.pkl"
        model_path = os.path.join(config.model_dir, model_filename)
    
        # Inisialisasi trainer
        trainer = CatBoostTrainer(
            logger=logger,
            model_path=model_path,          # ✅ Simpan otomatis ke path ini
            fe=fe,
            tune=Config.tune, # opsional: aktifkan tuning
            n_trials=Config.n_trials,
            objective_metric="profit_weighted_accuracy",
            min_samples_per_fold=50,
            gap_ratio=0.02,
            n_splits=5
        )
    
        # Training (df_features_all dilewatkan di sini!)
        model = trainer.train(
            X_train, 
            y_train, 
            task_type=task_type,
            df_features_all=X_train_raw  # ✅ untuk profit-weighted accuracy
        )
    
        # 🔎 Evaluasi dengan ModelEvaluator (bukan method trainer!)
        logger.info("PHASE 6B: Evaluasi metrik trading lengkap...")
        try:
    
            evaluator = ModelEvaluator(logger=logger)
            eval_results = evaluator.evaluate(
                model=model,                    # atau model_path jika pakai file
                X_test=X_test,
                y_test=y_test,
                df_features=X_test_raw,         # DataFrame dengan kolom 'close'
                task_type=task_type
            )
            logger.info(f"✅ Evaluasi selesai. Accuracy: {eval_results['accuracy']:.4f}, "
                   f"Profit-Weighted Acc: {eval_results['profit_weighted_accuracy']:.4f}")
        except Exception as e:
            logger.warning(f"Trading metrics evaluation skipped: {e}")
        logger.info("PHASE 6C: Evaluasi confidence & regime...")
        try:
            conf_result = trainer.predict_with_confidence(
                X_test,
                threshold=0.6,
                df_features=X_test_raw
            )
            high_conf_preds = conf_result["preds"]
            regime = conf_result["regime"]

            # Contoh: hitung akurasi hanya untuk high-confidence
            mask = high_conf_preds != None
            if np.any(mask):
                acc_high_conf = accuracy_score(y_test[mask], high_conf_preds[mask])
                logger.info(f"High-confidence accuracy: {acc_high_conf:.4f}")

            # Contoh: distribusi regime
            if regime is not None:
                logger.info(f"Market regime distribusi: {regime.value_counts().to_dict()}")
        except Exception as e:
            logger.warning(f"Confidence/regime evaluation skipped: {e}")
        # ✅ Export ke ONNX
        try:
            # Pastikan X_train berupa DataFrame atau array dengan kolom
            sample_input = X_train.iloc[:1] if isinstance(X_train, pd.DataFrame) else X_train[:1]
            trainer.export_to_onnx(
                X_sample=sample_input,
                prefix=os.path.join(config.model_dir, f"catboost_{task_type}")
            )
        except Exception as e:
            logger.error(f"ONNX export gagal: {e}")
    # --- PHASE 7: Evaluation ---
    logger.info("PHASE 7: Evaluation")
    evaluator = ModelEvaluator(logger=logger)
    
    if y_test is None or len(np.unique(y_test)) < 2:
        logger.warning("Test set has <2 classes, skipping detailed evaluation")
        results = {"warning": "insufficient_test_classes"}
    else:
        results = evaluator.evaluate(model, X_test, y_test, task_type=task_type)

    logger.info("=== Pipeline Completed Successfully ===")
    logger.info(f"Final Results: {results}")
    return results


# ==================== ARGUMENT PARSING ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EnhancedScalpingAI Trainer")

    # Data arguments dengan default dari config
    parser.add_argument("--data-dir", type=str, default=config.data_dir)
    parser.add_argument("--max-rows", type=int, default=config.max_rows)
    parser.add_argument("--session", type=str, default=config.session_filter)
    
    # Timeframe arguments - NEW
    parser.add_argument("--main-tf", type=str, default=None, 
                       help=f"Main timeframe (default: {config.main_timeframe})")
    parser.add_argument("--high-tf", nargs="+", default=None,
                       help="High timeframes for trend analysis. Example: --high-tf 30T 1H 4H")
    parser.add_argument("--high-tf-profile", type=str, default=None,
                       choices=["scalping", "day_trading", "swing", "all"],
                       help="Use predefined high timeframe profiles")

    # Signal & processing arguments dengan default dari config
    parser.add_argument("--signal-mode", type=str, default=config.signal_mode,
                       choices=["binary", "regression", "multiclass"])
    parser.add_argument("--split-method", type=str, default="timeseries",
                       choices=["timeseries", "random", "none"])
    parser.add_argument("--imbalance-method", type=str, default=Config.BALANCE_MODE,
                       choices=["smart_undersample", "undersample", "oversample", "none"])

    # Signal generation parameters dengan default dari config
    parser.add_argument("--tune-signal", action="store_true")
    parser.add_argument("--signal-trials", type=int, default=getattr(config, 'signal_tuning_trials', 20))
    parser.add_argument("--signal-future-bars", type=int, default=None)
    parser.add_argument("--signal-threshold", type=float, default=None)
    parser.add_argument("--signal-dynamic", action="store_true")
    parser.add_argument("--use-triple-barrier", action="store_true")

    # Model arguments
    parser.add_argument("--trainer", type=str, default="catboost",
                       choices=["catboost", "lightgbm", "ensemble"])
    parser.add_argument("--tune-ensemble", action="store_true")
    parser.add_argument("--n-trials", type=int, default=20)

    args = parser.parse_args()
    
    try:
        results = run(args)
        logger.info("Pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise