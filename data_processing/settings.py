import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class DataProcessingConfig:
    # ======================
    # Tentukan root project relatif ke file ini
    base_dir: Path = Path(__file__).resolve().parent.parent  

    # Default paths relatif ke base_dir
    data_dir: Path = base_dir / "data"
    output_dir: Path = base_dir / "outputs"
    model_dir: Path = base_dir / "outputs" / "models"  # FIXED: Consistent path
    fe_params_json: Path = base_dir / "outputs" / "fe_params.json"  # FIXED

    # ======================
    # DataLoader
    # ======================
    data_file: str = "data.csv"
    symbols: List[str] = None
    timeframe: str = "15T"
    session_filter: str = "all"
    chunk_size: int = 1_000_000
    num_worker: int = 4
    max_rows: Optional[int] = None

    # Multi-timeframe settings
    multi_timeframes: List[str] = None
    main_timeframe: str = '15T'
    use_multi_tf: bool = True
    high_tf_params: Optional[Dict] = None

    # ======================
    # Feature Engineering Controls
    # ======================
    use_scaled_high_tf: bool = True           # NEW: pilih metode high TF (scaled vs merge)
    nan_col_drop_threshold: float = 0.9       # NEW: drop kolom jika NaN >90%
    nan_col_convert_threshold: float = 0.75   # NEW: drop kolom non-numeric jika NaN >75%

    # ======================
    # Default Feature Engineering Params
    # ======================
    default_fe_params: Dict = None

    # ======================
    # Signal Generation
    # ======================
    signal_mode: str = "binary"
    label_mode: str = None
    label_future_bars: int = 8
    signal_threshold: float = 0.0005
    signal_dynamic: bool = True
    signal_triple_barrier: bool = True
    triple_barrier_params: Dict = None

    # ======================
    # Signal Tuning
    # ======================
    signal_tuning: bool = True
    signal_tuning_trials: int = 30
    signal_param_grid: Dict = None
    triple_barrier_param_grid: Dict = None

    def __post_init__(self):
        # Inisialisasi paths
        self.data_dir = Path(self.data_dir) if isinstance(self.data_dir, str) else self.data_dir
        self.output_dir = Path(self.output_dir) if isinstance(self.output_dir, str) else self.output_dir
        self.model_dir = Path(self.model_dir) if isinstance(self.model_dir, str) else self.model_dir
        self.fe_params_json = Path(self.fe_params_json) if isinstance(self.fe_params_json, str) else self.fe_params_json

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Inisialisasi multi-timeframes
        if self.multi_timeframes is None:
            self.multi_timeframes = ['15T', '1H']
        if self.main_timeframe not in self.multi_timeframes:
            self.multi_timeframes.append(self.main_timeframe)

        # Inisialisasi symbols
        if self.symbols is None:
            self.symbols = ["EURUSD"]

        # ======================
        # HIGH TF PARAMS - UPDATED untuk feature_engineering.py baru
        # ======================
        if self.high_tf_params is None:
            self.high_tf_params = {
                "ma_periods": [50, 100, 200],
                "use_sma": True,
                "use_ema": True,
                "rsi_period": 21,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "bb_window": 20,
                "bb_std": 2,
                "atr_period": 14,
                "stoch_k": 0,
                "stoch_d": 0,
                "psar_af": 0,
                "psar_af_max": 0,
            }

        # ======================
        # DEFAULT FE PARAMS - UPDATED
        # ======================
        if self.default_fe_params is None:
            self.default_fe_params = {
                "rsi_period": 7,
                "macd_fast": 5,
                "macd_slow": 13,
                "macd_signal": 4,
                "bb_window": 14,
                "bb_std": 2,
                "use_atr": True,
                "atr_period": 14,
                "atr_normalize": True,
                "atr_zscore_window": 30,
                "stoch_k": 9,
                "stoch_d": 3,
                "psar_af": 0.02,
                "psar_af_max": 0.2,
                "fillna_method": "ffill",
                "ma_periods": [9, 21, 50],
                "use_sma": True,
                "use_ema": True,
            }

        # ======================
        # TRIPLE BARRIER PARAMS
        # ======================
        if self.triple_barrier_params is None:
            self.triple_barrier_params = {
                "use_atr": True,
                "tp": 0.001,
                "sl": 0.001,
                "tp_atr_mult": 1.0,
                "sl_atr_mult": 1.0,
                "min_spread": 0.0001,
                "time_limit_bars": 24
            }

        # ======================
        # SIGNAL PARAM GRID
        # ======================
        if self.signal_param_grid is None:
            self.signal_param_grid = {
                "future_bars": [6, 10, 14, 18, 24],
                "threshold": [0.0005, 0.0015, 0.003, 0.005],
                "dynamic_threshold": [True, False],
                "use_triple_barrier": [True, False],
                "time_limit_bars": [12, 24, 36]
            }

        # ======================
        # TRIPLE BARRIER PARAM GRID
        # ======================
        if self.triple_barrier_param_grid is None:
            self.triple_barrier_param_grid = {
                "tp": [0.0008, 0.0015, 0.002],
                "sl": [0.0008, 0.0015, 0.002],
                "tp_atr_mult": [0.7, 1.1, 1.5],
                "sl_atr_mult": [0.7, 1.1, 1.5],
                "min_spread": [0.0, 0.0001, 0.0002],
                "time_limit_bars": [12, 24, 36]
            }

        self.label_mode = self.signal_mode if self.label_mode is None else self.label_mode
        self._validate_params()

    def _validate_params(self):
        """Validasi konsistensi parameters"""
        required_high_tf_params = ['use_sma', 'use_ema', 'bb_std']
        for param in required_high_tf_params:
            if param not in self.high_tf_params:
                raise ValueError(f"Missing required high_tf_params: {param}")

        required_fe_params = ['use_sma', 'use_ema', 'stoch_k', 'stoch_d', 'psar_af', 'psar_af_max']
        for param in required_fe_params:
            if param not in self.default_fe_params:
                raise ValueError(f"Missing required default_fe_params: {param}")

    def get_high_tf_params_for_timeframe(self, timeframe: str) -> Dict:
        base_params = self.high_tf_params.copy()
        if 'H' in timeframe or 'D' in timeframe:
            base_params.update({
                "rsi_period": 21,
                "ma_periods": [50, 100, 200],
                "bb_window": 20,
            })
        else:
            base_params.update({
                "rsi_period": 14,
                "ma_periods": [20, 50, 100],
                "bb_window": 14,
            })
        return base_params

    def get_fe_params_for_mode(self, mode: str = "base") -> Dict:
        return self.high_tf_params if mode == "high" else self.default_fe_params


if __name__ == "__main__":
    config = DataProcessingConfig(multi_timeframes=['15T', '1H'])
    print("=== CONFIG VALIDATION ===")
    print(f"Multi TFs: {config.multi_timeframes}")
    print(f"Main TF: {config.main_timeframe}")
    print(f"Use scaled high TF: {config.use_scaled_high_tf}")
    print(f"NaN drop threshold: {config.nan_col_drop_threshold}")
    print(f"NaN convert threshold: {config.nan_col_convert_threshold}")
    print("✅ Config validated")