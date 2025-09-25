import argparse
import os
import sys
from pathlib import Path
from utils.logger import setup_logger
from backtest_signal import BacktestWithSignal


def get_project_root():
    """Dapatkan path root project secara otomatis"""
    script_path = Path(__file__).resolve()
    return script_path.parent


def find_latest_model(model_dir=None):
    """Cari file model (.pkl) terbaru dengan path relatif yang aman"""
    if model_dir is None:
        model_dir = get_project_root() / "models"
    else:
        model_dir = Path(model_dir)

    if not model_dir.exists():
        return None

    list_of_models = list(model_dir.glob("*.pkl"))
    if not list_of_models:
        return None

    latest_model = max(list_of_models, key=os.path.getctime)
    return str(latest_model)


def find_data_directory(data_base_dir=None):
    """Cari direktori data dengan path relatif yang aman"""
    if data_base_dir is None:
        data_base_dir = get_project_root() / "data"
    else:
        data_base_dir = Path(data_base_dir)

    if not data_base_dir.exists():
        return None

    # 🔧 Skip folder `.ipynb_checkpoints`
    subdirs = [d for d in data_base_dir.iterdir() if d.is_dir() and d.name != ".ipynb_checkpoints"]

    if not subdirs:
        # Jika tidak ada subdir, cek file CSV langsung
        csv_files = [f for f in data_base_dir.glob("*.csv") if ".ipynb_checkpoints" not in str(f)]
        if csv_files:
            return str(data_base_dir)
        return None

    # Cari direktori dengan file CSV terbanyak
    best_dir = None
    max_csv_count = 0
    for dir_path in subdirs:
        csv_files = [f for f in dir_path.glob("*.csv") if ".ipynb_checkpoints" not in str(f)]
        if len(csv_files) > max_csv_count:
            max_csv_count = len(csv_files)
            best_dir = dir_path

    return str(best_dir) if best_dir else str(data_base_dir)


def main():
    logger = setup_logger("EnhancedScalpingAI")
    logger.info("Executor started")

    project_root = get_project_root()
    logger.info(f"Project root: {project_root}")

    parser = argparse.ArgumentParser(description="EnhancedScalpingAI Backtest Executor")
    parser.add_argument("--model", type=str, required=False, help="Path ke model .pkl")
    parser.add_argument("--fe_params", type=str, required=False, help="Path ke FE params .json")
    parser.add_argument("--data_dir", type=str, required=False, help="Path ke direktori data")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Direktori output hasil backtest")
    parser.add_argument("--timeframe", type=str, default="M15", help="Timeframe")
    parser.add_argument("--sessions", type=str, nargs="*", default=None, help="Filter sesi trading")
    parser.add_argument("--use_inverse_mapping", action="store_true", help="Gunakan inverse mapping")

    args = parser.parse_args()

    # Auto-detect model
    if args.model is None:
        args.model = find_latest_model()
        if args.model is None:
            logger.error("❌ Tidak ada model ditemukan di direktori 'models'")
            logger.error(f"   Mencari di: {project_root / 'models'}")
            return
        logger.info(f"🔍 Model terdeteksi otomatis: {args.model}")

    # Auto-detect FE params
    if args.fe_params is None and args.model is not None:
        model_path = Path(args.model)
        fe_params_candidate = model_path.parent / f"{model_path.stem}_fe_params.json"
        if fe_params_candidate.exists():
            args.fe_params = str(fe_params_candidate)
            logger.info(f"🔍 FE params terdeteksi otomatis: {args.fe_params}")

    # Auto-detect data directory
    if args.data_dir is None:
        args.data_dir = find_data_directory()
        if args.data_dir is None:
            logger.error("❌ Tidak ada data ditemukan di direktori 'data'")
            logger.error(f"   Mencari di: {project_root / 'data'}")
            return
        logger.info(f"📊 Data directory terdeteksi otomatis: {args.data_dir}")

    # Logging konfigurasi
    logger.info(f"Using timeframe: {args.timeframe}")
    if args.sessions:
        logger.info(f"Using session filter: {args.sessions}")
    logger.info(f"Inverse mapping: {'ON' if args.use_inverse_mapping else 'OFF'}")

    # Init backtester
    backtester = BacktestWithSignal(
        model_path=args.model,
        fe_params_path=args.fe_params,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        logger=logger,
        timeframe=args.timeframe,
        sessions=args.sessions,
        use_inverse_mapping=args.use_inverse_mapping,
    )

    # Jalankan backtest
    results = backtester.run()
    if results:
        logger.info(f"Final backtest results: {results}")
    else:
        logger.error("Backtest gagal atau tidak menghasilkan hasil.")


if __name__ == "__main__":
    main()