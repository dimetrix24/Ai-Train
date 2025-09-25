import os
import logging
import pandas as pd
import numpy as np
import glob
import csv
from io import StringIO
from data_processing.settings import DataProcessingConfig as Config


def clean_market_data(df: pd.DataFrame, logger: logging.Logger = None) -> pd.DataFrame:
    """
    Bersihkan data pasar dari anomali:
      - open/close harus dalam [low, high]
      - drop NaN
      - pastikan harga tidak negatif
    """
    logger = logger or logging.getLogger(__name__)

    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing expected column: {col}")
            raise KeyError(col)

    before = len(df)
    
    # 1. Hapus nilai negatif dari harga
    mask_negative = (
        (df["open"] <= 0) |
        (df["high"] <= 0) |
        (df["low"] <= 0) |
        (df["close"] <= 0)
    )
    negative_count = mask_negative.sum()
    
    # 2. Hapus invalid OHLC relationships
    mask_invalid = (
        (df["low"] > df["high"]) |
        (df["open"] > df["high"]) |
        (df["open"] < df["low"]) |
        (df["close"] > df["high"]) |
        (df["close"] < df["low"])
    )
    invalid = mask_invalid.sum()

    # Combine masks
    mask_to_remove = mask_negative | mask_invalid
    df = df[~mask_to_remove].dropna()
    after = len(df)

    logger.info(
        f"Cleaned market data: total={before}, removed_negative={negative_count}, "
        f"removed_invalid={invalid}, removed_nan={before - negative_count - invalid - after}, final={after}"
    )
    
    if negative_count > 0:
        logger.warning(f"Removed {negative_count} rows with negative/zero prices")
    
    return df


def validate_data(df: pd.DataFrame, logger: logging.Logger) -> bool:
    """
    Validasi format data setelah parsing/load.
    Return: True jika valid, False jika error.
    """
    logger.info("Validating data format after parsing...")

    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False

    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.error(f"Column {col} is not numeric")
            return False

    nan_count = df[required_cols].isna().sum()
    inf_count = df[required_cols].replace([np.inf, -np.inf], np.nan).isna().sum() - nan_count
    for col in required_cols:
        if nan_count[col] > 0:
            logger.warning(f"Found {nan_count[col]} NaN values in {col}")
        if inf_count[col] > 0:
            logger.error(f"Found {inf_count[col]} infinite values in {col}")
            return False

    # MODIFIED: Menghapus baris dengan nilai negatif alih-alih mengembalikan error
    negative_cols = []
    for col in ["open", "high", "low", "close"]:
        if (df[col] <= 0).any():
            negative_count = (df[col] <= 0).sum()
            logger.warning(f"Found {negative_count} non-positive values in {col}, removing them")
            negative_cols.append(col)
    
    # Hapus baris dengan nilai negatif
    if negative_cols:
        mask_negative = pd.Series(False, index=df.index)
        for col in negative_cols:
            mask_negative = mask_negative | (df[col] <= 0)
        
        df = df[~mask_negative]
        logger.warning(f"Removed {mask_negative.sum()} rows with non-positive values")

    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("Index is not DatetimeIndex")
        return False
    if not df.index.is_monotonic_increasing:
        logger.error("Index is not sorted in ascending order")
        return False
    if df.index.duplicated().any():
        logger.error("Found duplicate timestamps in index")
        return False

    if len(df) < 50:
        logger.error(f"Data too small: {len(df)} rows, minimum 50 required")
        return False

    logger.info("Data validation passed")
    return True


class DataLoader:
    def __init__(
        self,
        data_dir: str,
        symbols=None,
        timeframe: str = None,
        bar_intervals=None,
        logger: logging.Logger = None,
        drop_duplicates: bool = True,
    ):
        self.data_dir = data_dir
        self.symbols = symbols or []
        self.timeframe = timeframe
        self.bar_intervals = bar_intervals
        self.logger = logger or logging.getLogger(__name__)
        self.drop_duplicates = drop_duplicates
        self.config = Config()

    
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalisasi nama kolom agar konsisten (lowercase + alias)."""
        rename_map = {
            # ticker
            "ticker": "ticker", "<ticker>": "ticker",

            # tanggal/waktu
            "date": "date", "<dtyyyymmdd>": "date",
            "time": "time", "<time>": "time",
            "datetime": "datetime",

            # harga
            "open": "open", "<open>": "open",
            "high": "high", "<high>": "high",
            "low": "low", "<low>": "low",
            "close": "close", "<close>": "close",

            # volume
            "volume": "volume", "vol": "volume", "<vol>": "volume",
        }

        df.columns = [rename_map.get(str(c).strip().lower(), str(c).strip().lower()) for c in df.columns]
        return df

    def _parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse kolom date/time jadi datetime index, drop duplikat, sort index."""

        if "datetime" in df.columns:
            # Pastikan datetime dtype
            if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
                df["datetime"] = pd.to_datetime(
                    df["datetime"],
                    format="%Y-%m-%d %H:%M",  # sesuai format CSV kamu
                    errors="coerce"
                )

        elif "date" in df.columns and "time" in df.columns:
            # Gabung kolom date + time
            candidates = [
                ("%Y%m%d%H%M%S", df["date"].astype(str) + df["time"].astype(str).str.zfill(6)),
                ("%Y.%m.%d %H:%M", df["date"].astype(str) + " " + df["time"].astype(str)),
                ("%Y-%m-%d %H:%M:%S", df["date"].astype(str) + " " + df["time"].astype(str)),
            ]
            parsed = None
            for fmt, series in candidates:
                parsed = pd.to_datetime(series, format=fmt, errors="coerce")
                if parsed.notna().sum() > 0:
                    df["datetime"] = parsed
                    break
            df = df.drop(columns=["date", "time"], errors="ignore")

        else:
            raise KeyError("No datetime column found in input data")

        # Set datetime sebagai index, tetap simpan kolom
        df = df.set_index("datetime", drop=False)

        # Hilangkan nama index
        df.index.name = None

        # Sort by datetime
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()

        # Debug log
        if hasattr(self, "logger"):
            self.logger.info(
                f"[DataLoader] Datetime parsed: min={df['datetime'].min()}, max={df['datetime'].max()}, NaN={df['datetime'].isna().sum()}"
            )

        return df

    def _resample(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data ke timeframe yang diinginkan."""
        try:
            rule = str(timeframe).replace("T", "min")
            df_resampled = (
                df.resample(rule, label="right", closed="right")
                .agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum"
                })
                .dropna()
            )
            self.logger.info(f"Resampled data to timeframe={rule}, shape={df_resampled.shape}")
            return df_resampled
        except Exception as e:
            self.logger.warning(f"Failed to resample with timeframe={timeframe}: {e}")
            return df

    def _merge_multiple_csv_files(self, file_pattern: str = "*.csv", max_rows: int = None) -> pd.DataFrame:
        """
        Merge multiple CSV files dalam directory yang sama.
        
        Parameters:
        -----------
        file_pattern : str
            Pattern untuk mencari file CSV (default: "*.csv")
        max_rows : int, optional
            Maximum total rows untuk load (jika None, load semua)
            
        Returns:
        --------
        pd.DataFrame
            Merged dataframe dari semua file CSV
        """
        csv_files = glob.glob(os.path.join(self.data_dir, file_pattern))
        
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV files found matching pattern: {file_pattern}")
            
        if len(csv_files) == 1:
            self.logger.info(f"Found 1 CSV file: {csv_files[0]}")
            return self.load_single_file(csv_files[0], timeframe=None)
            
        self.logger.info(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
        
        all_dfs = []
        total_rows = 0
        
        for i, csv_file in enumerate(csv_files):
            if max_rows and total_rows >= max_rows:
                self.logger.info(f"Reached max_rows limit ({max_rows}), stopping file loading")
                break
                
            self.logger.info(f"Loading file {i+1}/{len(csv_files)}: {os.path.basename(csv_file)}")
            
            try:
                df_chunk = self.load_single_file(csv_file, timeframe=None)

                if df_chunk is not None and len(df_chunk) > 0:
                    nat_count = df_chunk.index.isna().sum()
                    if nat_count > 0:
                        self.logger.warning(
                            f"[DataLoader] Found {nat_count} NaT rows in {os.path.basename(csv_file)}, dropping them"
                        )
                        df_chunk = df_chunk.dropna(subset=["datetime"])

                    all_dfs.append(df_chunk)
                    total_rows += len(df_chunk)
                    self.logger.info(f"Loaded {len(df_chunk)} rows from {os.path.basename(csv_file)}, total: {total_rows}")
                    
            except Exception as e:
                self.logger.error(f"Failed to load {csv_file}: {e}")
                continue
        
        if not all_dfs:
            raise ValueError("No data loaded from any CSV files")
            
        # Merge semua dataframe
        merged_df = pd.concat(all_dfs, axis=0)
        
        # Drop duplicates dan sort index
        if self.drop_duplicates:
            before_merge = len(merged_df)
            merged_df = merged_df[~merged_df.index.duplicated(keep="last")]
            after_merge = len(merged_df)
            if before_merge != after_merge:
                self.logger.info(f"Removed {before_merge - after_merge} duplicate timestamps after merging")
        
        merged_df = merged_df.sort_index()
        
        self.logger.info(f"Merged {len(csv_files)} files, final shape: {merged_df.shape}")
        return merged_df

    

    def _detect_delimiter(self, filepath: str, sample_size: int = 1024) -> str:
        """
        Autodetect delimiter dari file CSV dengan membaca sample awal.
        Mendukung: ',', '\t', ';', ' ', dan lainnya.
        """
        with open(filepath, 'r') as f:
            sample = f.read(sample_size)
    
        # Gunakan csv.Sniffer untuk deteksi
        sniffer = csv.Sniffer()
        try:
            delimiter = sniffer.sniff(sample, delimiters=",;\t| ").delimiter
            self.logger.debug(f"Autodetected delimiter: '{repr(delimiter)}' for {os.path.basename(filepath)}")
            return delimiter
        except Exception as e:
            self.logger.warning(f"Failed to autodetect delimiter for {filepath}: {e}. Falling back to tab.")
            return '\t'  # fallback ke tab untuk data forex umumnya

    def load_single_file(self, filepath: str, timeframe: str = None) -> pd.DataFrame:
        """Load satu CSV file dengan autodetect delimiter dan format."""

        # Deteksi delimiter terlebih dahulu
        delimiter = self._detect_delimiter(filepath)

        # Preview file dengan delimiter yang terdeteksi
        try:
            df_preview = pd.read_csv(filepath, nrows=5, header=None, sep=delimiter)
        except Exception as e:
            self.logger.warning(f"Failed to read preview with detected delimiter '{repr(delimiter)}': {e}. Trying tab fallback.")
            delimiter = '\t'
            df_preview = pd.read_csv(filepath, nrows=5, header=None, sep=delimiter)

        num_cols = len(df_preview.columns)

        # CASE 1: Format tanpa header (datetime, open, high, low, close, volume)
        if num_cols == 6 and not str(df_preview.iloc[0, 0]).startswith("<"):
            self.logger.info(
                f"Detected headerless 6-column format with delimiter='{repr(delimiter)}': {os.path.basename(filepath)}"
            )
            df = pd.read_csv(filepath, header=None, sep=delimiter)
        
            if len(df.columns) != 6:
                raise ValueError(f"Expected 6 columns in headerless file, got {len(df.columns)}")

            df.columns = ["datetime", "open", "high", "low", "close", "volume"]

            # Parse datetime
            df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M", errors="coerce")
            df = df.dropna(subset=["datetime"])

        else:
            # CASE 2: Format dengan header (misal: MetaTrader)
            self.logger.info(
                f"Detected header format with delimiter='{repr(delimiter)}': {os.path.basename(filepath)}"
            )
            df = pd.read_csv(filepath, header=0, sep=delimiter)
            df = df.rename(columns=lambda x: str(x).strip().lower())
            rename_map = {
                "<ticker>": "ticker",
                "<dtyyyymmdd>": "date",
                "<time>": "time",
                "<open>": "open",
                "<high>": "high",
                "<low>": "low",
                "<close>": "close",
                "<vol>": "volume",
            }
            df = df.rename(columns=rename_map)
            df = self._parse_datetime(df)

        # --- Lanjutkan proses standar ---
        nat_count = df["datetime"].isna().sum()
        if nat_count > 0:
            self.logger.warning(
                f"[DataLoader] Found {nat_count} NaT rows in {os.path.basename(filepath)}, dropping them"
            )
            df = df.dropna(subset=["datetime"])

        df = df.set_index("datetime", drop=False).sort_index()
        df.index.name = None

        # Konversi kolom numerik
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Bersihkan anomali OHLC
        df = clean_market_data(df, logger=self.logger)

        # Resample jika perlu
        if timeframe:
            df = self._resample(df, timeframe)
            df = clean_market_data(df, logger=self.logger)

        # Validasi data
        if not validate_data(df, self.logger):
            df = clean_market_data(df, logger=self.logger)
            if not validate_data(df, self.logger):
                self.logger.error(
                    f"[DataLoader] Validation failed: datetime NaN={df['datetime'].isna().sum()}, "
                    f"rows={len(df)}, head={df.head().to_dict()}"
                )
                raise ValueError("Data validation failed, check logs for details")

        self.logger.info(
            f"Loaded data from {filepath} with shape {df.shape}, columns={df.columns.tolist()}"
        )
        return df

    def load_data(self, filepath: str = None, chunksize: int = None, max_rows: int = None, 
                 merge_multiple: bool = False, file_pattern: str = "*.csv") -> pd.DataFrame:
        """
        Load market data dari CSV file(s).
        
        Parameters:
        -----------
        filepath : str, optional
            Path ke file CSV tertentu. Jika None, akan cari file di data_dir
        merge_multiple : bool, default False
            Jika True, akan merge semua CSV files yang ditemukan
        file_pattern : str, default "*.csv"
            Pattern untuk mencari file CSV jika merge_multiple=True
            
        Returns:
        --------
        pd.DataFrame
            Loaded market data
        """
        if merge_multiple:
            return self._merge_multiple_csv_files(file_pattern, max_rows)
        else:
            if filepath is None:
                # Cari file CSV pertama di directory
                csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
                if not csv_files:
                    raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
                filepath = csv_files[0]
                self.logger.info(f"Auto-selected file: {filepath}")
            
            return self.load_single_file(filepath)

    def load_multi_tf_data(self, timeframes: list = None, filepath: str = None, chunksize: int = None, 
                          max_rows: int = None, merge_multiple: bool = False, file_pattern: str = "*.csv") -> dict:
        """
        Load market data and resample to multiple timeframes.
        
        Parameters:
        -----------
        timeframes : list, optional
            List of timeframes to resample to, e.g., ['5T', '30T', '1H']. 
            If None, use config.MULTI_TIMEFRAMES if USE_MULTI_TF=True, else [MAIN_TIMEFRAME].
        filepath : str, optional
            Path ke file CSV tertentu.
        ... other params same as load_data
        
        Returns:
        --------
        dict
            {tf: pd.DataFrame} for each timeframe
        """
        if timeframes is None:
            if self.config.use_multi_tf:
                timeframes = self.config.multi_timeframes
            else:
                timeframes = [self.config.main_timeframe]
        
        # Load raw data without resampling
        original_timeframe = self.timeframe
        self.timeframe = None
        raw_df = self.load_data(filepath=filepath, chunksize=chunksize, max_rows=max_rows, 
                               merge_multiple=merge_multiple, file_pattern=file_pattern)
        self.timeframe = original_timeframe
        
        multi_tf = {}
        for tf in timeframes:
            try:
                rule = str(tf).replace("T", "min")
                df_tf = raw_df.resample(rule, label="right", closed="right").agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum"
                }).dropna()
                
                # Clean after resample
                df_tf = clean_market_data(df_tf, logger=self.logger)
                
                # Extra log datetime range + NaT
                dt_min, dt_max = df_tf.index.min(), df_tf.index.max()
                nat_count = df_tf.index.isna().sum()
                self.logger.info(
                    f"[Multi-TF] {tf} datetime range: {dt_min} → {dt_max}, NaT={nat_count}, rows={len(df_tf)}"
                )

                nulls = df_tf.isna().sum()
                if nulls.sum() > 0:
                    nan_report = {col: int(n) for col, n in nulls.items() if n > 0}
                    self.logger.warning(f"[Multi-TF] {tf} has NaN values: {nan_report}")

                # Validate
                if not validate_data(df_tf, self.logger):
                    self.logger.warning(f"Validation failed for timeframe {tf}, skipping")
                    continue
                
                multi_tf[tf] = df_tf
                self.logger.info(f"Resampled to {tf} ({rule}), shape={df_tf.shape}")
                
            except Exception as e:
                self.logger.warning(f"Failed to resample to {tf}: {e}")
        
        if not multi_tf:
            raise ValueError("No valid timeframes resampled")
        
        self.logger.info(f"Loaded multi-TF data: {list(multi_tf.keys())}")
        return multi_tf