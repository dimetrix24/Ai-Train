# utils/logger.py
import logging
import os
import sys
import warnings
from datetime import datetime
from config.settings import Config

def setup_logger(name: str = "EnhancedScalpingAI", redirect_std: bool = True):
    """
    Setup unified logging:
      - dynamic per-run file (training_log_YYYYMMDD_HHMMSS.txt)
      - static file (train.log)
      - console stream
    Also:
      - attach handlers to root logger so library loggers propagate
      - enable captureWarnings -> warnings module -> logging
      - optionally redirect sys.stdout / sys.stderr into logger
      - adjust some common library logger levels (xgboost, lightgbm, sklearn, joblib)
    """
    # ensure output dir and logs folder
    log_dir = os.path.join(Config.OUTPUT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn_dynamic = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    fn_static = os.path.join(log_dir, "train.log")

    # formatter from config
    fmt = getattr(Config, "LOG_FORMAT", "%(asctime)s [%(levelname)-8s] %(name)-12s %(message)s")
    datefmt = getattr(Config, "LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")
    formatter = logging.Formatter(fmt, datefmt)

    # handlers
    fh_dyn = logging.FileHandler(fn_dynamic)
    fh_dyn.setLevel(logging.DEBUG)
    fh_dyn.setFormatter(formatter)

    fh_static = logging.FileHandler(fn_static)
    fh_static.setLevel(logging.DEBUG)
    fh_static.setFormatter(formatter)

    ch = logging.StreamHandler(sys.stdout)
    ch_level = getattr(logging, getattr(Config, "LOG_LEVEL", "INFO").upper(), logging.INFO)
    ch.setLevel(ch_level)
    ch.setFormatter(formatter)

    # ---- Attach to ROOT logger so other loggers propagate here ----
    root = logging.getLogger()
    # Avoid duplicate handlers if setup_logger called multiple times in the same process
    # We'll clear existing handlers to ensure consistent behavior.
    if root.handlers:
        root.handlers.clear()

    root.setLevel(logging.DEBUG)
    root.addHandler(fh_dyn)
    root.addHandler(fh_static)
    root.addHandler(ch)

    # Create/return a named app logger for convenience
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # allow messages to propagate to root handlers
    logger.propagate = True

    # capture warnings (module 'warnings') into logging
    logging.captureWarnings(True)
    warnings.simplefilter("default")  # show warnings so they are captured

    # Redirect stdout/stderr into logging (optional)
    if redirect_std:
        class StreamToLogger(object):
            def __init__(self, logger, level=logging.INFO):
                self.logger = logger
                self.level = level
            def write(self, message):
                message = message.strip()
                if message:
                    # avoid spamming empty newline-only writes
                    self.logger.log(self.level, message)
            def flush(self):
                pass

        # preserve originals if needed (not used here, but useful for debugging)
        try:
            sys.__stdout__ = sys.stdout
            sys.__stderr__ = sys.stderr
        except Exception:
            pass
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)

    # Recommend: make sure common library loggers propagate to root (they will, since root now has handlers)
    for lib in ("xgboost", "lightgbm", "sklearn", "joblib"):
        try:
            l = logging.getLogger(lib)
            # don't clear handlers (we want propagation), but set level to INFO so they emit
            l.setLevel(logging.INFO)
            l.propagate = True
        except Exception:
            pass

    logger.info(f"Logger initialized. Dynamic log: {fn_dynamic}, Static log: {fn_static}")
    return logger


def get_logger(name: str = "EnhancedScalpingAI"):
    return logging.getLogger(name)