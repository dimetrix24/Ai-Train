import logging
import os
from datetime import datetime
from typing import Optional
from config.settings import Config

def setup_logger(output_dir: Optional[str] = None, log_level: Optional[str] = None) -> logging.Logger:
    """Setup comprehensive logging for the application"""
    if log_level is None:
        log_level = Config.LOG_LEVEL
    if output_dir is None:
        output_dir = Config.OUTPUT_DIR
    
    # Convert string log level to numeric
    log_level_numeric = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger("EnhancedScalpingAI")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(Config.LOG_FORMAT, Config.LOG_DATE_FORMAT)
    
    # Ensure log directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"training_log_{timestamp}.txt")
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level_numeric)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger