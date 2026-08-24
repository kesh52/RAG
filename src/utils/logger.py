import os
import logging
from logging.handlers import RotatingFileHandler
from src.utils.config import config

def setup_logging():
    """Configures centralized logging for the application using rotating file and console handlers."""
    # 1. Fetch configurations
    log_level_str = config.get("logging.level", "INFO").upper()
    log_path = config.get("logging.file_path", "logs/pipeline.log")
    
    # Resolve relative paths relative to the project root directory
    if not os.path.isabs(log_path):
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        log_path = os.path.join(root_dir, log_path)

    # 2. Ensure log folder exists
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # 3. Configure logging level
    level = getattr(logging, log_level_str, logging.INFO)

    # 4. Define logging format
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # 5. Get root logger and configure handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logs in case of double imports
    if not root_logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)

        # Rotating File Handler (Max 5MB, keep 3 backup files)
        file_handler = RotatingFileHandler(
            log_path, 
            maxBytes=5 * 1024 * 1024, 
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)


# Automatically configure logging when the module is imported
setup_logging()

