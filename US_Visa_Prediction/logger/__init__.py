import logging
import os
from from_root import from_root
from datetime import datetime

# Generate the log file name with a timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the log directory and file path
log_dir = 'logs'
logs_dir_path = os.path.join(from_root(), log_dir)

# Ensure the directory exists before creating the log file
os.makedirs(logs_dir_path, exist_ok=True)

# Full path to the log file
logs_file_path = os.path.join(logs_dir_path, LOG_FILE)

# Set up logging configuration
logging.basicConfig(
    filename=logs_file_path,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)

# Test log statement
logging.info("Logging system initialized.")
