import logging
import os
import sys

def setup_logger(log_file='logs/app.log'):
    # Make path absolute relative to project root
    # __file__ points to logger.py inside src/utils
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    log_file = os.path.join(project_root, log_file)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("medical_qa_rag")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
