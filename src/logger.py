import logging

def setup_logger(log_filename="simulation.log"):
    """Initializes a dual-output logger for the terminal and a log file."""
    logger = logging.getLogger("DiffSWE")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to prevent duplicate printouts
    if logger.hasHandlers():
        logger.handlers.clear()

    # Terminal output (Clean, no timestamps)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    # File output (Detailed with timestamps)
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger