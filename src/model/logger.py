import logging
import sys


def set_logger(logger=None,
               log_level='info') -> logging.Logger:
    """
    Function to set up a logger for the application.
    :param logger: A logger object. [Default: None]
    :param log_level: Level of information to print out. [Default: info] [Options: info, debug]
    :return: A logger object.
    """
    # Create a default logger if not provided
    if logger is None:
        logger = logging.getLogger(__name__)

    # Determine log level
    if log_level == 'info':
        _level = logging.INFO
    elif log_level == 'debug':
        _level = logging.DEBUG
    else:
        raise ValueError(f"Log level {log_level} not recognized.")

    # Set the level in logger
    logger.setLevel(_level)

    # Set the log format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logfmt = logging.Formatter(log_format)

    # Set logger output to STDOUT and STDERR
    log_handler = logging.StreamHandler(stream=sys.stdout)
    log_handler.setLevel(_level)
    log_handler.setFormatter(logfmt)

    error_handler = logging.StreamHandler(stream=sys.stderr)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logfmt)

    # Add handler to the main logger
    logger.addHandler(log_handler)
    logger.addHandler(error_handler)

    return logger
