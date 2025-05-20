import os
import logging
from logging.handlers import TimedRotatingFileHandler


class Logger(logging.Logger):
    """
    A custom logger with timed file rotation and console output.

    This logger logs messages to both a console and a log file, with log rotation
    occurring daily. The log file is stored in the specified directory and
    is automatically rotated at midnight.

    Args:
        name (str): Logger name, which is also used as the log filename.
        log_dir (str): Directory where log files will be stored.
        level (int, optional): Logging level (default: logging.INFO).
        interval (int, optional): Rotation interval (default: 1).
        backupCount (int, optional): Number of backup log files to retain (default: 1).
        when (str, optional): Time unit for log rotation (default: "midnight").
        encoding (str, optional): Encoding for the log file (default: "utf-8").
    """

    def __init__(
        self,
        name: str,
        log_dir: str,
        level=logging.INFO,
        interval=1,
        backupCount=1,
        when="midnight",
        encoding="utf-8",
    ):
        """Initializes the logger.

        Args:
            name (str): Logger name and log filename.
            log_dir (str): Directory to store log files.
            level (_type_, optional): Logging level (default: logging.INFO).
            interval (int, optional): Rotation interval (default: 1).
            backupCount (int, optional): Number of backup logs to keep (default: 1).
            when (str, optional): Time unit for rotation (default: "midnight").
            encoding (str, optional): Encoding for the log file (default: "utf-8").
        """
        super().__init__(name=name, level=level)

        log_path = os.path.abspath(os.path.join(log_dir, name))
        os.makedirs(log_dir, exist_ok=True)

        log_format = logging.Formatter("%(levelname)s[%(asctime)s]:\t%(message)s")

        time_handler = TimedRotatingFileHandler(
            filename=log_path,
            when="midnight",
            interval=interval,
            backupCount=backupCount,
            encoding=encoding,
        )
        time_handler.setFormatter(log_format)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)

        self.addHandler(time_handler)
        self.addHandler(console_handler)
