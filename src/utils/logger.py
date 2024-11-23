import logging
import sys


class LoggerSingleton:
    _instance = None

    def __new__(cls, name: str):
        if cls._instance is None:
            cls._instance = super(LoggerSingleton, cls).__new__(cls)
            cls._instance._initialize_logger(name)
        return cls._instance

    def _initialize_logger(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Create a console handler if not already present
        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)

            # Create a formatter and set it for the handler
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)

            # Add the handler to the logger
            self.logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Retrieves the singleton logger instance.

    Args:
        name (str): The name of the logger, typically the module's `__name__`.

    Returns:
        logging.Logger: The singleton logger instance.
    """
    return LoggerSingleton(name).logger
