import logging
import colorlog


class Logger:
    @staticmethod
    def setup_logging(log_file=None):
        log_colors = {
            "DEBUG": "white",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        }
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors=log_colors,
        )

        handlers = [logging.StreamHandler()]
        if log_file:
            handlers.append(logging.FileHandler(log_file))

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=handlers,
        )

        for handler in handlers:
            handler.setFormatter(formatter)

    @staticmethod
    def info(*args):
        logging.info(" ".join(map(str, args)))

    @staticmethod
    def error(*args):
        logging.error(" ".join(map(str, args)))

    @staticmethod
    def debug(*args):
        logging.debug(" ".join(map(str, args)))

    @staticmethod
    def warning(*args):
        logging.warning(" ".join(map(str, args)))
