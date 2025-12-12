import logging


def init_logger(name: str, is_rank_zero: bool = True) -> logging.Logger:

    if is_rank_zero:
        logger = logging.getLogger(name=name)
        logger.setLevel(logging.INFO)

        if logger.hasHandlers():
            logger.handlers.clear()

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - \t%(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(stream_handler)

        logger.propagate = False
    else:
        logger = logging.getLogger()
        logger.addHandler(logging.NullHandler())

    return logger