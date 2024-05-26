from logging import getLogger, StreamHandler, DEBUG, INFO, Formatter, FileHandler


def create_logger(logger_name, log_file_path=None):
    logger = getLogger(logger_name)
    logger.setLevel(DEBUG)
    logger.propagate = False
    if log_file_path is not None:
        fh = FileHandler(log_file_path)
        fh.setLevel(INFO)
        fh_formatter = Formatter("%(asctime)s : %(levelname)s, %(funcName)s Message: %(message)s")
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)
        logger.info(log_file_path)
        logger.info("logging start")
    logger.info(logger_name)
    return logger
