import logging
import os
import pathlib


def safely_make_folders(folders):
    for fpath in folders:
        pathlib.Path(fpath).mkdir(parents=True, exist_ok=True)


def assert_files_exist(fpaths):
    for path in fpaths:
        if not os.path.exists(path):
            raise ValueError(f"File does not exist: {path}")


def set_up_logger(log_stream_list, log_file_list, level=logging.DEBUG):
    logger = logging.getLogger()
    logger.setLevel(level)

    log_formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s: %(message)s')

    handlers = []
    for log_file in log_file_list:
        flog_handler = logging.FileHandler(log_file)
        flog_handler.setFormatter(log_formatter)
        handlers.append(flog_handler)

    for log_stream in log_stream_list:
        tlog_handler = logging.StreamHandler(log_stream)
        tlog_handler.setFormatter(log_formatter)
        handlers.append(tlog_handler)

    logger.handlers = handlers
