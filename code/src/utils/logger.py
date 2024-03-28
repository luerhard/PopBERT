import logging
import sys

import src


def setup_logging(
    name="src",
    filename=src.PATH / "logfile.log",
    filelevel="INFO",
    streamlevel="WARNING",
    captureWarnings=True,
    filemode="a+",
    overwrite=False,
):
    """creates a logger"""

    datefmt = "%H:%M:%S"

    logging.captureWarnings(captureWarnings)
    log = logging.getLogger(name)
    if log.hasHandlers() and overwrite is False:
        return log

    log.propagate = False

    log.handlers = []
    log.setLevel(logging.DEBUG)

    if filelevel is not None:
        file_fmt = "[%(asctime)s.%(msecs)03d] %(filename)s [%(lineno)d] %(levelname)s - %(message)s"
        file_formatter = logging.Formatter(fmt=file_fmt, datefmt=datefmt)

        file_handler = logging.FileHandler(filename=filename, mode=filemode)
        file_handler.setLevel(getattr(logging, filelevel))
        file_handler.setFormatter(file_formatter)

        log.addHandler(file_handler)

    if streamlevel is not None:
        stream_fmt = "[%(asctime)s.%(msecs)03d] %(message)s"
        stream_formatter = logging.Formatter(fmt=stream_fmt, datefmt=datefmt)

        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setLevel(getattr(logging, streamlevel))
        stream_handler.setFormatter(stream_formatter)

        log.addHandler(stream_handler)

    return log


logger = setup_logging()
