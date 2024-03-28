from __future__ import annotations

from typing import Literal

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker as sm

import src


def make_engine(config_block: str = "DB") -> Engine:
    conf = src.config[config_block]
    connection_string = f"postgresql+psycopg://{conf['USER']}:{conf['PASSWORD']}@{conf['IP']}:{conf['PORT']}/{conf['DB']}"  # noqa: E501
    engine = create_engine(connection_string)
    return engine


def sessionmaker(engine: Literal["auto"] | Engine = "auto"):
    if engine == "auto":
        engine = make_engine()
    maker = sm(bind=engine)
    return maker
