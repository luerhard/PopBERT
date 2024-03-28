import datetime as dt
import logging
from collections.abc import Hashable

logger = logging.getLogger(__file__)


class SessionManager(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.last_vacuum = dt.datetime.now()
        self.vacuum_interval = dt.timedelta(seconds=60)
        self.ttl = dt.timedelta(seconds=3600)
        self.times: dict[Hashable, dt.datetime] = dict()

    def __setitem__(self, key, value) -> None:
        self.times[key] = dt.datetime.now()
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        now = dt.datetime.now()
        if (now - self.last_vacuum) > self.vacuum_interval:
            self.vacuum()
        self.times[key] = now
        return super().__getitem__(key)

    def vacuum(self):
        now = dt.datetime.now()
        for key in list(self.keys()):
            if (now - self.times[key]) > self.ttl:
                logger.warning("Cleaning %s from session cache", key)
                del self[key]
                del self.times[key]
