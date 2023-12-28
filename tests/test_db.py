from sqlalchemy import text
from sqlalchemy.orm.session import close_all_sessions

import src.db


class TestDatabase:
    @classmethod
    def setup_class(cls):
        cls.engine = src.db.make_engine()
        cls.Session = src.db.sessionmaker(engine=cls.engine)

    @classmethod
    def teardown_class(cls):
        close_all_sessions()
        cls.engine.dispose()

    def test_simple_connection(self):
        s = self.Session()
        s.execute(text("SELECT 1"))
        s.close()
        assert True
