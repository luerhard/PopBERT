# mypy: disable-error-code="assignment"

from sqlalchemy import BigInteger
from sqlalchemy import Column
from sqlalchemy import Date
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import text
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import relationship


class Base(DeclarativeBase):
    metadata = MetaData()


class ElectoralTerm(Base):
    __tablename__ = "electoral_terms"
    __table_args__ = {"schema": "open_discourse"}

    id = Column(BigInteger, primary_key=True)
    start_date = Column(BigInteger, nullable=False)
    end_date = Column(BigInteger)


class Faction(Base):
    __tablename__ = "factions"
    __table_args__ = {"schema": "open_discourse"}

    id = Column(BigInteger, primary_key=True)
    abbreviation = Column(String, nullable=False)
    full_name = Column(String, nullable=False)


class Politician(Base):
    __tablename__ = "politicians"
    __table_args__ = {"schema": "open_discourse"}

    id = Column(BigInteger, primary_key=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    birth_place = Column(String)
    birth_country = Column(String)
    birth_date = Column(Date)
    death_date = Column(Date)
    gender = Column(String)
    profession = Column(String)
    aristocracy = Column(String)
    academic_title = Column(String)


class Sentence(Base):
    __tablename__ = "sentences"
    __table_args__ = {"schema": "open_discourse"}

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('open_discourse.sentences_id_seq'::regclass)"),
    )
    speeches_id = Column(Integer)
    sentence_no = Column(Integer)
    sentence = Column(Text)
    pop_dict_score = Column(Integer)


class Speech(Base):
    __tablename__ = "speeches"
    __table_args__ = {"schema": "open_discourse"}

    id = Column(BigInteger, primary_key=True)
    session = Column(BigInteger, nullable=False)
    electoral_term: Mapped[int] = Column(
        ForeignKey("open_discourse.electoral_terms.id"),
        nullable=False,
    )
    first_name = Column(String)
    last_name = Column(String)
    politician_id: Mapped[int] = Column(
        ForeignKey("open_discourse.politicians.id"),
        nullable=False,
        index=True,
    )
    speech_content = Column(Text, nullable=False)
    faction_id: Mapped[int] = Column(
        ForeignKey("open_discourse.factions.id"),
        nullable=False,
        index=True,
    )
    document_url = Column(String, nullable=False)
    position_short = Column(String, nullable=False, index=True)
    position_long = Column(String)
    date = Column(Date)

    electoral_term1 = relationship("ElectoralTerm")
    faction = relationship("Faction")
    politician = relationship("Politician")


class ContributionsExtended(Base):
    __tablename__ = "contributions_extended"
    __table_args__ = {"schema": "open_discourse"}

    id = Column(BigInteger, primary_key=True)
    type = Column(String, nullable=False)
    first_name = Column(String)
    last_name = Column(String)
    politician_id: Mapped[int] = Column(ForeignKey("open_discourse.politicians.id"))
    content = Column(Text)
    speech_id: Mapped[int] = Column(ForeignKey("open_discourse.speeches.id"), nullable=False)
    text_position = Column(BigInteger, nullable=False)
    faction_id: Mapped[int] = Column(ForeignKey("open_discourse.factions.id"))

    faction = relationship("Faction")
    politician = relationship("Politician")
    speech = relationship("Speech")


class ContributionsSimplified(Base):
    __tablename__ = "contributions_simplified"
    __table_args__ = {"schema": "open_discourse"}

    id = Column(BigInteger, primary_key=True)
    text_position = Column(BigInteger, nullable=False)
    speech_id: Mapped[int] = Column(ForeignKey("open_discourse.speeches.id"))
    content = Column(String)

    speech = relationship("Speech")
