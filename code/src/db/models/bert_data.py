from collections import defaultdict

from sqlalchemy import BigInteger
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import UniqueConstraint
from sqlalchemy import func
from sqlalchemy.ext.hybrid import hybrid_method
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship

from .open_discourse import Base


class Sample(Base):
    __tablename__ = "samples"
    __table_args__ = (
        UniqueConstraint("speeches_id", "sentence_no"),
        {"schema": "bert_data", "extend_existing": True},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    used_in_batch = Column(Integer, nullable=True, default=None, index=True)
    time_created = Column(DateTime, server_default=func.now())
    speeches_id = Column(
        BigInteger,
        ForeignKey("open_discourse.speeches.id"),
        nullable=False,
        index=True,
    )
    sentence_no = Column(BigInteger, index=True)
    sentence_length = Column(Integer)
    text = Column(String)
    pop_dict_score = Column(Boolean, index=True)

    speech = relationship("Speech", viewonly=True)
    faction = relationship("Faction", secondary="open_discourse.speeches", viewonly=True)

    raw_labels = relationship("Label", back_populates="sample", uselist=True)

    @hybrid_property
    def n_coders(self):
        return len(self.raw_labels)

    @hybrid_method
    def labels(self, min_agree: float = 0.5, ignore_unsure: bool = True) -> frozenset:
        if self.n_coders == 0:
            return frozenset()

        d: dict[str, int] = defaultdict(int)
        for label in self.raw_labels:
            # only add populism if coder is sure or we do not skip unsures
            if ignore_unsure is True or label.unsure is False:
                d["antielite"] += label.pop_antielite
                d["pplcentr"] += label.pop_pplcentr

            d["eliteless"] += label.souv_eliteless
            d["pplmore"] += label.souv_pplmore

            d["left"] += label.ideol_left
            d["right"] += label.ideol_right

        return frozenset([key for key, n in d.items() if n / self.n_coders >= min_agree])


class Label(Base):
    __tablename__ = "labels"
    __table_args__ = (
        UniqueConstraint("username", "sample_id"),
        {"schema": "bert_data", "extend_existing": True},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    time_labeled = Column(DateTime)
    username = Column(String, index=True)
    sample_id = Column(BigInteger, ForeignKey("bert_data.samples.id"))

    pop_antielite = Column(Boolean)
    pop_pplcentr = Column(Boolean)

    souv_eliteless = Column(Boolean)
    souv_pplmore = Column(Boolean)

    ideol_left = Column(Boolean)
    ideol_right = Column(Boolean)

    unsure = Column(Boolean)

    sample = relationship("Sample", back_populates="raw_labels")


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = {"schema": "bert_data", "extend_existing": True}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    sample_id = Column(BigInteger, ForeignKey("bert_data.samples.id"), index=True)

    elite = Column(Float)
    pplcentr = Column(Float)

    left = Column(Float)
    right = Column(Float)
