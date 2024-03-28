# type: ignore
import datetime as dt

from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    version = Column(String, index=True, nullable=False)
    path = Column(String)
    date_added = Column(DateTime, default=dt.datetime.utcnow)


class Sample(Base):
    __tablename__ = "samples"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    text = Column(String, index=True)
    date_added = Column(DateTime, default=dt.datetime.utcnow)

    predictions = relationship("Prediction", back_populates="sample", uselist=True)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    sample_id = Column(Integer, ForeignKey(Sample.id), index=True)
    model_id = Column(Integer, ForeignKey(Model.id), index=True)

    pop_antielite = Column(Float)
    pop_pplcentr = Column(Float)

    souv_eliteless = Column(Float)
    souv_pplmore = Column(Float)

    ideol_left = Column(Float)
    ideol_right = Column(Float)

    sample = relationship(Sample, back_populates="predictions")
