from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import relationship


class Base(DeclarativeBase):
    metadata = MetaData()


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    lr = Column(Float)
    batch_size = Column(Integer)
    kfold = Column(Integer)
    weight_decay = Column(Float)
    clip = Column(Float)
    results = relationship("Result", uselist=True)


class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("models.id"))
    epoch = Column(Integer)
    loss = Column(Float)
    val_loss = Column(Float)
    score = Column(Float)
    best_threshold = Column(String)

    model = relationship("Model", uselist=False)
