from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload

from .models import Base
from .models import Model
from .models import Result


class ResultManager:
    def __init__(self, db_file, clear_db, logger) -> None:
        if clear_db:
            if db_file.is_file():
                db_file.unlink()
        self.logger = logger
        self.db_conn_string = f"sqlite:///{db_file.absolute()}"
        self.logger.warning("DB-Connection: %s", self.db_conn_string)
        self.db_engine = create_engine(self.db_conn_string)
        Base.metadata.create_all(self.db_engine)

    def get_model(
        self,
        model_name: str,
        batch_size: int,
        lr: float,
        weight_decay: float,
        clip: float,
        kfold: int,
    ):
        existed = True
        with Session(self.db_engine) as s:
            model = (
                s.query(Model)
                .options(joinedload(Model.results))
                .filter(
                    Model.name == model_name,
                    Model.batch_size == batch_size,
                    Model.lr == lr,
                    Model.kfold == kfold,
                    Model.weight_decay == weight_decay,
                    Model.clip == clip,
                )
                .one_or_none()
            )
        if not model:
            existed = False
            model = Model(
                name=model_name,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                kfold=kfold,
                clip=clip,
            )

        return existed, model

    def save_results(
        self,
        model_name: str,
        batch_size: int,
        epoch: int,
        lr: float,
        score: float,
        best_threshold: str,
        loss: float,
        val_loss: float,
        weight_decay: float,
        clip: float,
        kfold: int,
    ):
        _, model = self.get_model(
            model_name,
            batch_size,
            lr,
            weight_decay,
            clip,
            kfold,
        )
        with Session(self.db_engine) as s:
            result = (
                s.query(Result)
                .filter(
                    Result.model_id == model.id,
                    Result.epoch == epoch,
                )
                .one_or_none()
            )

            if result:
                raise Exception(
                    "Result already exists! {}".format(
                        "-".join((model_name, str(batch_size), str(lr))),
                    ),
                )
            result = Result(
                epoch=epoch,
                score=score,
                best_threshold=str(best_threshold),
                loss=loss,
                val_loss=val_loss,
            )
            model.results.append(result)

            s.add(model)
            s.commit()

    def report_best_model(self, per_model=False):
        with Session(self.db_engine) as s:
            best_result = (
                s.query(Result)
                .options(joinedload(Result.model))
                .order_by(Result.score.desc())
                .limit(1)
                .one()
            )

        self.logger.warning(
            "Current best model: %s(lr=%.0E, batch_size=%d, epoch=%d, fold=%d) -- f1: %.4f",
            best_result.model.name,
            best_result.model.lr,
            best_result.model.batch_size,
            best_result.epoch,
            best_result.model.kfold,
            best_result.score,
        )
