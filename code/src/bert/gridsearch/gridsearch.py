from __future__ import annotations

from collections.abc import Callable
from typing import Any
from typing import TypedDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import logging as tf_logging

import src
from .result_manager import ResultManager
from src.bert import training
from src.bert import utils as bert_utils
from src.bert.dataset import PBertDataset
from src.bert.module import BasePBertModel as PBertModel
from src.utils.logger import setup_logging

tf_logging.set_verbosity_error()

logger = setup_logging(
    name="gridsearch",
    overwrite=True,
    filelevel="DEBUG",
    streamlevel="DEBUG",
)


class ParamGrid(TypedDict):
    models: list[dict[str, Any]]
    lr: list[float]
    batch_size: list[int]
    weight_decay: list[float]
    clip: list[float]
    max_epochs: int


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class PBertGridSearch:
    def __init__(
        self,
        train: PBertDataset,
        test: PBertDataset,
        param_grid: ParamGrid,
        kfold: int,
        db_file=src.PATH / "gridsearch.db",
        clear_db=False,
    ) -> None:
        self.train_data = train
        self.test_data = test
        self.param_grid = param_grid
        self.kfold = kfold

        self.result_manager = ResultManager(db_file, clear_db, logger)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            logger.error("GridSearch on CPU will take forever!")

    def predict(self, model, dataset):
        y_true = []
        y_pred = []
        for batch in dataset:
            encodings = batch["encodings"]
            encodings = encodings.to(self.device)
            predictions = model.predict_proba(encodings)
            y_true.extend(batch["vote"])
            y_pred.extend(predictions)

        return np.array(y_true), np.array(y_pred)

    def score(self, model: PBertModel, test_data: DataLoader):
        with torch.inference_mode():
            # get score based on test data
            y_true, y_probas = self.predict(model, test_data)
            val_loss = model.loss_fn(torch.Tensor(y_true), torch.Tensor(y_probas))  # type: ignore
            val_loss = float(val_loss)
            thresh_finder = bert_utils.ThresholdFinder(
                type=model.model_type,
            )
            thresholds = thresh_finder.find_thresholds(y_true, y_probas)
            score = model.score(y_true, y_probas, {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5})
            score["best_threshold"] = thresholds
            score["val_loss"] = val_loss
        logger.warning(
            "%s: f1-macro=%.4f :: thresh=%s",
            str(model),
            score["score"],
            str(score["score_meta"]),
        )
        return score

    def train(
        self,
        model_name: str,
        model_fn: Callable[[], PBertModel],
        tokenizer_fn: Callable[[], BertTokenizer],
        lr: float,
        batch_size: int,
        max_epochs: int,
        weight_decay: float,
        clip: float,
    ):
        model = model_fn()
        tokenizer = tokenizer_fn()

        model = model.to(self.device)
        model.train()

        data: dict[str, DataLoader] = dict()
        for name, dataset in [
            ("train", self.train_data),
            ("test", self.test_data),
        ]:
            collate_fn = self.train_data.create_collate_fn(tokenizer)
            data[name] = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=1,
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            amsgrad=False,
            weight_decay=weight_decay,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=lr / 10,
        )

        logger.info("Start training for %s(lr=%.0E, batch_size=%d)", model_name, lr, batch_size)
        early_stopper = EarlyStopper(patience=4, min_delta=0.001)

        for epoch in range(1, max_epochs + 1):
            logger.warning("Current LR: %f", optimizer.state_dict()["param_groups"][0]["lr"])
            train_loss = training.train_epoch(
                model=model,
                dataloader=data["train"],
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                clip=clip,
            )
            eval_loss, score, thresholds = training.eval_epoch(model, data["test"])
            score = {
                "model_name": model_name,
                "epoch": epoch,
                "lr": lr,
                "batch_size": batch_size,
                "kfold": self.kfold,
                "score": score,
                "best_threshold": thresholds,
                "loss": train_loss,
                "val_loss": eval_loss,
                "weight_decay": weight_decay,
                "clip": clip,
            }
            logger.info("Epoch %d - config: %s", epoch, score)
            self.result_manager.save_results(**score)
            model.train()
            if early_stopper.early_stop(score["val_loss"]):
                logger.warning("Early stopping after %d epochs", epoch)
                break

        self.result_manager.report_best_model()

    def search(self):
        models = self.param_grid["models"]
        epochs = self.param_grid["max_epochs"]
        learning_rates = self.param_grid["lr"]
        batch_sizes = self.param_grid["batch_size"]
        weight_decays = self.param_grid["weight_decay"]
        clips = self.param_grid["clip"]

        for model_dict in models:
            for lr in learning_rates:
                for batch_size in batch_sizes:
                    for weight_decay in weight_decays:
                        for clip in clips:
                            existed, _ = self.result_manager.get_model(
                                model_name=model_dict["name"],
                                batch_size=batch_size,
                                lr=lr,
                                kfold=self.kfold,
                                weight_decay=weight_decay,
                                clip=clip,
                            )
                            if existed:
                                logger.info(
                                    "Skipping %s(lr=%.0E, batch_size=%d, decay=%f)",
                                    model_dict["name"],
                                    lr,
                                    batch_size,
                                    weight_decay,
                                )
                                continue

                            if not isinstance(
                                model_dict["label_strategy"],
                                type(self.train_data.strategy),
                            ):
                                self.train_data.strategy = model_dict["label_strategy"]
                                self.test_data.strategy = model_dict["label_strategy"]
                                self.train_data.apply_label_strategy()
                                self.test_data.apply_label_strategy()

                            self.train(
                                model_name=model_dict["name"],
                                model_fn=model_dict["model"],
                                tokenizer_fn=model_dict["tokenizer"],
                                lr=lr,
                                batch_size=batch_size,
                                max_epochs=epochs,
                                weight_decay=weight_decay,
                                clip=clip,
                            )
