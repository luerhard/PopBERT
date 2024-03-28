from __future__ import annotations

import os
import random
from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Literal
from typing import TypedDict

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import PreTrainedModel


class BatchDict(TypedDict):
    encodings: torch.Tensor
    labels: torch.Tensor
    vote: list


class BasePBertModel(metaclass=ABCMeta):
    model_type: Literal["single_task", "multi_task"]
    threshold_type: Literal[
        "single",
        "per_label",
        "per_layer",
        "per_label_layer",
        "per_label_layer_on_majority",
    ]
    name: str

    def predict_proba(self, encodings):
        _, probas = self(**encodings)
        probas = probas.to("cpu").detach().numpy()
        return probas

    def predict(self, encodings):
        probas = self.predict_proba(encodings)
        y_pred = []
        for proba in probas:
            vote = self.vote(proba)
            y_pred.append(vote)
        return y_pred

    @staticmethod
    def set_seed(seed: int = 1337):
        """Set all seeds to make results reproducible (deterministic mode).
        When seed is None, disables deterministic mode.
        :param seed: an integer to your choosing
        """
        if seed is not None:
            os.environ["PYTHONHASHSEED"] = str(seed)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            np.random.seed(seed)
            random.seed(seed)

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.name})"

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def to(self, device):
        pass

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def score(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: dict) -> dict[str, float]:
        pass


class SingleTaskMultiClassModel(BasePBertModel):
    model_type = "single_task"

    def vote(self, y_proba, threshold: dict):
        for dim, thresh in threshold.items():
            y_proba[:, dim] = np.where(y_proba[:, dim] > thresh, 1, 0)
        return np.argmax(y_proba)

    def score(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: dict):
        y_pred = np.apply_along_axis(self.vote, 1, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        return {"f1_score": f1, "score_meta": -1}


class MultiTaskMultiClassModel(BasePBertModel):
    model_type = "multi_task"

    @staticmethod
    def vote(y_proba):
        per_task_vote = np.apply_along_axis(np.argmax, 1, y_proba)
        values, counts = np.unique(per_task_vote, return_counts=True)
        major_vote = values[counts.argmax()]
        return major_vote

    def score(self, preds):
        y_probas_val = preds["y_probas_val"]
        y_true_val = preds["y_vote_val"]

        y_pred = [self.vote(item) for item in y_probas_val]
        f1 = f1_score(y_true_val, y_pred, average="macro", zero_division=0)
        return {"f1_score": f1, "score_meta": -1}


class SingleTaskMultiLabelModel(BasePBertModel):
    model_type: Literal["single_task", "multi_task"] = "single_task"

    @staticmethod
    def apply_thresh(y_proba, thresholds: dict):
        y_proba = y_proba.copy()
        for dim, thresh in thresholds.items():
            y_proba[:, dim] = np.where(y_proba[:, dim] > thresh, 1, 0)
        return y_proba

    def vote(self, y_proba, threshold=0.5):
        if isinstance(threshold, float):
            return np.where(y_proba > threshold, 1, 0)
        elif isinstance(threshold, dict):
            votes = y_proba.copy()
            for dim, thresh in threshold.items():
                votes[dim] = np.where(votes[dim] > thresh, 1, 0)
            return votes
        else:
            raise NotImplementedError("threshold must be float or dict!")

    def score(self, y_true: np.ndarray, y_pred: np.ndarray, thresholds: dict):
        y_pred = self.apply_thresh(y_pred, thresholds=thresholds)
        f1 = f1_score(y_true, y_pred, average="macro")
        return {"score": f1, "score_meta": thresholds}


class MultiTaskMultiLabelModel(BasePBertModel):
    model_type = "multi_task"

    def vote(self, y_proba, threshold=0.5):
        if isinstance(threshold, float):
            per_task_vote = np.where(y_proba > threshold, 1, 0)
        elif isinstance(threshold, dict):
            per_task_vote = y_proba.copy()
            if isinstance(next(iter(threshold.values())), dict):
                # per_label_layer case
                for layer, label in threshold.items():
                    for dim, thresh in label.items():
                        per_task_vote[layer, dim] = np.where(
                            per_task_vote[layer, dim] > thresh,
                            1,
                            0,
                        )
            else:
                # per_label case
                for dim, thresh in threshold.items():
                    per_task_vote[:, dim] = np.where(per_task_vote[:, dim] > thresh, 1, 0)
        else:
            raise TypeError("threshold has to be float, dict, or dict of dicts")

        major_vote = []
        for dim in range(per_task_vote.shape[1]):
            values, counts = np.unique(per_task_vote[:, dim], return_counts=True)
            dim_vote = values[counts.argmax()]
            major_vote.append(dim_vote)
        return major_vote

    def score(self, y_true: np.ndarray, y_pred: np.ndarray, thresholds: dict):
        y_pred = [self.vote(item, threshold=thresholds) for item in y_pred]
        score = f1_score(y_true, y_pred, average="macro")
        return {"score_meta": thresholds, "score": score}


class BertMultiClass(nn.Module, SingleTaskMultiClassModel):
    def __init__(self, num_labels: int, name: str = "deepset/gbert-base"):
        super().__init__()
        self.name = name
        self.num_labels = num_labels

        self.softmax = nn.Softmax(dim=1)

        self.bert = AutoModelForSequenceClassification.from_pretrained(
            self.name,
            num_labels=self.num_labels,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        pred = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        loss = None
        if labels is not None:
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(pred.logits, labels.long().view(-1))

        return loss, self.softmax(pred.logits)


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_labels)

    def forward(self, output):
        logits = self.linear(output)
        return logits


class BertMultiTaskClassification(nn.Module, MultiTaskMultiClassModel):
    def __init__(
        self,
        tasks: int,
        num_labels: int,
        freeze_bert=False,
        name: str = "deepset/gbert-base",
    ):
        super().__init__()

        self.num_labels = num_labels
        self.n_tasks = tasks
        self.name = name

        self.bert = AutoModel.from_pretrained(name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.config = self.bert.config
        self.config.num_labels = num_labels

        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.softmax = nn.Softmax(dim=1)

        self.heads = nn.ModuleList()
        for _ in range(self.n_tasks):
            head = ClassificationHead(self.config.hidden_size, self.num_labels)
            self.heads.append(head)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[float | None, torch.Tensor]:
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        pooled_output = outputs[1]

        # states  = outputs.hidden_states
        # last_hidden_states = tuple([states[i] for i in range(-4, 0)])
        # last_hidden_state = states[-1]
        # pooled_output = torch.cat(last_hidden_states, dim=-1)
        # pooled_output = pooled_output[:, 0, :]

        # pooled_output = torch.squeeze(
        #    torch.matmul(
        #        attention_mask.type(torch.float32).view(-1, 1, 512),
        #        last_hidden_state,
        #    ),
        #    1,
        # )

        pooled_output = self.dropout(pooled_output)

        if labels is not None:
            labels = labels.permute(1, 0)
            loss = 0.0

        logits_list = []
        for i, head in enumerate(self.heads):
            logits = head(pooled_output)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss += loss_fct(logits.view(-1, self.num_labels), labels[i].long().view(-1))
            logits_list.append(self.softmax(logits))

        logits = torch.stack(logits_list).permute(1, 0, 2)

        if labels is not None:
            return loss, logits
        return None, logits


class BertMultiTaskMultiLabel(nn.Module, MultiTaskMultiLabelModel):
    threshold_type = "per_label"  # type: ignore

    def __init__(
        self,
        num_tasks: int,
        num_labels: int,
        freeze_bert=False,
        name: str = "deepset/gbert-base",
    ):
        super().__init__()

        self.num_labels = num_labels
        self.n_tasks = num_tasks
        self.name = name

        self.bert = AutoModel.from_pretrained(name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.config = self.bert.config
        self.config.num_labels = num_labels

        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.sigmoid = nn.Sigmoid()

        self.heads = nn.ModuleList()
        for _ in range(self.n_tasks):
            head = ClassificationHead(self.config.hidden_size, self.num_labels)
            self.heads.append(head)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[float | None, torch.Tensor]:
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.permute(1, 0, 2)
            loss = 0.0

        pred_list = []
        for i, head in enumerate(self.heads):
            logits = head(pooled_output)
            if labels is not None:
                loss += loss_fct(logits.view(-1, self.num_labels), labels[i].float())
            pred_list.append(self.sigmoid(logits))

        preds = torch.stack(pred_list).permute(1, 0, 2)

        return loss, preds


class BertSingleTaskMultiLabel(nn.Module, SingleTaskMultiLabelModel):
    threshold_type = "per_label"  # type: ignore

    def __init__(self, num_labels: int, name: str = "deepset/gbert-base"):
        super().__init__()
        self.name = name
        self.num_labels = num_labels

        self.sigmoid = nn.Sigmoid()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            self.name,
            num_labels=self.num_labels,
        )

        self.loss_fn = BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        pred = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        loss = None
        if labels is not None:
            loss = self.loss_fn(pred.logits, labels.float())

        return loss, self.sigmoid(pred.logits)
