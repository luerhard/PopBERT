from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from typing import Literal

import numpy as np
import torch


class LabelStrategy(metaclass=ABCMeta):
    labels: list[str]

    def __init__(
        self,
        output_fmt: Literal["single_task"] | Literal["multi_task"] = "single_task",
    ):
        self.output_fmt = output_fmt

    @staticmethod
    @abstractmethod
    def create_label(row):
        pass

    @staticmethod
    @abstractmethod
    def vote(votes):
        pass

    @property
    def num_labels(self):
        return len(self.labels)

    def create_collator(self, tokenizer):
        def collate_fn(batch):
            nonlocal tokenizer

            text = [d["text"] for d in batch]
            vote = [d["vote"] for d in batch]
            if self.output_fmt == "multi_task":
                labels = np.array([d["labels"] for d in batch])
            else:
                labels = np.array(vote)

            encoding = tokenizer(text, padding=True, return_tensors="pt")

            return {
                "encodings": encoding,
                "text": text,
                "labels": torch.Tensor(labels),
                "vote": vote,
            }

        return collate_fn


class BaseMultiClassStrategy(LabelStrategy, metaclass=ABCMeta):
    @staticmethod
    def vote(votes):
        values, counts = np.unique(votes, return_counts=True)
        majority_vote = values[counts.argmax()]
        return majority_vote

    def create_collator(self, tokenizer):
        def collate_fn(batch):
            nonlocal tokenizer

            text = [d["text"] for d in batch]
            vote = [d["vote"] for d in batch]
            if self.output_fmt == "multi_task":
                labels = np.array([d["labels"] for d in batch])
            else:
                labels = np.array(vote)

            encoding = tokenizer(text, padding=True, return_tensors="pt")

            return {
                "encodings": encoding,
                "text": text,
                "labels": torch.LongTensor(labels),
                "vote": vote,
            }

        return collate_fn


class BaseMultiLabelStrategy(LabelStrategy):
    @staticmethod
    def vote(votes):
        votes = np.array(votes)

        majority_vote = []
        for col in range(votes.shape[1]):
            label_dim = votes[:, col]
            values, counts = np.unique(label_dim, return_counts=True)
            majority_vote_dim = values[counts.argmax()]
            majority_vote.append(majority_vote_dim)

        return tuple(majority_vote)


class BaseMin1MultiLabelStrategy(LabelStrategy, metaclass=ABCMeta):
    @staticmethod
    def vote(votes):
        votes = np.array(votes)

        majority_vote = []
        for col in range(votes.shape[1]):
            label_dim = votes[:, col]
            if 1 in label_dim:
                majority_vote.append(1)
            else:
                majority_vote.append(0)

        return tuple(majority_vote)


class BaseCertaintyLabelStrategy(LabelStrategy, metaclass=ABCMeta):
    @staticmethod
    def vote(votes):
        votes = np.array(votes)
        return np.mean(votes, axis=0)

    def create_collator(self, tokenizer):
        def collate_fn(batch):
            nonlocal tokenizer

            text = [d["text"] for d in batch]
            certainty = np.array([d["vote"] for d in batch])
            labels = np.where(certainty > 0.5, 1, 0)

            encoding = tokenizer(text, padding=True, return_tensors="pt")

            return {
                "encodings": encoding,
                "text": text,
                "labels": torch.Tensor(certainty),
                "vote": labels,
            }

        return collate_fn


class MLCertainPopBinIdeol(BaseCertaintyLabelStrategy):
    labels = ["pop", "left", "right"]

    @staticmethod
    def create_label(row):
        label = [0, 0, 0]

        any_pop = row["elite"] or row["centr"]

        if any_pop:
            label[0] = 1
        else:
            # TODO: should this return 0,0,0 if no populism?
            return tuple([0, 0, 0])

        if row["left"] and any_pop:
            label[1] = 1
        if row["right"] and any_pop:
            label[2] = 1

        return tuple(label)


class MLCertainPopIdeol(BaseCertaintyLabelStrategy):
    labels = ["elite", "centr", "left", "right"]

    @staticmethod
    def create_label(row):
        label = [0, 0, 0, 0]

        any_pop = row["elite"] or row["centr"]

        if not any_pop:
            return tuple([0, 0, 0, 0])

        if row["elite"]:
            label[0] = 1
        if row["centr"]:
            label[1] = 1
        if row["left"]:
            label[2] = 1
        if row["right"]:
            label[3] = 1

        return tuple(label)


class MLMin1PopBinIdeol(BaseMin1MultiLabelStrategy):
    labels = ["pop", "left", "right"]

    @staticmethod
    def create_label(row):
        label = [0, 0, 0]

        any_pop = row["elite"] or row["centr"]

        if any_pop:
            label[0] = 1
        else:
            return tuple([0, 0, 0])

        if row["left"] and any_pop:
            label[1] = 1
        if row["right"] and any_pop:
            label[2] = 1

        return tuple(label)


class MLMin1PopIdeol(BaseMin1MultiLabelStrategy):
    labels = ["elite", "centr", "left", "right"]

    @staticmethod
    def create_label(row):
        label = [0, 0, 0, 0]

        any_pop = row["elite"] or row["centr"]

        if not any_pop:
            return tuple([0, 0, 0, 0])

        if row["elite"]:
            label[0] = 1
        if row["centr"]:
            label[1] = 1
        if row["left"]:
            label[2] = 1
        if row["right"]:
            label[3] = 1

        return tuple(label)


class MLPopIdeol(BaseMultiLabelStrategy):
    labels = ["elite", "centr", "left", "right"]

    @staticmethod
    def create_label(row):
        label = [0, 0, 0, 0]

        any_pop = row["elite"] or row["centr"]
        if not any_pop:
            return tuple([0, 0, 0, 0])

        if row["elite"]:
            label[0] = 1
        if row["centr"]:
            label[1] = 1
        if row["left"] and any_pop:
            label[2] = 1
        if row["right"] and any_pop:
            label[3] = 1

        return tuple(label)


class MLPopBinIdeol(BaseMultiLabelStrategy):
    labels = ["pop", "left", "right"]

    @staticmethod
    def create_label(row):
        label = [0, 0, 0]

        any_pop = row["elite"] or row["centr"]
        if not any_pop:
            return tuple([0, 0, 0])

        if any_pop:
            label[0] = 1
        if row["left"] and any_pop:
            label[1] = 1
        if row["right"] and any_pop:
            label[2] = 1

        return tuple(label)


class MCPop(BaseMultiClassStrategy):
    labels = ["none", "elite", "centr"]

    @staticmethod
    def create_label(row):
        if row["centr"] and row["elite"]:
            return 2
        elif row["elite"]:
            return 1
        elif row["centr"]:
            return 2
        else:
            return 0


class MCPopBin(BaseMultiClassStrategy):
    labels = ["none", "pop"]

    @staticmethod
    def create_label(row):
        if row["centr"] or row["elite"]:
            return 1
        return 0


class MCPopBinIdeol(BaseMultiClassStrategy):
    labels = ["none", "pop", "pop_left", "pop_right"]

    @staticmethod
    def create_label(row):
        if (row["centr"] or row["elite"]) and row["left"] and row["right"]:
            # happens 6 times for 3 coders and 15 times for 5 coders
            return 2
        if (row["centr"] or row["elite"]) and row["left"]:
            return 2
        if (row["centr"] or row["elite"]) and row["right"]:
            return 3
        if row["centr"] or row["elite"]:
            return 1
        return 0


class MCPopIdeol(BaseMultiClassStrategy):
    labels = ["none", "elite", "elite_left", "elite_right", "centr", "centr_left", "centr_right"]

    @staticmethod
    def create_label(row):
        if row["centr"] and row["left"]:
            return 5
        if row["centr"] and row["right"]:
            return 6
        if row["centr"]:
            return 4

        if row["elite"] and row["left"]:
            return 2
        if row["elite"] and row["right"]:
            return 3
        if row["elite"]:
            return 1

        return 0


class IdeolLeftBin(BaseMultiClassStrategy):
    labels = ["none", "left"]

    @staticmethod
    def create_label(row):
        if row["left"]:
            return 1
        return 0


class IdeolRightBin(BaseMultiClassStrategy):
    labels = ["none", "right"]

    @staticmethod
    def create_label(row):
        if row["right"]:
            return 1
        return 0


class PopEliteBin(BaseMultiClassStrategy):
    labels = ["none", "elite"]

    @staticmethod
    def create_label(row):
        if row["elite"]:
            return 1
        return 0


class PopCentrBin(BaseMultiClassStrategy):
    labels = ["none", "centr"]

    @staticmethod
    def create_label(row):
        if row["centr"]:
            return 1
        return 0


class MLIdeol(BaseMultiLabelStrategy):
    labels = ["left", "right"]

    @staticmethod
    def create_label(row):
        label = [0, 0]
        if row["left"]:
            label[0] = 1
        if row["right"]:
            label[1] = 1

        return tuple(label)
