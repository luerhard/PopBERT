from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve


def add_predict_proba(model, df, text_col="text", classes: list | None = None):
    """adds prediction probabilities to a pandas DataFrame with data in it

    Args:
        model: The model that should predict the probabilites.
        df (pd.DataFrame): The DataFrame to be modified.
        text_col: The column in the DataFrame that includes the data.
        classes (list, optional): Optional column names for the resulting prediction.
    """

    with torch.inference_mode():
        probas = model.predict_proba(df[text_col]).round(4)

    new = pd.concat([df, pd.DataFrame(probas).set_index(df.index)], axis=1)

    if classes is None:
        classes = list(model.classes_)
    new.columns = list(df.columns) + classes

    return new


class ThresholdFinder:
    thresholds = [t / 100 for t in range(15, 75, 5)]

    def __init__(
        self,
        type: Literal["single_task"] | Literal["multi_task"] = "single_task",
    ) -> None:
        self.type = type

    @staticmethod
    def _single_vote(votes: np.ndarray):
        values, counts = np.unique(votes, return_counts=True)
        majority = values[counts.argmax()]
        return majority

    @staticmethod
    def _find_thresholds_single_task(y_true: np.ndarray, y_probas: np.ndarray) -> dict[int, float]:
        # find number of labels
        n_labels = y_true.shape[1]
        # treat every labels as binary classifier, find best thresh based on f1-score
        thresholds = dict()
        for dim in range(n_labels):
            yt = y_true[:, dim]
            yp = y_probas[:, dim]

            # sklearn
            precision, recall, thresh = precision_recall_curve(yt, yp)
            with np.errstate(divide="ignore", invalid="ignore"):
                fscores = (2 * precision * recall) / (precision + recall)
            ix = np.argmax(fscores)
            thresholds[dim] = thresh[ix]

        return thresholds

    @staticmethod
    def _find_thresholds_multi_task(y_true: np.ndarray, y_probas: np.ndarray):
        n_labels = y_probas[0].shape[1]
        n_coders = y_probas.shape[1]

        thresholds: dict[int, dict[int, float]] = defaultdict(dict)
        for coder in range(n_coders):
            for dim in range(n_labels):
                precision, recall, thresh = precision_recall_curve(
                    y_true[:, coder, dim],
                    y_probas[:, coder, dim],
                )
                fscores = (2 * precision * recall) / (precision + recall)
                ix = np.argmax(fscores)
                thresholds[coder][dim] = thresh[ix]

        return dict(thresholds)

    def find_thresholds(self, y_true, y_probas):
        if self.type == "single_task":
            return self._find_thresholds_single_task(y_true, y_probas)
        elif self.type == "multi_task":
            return self._find_thresholds_multi_task(y_true, y_probas)
