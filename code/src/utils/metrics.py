from collections.abc import Sequence

import numpy as np
import sklearn.metrics as m


def custom_f1_score(y_true, y_pred, labels: Sequence[Sequence]):
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_true, np.ndarray):
        y_pred = np.array(y_pred)

    f1_scores = []
    for combination in labels:
        if not isinstance(combination, np.ndarray):
            combination = np.array(combination)

        compare_func = lambda x: np.array_equal(x, combination)  # noqa

        y_true_bin = np.apply_along_axis(compare_func, 1, y_true)
        y_pred_bin = np.apply_along_axis(compare_func, 1, y_pred)

        f1_score = m.f1_score(y_true_bin, y_pred_bin, zero_division="warn")
        f1_scores.append(f1_score)

    return np.mean(f1_scores)
