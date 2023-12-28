import numpy as np
import pytest

from src.bert.utils import ThresholdFinder


@pytest.fixture
def sample_data_single_task():
    y_true = [
        (1, 1, 0),
        (1, 1, 0),
        (0, 0, 0),
        (1, 0, 1),
        (1, 0, 0),
        (1, 0, 0),
    ]

    y_pred = [
        (0.6, 0.7, 0.1),
        (0.4, 0.9, 0.03),
        (0.35, 0.3, 0.55),
        (0.7, 0.3, 0.6),
        (0.9, 0.2, 0.1),
        (0.6, 0.2, 0.1),
    ]

    return np.array(y_true), np.array(y_pred)


@pytest.fixture
def sample_data_multi_task():
    y_true = [
        (1, 1, 0),
        (1, 1, 0),
        (0, 0, 0),
        (1, 0, 1),
        (1, 0, 0),
        (1, 0, 0),
    ]

    y_labels = [
        [(1, 1, 0), (1, 1, 0), (1, 1, 0)],
        [(1, 1, 0), (1, 1, 0), (1, 0, 1)],
        [(0, 0, 0), (0, 0, 0), (0, 0, 0)],
        [(1, 0, 1), (1, 0, 1), (1, 0, 1)],
        [(1, 0, 0), (1, 0, 0), (1, 0, 0)],
        [(1, 0, 0), (1, 0, 0), (1, 0, 1)],
    ]

    y_pred = [
        [(0.6, 0.7, 0.1), (0.3, 0.7, 0.1), (0.4, 0.4, 0.1)],
        [(0.4, 0.9, 0.03), (0.7, 0.2, 0.1), (0.6, 0.4, 0.1)],
        [(0.35, 0.3, 0.55), (0.3, 0.3, 0.6), (0.05, 0.2, 0.3)],
        [(0.7, 0.3, 0.6), (0.4, 0.3, 0.6), (0.2, 0.2, 0.3)],
        [(0.9, 0.2, 0.1), (0.4, 0.25, 0.25), (0.5, 0.1, 0.1)],
        [(0.6, 0.2, 0.1), (0.6, 0.25, 0.25), (0.45, 0.1, 0.1)],
    ]

    y_pred = [np.array(obj) for obj in y_pred]

    return np.array(y_true), np.array(y_labels), np.array(y_pred)


@pytest.fixture
def sample_data_multi_task_4_coders():
    y_true = [
        (1, 1, 0),
        (1, 1, 0),
        (0, 0, 0),
        (1, 0, 1),
        (1, 0, 0),
        (1, 0, 0),
    ]

    y_labels = [
        [(1, 1, 0), (1, 1, 0), (1, 1, 0), (1, 1, 0)],
        [(1, 1, 0), (1, 1, 0), (1, 0, 1), (1, 0, 1)],
        [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
        [(1, 0, 1), (1, 0, 1), (1, 0, 1), (1, 0, 1)],
        [(1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0)],
        [(1, 0, 0), (1, 0, 0), (1, 0, 1), (1, 0, 1)],
    ]

    y_pred = [
        [(0.6, 0.7, 0.1), (0.3, 0.7, 0.1), (0.4, 0.4, 0.1), (0.4, 0.4, 0.1)],
        [(0.4, 0.9, 0.03), (0.7, 0.2, 0.1), (0.6, 0.4, 0.1), (0.6, 0.4, 0.1)],
        [(0.35, 0.3, 0.5), (0.3, 0.3, 0.6), (0.05, 0.2, 0.3), (0.05, 0.2, 0.3)],
        [(0.7, 0.3, 0.6), (0.4, 0.3, 0.6), (0.2, 0.2, 0.3), (0.2, 0.2, 0.3)],
        [(0.9, 0.2, 0.1), (0.4, 0.25, 0.25), (0.5, 0.1, 0.1), (0.5, 0.1, 0.1)],
        [(0.6, 0.2, 0.1), (0.6, 0.25, 0.25), (0.45, 0.1, 0.1), (0.45, 0.1, 0.1)],
    ]

    y_pred = [np.array(obj) for obj in y_pred]

    return np.array(y_true), np.array(y_labels), np.array(y_pred)


def test_find_threshold_per_label_single_task(sample_data_single_task):
    y_true, y_pred = sample_data_single_task
    thresh_finder = ThresholdFinder(method="per_label", type="single_task")

    thresholds = thresh_finder.find_thresholds(y_true, y_pred)

    assert thresholds == {0: 0.4, 1: 0.7, 2: 0.6}


def test_find_threshold_per_label_multi_task(sample_data_multi_task):
    _, y_labels, y_pred = sample_data_multi_task
    thresh_finder = ThresholdFinder(method="per_label", type="multi_task")

    thresholds = thresh_finder.find_thresholds(y_labels, y_pred)
    assert thresholds == {
        0: {0: 0.4, 1: 0.7, 2: 0.6},
        1: {0: 0.3, 1: 0.7, 2: 0.6},
        2: {0: 0.2, 1: 0.4, 2: 0.1},
    }


@pytest.mark.skip
def test_find_threshold_per_label_layer_on_majority(sample_data):
    thresh_finder = ThresholdFinder(method="per_label_layer_on_majority")

    thresholds = thresh_finder.find_thresholds(sample_data)
    assert thresholds == {
        0: {0: 0.35, 1: 0.3, 2: 0.55},
        1: {0: 0.3, 1: 0.3, 2: 0.25},
        2: {0: 0.15, 1: 0.2, 2: 0.15},
    }


@pytest.mark.skip
def test_find_threshold_per_layer(sample_data):
    thresh_finder = ThresholdFinder(method="per_layer")

    thresholds = thresh_finder.find_thresholds(sample_data)
    assert thresholds == {0: 0.55, 1: 0.3, 2: 0.2}
