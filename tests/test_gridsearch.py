from __future__ import annotations

import pytest
from transformers import AutoTokenizer

import src
import src.db.connect
from src.bert import module
from src.bert.dataset import PBertDataset
from src.bert.dataset import strategies
from src.bert.gridsearch import ParamGrid
from src.bert.gridsearch import PBertGridSearch

DEFAULT_ML_STRATEGY = strategies.MLCertainPopBinIdeol

EXCLUDE_CODERS: list[str] = []


@pytest.fixture
def ml_data():
    train = PBertDataset.from_disk(
        path=src.PATH / "data/bert/train.csv.zip",
        label_strategy=DEFAULT_ML_STRATEGY(),
        exclude_coders=EXCLUDE_CODERS,
    )

    train.df = train.df.iloc[:50, :]
    train.apply_label_strategy()

    test = PBertDataset.from_disk(
        path=src.PATH / "data/bert/test.csv.zip",
        label_strategy=DEFAULT_ML_STRATEGY(),
        exclude_coders=EXCLUDE_CODERS,
    )

    test.df = test.df.iloc[:50, :]
    test.apply_label_strategy()

    val = PBertDataset.from_disk(
        path=src.PATH / "data/bert/validation.csv.zip",
        label_strategy=DEFAULT_ML_STRATEGY(),
        exclude_coders=EXCLUDE_CODERS,
    )

    val.df = val.df.iloc[:50, :]
    val.apply_label_strategy()

    return train, test, val


def test_singletaskmin1(ml_data):
    train, test, val = ml_data

    param_grid = ParamGrid(
        models=[
            {
                "name": "BertMultiTaskMultiLabelSingleOutput(gbert-base)",
                "label_strategy": strategies.MLMin1PopBinIdeol(output_fmt="single_task"),
                "tokenizer": lambda: AutoTokenizer.from_pretrained("deepset/gbert-base"),
                "model": lambda: module.BertSingleTaskMultiLabel(
                    num_labels=train.num_labels,
                    name="deepset/gbert-base",
                ),
            },
        ],
        lr=[
            1e-5,
        ],
        batch_size=[
            2,
        ],
        max_epochs=1,
    )

    grid_search = PBertGridSearch(
        train=train,
        test=test,
        validation=val,
        param_grid=param_grid,
        clear_db=True,
        db_file=src.PATH / "tmp/test_grid.db",
    )

    grid_search.search()

    assert True


def test_multitaskmultilabel(ml_data):
    train, test, val = ml_data

    param_grid = ParamGrid(
        models=[
            {
                "name": "BertMultiTaskMultiLabel(gbert-base)",
                "label_strategy": strategies.MLPopIdeol(output_fmt="multi_task"),
                "tokenizer": lambda: AutoTokenizer.from_pretrained("deepset/gbert-base"),
                "model": lambda: module.BertMultiTaskMultiLabel(
                    num_labels=train.num_labels,
                    num_tasks=train.num_coders,
                    name="deepset/gbert-base",
                ),
            },
        ],
        lr=[
            1e-5,
        ],
        batch_size=[
            2,
        ],
        max_epochs=1,
    )

    grid_search = PBertGridSearch(
        train=train,
        test=test,
        validation=val,
        param_grid=param_grid,
        clear_db=True,
        db_file=src.PATH / "tmp/test_grid.db",
    )

    grid_search.search()

    assert True


def test_singletaskcertainty(ml_data):
    train, test, val = ml_data

    param_grid = ParamGrid(
        models=[
            {
                "name": "MLSingleTaskCertainty(gbert-base)",
                "label_strategy": strategies.MLCertainPopBinIdeol(output_fmt="single_task"),
                "tokenizer": lambda: AutoTokenizer.from_pretrained("deepset/gbert-base"),
                "model": lambda: module.BertSingleTaskMultiLabel(
                    num_labels=train.num_labels,
                    name="deepset/gbert-base",
                ),
            },
        ],
        lr=[
            1e-5,
        ],
        batch_size=[
            2,
        ],
        max_epochs=1,
    )

    grid_search = PBertGridSearch(
        train=train,
        test=test,
        validation=val,
        param_grid=param_grid,
        clear_db=True,
        db_file=src.PATH / "tmp/test_grid.db",
    )

    grid_search.search()

    assert True
