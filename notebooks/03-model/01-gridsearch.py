from __future__ import annotations

from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

import src.db.connect
from src.bert.dataset import PBertDataset
from src.bert.dataset import strategies
from src.bert.gridsearch import ParamGrid
from src.bert.gridsearch import PBertGridSearch

EXCLUDE_CODERS: list[str] = []

data = PBertDataset.from_disk(
    path=src.PATH / "data/bert/train.csv.zip",
    label_strategy=strategies.MLMin1PopIdeol(),
    exclude_coders=EXCLUDE_CODERS,
)

DEBUG = False
if DEBUG:
    MODEL_NAME = "deepset/gbert-base"
    N_SPLITS = 2
else:
    MODEL_NAME = "deepset/gbert-large"
    N_SPLITS = 5

param_grid = ParamGrid(
    models=[
        {
            "name": "MLSingleTaskMin1(gbert-large)",
            "label_strategy": strategies.MLMin1PopIdeol(output_fmt="single_task"),
            "tokenizer": lambda: AutoTokenizer.from_pretrained(MODEL_NAME),
            "model": lambda: AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=data.num_labels,
                problem_type="multi_label_classification",
            ),
        },
        {
            "name": "mBERT",
            "label_strategy": strategies.MLMin1PopIdeol(output_fmt="single_task"),
            "tokenizer": lambda: AutoTokenizer.from_pretrained("bert-base-multilingual-cased"),
            "model": lambda: AutoModelForSequenceClassification.from_pretrained(
                "bert-base-multilingual-cased",
                num_labels=data.num_labels,
                problem_type="multi_label_classification",
            ),
        },
        {
            "name": "xlm-roberta",
            "label_strategy": strategies.MLMin1PopIdeol(output_fmt="single_task"),
            "tokenizer": lambda: AutoTokenizer.from_pretrained("xlm-roberta-large"),
            "model": lambda: AutoModelForSequenceClassification.from_pretrained(
                "xlm-roberta-large",
                num_labels=data.num_labels,
                problem_type="multi_label_classification",
            ),
        },
    ],
    lr=[
        9e-6,
        4e-6,
        1e-5,
    ],
    batch_size=[
        # 4,
        8,
        16,
    ],
    weight_decay=[
        1e-2,
        5e-2,
    ],
    clip=[0.5, 1, 5],
    max_epochs=20,
)


# debug
if DEBUG:
    data = data.subset(list(range(100)))

split_by = data.df_labels.vote.apply(lambda x: sum(x))
kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=2)

for i, (train_index, test_index) in enumerate(kfold.split(data, y=split_by), 1):
    train = data.subset(train_index)
    test = data.subset(test_index)

    grid_search = PBertGridSearch(
        train=train,
        test=test,
        param_grid=param_grid,
        kfold=i,
        clear_db=False,
        db_file=src.PATH / "gridsearch.db",
    )

    grid_search.search()
