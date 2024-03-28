from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Query
from torch.utils.data import Dataset

from .strategies import LabelStrategy
from src.db.models import bert_data as bm


class PBertDataset(Dataset):
    all_coders = [
        "grabsch",
        "schadt",
        "richter",
        "riedel",
        "coudry",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        valid_coders: list[str],
        label_strategy: LabelStrategy,
        df_labels: pd.DataFrame | None = None,
    ):
        self.df = df
        self.valid_coders = valid_coders
        self.strategy = label_strategy

        if df is None and df_labels is None:
            raise Exception("Pass any data!")

        if df_labels is None:
            self.apply_label_strategy()
        else:
            self.df_labels = df_labels

    @classmethod
    def from_db(
        cls,
        engine,
        label_strategy: LabelStrategy,
        exclude_coders=[],
    ):
        valid_coders = cls.all_coders.copy()
        for excl in exclude_coders:
            valid_coders.remove(excl)

        query = (
            Query(bm.Label)  # type: ignore
            .join(bm.Sample)
            .with_entities(
                bm.Label.sample_id.label("id"),
                bm.Sample.text,
                bm.Label.username,
                bm.Label.pop_antielite.label("elite"),
                bm.Label.pop_pplcentr.label("centr"),
                bm.Label.ideol_left.label("left"),
                bm.Label.ideol_right.label("right"),
            )
            .filter(bm.Label.username.in_(valid_coders))
        )

        with engine.connect() as conn:
            df = pd.read_sql(query.statement, conn)

        return cls(df=df, valid_coders=valid_coders, label_strategy=label_strategy)

    @classmethod
    def from_disk(cls, path: Path, label_strategy: LabelStrategy, exclude_coders=[]):
        df = pd.read_csv(path)

        valid_coders = cls.all_coders.copy()
        for excl in exclude_coders:
            valid_coders.remove(excl)

        return cls(df=df, label_strategy=label_strategy, valid_coders=valid_coders)

    def to_disk(self, path: Path) -> None:
        self.df.to_csv(path, index=False)

    def apply_label_strategy(self):
        df = self.df.copy()

        # filter df by excluded coders
        df = df.loc[df.username.isin(self.valid_coders), :]

        # filter all samples that are not coded by every coder
        df = df.groupby(["id", "text"], sort=False).filter(
            lambda x: len(x) == len(self.valid_coders),
        )

        df["labels"] = df.apply(self.strategy.create_label, axis=1)
        # df = df[df.labels != -1]

        # ensure sort of coders is the same as in valid_coders
        coder_sort = CategoricalDtype(self.valid_coders, ordered=True)
        df["username"] = df["username"].astype(coder_sort)

        # aggregate codings
        df = (
            df.sort_values(["id", "text", "username"])
            .groupby(["id", "text"])["labels"]
            .agg(tuple)
            .reset_index()
        )

        # create majority vote column
        df["vote"] = df.labels.apply(self.strategy.vote)
        self.df_labels = df

    def subset(self, index):
        df_labels = self.df_labels.loc[index, :].copy()
        df_raw = self.df.loc[self.df.id.isin(df_labels.id), :].copy()
        return PBertDataset(
            df=df_raw,
            df_labels=df_labels,
            label_strategy=self.strategy,
            valid_coders=self.valid_coders,
        )

    def train_test_split(self, test_size=0.3, stratify: bool = True):
        stratify_by = self.df_labels.vote if stratify else None

        train, test = train_test_split(
            self.df_labels,
            test_size=test_size,
            random_state=1337,
            shuffle=True,
            stratify=stratify_by,
        )

        train_raw = self.df.loc[self.df.id.isin(train.id), :]
        test_raw = self.df.loc[self.df.id.isin(test.id), :]

        return (
            PBertDataset(
                df=train_raw,
                df_labels=train,
                label_strategy=self.strategy,
                valid_coders=self.valid_coders,
            ),
            PBertDataset(
                df=test_raw,
                df_labels=test,
                label_strategy=self.strategy,
                valid_coders=self.valid_coders,
            ),
        )

    def create_collate_fn(self, tokenizer):
        return self.strategy.create_collator(tokenizer)

    def __len__(self):
        return len(self.df_labels)

    @property
    def num_labels(self):
        return self.strategy.num_labels

    @property
    def num_coders(self):
        return len(self.valid_coders)

    @property
    def labels(self):
        return self.strategy.labels

    @property
    def coders(self):
        return self.valid_coders

    def __getitem__(self, idx):
        row = self.df_labels.iloc[idx]
        return {"text": row.text, "labels": row.labels, "vote": row.vote}
