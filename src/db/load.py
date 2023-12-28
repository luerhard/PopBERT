# type: ignore
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import pandas as pd
from sqlalchemy.orm import Query
from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload

import src.db.models.bert_data as bm


@lru_cache(maxsize=1)
def label_data(engine, batch: int | None = None) -> pd.DataFrame:
    """loads raw data for labels per coder

    Args:
        engine (sa.engine): engine to get data from.
        batch (int | None, optional): can be used as batch number to inspect data from a specific
            batch. If None, all data are used. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with columns [sample_id, username, labels] and
            types [int, str, frozenset]
    """
    with Session(engine) as s:
        query = s.query(bm.Label).with_entities(
            bm.Label.sample_id.label("id"),
            bm.Label.username,
            bm.Label.pop_antielite,
            bm.Label.pop_pplcentr,
            bm.Label.souv_eliteless,
            bm.Label.souv_pplmore,
            bm.Label.ideol_left,
            bm.Label.ideol_right,
        )

        if batch:
            query = query.join(bm.Sample).filter(bm.Sample.used_in_batch == batch)

    with engine.connect() as conn:
        df = pd.read_sql(query.statement, conn)

    def create_label_set(row):
        labels = set()
        if row.pop_antielite.bool():
            labels.add("antielite")
        if row.pop_pplcentr.bool():
            labels.add("pplcentr")
        if "antielite" not in labels and "pplcentr" not in labels:
            labels.add("no_pop")

        if row.souv_eliteless.bool():
            labels.add("eliteless")
        if row.souv_pplmore.bool():
            labels.add("pplmore")
        if "eliteless" not in labels and "pplmore" not in labels:
            labels.add("no_sov")

        if row.ideol_left.bool():
            labels.add("left")
        if row.ideol_right.bool():
            labels.add("right")
        if "left" not in labels and "right" not in labels:
            labels.add("no_ideol")

        return frozenset(labels)

    df = df.groupby(["id", "username"]).apply(create_label_set).reset_index()
    df.columns = ["sample_id", "username", "labels"]

    return df


@lru_cache(maxsize=1)
def samples_per_coder(engine, batch: int | None = None) -> pd.DataFrame:
    with Session(engine) as s:
        if batch is None:
            batch_filter = bm.Sample.used_in_batch != None  # noqa: E711
        else:
            batch_filter = bm.Sample.used_in_batch == batch

        query = (
            s.query(bm.Sample)
            .filter(batch_filter)
            .with_entities(bm.Sample.id.label("sample_id"), bm.Sample.text)
        )

    with engine.connect() as conn:
        samples = pd.read_sql(query.statement, conn)
    codes = label_data(engine, batch=batch)
    df = pd.merge(samples, codes, on="sample_id", how="inner")

    return df


@lru_cache(maxsize=1)
def certainty_data(engine):
    with Session(engine) as session:
        samples = (
            session.query(bm.Sample)
            .options(joinedload(bm.Sample.raw_labels))
            .filter(bm.Sample.used_in_batch != None)  # noqa: E711
        )

        rows = []
        for sample in samples.all():
            if sample.n_coders > 0:
                d = defaultdict(int)
                for label in sample.raw_labels:
                    d["n"] += 1

                    if label.unsure:
                        d["unsure"] += 1

                        d["antielite"] += label.pop_antielite * 0.5
                        d["pplcentr"] += label.pop_pplcentr * 0.5
                    else:
                        d["antielite"] += label.pop_antielite
                        d["pplcentr"] += label.pop_pplcentr

                    d["eliteless"] += label.souv_eliteless
                    d["pplmore"] += label.souv_pplmore

                    d["left"] += label.ideol_left
                    d["right"] += label.ideol_right

                labels = (
                    d["antielite"] / (d["n"] - (d["unsure"] * 0.5)),
                    d["pplcentr"] / (d["n"] - (d["unsure"] * 0.5)),
                    d["eliteless"] / d["n"],
                    d["pplmore"] / d["n"],
                    d["left"] / d["n"],
                    d["right"] / d["n"],
                )

                row = (sample.id, sample.text, labels)
                rows.append(row)

    return pd.DataFrame(rows, columns=["id", "text", "labels"])


@lru_cache(maxsize=1)
def classification_pop_data(engine):
    valid_coders = [
        "grabsch",
        "richter",
        "riedel",
        # "schadt",
        "coudry",
    ]

    query = (
        Query(bm.Label)
        .join(bm.Sample)
        .with_entities(
            bm.Label.sample_id.label("id"),
            bm.Sample.text,
            bm.Label.username,
            bm.Label.pop_antielite.label("antielite"),
            bm.Label.pop_pplcentr.label("pplcentr"),
        )
        .filter(bm.Label.username.in_(valid_coders))
    )

    with engine.connect() as conn:
        df = pd.read_sql(query.statement, conn)

    df = df.drop("username", axis=1).groupby(["id", "text"]).agg("mean")

    for col in ["antielite", "pplcentr"]:
        df[f"{col}_bin"] = df[col].apply(lambda x: 1 if x >= 0.5 else 0)

    df = df[~((df.antielite_bin == 1) & (df.pplcentr_bin == 1))]

    def set_labels(row):
        if row["antielite_bin"] == 1:
            return 1
        elif row["pplcentr_bin"] == 1:
            return 2
        else:
            return 0

    df["label"] = df.apply(set_labels, axis=1)

    df = df.reset_index()

    return df
