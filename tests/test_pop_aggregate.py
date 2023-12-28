import numpy as np
import pandas as pd


import src.pop_aggregate
from src.pop_aggregate import bollen_index
from src.pop_aggregate import goertz_index
from src.pop_aggregate import sartori_index

src.pop_aggregate.ELITE = "antielite"
src.pop_aggregate.CENTR = "pplcentr"


def test_bollen():
    s = pd.Series({"antielite": 0.3, "pplcentr": 0.1})
    val = bollen_index(s)
    assert val == 0.2


def test_bollen_nan():
    s = pd.Series({"antielite": 0.3, "pplcentr": np.nan})
    val = bollen_index(s)
    assert pd.isnull(val)


def test_goertz():
    s = pd.Series({"antielite": 0.3, "pplcentr": 0.1})
    val = goertz_index(s)
    assert val == 0.1


def test_goertz_nan():
    s = pd.Series({"antielite": 0.3, "pplcentr": np.nan})
    val = goertz_index(s)
    assert pd.isnull(val)


def test_sartori_one():
    s = pd.Series({"antielite": 0.3, "pplcentr": 0.1})
    val = sartori_index(s, threshold={"antielite": 0.2, "pplcentr": 0.09})
    assert val == 1


def test_sartori_zero():
    s = pd.Series({"antielite": 0.3, "pplcentr": 0.1})
    val = sartori_index(s, threshold={"antielite": 0.4, "pplcentr": 0.04})
    assert val == 0


def test_sartori_nan():
    s = pd.Series({"antielite": 0.3, "pplcentr": np.nan})
    val = sartori_index(s, threshold={"antielite": 0.2, "pplcentr": 0.2})
    assert pd.isnull(val)
