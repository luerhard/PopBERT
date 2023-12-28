import numpy as np
import pandas as pd

ELITE = "antielite"
CENTR = "pplcentr"


def multiplicative_index(row: pd.Series) -> float:
    return row[ELITE] * row[CENTR]


def bollen_index(row: pd.Series) -> float:
    return np.mean([row[ELITE], row[CENTR]])


def goertz_index(row: pd.Series) -> float:
    return np.min([row[ELITE], row[CENTR]])


def sartori_index(row: pd.Series, threshold: dict) -> int:
    values = np.array([row[ELITE], row[CENTR]])
    if sum(np.isnan(values)) > 0:
        return np.nan
    for key in [ELITE, CENTR]:
        val = row[key]
        if np.isnan(val):
            return np.nan
        if val < threshold[key]:
            return 0
    return 1

