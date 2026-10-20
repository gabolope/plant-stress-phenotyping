"""Recovery index: how much of the stress-induced loss a plant wins back, averaged over time."""
import numpy as np
import pandas as pd

from phenotyping.timeseries import PRE_STRESS_DAY


def relative_recovery(plant: pd.DataFrame, trait: str = "area") -> pd.Series:
    """r(t) = (x(t) - x_stress) / (x_pre - x_stress) for every day >= 0.

    r = 0 means nothing recovered since the stress, r = 1 means back to the pre-stress value.
    Empty when a reference photo is missing or the plant did not lose any `trait`,
    since then there is no loss to recover from.
    """
    values = plant.set_index("day")[trait]
    if PRE_STRESS_DAY not in values.index or 0.0 not in values.index:
        return pd.Series(dtype=float)
    pre, stressed = values[PRE_STRESS_DAY], values[0.0]
    if pre <= stressed:
        return pd.Series(dtype=float)
    after = values[values.index >= 0].sort_index()
    return (after - stressed) / (pre - stressed)


def recovery_index(plant: pd.DataFrame, trait: str = "area") -> float:
    """Time-averaged r(t): its trapezoidal area divided by the length of the recovery window.

    The trapezoid rule handles the irregular photo days (0, 1, 4, 7, ...) without resampling.
    NaN when r(t) is undefined or there is no photo after day 0.
    """
    r = relative_recovery(plant, trait)
    if len(r) < 2:
        return np.nan
    return np.trapezoid(r.to_numpy(), r.index.to_numpy()) / r.index.max()
