import math

import pandas as pd

from phenotyping.recovery import recovery_index
from phenotyping.timeseries import day_from_name


def plant(days, areas):
    return pd.DataFrame({"day": days, "area": areas})


def test_linear_recovery_averages_one_half():
    # Loses 6 cm2, then wins it back at a constant rate over 7 days.
    assert math.isclose(recovery_index(plant([-1, 0, 7], [10, 4, 10])), 0.5)


def test_irregular_days_use_the_trapezoid_rule():
    # r = 0, 1, 1 on days 0, 1, 7: area = 0.5 + 6 = 6.5 over 7 days.
    assert math.isclose(recovery_index(plant([-1, 0, 1, 7], [10, 4, 10, 10])), 6.5 / 7)


def test_unordered_rows_give_the_same_result():
    assert math.isclose(recovery_index(plant([7, -1, 1, 0], [10, 10, 10, 4])), 6.5 / 7)


def test_undefined_when_there_is_no_loss_or_no_follow_up():
    assert math.isnan(recovery_index(plant([-1, 0, 7], [4, 10, 12])))
    assert math.isnan(recovery_index(plant([-1, 0], [10, 4])))
    assert math.isnan(recovery_index(plant([0, 7], [4, 10])))


def test_day_from_name():
    assert day_from_name("cm13_0") == -1.0
    assert day_from_name("cm13_0d") == 0.0
    assert day_from_name("cm13_14d") == 14.0
