"""Turn the per-photo PlantCV tables of one tray into a single tidy time series."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

PRE_STRESS_DAY = -1.0  # the photo taken right before the heat shock


def day_from_name(stem: str) -> float:
    """'cm13_0' is the pre-stress photo; 'cm13_7d' was taken 7 days after the stress."""
    if stem.endswith("_0"):
        return PRE_STRESS_DAY
    match = re.search(r"_(\d+)d$", stem)
    if match is None:
        raise ValueError(f"Unrecognised photo name: {stem}")
    return float(match.group(1))


def load_experiment(
    folder: Path,
    first_plant: int,
    first_mutant: int,
    controls: list[int] = (),
    traits: tuple[str, ...] = ("area", "solidity"),
) -> pd.DataFrame:
    """One row per plant and photo, with day, genotype and treatment.

    Plants are numbered by their position in the tray, starting at `first_plant`;
    every plant from `first_mutant` on is SWAP A, the rest are wild type.
    """
    frames = []
    for csv in sorted(Path(folder).glob("*.csv")):
        photo = pd.read_csv(csv, usecols=list(traits))
        photo.insert(0, "plant", range(first_plant, first_plant + len(photo)))
        photo.insert(1, "day", day_from_name(csv.stem))
        frames.append(photo)
    data = pd.concat(frames, ignore_index=True)
    data["genotype"] = np.where(data["plant"] >= first_mutant, "SWAP A", "WT")
    data["treatment"] = np.where(data["plant"].isin(controls), "control", "heat")
    return data.sort_values(["plant", "day"], ignore_index=True)
