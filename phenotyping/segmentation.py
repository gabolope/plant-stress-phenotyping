"""Segment a tray photo with PlantCV and measure every plant in it."""
import warnings
from pathlib import Path

import pandas as pd
from plantcv import plantcv as pcv

LINEAR_TRAITS = ["perimeter", "width", "height", "longest_path", "ellipse_major_axis", "ellipse_minor_axis"]
AREA_TRAITS = ["area", "convex_hull_area"]


def measure_tray(
    image: Path,
    rows: int,
    cols: int,
    px_per_cm: float,
    first_plant: int = 1,
    channel: str = "a",
    threshold: int = 125,
    debug_dir: Path | None = None,
) -> pd.DataFrame:
    """Size and colour traits for each pot of a `rows` x `cols` tray, in cm and cm2.

    `channel` is the LAB channel that best separates leaves from soil (green-magenta 'a'
    by default) and `threshold` its cut-off; plants are the dark side of it. Pick both
    once per camera setup with `pcv.visualize.colorspaces` and a histogram of the channel.
    `px_per_cm` comes from measuring a pot of known diameter in the same photo.
    Pass `debug_dir` to save every intermediate image (mask, grid, labels).
    """
    pcv.params.debug = "print" if debug_dir else None
    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        pcv.params.debug_outdir = str(debug_dir)
    pcv.params.text_size = 10
    pcv.params.text_thickness = 20
    n_pots = rows * cols

    try:
        img, _, _ = pcv.readimage(str(image))
        gray = pcv.rgb2gray_lab(rgb_img=img, channel=channel)
        mask = pcv.threshold.binary(gray_img=gray, threshold=threshold, object_type="dark")
        mask = pcv.fill(bin_img=mask, size=200)  # drop specks of moss and soil
        rois = pcv.roi.auto_grid(mask=mask, nrows=rows, ncols=cols, img=img)
        labels, n_found = pcv.create_labels(mask=mask, rois=rois, roi_type="partial")
        if n_found != n_pots:
            warnings.warn(f"{image}: expected {n_pots} plants, found {n_found}")
        pcv.analyze.size(img=img, labeled_mask=labels, n_labels=n_pots)
        pcv.analyze.color(rgb_img=img, labeled_mask=labels, n_labels=n_pots, colorspaces="HSV")
        observations = pcv.outputs.observations
    finally:
        pcv.outputs.clear()

    traits = pd.DataFrame(
        {sample: {name: obs["value"] for name, obs in measured.items()} for sample, measured in observations.items()}
    ).T
    traits.index = pd.RangeIndex(first_plant, first_plant + len(traits), name="plant_id")
    for col in LINEAR_TRAITS:
        traits[col] = pd.to_numeric(traits[col]) / px_per_cm
    for col in AREA_TRAITS:
        traits[col] = pd.to_numeric(traits[col]) / px_per_cm**2
    return traits
