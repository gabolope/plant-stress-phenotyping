"""Recovery of rosette area after heat stress: WT vs SWAP A.

Reads the PlantCV tables in data/raw, applies the per-experiment cleaning in
experiments.toml, computes the recovery index of every plant, estimates the genotype
effect and writes results/ and img/.

    python run_analysis.py
"""
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf

from phenotyping.recovery import recovery_index, relative_recovery
from phenotyping.timeseries import PRE_STRESS_DAY, load_experiment

ROOT = Path(__file__).parent
TRAIT = "area"
GENOTYPES = ["WT", "SWAP A"]
COLORS = {"WT": "#1f51a2", "SWAP A": "#d95f02"}
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e4e4e4"


def clean(data: pd.DataFrame, exp: dict) -> pd.DataFrame:
    data = data[data["treatment"] == "heat"]
    if "min_baseline_area" in exp:
        pre = data[data["day"] == PRE_STRESS_DAY]
        data = data[~data["plant"].isin(pre.loc[pre["area"] < exp["min_baseline_area"], "plant"])]
    data = data[~data["plant"].isin(exp.get("outliers", []))]
    return data[data["day"] <= exp.get("max_day", float("inf"))]


def load_all(config: dict) -> pd.DataFrame:
    frames = []
    for name, exp in config.items():
        data = load_experiment(ROOT / "data" / "raw" / name, exp["first_plant"], exp["first_mutant"], exp.get("controls", []))
        frames.append(clean(data, exp).assign(experiment=name))
    return pd.concat(frames, ignore_index=True)


def genotype_effect(index: pd.DataFrame) -> pd.Series:
    """SWAP A minus WT, with experiment as a block so tray-to-tray differences don't leak in.

    HC3 errors because group sizes and variances differ between experiments.
    """
    # ponytail: fixed block effect, fine for 4 trays; switch to smf.mixedlm with a random intercept once there are ~8+
    fit = smf.ols("recovery_index ~ C(genotype, Treatment('WT')) + C(experiment)", index).fit(cov_type="HC3")
    term = "C(genotype, Treatment('WT'))[T.SWAP A]"
    low, high = fit.conf_int().loc[term]
    return pd.Series({"estimate": fit.params[term], "ci_low": low, "ci_high": high, "p_value": fit.pvalues[term], "n_plants": int(fit.nobs)})


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_curves(series: pd.DataFrame, path: Path):
    experiments = series["experiment"].unique()
    fig, axes = plt.subplots(1, len(experiments), figsize=(3.2 * len(experiments), 3.4), sharey=True)
    for ax, (name, exp_data) in zip(axes, series.groupby("experiment")):
        curves = pd.concat(
            {key: relative_recovery(plant, TRAIT) for key, plant in exp_data.groupby(["genotype", "plant"])},
            names=["genotype", "plant", "day"],
        ).rename("r").reset_index()
        for genotype in GENOTYPES:
            g = curves[curves["genotype"] == genotype]
            for _, one in g.groupby("plant"):
                ax.plot(one["day"], one["r"], color=COLORS[genotype], alpha=0.25, linewidth=1)
            mean = g.groupby("day")["r"].mean()
            ax.plot(mean.index, mean.values, color=COLORS[genotype], linewidth=2.5, marker="o", markersize=5, label=genotype)
        ax.axhline(0, color=MUTED, linewidth=1, linestyle=":")
        ax.axhline(1, color=MUTED, linewidth=1, linestyle="--")
        ax.set_title(name, color=INK, fontsize=11)
        ax.set_xlabel("Days after heat stress", color=INK)
        style(ax)
    axes[0].set_ylabel("Relative recovery r(t)", color=INK)
    axes[0].text(0.02, 1, "full recovery", transform=axes[0].get_yaxis_transform(), va="bottom", fontsize=8, color=MUTED)
    axes[-1].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_index(index: pd.DataFrame, effect: pd.Series, path: Path):
    experiments = sorted(index["experiment"].unique())
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, name in enumerate(experiments):
        for offset, genotype in zip((-0.17, 0.17), GENOTYPES):
            values = index.loc[(index["experiment"] == name) & (index["genotype"] == genotype), "recovery_index"]
            ax.scatter([i + offset] * len(values), values, color=COLORS[genotype], alpha=0.6, s=28,
                       edgecolor="white", linewidth=1, label=genotype if i == 0 else None)
            ax.hlines(values.mean(), i + offset - 0.12, i + offset + 0.12, color=INK, linewidth=2)
    ax.axhline(0, color=MUTED, linewidth=1, linestyle=":")
    ax.set_xticks(range(len(experiments)), experiments)
    ax.set_xlabel("Experiment (one tray each)", color=INK)
    ax.set_ylabel("Area recovery index", color=INK)
    ax.set_title(
        f"SWAP A - WT = {effect.estimate:+.2f} (95% CI {effect.ci_low:+.2f} to {effect.ci_high:+.2f}, p = {effect.p_value:.2f})",
        color=INK, fontsize=10, loc="left",
    )
    ax.legend(frameon=False, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    config = tomllib.loads((ROOT / "experiments.toml").read_text())
    series = load_all(config)
    index = (
        series.groupby(["experiment", "plant", "genotype"])
        .apply(recovery_index, trait=TRAIT, include_groups=False)
        .rename("recovery_index")
        .dropna()
        .reset_index()
    )
    effect = genotype_effect(index)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "img").mkdir(exist_ok=True)
    index.to_csv(ROOT / "results" / "recovery_index.csv", index=False)
    print(index.groupby(["experiment", "genotype"])["recovery_index"].agg(["count", "mean"]).round(3).unstack(), "\n")
    print(effect.round(3).to_string())

    plot_curves(series, ROOT / "img" / "recovery_curves.png")
    plot_index(index, effect, ROOT / "img" / "recovery_index.png")


if __name__ == "__main__":
    main()
