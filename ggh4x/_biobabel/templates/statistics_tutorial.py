"""End-to-end template ported from tutorials/Statistics.ipynb.

Builds tiny synthetic data and exercises theoretical density fits, rolling
kernels, signed differences, x/y summary stats, and run-length encoding.
Uses only small in-memory arrays/DataFrames seeded with a fixed RNG state
(illustrative only, not seed-identical to R) — no network access, no GUI,
no large datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ggplot2_py import ggplot, aes, geom_point

from ggh4x import (
    stat_theodensity,
    stat_rollingkernel,
    stat_difference,
    stat_funxy,
    stat_centroid,
    stat_midpoint,
    stat_rle,
)


def build_faithful() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    eruptions = np.concatenate([rng.normal(2, 0.3, 20), rng.normal(4.5, 0.4, 20)])
    waiting = np.concatenate([rng.normal(55, 5, 20), rng.normal(80, 5, 20)])
    group = np.where(eruptions > 3, "High", "Low")
    return pd.DataFrame({"eruptions": eruptions, "waiting": waiting, "group": group})


def main() -> None:
    faithful = build_faithful()

    # Theoretical density fit, split by group.
    ggplot(faithful, aes(x="eruptions", color="group")) + stat_theodensity(distri="gamma")

    # Model-free rolling-kernel trend.
    ggplot(faithful, aes(x="waiting", y="eruptions")) + geom_point() + stat_rollingkernel()

    # Signed-difference ribbon on a small hand-built example.
    dummy = pd.DataFrame({"x": [1, 2, 3, 4], "ymin": [1, 2, 1, 3], "ymax": [2, 1, 3, 1]})
    ggplot(dummy, aes(x="x", ymin="ymin", ymax="ymax")) + stat_difference()

    # Custom per-group x/y summary: draw each group's x-range at its mean y.
    ggplot(faithful, aes(x="waiting", y="eruptions", group="group")) + geom_point() + stat_funxy(
        aes(color="group"),
        funx=lambda v: [float(np.min(v)), float(np.max(v))],
        funy=np.mean,
        geom="line",
    )

    # Centroid vs range-midpoint labels layered on the same plot.
    (
        ggplot(faithful, aes(x="waiting", y="eruptions", group="group"))
        + geom_point()
        + stat_centroid(geom="point", color="dodgerblue", size=4)
        + stat_midpoint(geom="point", color="limegreen", size=4)
    )

    # Run-length-encoded interval rectangles.
    rdf = pd.DataFrame({"x": np.linspace(0, 10, 20)})
    rdf["y"] = np.cos(rdf["x"])
    rdf["lab"] = pd.cut(rdf["y"], 3).astype(str)
    ggplot(rdf, aes(x="x", y="y")) + stat_rle(aes(label="lab"), align="center") + geom_point()


if __name__ == "__main__":
    main()
