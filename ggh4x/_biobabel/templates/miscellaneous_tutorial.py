"""End-to-end template ported from tutorials/Miscellaneous.ipynb.

Builds tiny synthetic data and exercises multiple colour scales, extra
geoms (pointpath, polygonraster, text_aimed, tilemargin), a linear-
transform position, and axes drawn inside the panel. Uses only small
in-memory arrays/DataFrames — no network access, no GUI, no large datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ggplot2_py import (
    ggplot,
    aes,
    geom_point,
    geom_raster,
    coord_fixed,
    coord_cartesian,
    facet_wrap,
    theme,
    theme_void,
    element_line,
    scale_fill_viridis_c,
)

from ggh4x import (
    scale_colour_multi,
    geom_tilemargin,
    geom_pointpath,
    geom_polygonraster,
    geom_text_aimed,
    position_lineartrans,
    coord_axes_inside,
)


def build_iris_subsets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    iris = pd.DataFrame(
        {
            "Sepal.Width": [3.5, 3.0, 3.2, 3.1, 2.8, 3.3],
            "Sepal.Length": [5.1, 4.9, 4.7, 6.4, 6.9, 6.3],
            "Petal.Length": [1.4, 1.4, 1.3, 4.5, 4.9, 6.0],
            "Petal.Width": [0.2, 0.2, 0.2, 1.5, 1.5, 2.5],
            "Species": ["setosa", "setosa", "setosa", "versicolor", "versicolor", "virginica"],
        }
    )
    setosa = iris[iris["Species"] == "setosa"]
    versi = iris[iris["Species"] == "versicolor"]
    virg = iris[iris["Species"] == "virginica"]
    return iris, setosa, versi, virg


def build_raster_grid() -> pd.DataFrame:
    xs, ys = np.meshgrid(np.arange(4), np.arange(4))
    return pd.DataFrame({"x": xs.ravel(), "y": ys.ravel(), "value": (xs + ys).ravel().astype(float)})


def main() -> None:
    iris, setosa, versi, virg = build_iris_subsets()

    g = (
        ggplot(iris, aes("Sepal.Width", "Sepal.Length"))
        + geom_point(aes(swidth="Sepal.Width"), data=setosa)
        + geom_point(aes(pleng="Petal.Length"), data=versi)
        + geom_point(aes(pwidth="Petal.Width"), data=virg)
        + facet_wrap("Species", scales="free_x")
    )
    g + scale_colour_multi(
        aesthetics=["swidth", "pleng", "pwidth"],
        colours=[["black", "green"], ["gray", "red"], ["white", "blue"]],
    )

    pressure = pd.DataFrame({"temperature": [0, 20, 40, 60, 80], "pressure": [4, 10, 24, 57, 90]})
    ggplot(pressure, aes("temperature", "pressure")) + geom_pointpath(linesize=2, size=2, mult=1)

    dfv = build_raster_grid()
    gv = ggplot(dfv, aes("x", "y", fill="value")) + scale_fill_viridis_c(guide="none") + theme_void()
    gv + geom_polygonraster(position=position_lineartrans(shear=[0.2, 0.2])) + coord_fixed()
    gv + geom_polygonraster(position=position_lineartrans(angle=45)) + coord_fixed()

    mt = pd.DataFrame(
        {
            "mpg": [21.0, 22.8, 18.7, 14.3],
            "wt": [2.6, 2.3, 3.4, 3.6],
            "car": ["Mazda RX4", "Datsun 710", "Hornet Sportabout", "Duster 360"],
            "cyl_f": ["6", "4", "8", "8"],
        }
    )
    mpg_rng = mt["mpg"].max() - mt["mpg"].min()
    wt_rng = mt["wt"].max() - mt["wt"].min()
    (
        ggplot(mt, aes("mpg", "wt"))
        + geom_point(aes(colour="cyl_f"))
        + geom_text_aimed(aes(label="car"), hjust=-0.2, size=3, xend=mpg_rng / 2, yend=wt_rng / 2)
        + coord_cartesian(clip="off")
    )

    corr_df = pd.DataFrame(
        {"x": [0, 0, 1, 1], "y": [0, 1, 0, 1], "correlation": [1.0, 0.3, 0.3, 1.0]}
    )
    iris_df = iris.copy()
    iris_df["id"] = range(len(iris_df))
    gh = (
        ggplot(iris_df, aes("id", "id"))
        + geom_tilemargin(aes(species="Species"))
        + geom_raster(aes("x", "y", cor="correlation"), data=corr_df)
        + coord_fixed()
    )

    mpg3 = pd.DataFrame({"displ": [1.6, 2.0, 3.0, 4.0], "hwy": [33, 29, 22, 18]})
    mpg3["dx"] = mpg3["displ"] - mpg3["displ"].mean()
    mpg3["dy"] = mpg3["hwy"] - mpg3["hwy"].mean()
    ggplot(mpg3, aes("dx", "dy")) + geom_point() + theme(axis_line=element_line()) + coord_axes_inside()


if __name__ == "__main__":
    main()
