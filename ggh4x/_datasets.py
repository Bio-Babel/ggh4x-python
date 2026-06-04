"""Dataset loaders for tutorials/tests (R source: base ``datasets`` + ggplot2 datasets).

ggh4x bundles no data; its vignettes use standard R datasets. This module loads the
base-R datasets (bundled as CSVs in ``ggh4x/resources/``) and re-exports the ggplot2
datasets from ``ggplot2_py`` so tutorial/validation code is self-contained and R-faithful.
"""

from __future__ import annotations

from importlib import resources

import numpy as np
import pandas as pd

__all__ = [
    "load_iris",
    "load_mtcars",
    "load_faithful",
    "load_pressure",
    "load_volcano",
    "mpg",
    "diamonds",
    "economics",
]


def _resource_path(filename: str):
    return resources.files("ggh4x.resources").joinpath(filename)


def load_iris() -> pd.DataFrame:
    """Load the ``iris`` dataset (150x5), mirroring base R.

    Returns
    -------
    pandas.DataFrame
        Columns ``Sepal.Length``, ``Sepal.Width``, ``Petal.Length``, ``Petal.Width``,
        ``Species`` (``Species`` as a category).
    """
    with resources.as_file(_resource_path("iris.csv")) as p:
        df = pd.read_csv(p)
    df["Species"] = pd.Categorical(df["Species"])
    return df


def load_mtcars() -> pd.DataFrame:
    """Load the ``mtcars`` dataset (32x11), mirroring base R.

    Returns
    -------
    pandas.DataFrame
        Model name is the index (matching R rownames); 11 numeric columns.
    """
    with resources.as_file(_resource_path("mtcars.csv")) as p:
        df = pd.read_csv(p, index_col=0)
    df.index.name = "model"
    return df


def load_faithful() -> pd.DataFrame:
    """Load the ``faithful`` dataset (272x2), mirroring base R.

    Returns
    -------
    pandas.DataFrame
        Columns ``eruptions`` and ``waiting``.
    """
    with resources.as_file(_resource_path("faithful.csv")) as p:
        return pd.read_csv(p)


def load_pressure() -> pd.DataFrame:
    """Load the ``pressure`` dataset (19x2), mirroring base R.

    Returns
    -------
    pandas.DataFrame
        Columns ``temperature`` and ``pressure``.
    """
    with resources.as_file(_resource_path("pressure.csv")) as p:
        return pd.read_csv(p)


def load_volcano() -> np.ndarray:
    """Load the ``volcano`` matrix (87x61), mirroring base R.

    Returns
    -------
    numpy.ndarray
        Topographic heights as a float array.
    """
    with resources.as_file(_resource_path("volcano.csv")) as p:
        return pd.read_csv(p, header=None).to_numpy(dtype=float)


def _ggplot2_dataset(name: str) -> pd.DataFrame:
    from ggplot2_py import datasets as _ds

    return getattr(_ds, name)


# ggplot2 datasets re-exported from ggplot2_py (loaded lazily on attribute access).
def __getattr__(name: str):  # pragma: no cover - thin re-export
    if name in ("mpg", "diamonds", "economics"):
        return _ggplot2_dataset(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
