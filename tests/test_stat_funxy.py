"""R-parity tests for :mod:`ggh4x.stat_funxy`.

Reference values were produced by running the R package ``ggh4x`` live::

    library(ggh4x); library(ggplot2)
    p <- ggplot(iris, aes(Sepal.Width, Sepal.Length, colour = Species)) +
           geom_point() + stat_centroid()
    layer_data(p, 2)

The centroid / midpoint / quantile numbers below are copied verbatim from
that R session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import importlib

# Fetch the submodule explicitly: ``ggh4x.stat_funxy`` (attribute) resolves to the
# public ``stat_funxy`` *function* once the package __init__ exports it, so we go
# through importlib to reach the module object and its internal helpers.
m = importlib.import_module("ggh4x.stat_funxy")
from ggh4x._datasets import load_iris

from ggplot2_py import aes, geom_point, get_layer_data, ggplot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def iris() -> pd.DataFrame:
    return load_iris()


@pytest.fixture(scope="module")
def setosa_group(iris: pd.DataFrame) -> pd.DataFrame:
    """A single-group frame mapped to ``x``/``y`` plus extra columns."""
    sub = iris[iris["Species"] == "setosa"]
    return pd.DataFrame(
        {
            "x": sub["Sepal.Width"].to_numpy(),
            "y": sub["Sepal.Length"].to_numpy(),
            "xend": sub["Sepal.Width"].to_numpy(),
            "yend": sub["Sepal.Length"].to_numpy(),
            "group": 1,
            "colour": "#F8766D",
        }
    )


# R reference: layer_data(ggplot(iris,...) + stat_centroid(), 2)
R_CENTROID = {
    1: (3.428, 5.006, "#F8766D"),
    2: (2.770, 5.936, "#00BA38"),
    3: (2.974, 6.588, "#619CFF"),
}
# R reference: layer_data(... + stat_midpoint(), 2)
R_MIDPOINT = {
    1: (3.35, 5.05),
    2: (2.70, 5.95),
    3: (3.00, 6.40),
}


# ---------------------------------------------------------------------------
# compute_group: centroid
# ---------------------------------------------------------------------------
def test_compute_group_centroid_setosa(setosa_group: pd.DataFrame) -> None:
    st = m.StatFunxy()
    out = st.compute_group(
        setosa_group.drop(columns=["xend", "yend"]),
        None,
        funx=m._mean,
        funy=m._mean,
        argx={"na_rm": True},
        argy={"na_rm": True},
    )
    assert len(out) == 1
    assert out["x"].iloc[0] == pytest.approx(3.428)
    assert out["y"].iloc[0] == pytest.approx(5.006)
    # constant columns are recycled, not dropped
    assert out["group"].iloc[0] == 1
    assert out["colour"].iloc[0] == "#F8766D"


def test_compute_group_midpoint_setosa(setosa_group: pd.DataFrame) -> None:
    st = m.StatFunxy()
    out = st.compute_group(
        setosa_group.drop(columns=["xend", "yend"]),
        None,
        funx=m._midpoint,
        funy=m._midpoint,
        argx={"na_rm": True},
        argy={"na_rm": True},
    )
    assert out["x"].iloc[0] == pytest.approx(3.35)
    assert out["y"].iloc[0] == pytest.approx(5.05)


def test_compute_group_quantile_recycle(setosa_group: pd.DataFrame) -> None:
    """funx -> length 1, funy -> length 2: x recycled, others cropped."""
    st = m.StatFunxy()
    out = st.compute_group(
        setosa_group.drop(columns=["xend", "yend"]),
        None,
        funx=np.median,
        funy=np.quantile,
        argx={},
        argy={"q": [0.1, 0.9]},
    )
    assert len(out) == 2
    # x (median 3.4) recycled to length 2
    np.testing.assert_allclose(out["x"].to_numpy(), [3.4, 3.4])
    # y is the 10th/90th percentile (R type 7 == numpy linear)
    np.testing.assert_allclose(out["y"].to_numpy(), [4.59, 5.41])
    # other columns cropped to size = 2 then recycled
    assert list(out["group"]) == [1, 1]


def test_compute_group_crop_other_false(setosa_group: pd.DataFrame) -> None:
    """crop_other=False keeps full-length others; x/y (len 1) recycled."""
    st = m.StatFunxy()
    out = st.compute_group(
        setosa_group,
        None,
        funx=m._mean,
        funy=m._mean,
        argx={"na_rm": True},
        argy={"na_rm": True},
        crop_other=False,
    )
    # 50 setosa rows preserved
    assert len(out) == 50
    # x/y recycled to the centroid value on every row
    np.testing.assert_allclose(out["x"].to_numpy(), np.full(50, 3.428))
    np.testing.assert_allclose(out["y"].to_numpy(), np.full(50, 5.006))
    # xend/yend keep the original per-row values (R: 3.5, 5.1 first row)
    assert out["xend"].iloc[0] == pytest.approx(3.5)
    assert out["yend"].iloc[0] == pytest.approx(5.1)
    # R column order: other..., then x, y
    assert list(out.columns)[-2:] == ["x", "y"]
    assert "xend" in list(out.columns)[:-2]


def test_compute_group_identity_passthrough(setosa_group: pd.DataFrame) -> None:
    """Default funx/funy = identity leaves x/y unchanged (order-preserving)."""
    st = m.StatFunxy()
    g = setosa_group.drop(columns=["xend", "yend"])
    out = st.compute_group(g, None)
    assert len(out) == 50
    np.testing.assert_allclose(out["x"].to_numpy(), g["x"].to_numpy())
    np.testing.assert_allclose(out["y"].to_numpy(), g["y"].to_numpy())


# ---------------------------------------------------------------------------
# Full layer pipeline parity
# ---------------------------------------------------------------------------
def test_layer_centroid_all_groups(iris: pd.DataFrame) -> None:
    p = (
        ggplot(iris, aes("Sepal.Width", "Sepal.Length", colour="Species"))
        + geom_point()
        + m.stat_centroid()
    )
    d = get_layer_data(p, 2).sort_values("group").reset_index(drop=True)
    assert len(d) == 3
    for i, (_, row) in enumerate(d.iterrows(), start=1):
        ex, ey, ecol = R_CENTROID[i]
        assert row["x"] == pytest.approx(ex)
        assert row["y"] == pytest.approx(ey)
        assert row["group"] == i
        assert row["colour"] == ecol


def test_layer_midpoint_all_groups(iris: pd.DataFrame) -> None:
    p = (
        ggplot(iris, aes("Sepal.Width", "Sepal.Length", colour="Species"))
        + geom_point()
        + m.stat_midpoint()
    )
    d = get_layer_data(p, 2).sort_values("group").reset_index(drop=True)
    assert len(d) == 3
    for i, (_, row) in enumerate(d.iterrows(), start=1):
        ex, ey = R_MIDPOINT[i]
        assert row["x"] == pytest.approx(ex)
        assert row["y"] == pytest.approx(ey)


def test_layer_centroid_r_parity_correlation(iris: pd.DataFrame) -> None:
    """Pearson r >= 0.99 against the R centroid coordinates (Tier 1)."""
    p = (
        ggplot(iris, aes("Sepal.Width", "Sepal.Length", colour="Species"))
        + geom_point()
        + m.stat_centroid()
    )
    d = get_layer_data(p, 2).sort_values("group").reset_index(drop=True)
    py = np.concatenate([d["x"].to_numpy(), d["y"].to_numpy()])
    r_vals = np.array(
        [R_CENTROID[i][0] for i in (1, 2, 3)]
        + [R_CENTROID[i][1] for i in (1, 2, 3)]
    )
    r = np.corrcoef(py, r_vals)[0, 1]
    assert r >= 0.99


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------
def test_constructor_funx_not_callable() -> None:
    with pytest.raises(ValueError, match="must be a function"):
        m.stat_funxy(funx=5)


def test_constructor_funy_not_callable() -> None:
    with pytest.raises(ValueError, match="must be a function"):
        m.stat_funxy(funy="not a fun")


def test_constructor_argx_not_dict() -> None:
    with pytest.raises(ValueError, match="must be lists"):
        m.stat_funxy(argx=[1, 2])


def test_constructor_argx_unnamed() -> None:
    with pytest.raises(ValueError, match="named elements"):
        m.stat_funxy(argx={"": 1})


def test_constructor_argy_unnamed() -> None:
    with pytest.raises(ValueError, match="named elements"):
        m.stat_funxy(argy={"": 1})


def test_constructor_returns_layer() -> None:
    from ggplot2_py import Layer

    assert isinstance(m.stat_funxy(), Layer)
    assert isinstance(m.stat_centroid(), Layer)
    assert isinstance(m.stat_midpoint(), Layer)


def test_stat_required_aes() -> None:
    assert m.StatFunxy.required_aes == ["x", "y"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_na_aware_mean() -> None:
    """na_rm=True ignores NaN (mirrors R mean(na.rm=TRUE))."""
    x = np.array([1.0, 2.0, np.nan, 4.0])
    assert m._mean(x, na_rm=True) == pytest.approx(7.0 / 3.0)
    assert np.isnan(m._mean(x, na_rm=False))


def test_na_aware_midpoint() -> None:
    x = np.array([1.0, np.nan, 9.0])
    assert m._midpoint(x, na_rm=True) == pytest.approx(5.0)
    assert np.isnan(m._midpoint(x, na_rm=False))


def test_compute_group_centroid_with_nan(setosa_group: pd.DataFrame) -> None:
    """NaN in x is ignored when argx has na_rm=True."""
    g = setosa_group.drop(columns=["xend", "yend"]).copy()
    g.loc[g.index[0], "x"] = np.nan
    st = m.StatFunxy()
    out = st.compute_group(
        g, None, funx=m._mean, funy=m._mean,
        argx={"na_rm": True}, argy={"na_rm": True},
    )
    expected = np.nanmean(g["x"].to_numpy())
    assert out["x"].iloc[0] == pytest.approx(expected)


def test_compute_group_single_row() -> None:
    """A single-row group: mean is that row; no crash."""
    g = pd.DataFrame({"x": [3.0], "y": [5.0], "group": [1]})
    st = m.StatFunxy()
    out = st.compute_group(
        g, None, funx=m._mean, funy=m._mean,
        argx={"na_rm": True}, argy={"na_rm": True},
    )
    assert len(out) == 1
    assert out["x"].iloc[0] == pytest.approx(3.0)
    assert out["y"].iloc[0] == pytest.approx(5.0)
