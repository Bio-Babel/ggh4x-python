"""Slice 0 foundation tests: _vctrs, _rlang, _borrowed_ggplot2, _utils, _datasets.

R-parity values were captured from a live ggrepel-dev R session (vctrs + ggh4x internals).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggh4x import _borrowed_ggplot2 as bg
from ggh4x import _datasets, _rlang, _utils, _vctrs


# --- _borrowed_ggplot2.id / id_var (facet panel ordering) --------------------
def test_id_two_vars_radix_order():
    # expand.grid(year, drv) with year fastest; R id() -> 1,4,2,5,3,6 (n=6)
    df = pd.DataFrame(
        {"year": [1999, 2008, 1999, 2008, 1999, 2008], "drv": ["4", "4", "f", "f", "r", "r"]}
    )
    out = bg.id(df)
    assert list(out) == [1, 4, 2, 5, 3, 6]
    assert out.n == 6


def test_id_var_factor_sorted_levels():
    out = bg.id_var(pd.Categorical(["b", "a", "a", "c"]))
    assert list(out) == [2, 1, 1, 3]
    assert out.n == 3


def test_id_var_non_factor_sorts_na_last():
    out = bg.id_var([3.0, 1.0, np.nan, 1.0])
    # sorted unique = [1,3], NA last -> levels [1,3,NA]; ids: 3->2,1->1,NA->3,1->1
    assert list(out) == [2, 1, 3, 1]
    assert out.n == 3


def test_id_single_var_delegates_to_id_var():
    out = bg.id(pd.DataFrame({"x": ["b", "a", "a"]}))
    assert list(out) == [2, 1, 1]


def test_id_empty_vars_returns_seq():
    out = bg.id(pd.DataFrame(index=range(4)))
    assert list(out) == [1, 2, 3, 4]
    assert out.n == 4


def test_snake_class():
    assert bg.snake_class("FacetGrid2") == "facet_grid2"
    assert bg.snake_class("StripNested") == "strip_nested"
    assert bg.snake_class("CoordAxesInside") == "coord_axes_inside"


def test_empty_and_is_zero():
    assert bg.empty(None)
    assert bg.empty(pd.DataFrame())
    assert not bg.empty(pd.DataFrame({"a": [1]}))
    assert bg.is_zero(None)


def test_ulevels_factor_and_numeric():
    assert list(bg.ulevels(pd.Categorical(["b", "a"], categories=["a", "b", "c"]))) == ["a", "b", "c"]
    assert list(bg.ulevels([3, 1, 2, 1])) == [1, 2, 3]


# --- _vctrs ------------------------------------------------------------------
def test_vec_interleave():
    assert list(_vctrs.vec_interleave([1, 2, 3], [4, 5, 6])) == [1, 4, 2, 5, 3, 6]


def test_vec_unrep_runs():
    out = _vctrs.vec_unrep([1, 1, 2, 3, 3, 3])
    assert list(out["key"]) == [1, 2, 3]
    assert list(out["times"]) == [2, 1, 3]


def test_vec_unrep_empty():
    out = _vctrs.vec_unrep([])
    assert len(out) == 0


def test_vec_rep_each():
    assert list(_vctrs.vec_rep_each(["a", "b"], [2, 3])) == ["a", "a", "b", "b", "b"]


def test_vec_match():
    # 0-based; absent -> -1
    assert list(_vctrs.vec_match(["b", "z", "a"], ["a", "b", "c"])) == [1, -1, 0]


def test_vec_unique_and_count():
    assert list(_vctrs.vec_unique([3, 1, 3, 2, 1])) == [3, 1, 2]
    assert _vctrs.vec_unique_count([3, 1, 3, 2, 1]) == 3


def test_vec_rbind_fills_missing():
    a = pd.DataFrame({"x": [1], "y": [2]})
    b = pd.DataFrame({"x": [3], "z": [4]})
    out = _vctrs.vec_rbind(a, b)
    assert list(out["x"]) == [1, 3]
    assert pd.isna(out.loc[0, "z"]) and out.loc[1, "z"] == 4


def test_vec_recycle_common():
    out = _vctrs.vec_recycle_common([1], [1, 2, 3])
    assert list(out[0]) == [1, 1, 1]
    with pytest.raises(ValueError):
        _vctrs.vec_recycle_common([1, 2], [1, 2, 3])


def test_vec_group_loc():
    out = _vctrs.vec_group_loc(["a", "b", "a", "a"])
    assert list(out["key"]) == ["a", "b"]
    assert list(out.loc[0, "loc"]) == [0, 2, 3]


# --- _rlang ------------------------------------------------------------------
def test_arg_match0_ok():
    assert _rlang.arg_match0("y", ["x", "y", "z"]) == "y"


def test_arg_match0_rejects():
    with pytest.raises(ValueError, match="must be one of"):
        _rlang.arg_match0("q", ["x", "y"])


def test_value_or():
    assert _rlang.value_or(None, 5) == 5
    assert _rlang.value_or(3, 5) == 3


# --- _datasets ---------------------------------------------------------------
def test_dataset_shapes_match_r():
    assert _datasets.load_iris().shape == (150, 5)
    assert _datasets.load_mtcars().shape == (32, 11)
    assert _datasets.load_faithful().shape == (272, 2)
    assert _datasets.load_pressure().shape == (19, 2)
    assert _datasets.load_volcano().shape == (87, 61)


def test_faithful_columns_and_means():
    df = _datasets.load_faithful()
    assert list(df.columns) == ["eruptions", "waiting"]
    # R: mean(faithful$waiting) = 70.897059..., mean(eruptions) = 3.487783...
    assert df["waiting"].mean() == pytest.approx(70.8970588, abs=1e-6)
    assert df["eruptions"].mean() == pytest.approx(3.4877830, abs=1e-6)


def test_iris_species_categorical():
    df = _datasets.load_iris()
    assert isinstance(df["Species"].dtype, pd.CategoricalDtype)
    assert set(df["Species"].cat.categories) == {"setosa", "versicolor", "virginica"}


def test_ggplot2_datasets_reexport():
    assert _datasets.mpg.shape == (234, 11)
    assert _datasets.economics.shape[1] == 6


# --- _utils ------------------------------------------------------------------
def test_seq_range():
    assert list(_utils.seq_range([3, 1, 2], step=1)) == [1, 2, 3]
    assert list(_utils.seq_range([0, 10], length_out=3)) == [0.0, 5.0, 10.0]
