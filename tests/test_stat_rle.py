"""R-parity tests for ``ggh4x.stat_rle`` (StatRle / stat_rle).

All expected values were produced by running the R ``ggh4x`` package live
(``StatRle$compute_group`` and the full ``layer_data`` build) and pasted here as
fixtures.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import ggplot2_py as gg
from ggplot2_py import aes, ggplot

from ggh4x.stat_rle import StatRle, _vec_unrep, stat_rle


# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------


def _sin_df():
    """The canonical example data from the R docs (20-point sine)."""
    x = np.linspace(0, 10, 20)
    return pd.DataFrame({"x": x, "label": np.sin(x * 2) > 0})


# R: layer_data() of stat_rle on the 20-point sine, per alignment.
# Columns: start, end, start_id, end_id, run_id, runlength, runvalue
_SIN_NONE = {
    "start": [0.0, 0.5263158, 1.5789474, 3.1578947, 4.7368421, 6.3157895, 7.8947368, 9.4736842],
    "end": [0.0, 1.052632, 2.631579, 4.210526, 5.789474, 7.368421, 8.947368, 10.0],
    "start_id": [1, 2, 4, 7, 10, 13, 16, 19],
    "end_id": [1, 3, 6, 9, 12, 15, 18, 20],
    "run_id": [1, 2, 3, 4, 5, 6, 7, 8],
    "runlength": [1, 2, 3, 3, 3, 3, 3, 2],
    "runvalue": [False, True, False, True, False, True, False, True],
}
_SIN_CENTRE = {
    "start": [0.0, 0.2631579, 1.3157895, 2.8947368, 4.4736842, 6.0526316, 7.6315789, 9.2105263],
    "end": [0.2631579, 1.3157895, 2.8947368, 4.4736842, 6.0526316, 7.6315789, 9.2105263, 10.0],
    "start_id": [1, 2, 4, 7, 10, 13, 16, 19],
    "end_id": [1, 3, 6, 9, 12, 15, 18, 20],
}
_SIN_START = {
    "start": [0.0, 0.5263158, 1.5789474, 3.1578947, 4.7368421, 6.3157895, 7.8947368, 9.4736842],
    "end": [0.5263158, 1.5789474, 3.1578947, 4.7368421, 6.3157895, 7.8947368, 9.4736842, 10.0],
}
_SIN_END = {
    "start": [0.0, 0.0, 1.052632, 2.631579, 4.210526, 5.789474, 7.368421, 8.947368],
    "end": [0.0, 1.052632, 2.631579, 4.210526, 5.789474, 7.368421, 8.947368, 10.0],
}


# --------------------------------------------------------------------------
# _vec_unrep (NA-as-equal)
# --------------------------------------------------------------------------


def test_vec_unrep_basic_runs():
    res = _vec_unrep([True, True, False, True, True, True])
    assert list(res["key"]) == [True, False, True]
    assert list(res["times"]) == [2, 1, 3]


def test_vec_unrep_na_as_equal():
    # Two consecutive NA collapse into a single run (unlike base::rle).
    res = _vec_unrep([True, True, False, np.nan, np.nan, False, False])
    assert list(res["times"]) == [2, 1, 2, 2]
    keys = list(res["key"])
    assert keys[0] is True or keys[0] == True  # noqa: E712
    assert keys[1] == False  # noqa: E712
    assert pd.isna(keys[2])
    assert keys[3] == False  # noqa: E712


def test_vec_unrep_all_na_single_run():
    res = _vec_unrep([np.nan, np.nan, np.nan])
    assert len(res) == 1
    assert list(res["times"]) == [3]
    assert pd.isna(res["key"].iloc[0])


def test_vec_unrep_na_between_equal_values_splits():
    # R: vec_unrep(c(1, NA, 1)) -> three runs (the NA separates the 1s).
    res = _vec_unrep([1.0, np.nan, 1.0])
    assert list(res["times"]) == [1, 1, 1]
    assert res["key"].iloc[0] == 1.0
    assert pd.isna(res["key"].iloc[1])
    assert res["key"].iloc[2] == 1.0


def test_vec_unrep_empty():
    res = _vec_unrep([])
    assert len(res) == 0
    assert list(res.columns) == ["key", "times"]


def test_vec_unrep_none_raises():
    with pytest.raises(ValueError):
        _vec_unrep(None)


def test_vec_unrep_preserves_categorical():
    cat = pd.Categorical(["lo", "lo", "hi", "hi"], categories=["lo", "mid", "hi"])
    res = _vec_unrep(pd.Series(cat))
    assert list(res["key"]) == ["lo", "hi"]
    assert list(res["times"]) == [2, 2]
    # Category levels preserved on the key column.
    assert list(res["key"].cat.categories) == ["lo", "mid", "hi"]


# --------------------------------------------------------------------------
# compute_group — all four alignments vs R
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "align,expected",
    [("none", _SIN_NONE), ("centre", _SIN_CENTRE), ("start", _SIN_START), ("end", _SIN_END)],
)
def test_compute_group_alignments(align, expected):
    st = StatRle()
    res = st.compute_group(_sin_df(), None, align=align)
    for col, exp in expected.items():
        if col in ("start", "end"):
            np.testing.assert_allclose(res[col].to_numpy(), exp, rtol=1e-6, atol=1e-9)
        else:
            assert list(res[col]) == exp


def test_compute_group_returns_exact_columns():
    st = StatRle()
    res = st.compute_group(_sin_df(), None, align="none")
    assert list(res.columns) == [
        "start",
        "end",
        "start_id",
        "end_id",
        "run_id",
        "runlength",
        "runvalue",
    ]


def test_compute_group_id_columns_are_integer():
    st = StatRle()
    res = st.compute_group(_sin_df(), None, align="none")
    for col in ("start_id", "end_id", "run_id", "runlength"):
        assert pd.api.types.is_integer_dtype(res[col]), col


# --------------------------------------------------------------------------
# compute_group — ordering + NA, edge cases vs R
# --------------------------------------------------------------------------


def test_compute_group_sorts_by_x_and_na_as_equal_none():
    # R fixture: UNSORTED_NA_NONE
    data = pd.DataFrame(
        {"x": [5, 1, 2, 4, 3, 6, 7], "label": ["a", "a", np.nan, np.nan, "b", "b", "b"]}
    )
    res = StatRle().compute_group(data, None, align="none")
    np.testing.assert_allclose(res["start"].to_numpy(), [1, 2, 3, 4, 5, 6], atol=1e-9)
    np.testing.assert_allclose(res["end"].to_numpy(), [1, 2, 3, 4, 5, 7], atol=1e-9)
    assert list(res["start_id"]) == [1, 2, 3, 4, 5, 6]
    assert list(res["end_id"]) == [1, 2, 3, 4, 5, 7]
    assert list(res["run_id"]) == [1, 2, 3, 4, 5, 6]
    assert list(res["runlength"]) == [1, 1, 1, 1, 1, 2]
    rv = list(res["runvalue"])
    assert rv[0] == "a" and pd.isna(rv[1]) and rv[2] == "b"
    assert pd.isna(rv[3]) and rv[4] == "a" and rv[5] == "b"


def test_compute_group_sorts_by_x_and_na_as_equal_centre():
    # R fixture: UNSORTED_NA_CENTRE
    data = pd.DataFrame(
        {"x": [5, 1, 2, 4, 3, 6, 7], "label": ["a", "a", np.nan, np.nan, "b", "b", "b"]}
    )
    res = StatRle().compute_group(data, None, align="centre")
    np.testing.assert_allclose(
        res["start"].to_numpy(), [1, 1.5, 2.5, 3.5, 4.5, 5.5], atol=1e-9
    )
    np.testing.assert_allclose(
        res["end"].to_numpy(), [1.5, 2.5, 3.5, 4.5, 5.5, 7], atol=1e-9
    )


def test_compute_group_single_point():
    res = StatRle().compute_group(pd.DataFrame({"x": [1], "label": ["z"]}), None, align="none")
    assert len(res) == 1
    assert res["start"].iloc[0] == 1 and res["end"].iloc[0] == 1
    assert res["start_id"].iloc[0] == 1 and res["end_id"].iloc[0] == 1
    assert res["run_id"].iloc[0] == 1 and res["runlength"].iloc[0] == 1
    assert res["runvalue"].iloc[0] == "z"


def test_compute_group_all_same_value():
    # R fixture: ALLSAME
    res = StatRle().compute_group(
        pd.DataFrame({"x": [1, 2, 3, 4, 5], "label": ["z"] * 5}), None, align="none"
    )
    assert len(res) == 1
    assert res["start"].iloc[0] == 1 and res["end"].iloc[0] == 5
    assert res["start_id"].iloc[0] == 1 and res["end_id"].iloc[0] == 5
    assert res["runlength"].iloc[0] == 5 and res["runvalue"].iloc[0] == "z"


def test_compute_group_all_na_single_run():
    res = StatRle().compute_group(
        pd.DataFrame({"x": [1, 2, 3], "label": [np.nan, np.nan, np.nan]}), None, align="none"
    )
    assert len(res) == 1
    assert res["start"].iloc[0] == 1 and res["end"].iloc[0] == 3
    assert res["runlength"].iloc[0] == 3
    assert pd.isna(res["runvalue"].iloc[0])


def test_compute_group_three_runs_start_align():
    # R fixture: THREERUN_START
    data = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "label": ["a", "a", "b", "b", "b", "c"]})
    res = StatRle().compute_group(data, None, align="start")
    np.testing.assert_allclose(res["start"].to_numpy(), [1, 3, 6], atol=1e-9)
    np.testing.assert_allclose(res["end"].to_numpy(), [3, 6, 6], atol=1e-9)
    assert list(res["start_id"]) == [1, 3, 6]
    assert list(res["end_id"]) == [2, 5, 6]
    assert list(res["runlength"]) == [2, 3, 1]


def test_compute_group_three_runs_end_align():
    # R fixture: THREERUN_END
    data = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "label": ["a", "a", "b", "b", "b", "c"]})
    res = StatRle().compute_group(data, None, align="end")
    np.testing.assert_allclose(res["start"].to_numpy(), [1, 2, 5], atol=1e-9)
    np.testing.assert_allclose(res["end"].to_numpy(), [2, 5, 6], atol=1e-9)


def test_compute_group_preserves_factor_runvalue_levels():
    cat = pd.Categorical(["lo", "lo", "hi", "hi"], categories=["lo", "mid", "hi"])
    data = pd.DataFrame({"x": [1, 2, 3, 4], "label": pd.Series(cat)})
    res = StatRle().compute_group(data, None, align="none")
    assert list(res["runvalue"]) == ["lo", "hi"]
    assert isinstance(res["runvalue"].dtype, pd.CategoricalDtype)
    assert list(res["runvalue"].cat.categories) == ["lo", "mid", "hi"]


# --------------------------------------------------------------------------
# setup_params — orientation -> flipped_aes
# --------------------------------------------------------------------------


def test_setup_params_orientation_y_sets_flipped():
    p = StatRle().setup_params(pd.DataFrame(), {"orientation": "y"})
    assert p["flipped_aes"] is True


def test_setup_params_orientation_x_not_flipped():
    p = StatRle().setup_params(pd.DataFrame(), {"orientation": "x"})
    assert p["flipped_aes"] is False


def test_setup_params_missing_orientation_not_flipped():
    p = StatRle().setup_params(pd.DataFrame(), {})
    assert p["flipped_aes"] is False


# --------------------------------------------------------------------------
# class attributes / default_aes
# --------------------------------------------------------------------------


def test_required_aes():
    assert StatRle.required_aes == ["x", "label"]


def test_dropped_aes():
    assert StatRle.dropped_aes == ["x", "label"]


def test_extra_params():
    assert StatRle.extra_params == ["na_rm", "orientation", "align"]


def test_default_aes_keys():
    assert set(StatRle.default_aes.keys()) == {"xmin", "xmax", "ymin", "ymax", "fill"}


def test_default_aes_string_refs():
    from ggplot2_py.aes import AfterStat

    assert StatRle.default_aes["xmin"] == AfterStat("start")
    assert StatRle.default_aes["xmax"] == AfterStat("end")
    assert StatRle.default_aes["fill"] == AfterStat("runvalue")


def test_default_aes_inf_callables():
    # ymin/ymax are constant computed aes (-Inf / +Inf) implemented as callables.
    ymin = StatRle.default_aes["ymin"].x
    ymax = StatRle.default_aes["ymax"].x
    assert callable(ymin) and callable(ymax)
    df = pd.DataFrame({"a": [1, 2, 3]})
    np.testing.assert_array_equal(np.asarray(ymin(df)), np.full(3, -np.inf))
    np.testing.assert_array_equal(np.asarray(ymax(df)), np.full(3, np.inf))


def test_stat_registered():
    # __init_subclass__ registers under name[4:].
    from ggplot2_py.stat import Stat

    assert Stat._registry.get("Rle") is StatRle
    assert Stat._registry.get("rle") is StatRle


# --------------------------------------------------------------------------
# constructor — align normalization + validation
# --------------------------------------------------------------------------


def test_constructor_normalizes_center_to_centre():
    layer = stat_rle(aes(x="x", label="label"), align="center")
    assert layer.stat_params["align"] == "centre"


def test_constructor_keeps_centre():
    layer = stat_rle(aes(x="x", label="label"), align="centre")
    assert layer.stat_params["align"] == "centre"


def test_constructor_default_align_none():
    layer = stat_rle(aes(x="x", label="label"))
    assert layer.stat_params["align"] == "none"


def test_constructor_invalid_align_raises():
    with pytest.raises(Exception):
        stat_rle(aes(x="x", label="label"), align="middle")


def test_constructor_default_geom_is_rect():
    layer = stat_rle(aes(x="x", label="label"))
    assert layer.geom.__class__.__name__.lower().endswith("rect")


def test_constructor_orientation_param():
    layer = stat_rle(aes(x="x", label="label"), orientation="y")
    assert layer.stat_params["orientation"] == "y"


# --------------------------------------------------------------------------
# full pipeline build (layer_data) vs R
# --------------------------------------------------------------------------


def test_full_pipeline_matches_r():
    df = _sin_df().rename(columns={"label": "lab"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = ggplot(df) + stat_rle(aes(x="x", label="lab"), align="none", geom="rect")
        d = gg.layer_data(p)
    # computed columns
    np.testing.assert_allclose(d["start"].to_numpy(), _SIN_NONE["start"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(d["end"].to_numpy(), _SIN_NONE["end"], rtol=1e-6, atol=1e-9)
    assert list(d["start_id"]) == _SIN_NONE["start_id"]
    assert list(d["end_id"]) == _SIN_NONE["end_id"]
    # default_aes mapping
    np.testing.assert_allclose(d["xmin"].to_numpy(), _SIN_NONE["start"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(d["xmax"].to_numpy(), _SIN_NONE["end"], rtol=1e-6, atol=1e-9)
    assert np.all(np.isneginf(d["ymin"].to_numpy().astype(float)))
    assert np.all(np.isposinf(d["ymax"].to_numpy().astype(float)))


def test_full_pipeline_drops_na_labels_at_layer():
    # In the full pipeline remove_missing(finite=True) strips NA labels
    # (even with na_rm=False, with a warning). R: runs a,a / b,b survive.
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "lab": ["a", "a", np.nan, np.nan, "b", "b"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = ggplot(df) + stat_rle(aes(x="x", label="lab"), geom="rect")
        d = gg.layer_data(p)
    np.testing.assert_allclose(d["start"].to_numpy(), [1, 5], atol=1e-9)
    np.testing.assert_allclose(d["end"].to_numpy(), [2, 6], atol=1e-9)
    assert list(d["start_id"]) == [1, 3]
    assert list(d["end_id"]) == [2, 4]
    assert list(d["runlength"]) == [2, 2]
    assert list(d["runvalue"]) == ["a", "b"]


def test_full_pipeline_fill_colours_match_r():
    # R assigns #F8766D / #00BFC4 to the two-level discrete fill scale.
    df = _sin_df().rename(columns={"label": "lab"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = ggplot(df) + stat_rle(aes(x="x", label="lab"), align="none", geom="rect")
        d = gg.layer_data(p)
    fills = list(d["fill"])
    # FALSE runs -> first hue, TRUE runs -> second hue
    assert fills[0] == "#F8766D"  # runvalue False
    assert fills[1] == "#00BFC4"  # runvalue True


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
