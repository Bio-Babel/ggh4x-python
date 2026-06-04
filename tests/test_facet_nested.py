"""Tests for :mod:`ggh4x.facet_nested` and :mod:`ggh4x.facet_nested_wrap`.

Reference values are taken from R (ggplot2 4.0.2 + ggh4x via the ggrepel-dev
Rscript) by tracing ``ggh4x:::add_nest_indicator`` and dumping the assembled
panel-gtable layout (``ggplotGrob``).  The load-bearing anchors baked in here:

* **Merged strip spans + z-order.**  For ``facet_nested(~ vs + cyl)`` (top) the
  multi-child parent strips ``strip-t-1`` / ``strip-t-2`` span several columns and
  carry the nest line at ``z = 3`` (base ``2`` + offset ``1``), while the
  single-child strips stay at ``z = 2``.  Mirror cases verified: rows nesting
  (right strips, ``z = 4``, secondary offset ``2``), ``switch="x"`` (bottom
  strips, ``z = 4``) and the wrap facet.
* **Nest-line z-offset math** (``facet_nested.R:300-307`` / ``343-350``).  Traced
  from R: top -> ``nlevels(2) - t(1) = 1`` (flip); right -> ``l(1) = 2`` (no flip,
  secondary); bottom -> ``t(1) = 2`` (no flip, secondary).
* **map_data / vars_combine nesting rule.**  Verified cell-for-cell against R:
  the partial (outer-var-only) layer maps to the blank-inner panels and
  ``vars_combine`` blank-fills (``""``) the absent inner column.

The default ``nest_line`` (``element_line(inherit_blank=True)``) draws nothing
under the default theme (the ``ggh4x.facet.nestline`` element is blank), turns on
when the theme sets it, and an explicit ``element_line`` always draws -- all
verified against R.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ggplot2_py import aes, geom_point, ggplot, ggplotGrob, theme
from ggplot2_py.theme_elements import ElementBlank, ElementLine, element_blank, element_line
from gtable_py import Gtable

from ggh4x.facet_grid2 import FacetGrid2
from ggh4x.facet_wrap2 import FacetWrap2
from ggh4x.facet_nested import (
    FacetNested,
    add_nest_indicator,
    facet_nested,
)
from ggh4x.facet_nested_wrap import FacetNestedWrap, facet_nested_wrap
from ggh4x.strip_nested import StripNested


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _mtcars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mpg": [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4,
                    22.8, 19.2, 10.4, 32.4, 15.0, 33.9],
            "wt": [2.62, 2.88, 2.32, 3.21, 3.44, 3.46, 3.57, 3.19,
                   3.15, 3.44, 5.25, 2.20, 3.57, 1.84],
            "vs": [0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
            "cyl": [6, 6, 4, 6, 8, 6, 8, 4, 4, 6, 8, 4, 8, 4],
        }
    )


def _mpg() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "displ": [1.8, 2.0, 2.8, 3.1, 4.7, 5.2, 3.8, 2.5, 6.2, 5.7, 4.0, 3.3],
            "hwy": [29, 31, 26, 27, 17, 15, 18, 20, 16, 18, 19, 21],
            "cyl": [4, 4, 6, 6, 8, 8, 6, 4, 8, 8, 6, 6],
            "drv": ["f", "f", "f", "f", "4", "4", "4", "4", "r", "r", "r", "r"],
        }
    )


@pytest.fixture
def mtcars() -> pd.DataFrame:
    return _mtcars()


@pytest.fixture
def mpg() -> pd.DataFrame:
    return _mpg()


# ---------------------------------------------------------------------------
# Helpers: capture the assembled panel table (post finish_panels)
# ---------------------------------------------------------------------------
def _capture_panel_table(plot) -> Gtable:
    captured = {}
    facet = plot.facet
    orig = facet.draw_panels

    def _wrap(*args, **kwargs):
        out = orig(*args, **kwargs)
        captured["gt"] = out
        return out

    facet.draw_panels = _wrap
    try:
        ggplotGrob(plot)
    finally:
        facet.draw_panels = orig
    return captured["gt"]


def _layout(gt: Gtable) -> pd.DataFrame:
    lay = gt.layout
    if isinstance(lay, dict):
        return pd.DataFrame({k: list(v) for k, v in lay.items()})
    return lay


def _strip_rows(gt: Gtable) -> pd.DataFrame:
    df = _layout(gt)
    return df[df["name"].astype(str).str.startswith("strip-")].reset_index(drop=True)


def _has_nester(gt: Gtable) -> bool:
    for g in gt.grobs:
        if hasattr(g, "layout"):
            names = g.layout.get("name", []) if isinstance(g.layout, dict) else []
            if "nester" in list(names):
                return True
    return False


def _strip_z(gt: Gtable) -> dict:
    df = _strip_rows(gt)
    return {str(n): int(float(z)) for n, z in zip(df["name"], df["z"])}


# ---------------------------------------------------------------------------
# Constructors / class wiring
# ---------------------------------------------------------------------------
def test_facet_nested_returns_facetnested():
    f = facet_nested(cols="vs + cyl")
    assert isinstance(f, FacetNested)
    assert isinstance(f, FacetGrid2)
    assert isinstance(f.strip, StripNested)  # R default strip = "nested"


def test_facet_nested_wrap_returns_facetnestedwrap():
    f = facet_nested_wrap("cyl + drv")
    assert isinstance(f, FacetNestedWrap)
    assert isinstance(f, FacetWrap2)
    assert isinstance(f.strip, StripNested)


def test_default_params_present():
    f = facet_nested(cols="vs + cyl")
    assert "nest_line" in f.params
    assert f.params["solo_line"] is False
    assert f.params["resect"] is not None
    # default nest_line is an inherit-blank element_line (R constructor default).
    assert isinstance(f.params["nest_line"], ElementLine)
    assert f.params["nest_line"].inherit_blank is True


def test_nest_line_logical_coercion():
    assert isinstance(facet_nested(cols="vs", nest_line=True).params["nest_line"], ElementLine)
    assert isinstance(facet_nested(cols="vs", nest_line=False).params["nest_line"], ElementBlank)
    assert isinstance(
        facet_nested(cols="vs", nest_line=element_blank()).params["nest_line"], ElementBlank
    )


def test_nest_line_invalid_raises():
    with pytest.raises(Exception):
        facet_nested(cols="vs", nest_line="not-an-element")


def test_solo_line_param():
    f = facet_nested(cols="vs + cyl", solo_line=True)
    assert f.params["solo_line"] is True


def test_bleed_deprecation_warns_and_sets_strip_param():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        f = facet_nested(cols="vs + cyl", bleed=True)
    assert any(issubclass(w.category, DeprecationWarning) for w in rec)
    assert f.strip.params["bleed"] is True


def test_wrap_bleed_deprecation():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        f = facet_nested_wrap("cyl + drv", bleed=True)
    assert any(issubclass(w.category, DeprecationWarning) for w in rec)
    assert f.strip.params["bleed"] is True


# ---------------------------------------------------------------------------
# vars_combine: blank-fill of partial layers (R: facet_nested.R:188-234)
# ---------------------------------------------------------------------------
def _vc_data():
    df_full = pd.DataFrame(
        dict(mpg=[21, 22, 18, 15, 33, 19], wt=[2.6, 2.3, 3.4, 3.6, 1.8, 3.4],
             vs=[0, 1, 0, 0, 1, 1], cyl=[6, 4, 8, 8, 4, 6])
    )
    df_partial = pd.DataFrame(dict(mpg=[25, 16], wt=[3, 4], vs=[1, 0]))
    return df_full, df_partial


def test_vars_combine_full_only_keeps_numeric():
    df_full, _ = _vc_data()
    f = facet_nested(cols="vs + cyl")
    vc = f.vars_combine([df_full, df_full], vars_=f.params["cols"])
    # has_all base only (R: 4 rows); cyl stays numeric.
    assert len(vc) == 4
    assert set(map(int, vc["cyl"])) == {4, 6, 8}


def test_vars_combine_partial_blank_fills():
    df_full, df_partial = _vc_data()
    f = facet_nested(cols="vs + cyl")
    vc = f.vars_combine([df_full, df_full, df_partial], vars_=f.params["cols"])
    # R returns 4 base + 8 blank-filled rows (df.grid over all 4 base rows x 2 vs).
    assert len(vc) == 12
    cyl = [str(c) for c in vc["cyl"]]
    assert cyl.count("") == 8
    # the blank-filled vs values are 0,0,0,0,1,1,1,1 (R df.grid order).
    blanks = vc[vc["cyl"].astype(str) == ""]
    assert list(map(int, blanks["vs"])) == [0, 0, 0, 0, 1, 1, 1, 1]


def test_vars_combine_requires_one_full_layer():
    df_partial = pd.DataFrame(dict(mpg=[1], vs=[0]))  # only vs, never cyl
    f = facet_nested(cols="vs + cyl")
    with pytest.raises(Exception):
        f.vars_combine([df_partial, df_partial], vars_=f.params["cols"])


# ---------------------------------------------------------------------------
# map_data: per-direction missing rule (R: facet_nested.R:130-187)
# ---------------------------------------------------------------------------
def test_map_data_full_layer_panels():
    df_full, df_partial = _vc_data()
    f = facet_nested(cols="vs + cyl")
    params = dict(f.params)
    params["_possible_columns"] = sorted(
        set(list(df_full.columns) + list(df_partial.columns))
    )
    layout = f.compute_layout([df_full, df_full, df_partial], params)
    md = f.map_data(df_full.copy(), layout, params)
    # R gold standard: PANEL = 2 5 3 3 5 6.
    assert list(map(int, md["PANEL"])) == [2, 5, 3, 3, 5, 6]


def test_map_data_partial_layer_to_blank_panels():
    df_full, df_partial = _vc_data()
    f = facet_nested(cols="vs + cyl")
    params = dict(f.params)
    params["_possible_columns"] = sorted(
        set(list(df_full.columns) + list(df_partial.columns))
    )
    layout = f.compute_layout([df_full, df_full, df_partial], params)
    md = f.map_data(df_partial.copy(), layout, params)
    # R gold standard: vs=1 -> PANEL 4, vs=0 -> PANEL 1 (the blank-cyl panels).
    assert list(map(int, md["PANEL"])) == [4, 1]


def test_map_data_empty_data():
    f = facet_nested(cols="vs + cyl")
    empty = pd.DataFrame({"mpg": [], "wt": [], "vs": [], "cyl": []})
    layout = pd.DataFrame({"PANEL": [1], "vs": [0], "cyl": [6]})
    md = f.map_data(empty, layout, dict(f.params, _possible_columns=["vs", "cyl"]))
    assert "PANEL" in md.columns
    assert len(md) == 0


def test_map_data_no_facet_vars():
    f = facet_nested()  # no rows/cols
    data = pd.DataFrame({"mpg": [1, 2], "wt": [3, 4]})
    layout = pd.DataFrame({"PANEL": [1]})
    md = f.map_data(data, layout, dict(f.params, _possible_columns=["mpg", "wt"]))
    assert list(md["PANEL"]) == [1, 1]


# ---------------------------------------------------------------------------
# Nest indicator: merged strip spans + z-offsets (R gold standard)
# ---------------------------------------------------------------------------
def test_grid_top_merged_spans_and_z(mtcars):
    plot = (
        ggplot(mtcars, aes("mpg", "wt")) + geom_point()
        + facet_nested(cols="vs + cyl", nest_line=element_line(colour="red"))
    )
    gt = _capture_panel_table(plot)
    sr = _strip_rows(gt)
    # Two merged (multi-child) parent strips span >1 column (l != r).
    merged = sr[sr["l"] != sr["r"]]
    assert len(merged) == 2
    # Merged parents carry the nest line at z = 3; single-child strips at z = 2.
    z = _strip_z(gt)
    for n, zz in z.items():
        if (sr.loc[sr["name"] == n, "l"].iloc[0]
                != sr.loc[sr["name"] == n, "r"].iloc[0]):
            assert zz == 3, (n, zz)
        else:
            assert zz == 2, (n, zz)
    assert _has_nester(gt)


def test_grid_top_nester_on_outer_layer(mtcars):
    plot = (
        ggplot(mtcars, aes("mpg", "wt")) + geom_point()
        + facet_nested(cols="vs + cyl", nest_line=element_line(colour="red"))
    )
    gt = _capture_panel_table(plot)
    df = _layout(gt)
    # Find a merged strip sub-gtable and confirm the nester sits on the outer
    # (t == 1) layer row of the 2x1 sub-gtable (R vec_slice(layout, 1)).
    found = False
    for i in df.index:
        nm = str(df["name"][i])
        if not nm.startswith("strip-t"):
            continue
        g = gt.grobs[i]
        if not hasattr(g, "layout"):
            continue
        sub = pd.DataFrame({k: list(v) for k, v in g.layout.items()})
        if "nester" in list(sub["name"]):
            assert g.shape == (2, 1)
            nrow = sub[sub["name"] == "nester"]
            assert int(nrow["t"].iloc[0]) == 1
            found = True
    assert found


def test_grid_right_rows_nesting_z(mtcars):
    plot = (
        ggplot(mtcars, aes("mpg", "wt")) + geom_point()
        + facet_nested(rows="vs + cyl", nest_line=element_line(colour="blue"))
    )
    gt = _capture_panel_table(plot)
    sr = _strip_rows(gt)
    # Right strips (secondary): merged parents span >1 row (t != b), z = 4.
    merged = sr[sr["t"] != sr["b"]]
    assert len(merged) == 2
    z = _strip_z(gt)
    for n in merged["name"]:
        assert z[str(n)] == 4
    assert all(str(n).startswith("strip-r") for n in sr["name"])


def test_grid_switch_bottom_z(mtcars):
    plot = (
        ggplot(mtcars, aes("mpg", "wt")) + geom_point()
        + facet_nested(cols="vs + cyl", switch="x", nest_line=element_line(colour="red"))
    )
    gt = _capture_panel_table(plot)
    sr = _strip_rows(gt)
    assert all(str(n).startswith("strip-b") for n in sr["name"])
    # Bottom strips (secondary): merged parents at z = 4 (base 2 + offset t(1)=2).
    merged = sr[sr["l"] != sr["r"]]
    z = _strip_z(gt)
    for n in merged["name"]:
        assert z[str(n)] == 4


def test_wrap_merged_z(mpg):
    plot = (
        ggplot(mpg, aes("displ", "hwy")) + geom_point()
        + facet_nested_wrap("cyl + drv", nest_line=element_line(colour="green"))
    )
    gt = _capture_panel_table(plot)
    sr = _strip_rows(gt)
    z = _strip_z(gt)
    merged = sr[sr["l"] != sr["r"]]
    # Each multi-child parent strip carries the line at z = 3 (top, offset 1).
    assert len(merged) >= 1
    for n in merged["name"]:
        assert z[str(n)] == 3
    assert _has_nester(gt)


def test_solo_line_drops_inner_layer(mtcars):
    # With solo_line=True, the line is also placed on single-child parents, but
    # the innermost layer strips are dropped (R: is_inner via t == nrow).
    plot = (
        ggplot(mtcars, aes("mpg", "wt")) + geom_point()
        + facet_nested(cols="vs + cyl", solo_line=True, nest_line=element_line(colour="red"))
    )
    gt = _capture_panel_table(plot)
    sr = _strip_rows(gt)
    z = _strip_z(gt)
    # Only the outer (parent) layer strips get bumped to z = 3; inner stay z = 2.
    bumped = [n for n, zz in z.items() if zz == 3]
    # The merged (l != r) parents must be among the bumped strips.
    merged_names = set(str(n) for n in sr.loc[sr["l"] != sr["r"], "name"])
    assert merged_names.issubset(set(bumped))


# ---------------------------------------------------------------------------
# nest_line / theme resolution (R-faithful blank default)
# ---------------------------------------------------------------------------
def test_default_nest_line_draws_nothing(mtcars):
    plot = (
        ggplot(mtcars, aes("mpg", "wt")) + geom_point()
        + facet_nested(cols="vs + cyl")  # default inherit-blank line + blank theme
    )
    gt = _capture_panel_table(plot)
    assert not _has_nester(gt)


def test_theme_turns_nest_line_on(mtcars):
    plot = (
        ggplot(mtcars, aes("mpg", "wt")) + geom_point()
        + facet_nested(cols="vs + cyl")
        + theme(**{"ggh4x.facet.nestline": element_line(colour="red")})
    )
    gt = _capture_panel_table(plot)
    assert _has_nester(gt)


def test_blank_nest_line_draws_nothing(mtcars):
    plot = (
        ggplot(mtcars, aes("mpg", "wt")) + geom_point()
        + facet_nested(cols="vs + cyl", nest_line=element_blank())
    )
    gt = _capture_panel_table(plot)
    assert not _has_nester(gt)


def test_add_nest_indicator_blank_is_identity():
    # add_nest_indicator on a blank nest_line returns the gtable untouched.
    gt = Gtable()
    params = {"nest_line": element_blank(), "solo_line": False, "resect": None}
    out = add_nest_indicator(gt, params, None)
    assert out is gt


def test_add_nest_indicator_none_is_identity():
    gt = Gtable()
    params = {"nest_line": None, "solo_line": False, "resect": None}
    out = add_nest_indicator(gt, params, None)
    assert out is gt


# ---------------------------------------------------------------------------
# End-to-end: renders to a Gtable
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "facet",
    [
        facet_nested(cols="vs + cyl", nest_line=element_line(colour="red")),
        facet_nested(rows="vs + cyl", nest_line=element_line(colour="red")),
        facet_nested(cols="vs + cyl", switch="x", nest_line=element_line(colour="red")),
        facet_nested(cols="vs + cyl", solo_line=True, nest_line=element_line(colour="red")),
        facet_nested(cols="vs + cyl"),  # default (blank) nest line
    ],
)
def test_grid_renders_to_gtable(mtcars, facet):
    g = ggplotGrob(ggplot(mtcars, aes("mpg", "wt")) + geom_point() + facet)
    assert isinstance(g, Gtable)


def test_wrap_renders_to_gtable(mpg):
    g = ggplotGrob(
        ggplot(mpg, aes("displ", "hwy")) + geom_point()
        + facet_nested_wrap("cyl + drv", nest_line=element_line(colour="green"))
    )
    assert isinstance(g, Gtable)


def test_resect_compound_unit_active():
    # The active line unit is unit(c(0,1),"npc") + c(1,-1)*resect -- a length-2
    # compound; verify a non-zero resect still renders end-to-end.
    from grid_py import Unit

    mt = _mtcars()
    g = ggplotGrob(
        ggplot(mt, aes("mpg", "wt")) + geom_point()
        + facet_nested(
            cols="vs + cyl",
            nest_line=element_line(colour="red"),
            resect=Unit(5, "mm"),
        )
    )
    assert isinstance(g, Gtable)
