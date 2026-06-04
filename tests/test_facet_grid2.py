"""Tests for :mod:`ggh4x.facet_grid2`.

Reference values are taken from R (ggplot2 4.0.2 + ggh4x 0.3.1.9000) and baked in
as regression anchors:

- The ``draw_panels`` gtable layout table (every cell's ``t``/``l``/``b``/``r``/
  ``z``/``clip``/``name``, plus widths/heights unit *types* and ``null`` counts)
  was dumped from R by tracing ``FacetGrid2$draw_panels`` and is reproduced here
  cell-for-cell.  Cases: ``facet_grid2(vs ~ cyl)`` (2x3), ``scales="free"``,
  ``switch="both"`` and ``axes="all"``.
- ``.match_facet_arg`` / ``.validate_independent`` behaviour verified against the
  R helpers (including the independent => free abort and the space / remove_labels
  overrides + warnings).

The single known numeric deviation is the rendered bottom-axis band height
(R 0.4718 cm vs Python 0.4132 cm); this originates in the shared ggplot2_py axis
label-rendering pipeline (font metrics), not the facet port, so absolute cm values
are *not* asserted -- only unit types and ``null`` counts, which match exactly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ggplot2_py import aes, geom_point, ggplot, ggplotGrob

from ggh4x._facet_helpers import (
    AspectRatio,
    _match_facet_arg,
    _validate_independent,
)
from ggh4x.facet_grid2 import FacetGrid2, facet_grid2, new_grid_facets


# ---------------------------------------------------------------------------
# Fixtures: mtcars subset (matching R's datasets::mtcars columns we use)
# ---------------------------------------------------------------------------
def _mtcars() -> pd.DataFrame:
    rows = [
        (21.0, 6, 160.0, 0), (21.0, 6, 160.0, 0), (22.8, 4, 108.0, 1),
        (21.4, 6, 258.0, 1), (18.7, 8, 360.0, 0), (18.1, 6, 225.0, 1),
        (14.3, 8, 360.0, 0), (24.4, 4, 146.7, 1), (22.8, 4, 140.8, 1),
        (19.2, 6, 167.6, 1), (17.8, 6, 167.6, 1), (16.4, 8, 275.8, 0),
        (17.3, 8, 275.8, 0), (15.2, 8, 275.8, 0), (10.4, 8, 472.0, 0),
        (10.4, 8, 460.0, 0), (14.7, 8, 440.0, 0), (32.4, 4, 78.7, 1),
        (30.4, 4, 75.7, 1), (33.9, 4, 71.1, 1), (21.5, 4, 120.1, 1),
        (15.5, 8, 318.0, 0), (15.2, 8, 304.0, 0), (13.3, 8, 350.0, 0),
        (19.2, 8, 400.0, 0), (27.3, 4, 79.0, 1), (26.0, 4, 120.3, 0),
        (30.4, 4, 95.1, 1), (15.8, 8, 351.0, 0), (19.7, 6, 145.0, 0),
        (15.0, 8, 301.0, 0), (21.4, 4, 121.0, 1),
    ]
    return pd.DataFrame(rows, columns=["mpg", "cyl", "disp", "vs"])


@pytest.fixture
def mtcars() -> pd.DataFrame:
    return _mtcars()


# ---------------------------------------------------------------------------
# Helper: capture the draw_panels gtable + normalise its layout
# ---------------------------------------------------------------------------
def _capture_panel_table(plot) -> object:
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


def _layout_cells(gt) -> set:
    lay = gt.layout
    if isinstance(lay, dict):
        df = pd.DataFrame({k: list(v) for k, v in lay.items()})
    else:
        df = lay
    cells = set()
    for _, row in df.iterrows():
        cells.add(
            (
                int(row["t"]), int(row["l"]), int(row["b"]), int(row["r"]),
                int(float(row["z"])), str(row["clip"]), str(row["name"]),
            )
        )
    return cells


def _unit_types(unit) -> list:
    return list(unit.units_list) if hasattr(unit, "units_list") else []


def _null_count(unit) -> int:
    return sum(1 for u in _unit_types(unit) if u == "null")


# ---------------------------------------------------------------------------
# R gold-standard layout cells (traced from FacetGrid2$draw_panels)
# ---------------------------------------------------------------------------
GRID2_BASIC_CELLS = {
    (1, 2, 1, 2, 3, "off", "axis-t-1-1"), (1, 6, 1, 6, 3, "off", "axis-t-2-1"),
    (1, 10, 1, 10, 3, "off", "axis-t-3-1"),
    (2, 2, 2, 2, 2, "on", "strip-t-1"), (2, 6, 2, 6, 2, "on", "strip-t-2"),
    (2, 10, 2, 10, 2, "on", "strip-t-3"),
    (3, 1, 3, 1, 3, "off", "axis-l-1-1"), (3, 2, 3, 2, 1, "on", "panel-1-1"),
    (3, 3, 3, 3, 3, "off", "axis-r-1-1"), (3, 5, 3, 5, 3, "off", "axis-l-1-2"),
    (3, 6, 3, 6, 1, "on", "panel-2-1"), (3, 7, 3, 7, 3, "off", "axis-r-1-2"),
    (3, 9, 3, 9, 3, "off", "axis-l-1-3"), (3, 10, 3, 10, 1, "on", "panel-1-2"),
    (3, 11, 3, 11, 2, "on", "strip-r-1"), (3, 12, 3, 12, 3, "off", "axis-r-1-3"),
    (4, 2, 4, 2, 3, "off", "axis-b-1-1"), (4, 6, 4, 6, 3, "off", "axis-b-2-1"),
    (4, 10, 4, 10, 3, "off", "axis-b-3-1"),
    (6, 2, 6, 2, 3, "off", "axis-t-1-2"), (6, 6, 6, 6, 3, "off", "axis-t-2-2"),
    (6, 10, 6, 10, 3, "off", "axis-t-3-2"),
    (7, 1, 7, 1, 3, "off", "axis-l-2-1"), (7, 2, 7, 2, 1, "on", "panel-2-2"),
    (7, 3, 7, 3, 3, "off", "axis-r-2-1"), (7, 5, 7, 5, 3, "off", "axis-l-2-2"),
    (7, 6, 7, 6, 1, "on", "panel-1-3"), (7, 7, 7, 7, 3, "off", "axis-r-2-2"),
    (7, 9, 7, 9, 3, "off", "axis-l-2-3"), (7, 10, 7, 10, 1, "on", "panel-2-3"),
    (7, 11, 7, 11, 2, "on", "strip-r-2"), (7, 12, 7, 12, 3, "off", "axis-r-2-3"),
    (8, 2, 8, 2, 3, "off", "axis-b-1-2"), (8, 6, 8, 6, 3, "off", "axis-b-2-2"),
    (8, 10, 8, 10, 3, "off", "axis-b-3-2"),
}

# ``scales="free"`` produces the same panel/strip/axis cell topology as basic
# (the axes are still only at the margins; free only affects scale training).
GRID2_FREE_CELLS = GRID2_BASIC_CELLS

# switch="both" -> col strips move to the bottom (strip-b) and row strips to the
# left (strip-l) of the panels. Generated from the R draw_panels dump.
GRID2_SWITCH_CELLS = {
    (1, 3, 1, 3, 3, "off", "axis-t-1-1"),
    (1, 7, 1, 7, 3, "off", "axis-t-2-1"),
    (1, 11, 1, 11, 3, "off", "axis-t-3-1"),
    (2, 1, 2, 1, 3, "off", "axis-l-1-1"),
    (2, 2, 2, 2, 2, "on", "strip-l-1"),
    (2, 3, 2, 3, 1, "on", "panel-1-1"),
    (2, 4, 2, 4, 3, "off", "axis-r-1-1"),
    (2, 6, 2, 6, 3, "off", "axis-l-1-2"),
    (2, 7, 2, 7, 1, "on", "panel-2-1"),
    (2, 8, 2, 8, 3, "off", "axis-r-1-2"),
    (2, 10, 2, 10, 3, "off", "axis-l-1-3"),
    (2, 11, 2, 11, 1, "on", "panel-1-2"),
    (2, 12, 2, 12, 3, "off", "axis-r-1-3"),
    (3, 3, 3, 3, 3, "off", "axis-b-1-1"),
    (3, 7, 3, 7, 3, "off", "axis-b-2-1"),
    (3, 11, 3, 11, 3, "off", "axis-b-3-1"),
    (5, 3, 5, 3, 3, "off", "axis-t-1-2"),
    (5, 7, 5, 7, 3, "off", "axis-t-2-2"),
    (5, 11, 5, 11, 3, "off", "axis-t-3-2"),
    (6, 1, 6, 1, 3, "off", "axis-l-2-1"),
    (6, 2, 6, 2, 2, "on", "strip-l-2"),
    (6, 3, 6, 3, 1, "on", "panel-2-2"),
    (6, 4, 6, 4, 3, "off", "axis-r-2-1"),
    (6, 6, 6, 6, 3, "off", "axis-l-2-2"),
    (6, 7, 6, 7, 1, "on", "panel-1-3"),
    (6, 8, 6, 8, 3, "off", "axis-r-2-2"),
    (6, 10, 6, 10, 3, "off", "axis-l-2-3"),
    (6, 11, 6, 11, 1, "on", "panel-2-3"),
    (6, 12, 6, 12, 3, "off", "axis-r-2-3"),
    (7, 3, 7, 3, 2, "on", "strip-b-1"),
    (7, 7, 7, 7, 2, "on", "strip-b-2"),
    (7, 11, 7, 11, 2, "on", "strip-b-3"),
    (8, 3, 8, 3, 3, "off", "axis-b-1-2"),
    (8, 7, 8, 7, 3, "off", "axis-b-2-2"),
    (8, 11, 8, 11, 3, "off", "axis-b-3-2"),
}

# axes="all": layout cell positions are identical to basic (the inner axis bands
# already exist as 0-cm cells); axes="all" only fills them with non-zero grobs.
GRID2_AXESALL_CELLS = GRID2_BASIC_CELLS


# ---------------------------------------------------------------------------
# Constructor / params
# ---------------------------------------------------------------------------
def test_constructor_returns_facetgrid2():
    f = facet_grid2("vs", "cyl")
    assert isinstance(f, FacetGrid2)
    assert f.strip is not None
    assert f.shrink is True


def test_params_normalised():
    f = facet_grid2("vs", "cyl", scales="free", axes="all", independent="all")
    assert f.params["free"] == {"x": True, "y": True}
    assert f.params["axes"] == {"x": True, "y": True}
    assert f.params["independent"] == {"x": True, "y": True}
    assert f.params["rows"] == ["vs"]
    assert f.params["cols"] == ["cyl"]


def test_switch_none_maps_to_none():
    assert facet_grid2("vs", "cyl").params["switch"] is None
    assert facet_grid2("vs", "cyl", switch="both").params["switch"] == "both"


# ---------------------------------------------------------------------------
# .match_facet_arg parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        ("fixed", {"x": False, "y": False}),
        ("free_x", {"x": True, "y": False}),
        ("free_y", {"x": False, "y": True}),
        ("free", {"x": True, "y": True}),
        (False, {"x": False, "y": False}),
        (True, {"x": True, "y": True}),
    ],
)
def test_match_facet_arg(value, expected):
    opts = ["fixed", "free_x", "free_y", "free"]
    assert _match_facet_arg(value, opts) == expected


def test_match_facet_arg_invalid():
    with pytest.raises(ValueError):
        _match_facet_arg("bogus", ["fixed", "free_x", "free_y", "free"])


# ---------------------------------------------------------------------------
# .validate_independent parity
# ---------------------------------------------------------------------------
def test_validate_independent_requires_free():
    with pytest.raises(ValueError):
        _validate_independent(
            {"x": True, "y": False},
            {"x": False, "y": False},
            {"x": False, "y": False},
            {"x": False, "y": False},
        )


def test_validate_independent_overrides_space_and_rmlab():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = _validate_independent(
            {"x": True, "y": False},
            {"x": True, "y": False},
            {"x": True, "y": False},  # space free on x -> forced off
            {"x": True, "y": False},  # rmlab on x -> forced off
        )
    assert out["space_free"]["x"] is False
    assert out["rmlab"]["x"] is False
    assert len(rec) == 2  # one warn for space, one for rmlab


def test_validate_independent_noop_when_not_independent():
    out = _validate_independent(
        {"x": False, "y": False},
        {"x": True, "y": True},
        {"x": True, "y": True},
        {"x": True, "y": True},
    )
    assert out["space_free"] == {"x": True, "y": True}
    assert out["rmlab"] == {"x": True, "y": True}


# ---------------------------------------------------------------------------
# compute_layout
# ---------------------------------------------------------------------------
def test_compute_layout_has_render_column(mtcars):
    f = facet_grid2("vs", "cyl")
    layout = f.compute_layout([mtcars], f.params)
    assert "_render" in layout.columns
    assert set(layout.columns) >= {"PANEL", "ROW", "COL", "SCALE_X", "SCALE_Y", "_render"}
    # 2 vs x 3 cyl = 6 panels.
    assert len(layout) == 6
    assert int(layout["ROW"].max()) == 2
    assert int(layout["COL"].max()) == 3
    # fixed scales -> all SCALE ids are 1.
    assert set(layout["SCALE_X"]) == {1}
    assert set(layout["SCALE_Y"]) == {1}


def test_compute_layout_independent_scales(mtcars):
    f = facet_grid2("vs", "cyl", scales="free", independent="all")
    layout = f.compute_layout([mtcars], f.params)
    # independent => one scale per panel.
    assert sorted(layout["SCALE_X"]) == list(range(1, len(layout) + 1))
    assert sorted(layout["SCALE_Y"]) == list(range(1, len(layout) + 1))


def test_compute_layout_free_scales_per_axis(mtcars):
    f = facet_grid2("vs", "cyl", scales="free")
    layout = f.compute_layout([mtcars], f.params)
    # free (not independent): SCALE_X == COL, SCALE_Y == ROW.
    assert list(layout["SCALE_X"]) == list(layout["COL"])
    assert list(layout["SCALE_Y"]) == list(layout["ROW"])


def test_compute_layout_duplicate_var_aborts(mtcars):
    f = facet_grid2("cyl", "cyl")
    with pytest.raises(ValueError):
        f.compute_layout([mtcars], f.params)


# ---------------------------------------------------------------------------
# draw_panels layout-table parity (the acceptance test)
# ---------------------------------------------------------------------------
def test_draw_panels_basic_layout(mtcars):
    p = ggplot(mtcars, aes("disp", "mpg")) + geom_point() + facet_grid2("vs", "cyl")
    gt = _capture_panel_table(p)
    assert _layout_cells(gt) == GRID2_BASIC_CELLS


def test_draw_panels_free_layout(mtcars):
    p = (
        ggplot(mtcars, aes("disp", "mpg"))
        + geom_point()
        + facet_grid2("vs", "cyl", scales="free")
    )
    gt = _capture_panel_table(p)
    assert _layout_cells(gt) == GRID2_FREE_CELLS


def test_draw_panels_switch_layout(mtcars):
    p = (
        ggplot(mtcars, aes("disp", "mpg"))
        + geom_point()
        + facet_grid2("vs", "cyl", switch="both")
    )
    gt = _capture_panel_table(p)
    assert _layout_cells(gt) == GRID2_SWITCH_CELLS


def test_draw_panels_axes_all_layout(mtcars):
    p = (
        ggplot(mtcars, aes("disp", "mpg"))
        + geom_point()
        + facet_grid2("vs", "cyl", axes="all")
    )
    gt = _capture_panel_table(p)
    # The cell topology equals basic; axes="all" only fills the existing inner
    # axis bands with non-zero grobs (verified separately via cm sizes).
    assert _layout_cells(gt) == GRID2_AXESALL_CELLS

    # Prove the inner x-axis band is actually drawn: R turns height row 4
    # (the inner axis-b row, 0-based index 3) from 0 cm (basic) into a non-zero
    # cm; Python must do the same.
    heights = gt._heights if hasattr(gt, "_heights") else gt.heights
    vals = list(heights.values)
    assert vals[3] > 0.0

    # And basic must keep that inner band at 0 cm.
    p_basic = (
        ggplot(mtcars, aes("disp", "mpg")) + geom_point() + facet_grid2("vs", "cyl")
    )
    gt_basic = _capture_panel_table(p_basic)
    hb = gt_basic._heights if hasattr(gt_basic, "_heights") else gt_basic.heights
    vb = list(hb.values)
    assert vb[3] == 0.0


# ---------------------------------------------------------------------------
# widths / heights unit-type + null-count parity
# ---------------------------------------------------------------------------
def test_draw_panels_size_unit_types(mtcars):
    p = ggplot(mtcars, aes("disp", "mpg")) + geom_point() + facet_grid2("vs", "cyl")
    gt = _capture_panel_table(p)
    widths = gt._widths if hasattr(gt, "_widths") else gt.widths
    heights = gt._heights if hasattr(gt, "_heights") else gt.heights
    # R: widths 3 null, heights 2 null (one per panel column / row).
    assert _null_count(widths) == 3
    assert _null_count(heights) == 2
    # R width unit types (per cm/null/points pattern).
    assert _unit_types(widths) == [
        "cm", "null", "cm", "points", "cm", "null", "cm", "points",
        "cm", "null", "cm", "cm",
    ]
    assert _unit_types(heights) == [
        "cm", "cm", "null", "cm", "points", "cm", "null", "cm",
    ]


def test_draw_panels_respect_false_default(mtcars):
    p = ggplot(mtcars, aes("disp", "mpg")) + geom_point() + facet_grid2("vs", "cyl")
    gt = _capture_panel_table(p)
    respect = getattr(gt, "respect", getattr(gt, "_respect", None))
    assert respect is False


# ---------------------------------------------------------------------------
# setup_aspect_ratio struct
# ---------------------------------------------------------------------------
def test_setup_aspect_ratio_struct(mtcars):
    f = facet_grid2("vs", "cyl")
    # No aspect, free both -> 1 / respect False.
    ar = f.setup_aspect_ratio(_FakeCoord(None), {"x": True, "y": True}, None, [{}])
    assert isinstance(ar, AspectRatio)
    assert ar.value == 1.0 and ar.respect is False
    # Coord supplies an aspect when not free -> respect True.
    ar2 = f.setup_aspect_ratio(_FakeCoord(2.0), {"x": False, "y": False}, None, [{}])
    assert ar2.value == 2.0 and ar2.respect is True


class _FakeCoord:
    clip = "on"

    def __init__(self, aspect_val):
        self._aspect = aspect_val

    def aspect(self, ranges):
        return self._aspect

    def is_free(self):
        return True


# ---------------------------------------------------------------------------
# new_grid_facets exposes the same object
# ---------------------------------------------------------------------------
def test_new_grid_facets_direct():
    f = new_grid_facets(
        "vs", "cyl", "free", "fixed", "all", "none", "all",
        True, "label_value", True, None, True, False, True, "vanilla",
    )
    assert isinstance(f, FacetGrid2)
    assert f.params["free"] == {"x": True, "y": True}
    assert f.params["independent"] == {"x": True, "y": True}
