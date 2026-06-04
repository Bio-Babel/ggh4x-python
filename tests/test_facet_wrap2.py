r"""Tests for :mod:`ggh4x.facet_wrap2`.

Reference values are taken from R (ggplot2 4.0.2 + ggh4x 0.3.1.9000) and baked in
as regression anchors:

- The ``draw_panels`` gtable layout table (every cell's ``t``/``l``/``b``/``r``/
  ``z``/``clip``/``name``, plus widths/heights unit *types* and ``null`` counts)
  dumped from R by tracing ``FacetWrap2$draw_panels``.  Cases:
  ``facet_wrap2(~cyl)`` (2x2 over the 4 ``cyl`` levels) and ``facet_wrap2(~class)``
  (7 panels in a 3x3 grid, exercising the dangling-cell / marginal-axis
  re-placement branch -- the highest-bug-density part of the port).
- The ``empties``-mask ``diff`` helpers (``_diff_down`` / ``_diff_right``) verified
  against R's ``apply(empties, MARGIN, \(x) c(diff(x)==t, FALSE))`` directly.

The single known numeric deviation is the rendered bottom-axis band height
(R 0.4718 cm vs Python 0.4132 cm) from the shared ggplot2_py axis label-rendering
pipeline; absolute cm values are not asserted -- only unit types / ``null`` counts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggplot2_py import aes, geom_point, ggplot, ggplotGrob
from ggplot2_py.coord import CoordCartesian, CoordFlip

from ggh4x._facet_helpers import AspectRatio
from ggh4x.facet_wrap2 import (
    FacetWrap2,
    _diff_down,
    _diff_right,
    facet_wrap2,
    new_wrap_facets,
)


# ---------------------------------------------------------------------------
# Fixtures: mpg subset (cyl spans 4,5,6,8; class has 7 levels like R's mpg)
# ---------------------------------------------------------------------------
def _mpg() -> pd.DataFrame:
    classes = [
        "2seater", "compact", "midsize", "minivan", "pickup", "subcompact", "suv"
    ]
    rows = []
    # cyl must span all four levels (4,5,6,8) -> ~cyl makes the same 2x2 grid R does.
    for j, cy in enumerate([4, 5, 6, 8]):
        rows.append((2.0 + j * 0.3, 20 + j, cy, "compact"))
    for i, cl in enumerate(classes):
        rows.append((2.0 + i * 0.3, 20 + i, [4, 6, 8, 4, 6, 8, 8][i], cl))
        rows.append((3.0 + i * 0.2, 25 + i, [4, 6, 8, 4, 8, 8, 8][i], cl))
    return pd.DataFrame(rows, columns=["displ", "hwy", "cyl", "class"])


@pytest.fixture
def mpg() -> pd.DataFrame:
    return _mpg()


# ---------------------------------------------------------------------------
# Helpers
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
# R gold-standard layout cells (traced from FacetWrap2$draw_panels)
# ---------------------------------------------------------------------------
WRAP2_BASIC_CELLS = {
    (1, 2, 1, 2, 3, "off", "axis-t-1-1"), (1, 6, 1, 6, 3, "off", "axis-t-2-1"),
    (2, 2, 2, 2, 2, "on", "strip-t-1-1"), (2, 6, 2, 6, 2, "on", "strip-t-2-2"),
    (3, 1, 3, 1, 3, "off", "axis-l-1-1"), (3, 2, 3, 2, 1, "on", "panel-1"),
    (3, 3, 3, 3, 3, "off", "axis-r-1-1"), (3, 5, 3, 5, 3, "off", "axis-l-1-2"),
    (3, 6, 3, 6, 1, "on", "panel-2"), (3, 7, 3, 7, 3, "off", "axis-r-1-2"),
    (4, 2, 4, 2, 3, "off", "axis-b-1-1"), (4, 6, 4, 6, 3, "off", "axis-b-2-1"),
    (6, 2, 6, 2, 3, "off", "axis-t-1-2"), (6, 6, 6, 6, 3, "off", "axis-t-2-2"),
    (7, 2, 7, 2, 2, "on", "strip-t-3-3"), (7, 6, 7, 6, 2, "on", "strip-t-4-4"),
    (8, 1, 8, 1, 3, "off", "axis-l-2-1"), (8, 2, 8, 2, 1, "on", "panel-3"),
    (8, 3, 8, 3, 3, "off", "axis-r-2-1"), (8, 5, 8, 5, 3, "off", "axis-l-2-2"),
    (8, 6, 8, 6, 1, "on", "panel-4"), (8, 7, 8, 7, 3, "off", "axis-r-2-2"),
    (9, 2, 9, 2, 3, "off", "axis-b-1-2"), (9, 6, 9, 6, 3, "off", "axis-b-2-2"),
}

# 7 classes -> 3x3 grid with the bottom row holding a single dangling panel.
WRAP2_CLASS_CELLS = {
    (1, 2, 1, 2, 3, "off", "axis-t-1-1"), (1, 6, 1, 6, 3, "off", "axis-t-2-1"),
    (1, 10, 1, 10, 3, "off", "axis-t-3-1"),
    (2, 2, 2, 2, 2, "on", "strip-t-1-1"), (2, 6, 2, 6, 2, "on", "strip-t-2-2"),
    (2, 10, 2, 10, 2, "on", "strip-t-3-3"),
    (3, 1, 3, 1, 3, "off", "axis-l-1-1"), (3, 2, 3, 2, 1, "on", "panel-1"),
    (3, 3, 3, 3, 3, "off", "axis-r-1-1"), (3, 5, 3, 5, 3, "off", "axis-l-1-2"),
    (3, 6, 3, 6, 1, "on", "panel-2"), (3, 7, 3, 7, 3, "off", "axis-r-1-2"),
    (3, 9, 3, 9, 3, "off", "axis-l-1-3"), (3, 10, 3, 10, 1, "on", "panel-3"),
    (3, 11, 3, 11, 3, "off", "axis-r-1-3"),
    (4, 2, 4, 2, 3, "off", "axis-b-1-1"), (4, 6, 4, 6, 3, "off", "axis-b-2-1"),
    (4, 10, 4, 10, 3, "off", "axis-b-3-1"),
    (6, 2, 6, 2, 3, "off", "axis-t-1-2"), (6, 6, 6, 6, 3, "off", "axis-t-2-2"),
    (6, 10, 6, 10, 3, "off", "axis-t-3-2"),
    (7, 2, 7, 2, 2, "on", "strip-t-4-4"), (7, 6, 7, 6, 2, "on", "strip-t-5-5"),
    (7, 10, 7, 10, 2, "on", "strip-t-6-6"),
    (8, 1, 8, 1, 3, "off", "axis-l-2-1"), (8, 2, 8, 2, 1, "on", "panel-4"),
    (8, 3, 8, 3, 3, "off", "axis-r-2-1"), (8, 5, 8, 5, 3, "off", "axis-l-2-2"),
    (8, 6, 8, 6, 1, "on", "panel-5"), (8, 7, 8, 7, 3, "off", "axis-r-2-2"),
    (8, 9, 8, 9, 3, "off", "axis-l-2-3"), (8, 10, 8, 10, 1, "on", "panel-6"),
    (8, 11, 8, 11, 3, "off", "axis-r-2-3"),
    (9, 2, 9, 2, 3, "off", "axis-b-1-2"), (9, 6, 9, 6, 3, "off", "axis-b-2-2"),
    (9, 10, 9, 10, 3, "off", "axis-b-3-2"),
    (11, 2, 11, 2, 3, "off", "axis-t-1-3"), (11, 6, 11, 6, 3, "off", "axis-t-2-3"),
    (11, 10, 11, 10, 3, "off", "axis-t-3-3"),
    (12, 2, 12, 2, 2, "on", "strip-t-7-7"),
    (13, 1, 13, 1, 3, "off", "axis-l-3-1"), (13, 2, 13, 2, 1, "on", "panel-7"),
    (13, 3, 13, 3, 3, "off", "axis-r-3-1"), (13, 5, 13, 5, 3, "off", "axis-l-3-2"),
    (13, 7, 13, 7, 3, "off", "axis-r-3-2"), (13, 9, 13, 9, 3, "off", "axis-l-3-3"),
    (13, 11, 13, 11, 3, "off", "axis-r-3-3"),
    (14, 2, 14, 2, 3, "off", "axis-b-1-3"), (14, 6, 14, 6, 3, "off", "axis-b-2-3"),
    (14, 10, 14, 10, 3, "off", "axis-b-3-3"),
}


# ---------------------------------------------------------------------------
# Constructor / params
# ---------------------------------------------------------------------------
def test_constructor_returns_facetwrap2():
    f = facet_wrap2("cyl")
    assert isinstance(f, FacetWrap2)
    assert f.strip is not None
    assert f.shrink is True


def test_params_normalised():
    f = facet_wrap2("class", axes="all", remove_labels="all")
    assert f.params["axes"] == {"x": True, "y": True}
    assert f.params["rmlab"] == {"x": True, "y": True}
    assert f.params["facets"] == ["class"]
    # trim_blank default -> dim None.
    assert f.params["dim"] is None


def test_trim_blank_false_sets_dim():
    f = facet_wrap2("class", nrow=4, ncol=3, trim_blank=False)
    assert f.params["dim"] == [4, 3]


# ---------------------------------------------------------------------------
# empties-mask diff helpers (R: apply(empties, MARGIN, \(x) c(diff(x)==t, FALSE)))
# ---------------------------------------------------------------------------
def test_diff_down_bottom_and_top():
    # 3x3 grid, last row only col 1 occupied (panels at all of row1,row2 + (3,1)).
    empties = np.array(
        [
            [False, False, False],
            [False, False, False],
            [False, True, True],
        ]
    )
    # bottom_empty: panel whose cell below is empty -> row 2 (0-based) cols 1,2.
    be = _diff_down(empties, target=1, append_last=False)
    assert be[1, 1] and be[1, 2]
    assert be.sum() == 2
    # top_empty: panel whose cell above is empty -> none here.
    te = _diff_down(empties, target=-1, append_last=True)
    assert te.sum() == 0


def test_diff_right_left_and_right():
    empties = np.array(
        [
            [False, False, False],
            [False, False, False],
            [False, True, True],
        ]
    )
    # right_empty: panel whose right neighbour is empty -> (2,0).
    re_ = _diff_right(empties, target=1, append_last=False)
    assert re_[2, 0]
    assert re_.sum() == 1
    # left_empty: panel whose left neighbour is empty -> none (col 0 is filled).
    le = _diff_right(empties, target=-1, append_last=True)
    assert le.sum() == 0


def test_diff_down_single_row_safe():
    assert _diff_down(np.array([[False, True]]), 1, False).sum() == 0


# ---------------------------------------------------------------------------
# setup_layout (CoordFlip swap)
# ---------------------------------------------------------------------------
def test_setup_layout_noop_for_cartesian():
    f = facet_wrap2("cyl", scales="free")
    layout = pd.DataFrame(
        {"PANEL": [1, 2], "ROW": [1, 1], "COL": [1, 2], "SCALE_X": [1, 2], "SCALE_Y": [1, 1]}
    )
    out = f.setup_layout(layout, CoordCartesian(), f.params)
    assert list(out["SCALE_X"]) == [1, 2]


def test_setup_layout_flip_swaps_scales():
    f = facet_wrap2("cyl", scales="free")
    layout = pd.DataFrame(
        {"PANEL": [1, 2, 3], "ROW": [1, 1, 2], "COL": [1, 2, 1],
         "SCALE_X": [1, 2, 3], "SCALE_Y": [1, 1, 1]}
    )
    out = f.setup_layout(layout, CoordFlip(), f.params)
    # scales="free" -> both free -> both SCALE_* become per-panel under CoordFlip.
    assert list(out["SCALE_X"]) == [1, 2, 3]
    assert list(out["SCALE_Y"]) == [1, 2, 3]


def test_setup_layout_flip_free_x_only():
    f = facet_wrap2("cyl", scales="free_x")
    layout = pd.DataFrame(
        {"PANEL": [1, 2, 3], "ROW": [1, 1, 2], "COL": [1, 2, 1],
         "SCALE_X": [1, 2, 3], "SCALE_Y": [1, 1, 1]}
    )
    out = f.setup_layout(layout, CoordFlip(), f.params)
    # free_x only -> SCALE_X per-panel, SCALE_Y forced to all 1.
    assert list(out["SCALE_X"]) == [1, 2, 3]
    assert set(out["SCALE_Y"]) == {1}


# ---------------------------------------------------------------------------
# draw_panels layout-table parity (the acceptance test)
# ---------------------------------------------------------------------------
def test_draw_panels_basic_layout(mpg):
    p = ggplot(mpg, aes("displ", "hwy")) + geom_point() + facet_wrap2("cyl")
    gt = _capture_panel_table(p)
    assert _layout_cells(gt) == WRAP2_BASIC_CELLS


def test_draw_panels_class_dangling_layout(mpg):
    # 7 panels in a 3x3 grid -> the dangling bottom panel keeps its marginal axes
    # via the empties re-placement branch.
    p = ggplot(mpg, aes("displ", "hwy")) + geom_point() + facet_wrap2("class")
    gt = _capture_panel_table(p)
    assert _layout_cells(gt) == WRAP2_CLASS_CELLS


def test_draw_panels_class_replaces_marginal_axes(mpg):
    # The dangling panel (panel-7, row 3 col 1) must keep its bottom axis, and
    # the panels above the holes (row 2, cols 2 & 3) get a re-placed bottom-axis
    # grob.  The bottom axis grobs (axis-b) at the dangling row + the
    # hole-bordering positions must exist (R draws axis-b-1-3/2-3/3-3 in the last
    # panel row even though only col-1 has a real panel).
    p = ggplot(mpg, aes("displ", "hwy")) + geom_point() + facet_wrap2("class")
    gt = _capture_panel_table(p)
    cells = _layout_cells(gt)
    names = {c[6] for c in cells}
    # Last panel-row bottom axes (re-placed at all three columns).
    assert {"axis-b-1-3", "axis-b-2-3", "axis-b-3-3"} <= names
    # One null height row per panel row (3 rows).
    heights = gt._heights if hasattr(gt, "_heights") else gt.heights
    null_idx = [i for i, u in enumerate(_unit_types(heights)) if u == "null"]
    assert len(null_idx) == 3
    # The bottom band of the final row is non-zero (dangling panel's axis).
    vals = list(heights.values)
    assert vals[-1] > 0.0


def test_draw_panels_basic_size_unit_types(mpg):
    p = ggplot(mpg, aes("displ", "hwy")) + geom_point() + facet_wrap2("cyl")
    gt = _capture_panel_table(p)
    widths = gt._widths if hasattr(gt, "_widths") else gt.widths
    heights = gt._heights if hasattr(gt, "_heights") else gt.heights
    # R: widths 2 null (2 panel cols), heights 2 null (2 panel rows).
    assert _null_count(widths) == 2
    assert _null_count(heights) == 2
    assert _unit_types(widths) == ["cm", "null", "cm", "points", "cm", "null", "cm"]
    assert _unit_types(heights) == [
        "cm", "cm", "null", "cm", "points", "cm", "cm", "null", "cm",
    ]


def test_draw_panels_respect_false_default(mpg):
    p = ggplot(mpg, aes("displ", "hwy")) + geom_point() + facet_wrap2("cyl")
    gt = _capture_panel_table(p)
    respect = getattr(gt, "respect", getattr(gt, "_respect", None))
    assert respect is False


# ---------------------------------------------------------------------------
# setup_aspect_ratio struct
# ---------------------------------------------------------------------------
def test_setup_aspect_ratio_struct():
    f = facet_wrap2("cyl")
    ar = f.setup_aspect_ratio(_FakeCoord(None), {"x": True, "y": True}, None, [{}])
    assert isinstance(ar, AspectRatio)
    assert ar.value == 1.0 and ar.respect is False
    ar2 = f.setup_aspect_ratio(_FakeCoord(0.5), {"x": False, "y": False}, None, [{}])
    assert ar2.value == 0.5 and ar2.respect is True


class _FakeCoord:
    clip = "on"

    def __init__(self, aspect_val):
        self._aspect = aspect_val

    def aspect(self, ranges):
        return self._aspect

    def is_free(self):
        return True


# ---------------------------------------------------------------------------
# new_wrap_facets direct
# ---------------------------------------------------------------------------
def test_new_wrap_facets_direct():
    f = new_wrap_facets(
        "class", None, None, "fixed", "all", "none",
        True, "label_value", True, True, "h", "top", "vanilla", True,
    )
    assert isinstance(f, FacetWrap2)
    assert f.params["axes"] == {"x": True, "y": True}
    assert f.params["facets"] == ["class"]
