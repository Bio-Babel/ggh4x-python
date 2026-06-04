"""Tests for facet foundation helpers.

Reference values are taken from R (ggplot2 4.0.2) and baked in as regression
anchors:

- ``id`` / ``id_var`` reference values produced by ``ggplot2:::id`` (see the
  R-parity harness in the porting session).
- ``panel_cols`` / ``panel_rows`` verified against ``ggplot2:::panel_cols`` /
  ``ggplot2:::panel_rows`` on a 2x3 ``facet_grid(vs ~ cyl)`` gtable.
- ``render_axes`` structure verified against ``ggplot2:::render_axes`` (both
  ``transpose`` modes, incl. the ``x=NULL`` asymmetry).
- ``weave_tables_row`` / ``weave_tables_col`` / ``weave_panel_rows`` /
  ``weave_panel_cols`` cell positions, names and z-order verified cell-by-cell
  against the R implementations on a synthetic 2x3 panel gtable.
- ``split_heights_cm`` / ``split_widths_cm`` group-max values verified against
  ``ggh4x:::split_heights_cm`` (incl. dropped factor levels & group ordering).
- ``df_grid`` cross-product verified against ``ggh4x:::df.grid``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from grid_py import Unit, convert_height, convert_width, null_grob, rect_grob
from gtable_py import Gtable, gtable_add_grob

from ggh4x._borrowed_ggplot2 import id, id_var
from ggh4x._facet_utils import (
    df_grid,
    panel_cols,
    panel_rows,
    render_axes,
    split_heights_cm,
    split_widths_cm,
    weave_panel_cols,
    weave_panel_rows,
    weave_tables_col,
    weave_tables_row,
)


# --- id() / id_var() parity (R: ggplot2:::id) --------------------------------

ID_CASES = [
    # (DataFrame, drop, expected_ids, expected_n)
    (pd.DataFrame({"a": [1, 1, 2, 2, 3, 3], "b": [1, 2, 1, 2, 1, 2]}), False,
     [1, 2, 3, 4, 5, 6], 6),
    (pd.DataFrame({"a": [1, 1, 2, 2, 3, 3], "b": [1, 2, 1, 2, 1, 2]}), True,
     [1, 2, 3, 4, 5, 6], 6),
    (pd.DataFrame({"a": pd.Categorical(["x", "y", "x"], categories=["y", "x"]),
                   "b": pd.Categorical(["p", "p", "q"])}), False,
     [3, 1, 4], 4),
    (pd.DataFrame({"a": [1, 1, 3], "b": [1, 2, 2]}), True, [1, 2, 3], 3),
    (pd.DataFrame({"a": [1, 1, 3], "b": [1, 2, 2]}), False, [1, 2, 4], 4),
    (pd.DataFrame({"a": [1, 2, 1, 2], "b": [1, 1, 2, 2], "c": [1, 1, 1, 2]}), False,
     [1, 5, 3, 8], 8),
]


@pytest.mark.parametrize("df, drop, expected_ids, expected_n", ID_CASES)
def test_id_matches_r(df, drop, expected_ids, expected_n):
    out = id(df, drop=drop)
    assert list(map(int, out)) == expected_ids
    assert int(out.n) == expected_n


def test_id_var_single_reversed_factor():
    # R: ggplot2:::id(data.frame(a=c(3,1,2,1))) -> 3,1,2,1 ; n=3
    out = id(pd.DataFrame({"a": [3, 1, 2, 1]}), drop=False)
    assert list(map(int, out)) == [3, 1, 2, 1]
    assert int(out.n) == 3


# --- panel_cols / panel_rows (R: ggplot2:::panel_cols/panel_rows) ------------

class _FakeGT:
    """Minimal stand-in carrying a gtable-style ``.layout`` dict of parallel lists."""

    def __init__(self, layout: dict):
        self.layout = layout


def _grid_2x3_layout() -> _FakeGT:
    """Layout rows mirroring a real ``facet_grid(vs ~ cyl)`` gtable (2 rows x 3 cols).

    Panel cells live at the column/row positions ggplot2 produced: cols 7,9,11
    and rows 10,12. Non-panel rows (axes/strips/background) must be ignored.
    """
    names, t, l, b, r = [], [], [], [], []
    names.append("background"); t.append(1); l.append(1); b.append(18); r.append(13)
    for ri, row in enumerate((10, 12)):
        for ci, col in enumerate((7, 9, 11)):
            names.append(f"panel-{ci + 1}-{ri + 1}")
            t.append(row); l.append(col); b.append(row); r.append(col)
    names.append("axis-l-1"); t.append(10); l.append(6); b.append(10); r.append(6)
    names.append("strip-t-1"); t.append(9); l.append(7); b.append(9); r.append(7)
    return _FakeGT({"name": names, "t": t, "l": l, "b": b, "r": r})


def test_panel_cols_matches_r():
    pc = panel_cols(_grid_2x3_layout())
    assert list(zip(pc["l"], pc["r"])) == [(7, 7), (9, 9), (11, 11)]


def test_panel_rows_matches_r():
    pr = panel_rows(_grid_2x3_layout())
    assert list(zip(pr["t"], pr["b"])) == [(10, 10), (12, 12)]


def test_panel_helpers_ignore_non_panel_grobs():
    gt = _FakeGT({"name": ["background", "axis-l", "guide-box"],
                  "t": [1, 2, 3], "l": [1, 2, 3], "b": [1, 2, 3], "r": [1, 2, 3]})
    assert len(panel_cols(gt)) == 0
    assert len(panel_rows(gt)) == 0


# --- render_axes (R: ggplot2:::render_axes) ----------------------------------

class _FakeCoord:
    """Minimal coord exposing render_axis_h / render_axis_v.

    Returns sentinel marker strings so the loop / transpose plumbing can be
    asserted without depending on the real grob pipeline.
    """

    def render_axis_h(self, pp, theme):
        return {"top": f"top:{pp}", "bottom": f"bottom:{pp}"}

    def render_axis_v(self, pp, theme):
        return {"left": f"left:{pp}", "right": f"right:{pp}"}


def test_render_axes_non_transpose_groups_per_panel():
    # R: render_axes(x, y, coord, theme, FALSE) -> $x[[i]]$top/bottom per panel.
    coord = _FakeCoord()
    ax = render_axes([1, 2], [1, 2], coord, theme=None, transpose=False)
    assert set(ax.keys()) == {"x", "y"}
    assert len(ax["x"]) == 2 and len(ax["y"]) == 2
    assert ax["x"][0] == {"top": "top:1", "bottom": "bottom:1"}
    assert ax["x"][1] == {"top": "top:2", "bottom": "bottom:2"}
    assert ax["y"][0] == {"left": "left:1", "right": "right:1"}


def test_render_axes_transpose_groups_per_side():
    # R transpose=TRUE -> $x$top/$x$bottom/$y$left/$y$right as per-panel lists.
    coord = _FakeCoord()
    ax = render_axes([1, 2, 3], [1, 2, 3], coord, theme=None, transpose=True)
    assert set(ax.keys()) == {"x", "y"}
    assert set(ax["x"].keys()) == {"top", "bottom"}
    assert set(ax["y"].keys()) == {"left", "right"}
    assert ax["x"]["bottom"] == ["bottom:1", "bottom:2", "bottom:3"]
    assert ax["x"]["top"] == ["top:1", "top:2", "top:3"]
    assert ax["y"]["left"] == ["left:1", "left:2", "left:3"]


def test_render_axes_x_none_non_transpose_omits_x():
    # R: when x=NULL & !transpose, axes$x is never assigned -> only 'y' present.
    coord = _FakeCoord()
    ax = render_axes(None, [1, 2], coord, theme=None, transpose=False)
    assert set(ax.keys()) == {"y"}
    assert len(ax["y"]) == 2


def test_render_axes_x_none_transpose_keeps_empty_x():
    # R: transpose block rebuilds from lapply(NULL, ...) -> x present, empty lists.
    coord = _FakeCoord()
    ax = render_axes(None, [1, 2], coord, theme=None, transpose=True)
    assert set(ax.keys()) == {"x", "y"}
    assert ax["x"]["top"] == [] and ax["x"]["bottom"] == []
    assert len(ax["y"]["left"]) == 2


# --- weave_tables_row / weave_tables_col (R: ggplot2 internals) --------------

def _panel_table_2x3() -> Gtable:
    """A bare 2 (rows) x 3 (cols) gtable of named panel cells, byrow placement."""
    gt = Gtable(widths=Unit([1, 1, 1], "null"), heights=Unit([1, 1], "null"))
    for r in (1, 2):
        for c in (1, 2, 3):
            gt = gtable_add_grob(gt, rect_grob(), t=r, l=c,
                                 name=f"panel-{c}-{r}", z=1)
    return gt


def _cells(gt: Gtable) -> set:
    lay = gt.layout
    return {
        (n, t, l, b, r, z)
        for n, t, l, b, r, z in zip(
            lay["name"], lay["t"], lay["l"], lay["b"], lay["r"], lay["z"]
        )
    }


def test_weave_tables_col_left_matches_r():
    # R weave_tables_col(.., -1, .., "axis-l", 3) on a 2x3 panel table.
    gm = [[rect_grob() for _ in range(3)] for _ in range(2)]
    gt = weave_tables_col(_panel_table_2x3(), gm, -1,
                          Unit([0.5, 0.5, 0.5], "cm"), "axis-l", 3)
    cells = _cells(gt)
    # Panels shifted to even columns; axes occupy odd columns at z=3.
    assert ("panel-1-1", 1, 2, 1, 2, 1) in cells
    assert ("panel-3-2", 2, 6, 2, 6, 1) in cells
    expected_axes = {
        ("axis-l-1-1", 1, 1, 1, 1, 3), ("axis-l-1-2", 1, 3, 1, 3, 3),
        ("axis-l-1-3", 1, 5, 1, 5, 3), ("axis-l-2-1", 2, 1, 2, 1, 3),
        ("axis-l-2-2", 2, 3, 2, 3, 3), ("axis-l-2-3", 2, 5, 2, 5, 3),
    }
    assert expected_axes <= cells
    assert len(gt.widths) == 6 and len(gt.heights) == 2


def test_weave_tables_row_bottom_matches_r():
    # R weave_tables_row(.., 0, .., "axis-b", 3) on a 2x3 panel table.
    gm = [[rect_grob() for _ in range(3)] for _ in range(2)]
    gt = weave_tables_row(_panel_table_2x3(), gm, 0,
                          Unit([0.7, 0.7], "cm"), "axis-b", 3)
    cells = _cells(gt)
    assert ("panel-1-1", 1, 1, 1, 1, 1) in cells
    assert ("panel-1-2", 3, 1, 3, 1, 1) in cells
    expected_axes = {
        ("axis-b-1-1", 2, 1, 2, 1, 3), ("axis-b-2-1", 2, 2, 2, 2, 3),
        ("axis-b-3-1", 2, 3, 2, 3, 3), ("axis-b-1-2", 4, 1, 4, 1, 3),
        ("axis-b-2-2", 4, 2, 4, 2, 3), ("axis-b-3-2", 4, 3, 4, 3, 3),
    }
    assert expected_axes <= cells
    assert len(gt.heights) == 4 and len(gt.widths) == 3


def test_weave_tables_col_no_table2_only_inserts_columns():
    gt = weave_tables_col(_panel_table_2x3(), None, -1,
                          Unit([0.5, 0.5, 0.5], "cm"), "axis-l", 3)
    # Only empty columns added; no axis grobs.
    assert len(gt.widths) == 6
    assert not any(str(n).startswith("axis") for n in gt.layout["name"])


# --- weave_panel_rows / weave_panel_cols (R: ggh4x utils_gtable.R) -----------

def _panel_df_2x3() -> pd.DataFrame:
    df = pd.DataFrame({
        "t": [1, 1, 1, 2, 2, 2], "b": [1, 1, 1, 2, 2, 2],
        "l": [1, 2, 3, 1, 2, 3], "r": [1, 2, 3, 1, 2, 3],
    })
    df["grobs"] = [rect_grob() for _ in range(6)]
    return df


def test_weave_panel_rows_matches_r():
    gt = weave_panel_rows(_panel_table_2x3(), _panel_df_2x3(), -1,
                          Unit([0.5, 0.5], "cm"), "strip-t", z=2)
    cells = _cells(gt)
    assert ("panel-1-1", 2, 1, 2, 1, 1) in cells
    assert ("panel-3-2", 4, 3, 4, 3, 1) in cells
    # All strips land on row 1, two per panel column.
    expected = {
        ("strip-t-1-1", 1, 1, 1, 1, 2), ("strip-t-4-4", 1, 1, 1, 1, 2),
        ("strip-t-2-2", 1, 2, 1, 2, 2), ("strip-t-5-5", 1, 2, 1, 2, 2),
        ("strip-t-3-3", 1, 3, 1, 3, 2), ("strip-t-6-6", 1, 3, 1, 3, 2),
    }
    assert expected <= cells
    assert len(gt.heights) == 4 and len(gt.widths) == 3


def test_weave_panel_cols_matches_r():
    gt = weave_panel_cols(_panel_table_2x3(), _panel_df_2x3(), 0,
                          Unit([0.6, 0.6, 0.6], "cm"), "strip-r", z=2)
    cells = _cells(gt)
    assert ("panel-1-1", 1, 1, 1, 1, 1) in cells
    assert ("panel-3-1", 1, 5, 1, 5, 1) in cells
    expected = {
        ("strip-r-1-1", 1, 2, 1, 2, 2), ("strip-r-4-4", 1, 2, 1, 2, 2),
        ("strip-r-2-2", 1, 4, 1, 4, 2), ("strip-r-5-5", 1, 4, 1, 4, 2),
        ("strip-r-3-3", 1, 6, 1, 6, 2), ("strip-r-6-6", 1, 6, 1, 6, 2),
    }
    assert expected <= cells
    assert len(gt.widths) == 6 and len(gt.heights) == 2


# --- split_heights_cm / split_widths_cm (R: ggh4x utils_grid.R) --------------

def _grobs_for_split() -> list:
    return [
        rect_grob(height=Unit(1, "cm"), width=Unit(2, "cm")),
        rect_grob(height=Unit(3, "cm"), width=Unit(1, "cm")),
        rect_grob(height=Unit(2, "cm"), width=Unit(5, "cm")),
        rect_grob(height=Unit(4, "cm"), width=Unit(0.5, "cm")),
    ]


def _cm_h(u: Unit) -> list:
    return [round(float(v), 3) for v in convert_height(u, "cm", True)]


def _cm_w(u: Unit) -> list:
    return [round(float(v), 3) for v in convert_width(u, "cm", True)]


def test_split_heights_cm_group_max_matches_r():
    # R: split_heights_cm -> max height per group -> [3, 4].
    out = split_heights_cm(_grobs_for_split(), ["a", "a", "b", "b"])
    assert _cm_h(out) == [3.0, 4.0]


def test_split_widths_cm_group_max_matches_r():
    # R: split_widths_cm -> max width per group -> [2, 5].
    out = split_widths_cm(_grobs_for_split(), ["a", "a", "b", "b"])
    assert _cm_w(out) == [2.0, 5.0]


def test_split_orders_by_sorted_unique_keys():
    # R split() orders groups by sort(unique(split)) -> group 'a' first.
    out = split_heights_cm(_grobs_for_split(), ["b", "b", "a", "a"])
    assert _cm_h(out) == [4.0, 3.0]


def test_split_drops_unused_factor_levels():
    # R split(.., drop = TRUE) drops the unused 'c' level.
    split = pd.Categorical(["a", "a", "b", "b"], categories=["a", "b", "c"])
    out = split_heights_cm(_grobs_for_split(), split)
    assert len(out) == 2 and _cm_h(out) == [3.0, 4.0]


def test_split_respects_factor_level_order():
    # Factor level order (not sorted-unique) drives group ordering.
    split = pd.Categorical(["a", "a", "b", "b"], categories=["b", "a"])
    out = split_heights_cm(_grobs_for_split(), split)
    # 'b' level first -> max(2,4)=4 ; then 'a' -> max(1,3)=3.
    assert _cm_h(out) == [4.0, 3.0]


# --- df_grid (R: ggh4x df.grid) ----------------------------------------------

def test_df_grid_cross_product_matches_r():
    a = pd.DataFrame({"x": ["p", "q"]})
    b = pd.DataFrame({"y": [10, 20, 30]})
    out = df_grid(a, b)
    # R expand.grid(i_a, i_b): i_a fastest -> a tiled, b repeated.
    assert list(out["x"]) == ["p", "q", "p", "q", "p", "q"]
    assert list(out["y"]) == [10, 10, 20, 20, 30, 30]
    assert len(out) == 6


def test_df_grid_multi_column_a():
    a = pd.DataFrame({"x": ["p", "q"], "z": [1, 2]})
    b = pd.DataFrame({"y": [10, 20, 30]})
    out = df_grid(a, b)
    assert list(out.columns) == ["x", "z", "y"]
    assert list(out["z"]) == [1, 2, 1, 2, 1, 2]


def test_df_grid_empty_a_returns_b():
    a = pd.DataFrame({"x": []})
    b = pd.DataFrame({"y": [10, 20, 30]})
    out = df_grid(a, b)
    assert list(out["y"]) == [10, 20, 30]
    assert "x" not in out.columns


def test_df_grid_none_a_returns_b():
    b = pd.DataFrame({"y": [10, 20, 30]})
    assert df_grid(None, b).equals(b)


def test_df_grid_none_b_returns_a():
    a = pd.DataFrame({"x": ["p", "q"]})
    assert df_grid(a, None).equals(a)
