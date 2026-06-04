r"""Tests for :mod:`ggh4x.facet_manual`.

Reference values are taken from R (ggplot2 4.0.2 + ggh4x 0.3.1.9000) run through
``Rscript`` and baked in as regression anchors.

- **validate_design** char path: ``"A##\nAB#\n#BC\n##C"`` -> the 4x3 integer
  matrix ``[[1,nan,nan],[1,2,nan],[nan,2,3],[nan,nan,3]]`` with
  ``design_names = ["A","B","C"]``.  Matrix path renumbers / trims.

- **compute_layout** (char design ``~cyl``)::

      .TOP .RIGHT .BOTTOM .LEFT PANEL cyl SCALE_X SCALE_Y
         1      1       2     1     1   4       1       1
         2      2       3     2     2   6       1       1
         3      3       4     3     3   8       1       1

- **draw_panels** gtable panel spans (char design ``~cyl``) match R exactly::

      panel-1: t=10 b=14 l=7  r=7
      panel-2: t=14 b=19 l=11 r=11
      panel-3: t=19 b=22 l=15 r=15

  plus 12 axis grobs and 3 strip grobs.

- Matrix design ``[[1,2],[3,3]]`` (``~cyl``): panel-3 spans columns ``l=7 r=11``.

- **do_purge** / **restrict_axes** verified against R's ``vec_unique`` / split
  semantics directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggplot2_py import aes, geom_point, ggplot, ggplotGrob, ggplot_build
from ggplot2_py.facet import FacetNull
from grid_py import Unit, null_grob

from ggh4x.facet_manual import (
    FacetManual,
    _do_purge,
    _restrict_axes,
    _validate_design,
    facet_manual,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
DESIGN_CHAR = "\n A##\n AB#\n #BC\n ##C\n"


def _mtcars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mpg": [21, 22.8, 18.7, 21.4, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 33.9],
            "wt": [2.6, 2.3, 3.4, 3.2, 3.5, 3.6, 3.2, 3.1, 3.4, 3.4, 4.1, 1.9],
            "cyl": [6, 4, 8, 6, 6, 8, 4, 4, 6, 8, 6, 4],
        }
    )


@pytest.fixture
def mtcars() -> pd.DataFrame:
    return _mtcars()


def _panel_frame(g) -> pd.DataFrame:
    lay = g.layout
    if not isinstance(lay, pd.DataFrame):
        lay = pd.DataFrame({k: list(v) for k, v in lay.items()})
    return lay


def _panels(g) -> pd.DataFrame:
    df = _panel_frame(g)
    mask = df["name"].astype(str).str.match("^panel")
    return df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# validate_design
# ---------------------------------------------------------------------------
def test_validate_design_char():
    d = _validate_design(DESIGN_CHAR, trim=True)
    expected = np.array(
        [
            [1, np.nan, np.nan],
            [1, 2, np.nan],
            [np.nan, 2, 3],
            [np.nan, np.nan, 3],
        ]
    )
    assert np.array_equal(d.matrix, expected, equal_nan=True)
    assert d.names == ["A", "B", "C"]


def test_validate_design_matrix():
    d = _validate_design(np.array([[1, 2], [3, 3]]), trim=True)
    assert np.array_equal(d.matrix, np.array([[1.0, 2.0], [3.0, 3.0]]))
    assert d.names is None


def test_validate_design_renumbers_and_sorts():
    # Non-contiguous numeric ids get renumbered by sort(unique).
    d = _validate_design(np.array([[5, 5], [9, 2]]), trim=True)
    # sort(unique) = [2,5,9] -> match -> 2->1, 5->2, 9->3
    assert np.array_equal(d.matrix, np.array([[2.0, 2.0], [3.0, 1.0]]))


def test_validate_design_trims_empty():
    d = _validate_design("\n #A#\n #B#\n", trim=True)
    # Empty left/right columns trimmed -> single column.
    assert d.matrix.shape == (2, 1)


def test_validate_design_none_raises():
    with pytest.raises(ValueError):
        _validate_design(None)


def test_validate_design_non_rectangular_raises():
    with pytest.raises(ValueError):
        _validate_design("\n AB\n ABC\n")


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------
def test_facet_manual_returns_facetmanual():
    fm = facet_manual("cyl", DESIGN_CHAR)
    assert isinstance(fm, FacetManual)
    assert fm.params["nrow"] == 4
    assert fm.params["ncol"] == 3


def test_facet_manual_empty_facets_returns_null():
    fm = facet_manual([], DESIGN_CHAR)
    assert isinstance(fm, FacetNull)


def test_facet_manual_widths_heights_coerced():
    fm = facet_manual("cyl", np.array([[1, 2], [3, 3]]), widths=[2, 1], heights=[2, 1])
    assert fm.params["widths"] is not None
    assert fm.params["heights"] is not None
    # recycled to ncol / nrow
    assert len(fm.params["widths"]) == 2
    assert len(fm.params["heights"]) == 2


# ---------------------------------------------------------------------------
# compute_layout (char design)
# ---------------------------------------------------------------------------
def test_compute_layout_char(mtcars):
    fm = facet_manual("cyl", DESIGN_CHAR)
    layout = fm.compute_layout([mtcars], fm.params)
    assert list(layout[".TOP"]) == [1, 2, 3]
    assert list(layout[".RIGHT"]) == [1, 2, 3]
    assert list(layout[".BOTTOM"]) == [2, 3, 4]
    assert list(layout[".LEFT"]) == [1, 2, 3]
    assert list(layout["PANEL"].astype(int)) == [1, 2, 3]
    assert list(layout["cyl"]) == [4, 6, 8]
    assert list(layout["SCALE_X"]) == [1, 1, 1]


def test_compute_layout_matrix(mtcars):
    fm = facet_manual("cyl", np.array([[1, 2], [3, 3]]))
    layout = fm.compute_layout([mtcars], fm.params)
    assert list(layout[".TOP"]) == [1, 1, 2]
    assert list(layout[".RIGHT"]) == [1, 2, 2]
    assert list(layout[".BOTTOM"]) == [1, 1, 2]
    assert list(layout[".LEFT"]) == [1, 2, 1]
    assert list(layout["cyl"]) == [4, 6, 8]


def test_compute_layout_free_scales(mtcars):
    fm = facet_manual("cyl", DESIGN_CHAR, scales="free")
    layout = fm.compute_layout([mtcars], fm.params)
    assert list(layout["SCALE_X"]) == [1, 2, 3]
    assert list(layout["SCALE_Y"]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# map_data
# ---------------------------------------------------------------------------
def test_map_data_assigns_panel(mtcars):
    fm = facet_manual("cyl", DESIGN_CHAR)
    layout = fm.compute_layout([mtcars], fm.params)
    mapped = fm.map_data(mtcars, layout, fm.params)
    assert "PANEL" in mapped.columns
    # cyl 4 -> panel 1, 6 -> 2, 8 -> 3
    cyl_to_panel = dict(zip(mapped["cyl"], mapped["PANEL"].astype(int)))
    assert cyl_to_panel[4] == 1
    assert cyl_to_panel[6] == 2
    assert cyl_to_panel[8] == 3


# ---------------------------------------------------------------------------
# draw_panels (full render) -- the multi-cell span gtable
# ---------------------------------------------------------------------------
def test_draw_panels_char_spans(mtcars):
    p = ggplot(mtcars, aes("mpg", "wt")) + geom_point() + facet_manual("cyl", DESIGN_CHAR)
    g = ggplotGrob(p)
    panels = _panels(g)
    rows = {
        name: (int(t), int(b), int(l), int(r))
        for name, t, b, l, r in zip(
            panels["name"], panels["t"], panels["b"], panels["l"], panels["r"]
        )
    }
    # R gold standard.
    assert rows["panel-1"] == (10, 14, 7, 7)
    assert rows["panel-2"] == (14, 19, 11, 11)
    assert rows["panel-3"] == (19, 22, 15, 15)

    df = _panel_frame(g)
    assert int(df["name"].astype(str).str.contains("axis").sum()) == 12
    assert int(df["name"].astype(str).str.contains("strip").sum()) == 3


def test_draw_panels_matrix_span(mtcars):
    p = (
        ggplot(mtcars, aes("mpg", "wt"))
        + geom_point()
        + facet_manual("cyl", np.array([[1, 2], [3, 3]]))
    )
    g = ggplotGrob(p)
    panels = _panels(g)
    rows = {
        name: (int(t), int(b), int(l), int(r))
        for name, t, b, l, r in zip(
            panels["name"], panels["t"], panels["b"], panels["l"], panels["r"]
        )
    }
    # panel-3 spans columns 7..11 (the wide bottom panel).
    assert rows["panel-1"] == (10, 10, 7, 7)
    assert rows["panel-2"] == (10, 10, 11, 11)
    assert rows["panel-3"] == (15, 15, 7, 11)


def test_draw_panels_free_scales(mtcars):
    p = (
        ggplot(mtcars, aes("mpg", "wt"))
        + geom_point()
        + facet_manual("cyl", DESIGN_CHAR, scales="free")
    )
    g = ggplotGrob(p)
    assert len(_panels(g)) == 3


def test_draw_panels_axes_all_and_remove_labels(mtcars):
    # axes="all" repeats inner axes; remove_labels="all" purges their labels
    # via the restrict_axes(purge_guide_labels) branch.
    p = (
        ggplot(mtcars, aes("mpg", "wt"))
        + geom_point()
        + facet_manual("cyl", DESIGN_CHAR, axes="all", remove_labels="all")
    )
    g = ggplotGrob(p)
    assert len(_panels(g)) == 3


def test_draw_panels_transposed_with_sizes(mtcars):
    design = np.array([[1, 2], [3, 3]]).T  # transpose
    p = (
        ggplot(mtcars, aes("mpg", "wt"))
        + geom_point()
        + facet_manual("cyl", design, widths=[2, 1], heights=[2, 1], respect=True)
    )
    g = ggplotGrob(p)
    assert getattr(g, "respect", None) is True
    # Renders without error and produces 3 panels.
    assert len(_panels(g)) == 3


# ---------------------------------------------------------------------------
# do_purge (R facet_manual.R:455-466)
# ---------------------------------------------------------------------------
def test_do_purge_one_to_one():
    # Distinct (a,b) with a, b each unique -> True.
    assert _do_purge([1, 2, 3], [1, 2, 3]) is True
    # Repeated a-value -> not 1:1.
    assert _do_purge([1, 1, 2], [1, 2, 3]) is False


def test_do_purge_single_pair():
    assert _do_purge([1], [1], check_disjoint=True) is True


def test_do_purge_check_disjoint():
    # Non-overlapping spans (a=top, b=bottom) -> True.
    assert _do_purge([1, 3], [2, 4], check_disjoint=True) is True
    # Overlapping spans -> False.
    assert _do_purge([1, 2], [3, 4], check_disjoint=True) is False


# ---------------------------------------------------------------------------
# restrict_axes (R facet_manual.R:442-453)
# ---------------------------------------------------------------------------
def test_restrict_axes_value_restrictor():
    axes = ["a", "b", "c"]
    # group by `by`; keep min(position) per group.
    # by = [1,1,2]; position = [1,2,1]: group 1 keeps position 1 (index 0),
    # group 2 keeps index 2. Index 1 is blanked.
    blank = null_grob()
    out = _restrict_axes(axes, position=[1, 2, 1], by=[1, 1, 2], which_fun=min, restrictor=blank)
    assert out[0] == "a"
    assert out[1] is blank
    assert out[2] == "c"


def test_restrict_axes_callable_restrictor():
    calls = []

    def purger(g):
        calls.append(g)
        return "PURGED"

    axes = ["a", "b", "c"]
    out = _restrict_axes(axes, position=[1, 2, 1], by=[1, 1, 2], which_fun=min, restrictor=purger)
    assert out[1] == "PURGED"
    assert calls == ["b"]
