r"""Tests for :mod:`ggh4x.panel_scales`.

Reference values are taken from R (ggplot2 4.0.2 + ggh4x 0.3.1.9000) run through
``Rscript`` and baked in as regression anchors:

- **force_panelsizes acceptance test** (the flagged biggest risk): the gtable
  panel-row heights / panel-col widths must come through as ``"null"`` units in
  the forced ratio.  R gold standard for
  ``facet_grid2(rows=vs, cols=am) + force_panelsizes(rows=c(2,1), cols=c(2,1))``::

      panel rows t: 10 14   ; panel cols l: 7 11
      heights at panel rows: 2null 1null
      widths  at panel cols: 2null 1null
      respect: FALSE

- **force_panelsizes total_width** for ``facet_grid(. ~ cyl)`` with
  ``cols = c(1,2,1)`` and ``total_width = unit(12, "cm")`` -> panel col widths
  ``2.90335, 5.80670, 2.90335`` cm (ratio 1:2:1) summing to ``11.61339`` cm.

- **facetted_pos_scales** reverses the targeted panel's range (negative y-range)
  while leaving the others positive; the facet class becomes
  ``FreeScaledFacetWrap``.

- **scale_x_facet** by predicate targets the matching panel's ``SCALE_X`` id and
  applies its limits there only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggplot2_py import (
    aes,
    annotate,
    facet_grid,
    facet_wrap,
    geom_point,
    ggplot,
    ggplotGrob,
    ggplot_build,
    scale_y_continuous,
)
from grid_py import Unit, convert_width, unit_type

from ggh4x._facet_utils import panel_cols, panel_rows
from ggh4x.facet_grid2 import facet_grid2
from ggh4x.panel_scales import (
    FacettedPosScales,
    ForcedSize,
    ScaleFacet,
    at_panel,
    check_facetted_scale,
    facetted_pos_scales,
    force_panelsizes,
    is_null_unit,
    scale_x_facet,
    scale_y_facet,
    should_transform,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _mtcars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "disp": [160, 160, 108, 258, 360, 225, 360, 146, 140, 167, 200, 300],
            "mpg": [21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4],
            "vs": [0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
            "am": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "cyl": [6, 6, 4, 6, 8, 6, 8, 4, 4, 6, 8, 8],
        }
    )


@pytest.fixture
def mtcars() -> pd.DataFrame:
    return _mtcars()


def _scalar(x) -> float:
    return float(np.asarray(x).ravel()[0])


def _null_count(u: Unit, idx: int) -> int:
    """Numeric ``"null"`` value at element *idx* of *u* (0 if not a null unit)."""
    t = unit_type(u[idx])
    t = t[0] if isinstance(t, (list, tuple)) else t
    return float(u[idx]._values[0]) if t == "null" else 0.0


# ---------------------------------------------------------------------------
# force_panelsizes -- constructor / is_null_unit
# ---------------------------------------------------------------------------
def test_force_panelsizes_coerces_numeric_to_null():
    obj = force_panelsizes(rows=[2, 1], cols=[2, 1])
    assert isinstance(obj, ForcedSize)
    assert is_null_unit(obj.rows)
    assert is_null_unit(obj.cols)
    assert [v for v in obj.rows._values] == [2.0, 1.0]


def test_is_null_unit():
    assert is_null_unit(Unit([2, 1], "null"))
    assert not is_null_unit(Unit([2, 1], "cm"))
    assert not is_null_unit(5)


def test_force_panelsizes_total_width_requires_relative_cols():
    with pytest.raises(ValueError):
        force_panelsizes(cols=Unit([2, 1], "cm"), total_width=Unit(10, "cm"))


def test_force_panelsizes_total_width_must_be_unit():
    with pytest.raises(ValueError):
        force_panelsizes(cols=[1, 1], total_width=10)


def test_force_panelsizes_empty_returns_plot_unchanged(mtcars):
    p = ggplot(mtcars, aes("disp", "mpg")) + geom_point() + facet_grid2(cols="cyl")
    facet_before = p.facet
    p2 = p + ForcedSize()  # all None
    assert p2.facet is facet_before


# ---------------------------------------------------------------------------
# force_panelsizes -- the acceptance test (flagged biggest risk)
# ---------------------------------------------------------------------------
def test_force_panelsizes_null_ratio_2_to_1(mtcars):
    """Panel heights 2:1 and widths 2:1 as null units (R gold standard)."""
    p = (
        ggplot(mtcars, aes("disp", "mpg"))
        + geom_point()
        + facet_grid2(rows="vs", cols="am")
        + force_panelsizes(rows=[2, 1], cols=[2, 1])
    )
    assert type(p.facet).__name__ == "ForcedFacetGrid2"

    g = ggplotGrob(p)
    prows = panel_rows(g)
    pcols = panel_cols(g)
    t_pos = [int(v) for v in prows["t"]]
    l_pos = [int(v) for v in pcols["l"]]

    # R: panel rows t = 10 14 ; panel cols l = 7 11
    assert t_pos == [10, 14]
    assert l_pos == [7, 11]

    # Heights at panel rows are null units in ratio 2:1.
    h0 = _null_count(g.heights, t_pos[0] - 1)
    h1 = _null_count(g.heights, t_pos[1] - 1)
    assert h0 == 2.0 and h1 == 1.0
    # Widths at panel cols are null units in ratio 2:1.
    w0 = _null_count(g.widths, l_pos[0] - 1)
    w1 = _null_count(g.widths, l_pos[1] - 1)
    assert w0 == 2.0 and w1 == 1.0


def test_force_panelsizes_respect(mtcars):
    p = (
        ggplot(mtcars, aes("disp", "mpg"))
        + geom_point()
        + facet_grid2(cols="cyl")
        + force_panelsizes(cols=[1, 1, 1], respect=True)
    )
    g = ggplotGrob(p)
    assert getattr(g, "respect", None) is True


def test_force_panelsizes_recycles(mtcars):
    # cols length 1 recycled over 3 panel columns.
    p = (
        ggplot(mtcars, aes("disp", "mpg"))
        + geom_point()
        + facet_grid2(cols="cyl")
        + force_panelsizes(cols=[3])
    )
    g = ggplotGrob(p)
    pcols = panel_cols(g)
    l_pos = [int(v) for v in pcols["l"]]
    for l in l_pos:
        assert _null_count(g.widths, l - 1) == 3.0


# ---------------------------------------------------------------------------
# force_panelsizes -- total_width branch (R: 2.90335, 5.80670, 2.90335)
# ---------------------------------------------------------------------------
def test_force_panelsizes_total_width(mtcars):
    p = (
        ggplot(mtcars, aes("disp", "mpg"))
        + geom_point()
        + facet_grid(cols="cyl")
        + force_panelsizes(cols=[1, 2, 1], total_width=Unit(12, "cm"))
    )
    g = ggplotGrob(p)
    pcols = panel_cols(g)
    widths_cm = [
        _scalar(convert_width(g.widths[int(l) - 1], "cm", valueOnly=True))
        for l in pcols["l"]
    ]
    # Ratio 1:2:1.
    assert widths_cm[1] == pytest.approx(2 * widths_cm[0], rel=1e-6)
    assert widths_cm[2] == pytest.approx(widths_cm[0], rel=1e-6)
    # R sum was 11.61339 cm (12 minus inter-panel spacing).
    assert sum(widths_cm) == pytest.approx(11.61339, abs=1e-3)
    assert widths_cm[0] == pytest.approx(2.90335, abs=1e-3)


# ---------------------------------------------------------------------------
# facetted_pos_scales
# ---------------------------------------------------------------------------
def _iris() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sepal_Width": [3.5, 3.0, 3.2, 3.2, 3.2, 3.1, 3.3, 2.7, 3.0],
            "Sepal_Length": [5.1, 4.9, 4.7, 7.0, 6.4, 6.9, 6.3, 5.8, 7.1],
            "Species": (
                ["setosa"] * 3 + ["versicolor"] * 3 + ["virginica"] * 3
            ),
        }
    )


def test_facetted_pos_scales_constructor_validates():
    obj = facetted_pos_scales(y=[None, scale_y_continuous(), None])
    assert isinstance(obj, FacettedPosScales)
    # Wrong-aesthetic scale rejected.
    with pytest.raises(ValueError):
        facetted_pos_scales(y=[scale_y_continuous(), "notascale"])


def test_check_facetted_scale():
    assert check_facetted_scale([None, scale_y_continuous()], "y")
    assert not check_facetted_scale([scale_y_continuous()], "x")  # wrong aes


def test_facetted_pos_scales_reverses_target_panel():
    df = _iris()
    p = (
        ggplot(df, aes("Sepal_Width", "Sepal_Length"))
        + geom_point()
        + facet_wrap("Species", scales="free_y")
    )
    p2 = p + facetted_pos_scales(
        y=[None, scale_y_continuous(transform="reverse"), None]
    )
    assert type(p2.facet).__name__ == "FreeScaledFacetWrap"

    b = ggplot_build(p2)
    ranges = []
    for pp in b.layout.panel_params:
        yr = pp.get("y.range") if isinstance(pp, dict) else getattr(pp, "y_range", None)
        ranges.append([float(v) for v in yr])
    # Panel 2 (reverse) is negative; panels 1, 3 positive.
    assert ranges[1][0] < 0 and ranges[1][1] < 0
    assert ranges[0][0] > 0 and ranges[2][0] > 0


def test_facetted_pos_scales_empty_is_noop():
    df = _iris()
    p = (
        ggplot(df, aes("Sepal_Width", "Sepal_Length"))
        + geom_point()
        + facet_wrap("Species", scales="free_y")
    )
    facet_before = p.facet
    p2 = p + facetted_pos_scales(y=[None, None, None])
    assert p2.facet is facet_before


# ---------------------------------------------------------------------------
# should_transform
# ---------------------------------------------------------------------------
def test_should_transform_none_and_discrete():
    assert should_transform(None, ["x", "y"]) == []
    sc = scale_y_continuous()
    cols = should_transform(sc, ["x", "y", "ymin"])
    # y-position aesthetics intersected with columns.
    assert "y" in cols and "ymin" in cols and "x" not in cols


# ---------------------------------------------------------------------------
# scale_x_facet / scale_y_facet
# ---------------------------------------------------------------------------
def _mt_facet() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "disp": [160, 108, 258, 360, 225, 146, 140, 167, 200, 300, 120, 90],
            "mpg": [21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 33.9],
            "cyl": [6, 4, 6, 8, 6, 8, 4, 4, 6, 8, 6, 4],
        }
    )


def test_scale_x_facet_lowers_to_freescaled():
    df = _mt_facet()
    p = ggplot(df, aes("disp", "mpg")) + geom_point() + facet_wrap("cyl", scales="free")
    p2 = p + scale_x_facet("cyl == 8", limits=[200, 600])
    assert isinstance(scale_x_facet("cyl == 8", limits=[200, 600]), ScaleFacet)
    assert type(p2.facet).__name__ == "FreeScaledFacetWrap"
    assert len(p2.facet.new_x_scales) == 1
    assert p2.facet.new_x_scales.lhs == ["cyl == 8"]


def test_scale_x_facet_applies_limits_to_target_panel():
    df = _mt_facet()
    p = ggplot(df, aes("disp", "mpg")) + geom_point() + facet_wrap("cyl", scales="free")
    p2 = p + scale_x_facet("cyl == 8", limits=[200, 600])
    b = ggplot_build(p2)
    layout = b.layout.layout
    # Find the panel whose cyl == 8.
    target = layout.loc[layout["cyl"] == 8]
    sx = int(target["SCALE_X"].iloc[0])
    pp = b.layout.panel_params[sx - 1]
    xr = pp.get("x.range") if isinstance(pp, dict) else getattr(pp, "x_range", None)
    xr = [float(v) for v in xr]
    # Limits 200-600 expand outward; range should bracket [200, 600].
    assert xr[0] <= 200 + 1e-6 and xr[1] >= 600 - 1e-6
    assert xr[0] >= 150 and xr[1] <= 650  # not the default disp domain


def test_scale_facet_stacking_first_wins():
    df = _mt_facet()
    p = ggplot(df, aes("disp", "mpg")) + geom_point() + facet_wrap("cyl", scales="free")
    p2 = (
        p
        + scale_y_facet("cyl == 4", limits=[10, 50])
        + scale_y_facet("cyl == 4", limits=[0, 100])
    )
    # Both appended; lhs list length 2.
    assert len(p2.facet.new_y_scales) == 2
    assert p2.facet.new_y_scales.lhs == ["cyl == 4", "cyl == 4"]

    # First-added wins: cyl==4 panel uses limits 10-50 (R y.range == [8, 52]),
    # NOT the later 0-100.
    b = ggplot_build(p2)
    layout = b.layout.layout
    sy = int(layout.loc[layout["cyl"] == 4, "SCALE_Y"].iloc[0])
    pp = b.layout.panel_params[sy - 1]
    yr = pp.get("y.range") if isinstance(pp, dict) else getattr(pp, "y_range", None)
    yr = [float(v) for v in yr]
    assert yr == pytest.approx([8.0, 52.0], abs=1e-6)


def test_scale_facet_rejects_facetnull():
    df = _mt_facet()
    p = ggplot(df, aes("disp", "mpg")) + geom_point()
    with pytest.raises(ValueError):
        _ = p + scale_x_facet("PANEL == 1", limits=[0, 1])


# ---------------------------------------------------------------------------
# at_panel
# ---------------------------------------------------------------------------
def test_at_panel_clones_geom_and_renders():
    df = _mt_facet()
    p = ggplot(df, aes("disp", "mpg")) + geom_point() + facet_wrap("cyl")
    anno = annotate("text", x=200, y=20, label="hi")
    p2 = p + at_panel(anno, "PANEL == 1")
    # Geom is a clone (still a text geom) and renders.
    assert "Text" in type(p2.layers[-1].geom).__name__
    g = ggplotGrob(p2)
    assert g is not None


def test_at_panel_missing_expr_raises():
    df = _mt_facet()
    anno = annotate("text", x=1, y=1, label="x")
    with pytest.raises(ValueError):
        at_panel(anno, None)


def test_at_panel_bare_list_recurses():
    anno = annotate("text", x=1, y=1, label="x")
    out = at_panel([anno], "PANEL == 1")
    assert isinstance(out, list) and len(out) == 1
