"""Tests for the ggh4x coord/position ports.

Covers:

* :mod:`ggh4x.position_lineartrans` — matrix construction (all
  scale/shear/angle/custom-M branches) and transformed coordinates, verified
  against R ggh4x gold-standard fixtures (column-major matrix decode).
* :mod:`ggh4x.position_disjoint_ranges` — sweep-line disjoint-bin assignment for
  all three group branches plus the strict-``<`` boundary semantics, verified
  against R ``PositionDisjointRanges$compute_panel`` output.
* :mod:`ggh4x.coord_axes_inside` — constructor / theme building, ``is_free`` /
  ``aspect``, the ``_replace_vp_coord`` helper, and the full
  ``render_bg`` pipeline (interior-axis repositioning to the origin NPC
  coordinate, including ``oob_squish`` clamping).

The R fixtures embedded below were produced with ggplot2 4.0.2 + ggh4x via the
ggrepel-dev Rscript (see the task R-comparison scripts).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ggh4x.coord_axes_inside import (
    CoordAxesInside,
    _replace_vp_coord,
    coord_axes_inside,
)
from ggh4x.position_disjoint_ranges import (
    PositionDisjointRanges,
    position_disjoint_ranges,
)
from ggh4x.position_lineartrans import (
    PositionLinearTrans,
    position_lineartrans,
)


# ===========================================================================
# PositionLinearTrans
# ===========================================================================

# R fixtures: M flattened row-major [r0c0, r0c1, r1c0, r1c1].
_R_LINEARTRANS_M = {
    "identity": ([1, 0, 0, 1], {}),
    "scale2": ([2, 0, 0, 2], {"scale": (2, 2)}),
    "squeeze": ([2, 0, 0, 0.5], {"scale": (2, 0.5)}),
    "shear_v10": ([1, 0, 0.1, 1], {"shear": (0.1, 0)}),
    "shear_h2": ([1, 2, 0, 1], {"shear": (0, 2)}),
    "shear_both": ([1, 0.7, 0.3, 1], {"shear": (0.3, 0.7)}),
    "angle30": (
        [0.8660254038, 0.5, -0.5, 0.8660254038],
        {"angle": 30},
    ),
    "angle90": ([0, 1, -1, 0], {"angle": 90}),
    "scale_shear_angle": (
        [1.5202795796, 1.3435028843, -1.3081475452, -0.6363961031],
        {"scale": (2, 0.5), "shear": (0.3, 0.7), "angle": 45},
    ),
}


@pytest.mark.parametrize("name", list(_R_LINEARTRANS_M))
def test_lineartrans_setup_params_matrix(name):
    """setup_params reproduces R's column-major matrix for every branch."""
    expected, kwargs = _R_LINEARTRANS_M[name]
    pos = position_lineartrans(**kwargs)
    M = np.asarray(pos.setup_params(pd.DataFrame({"x": [0.0], "y": [0.0]}))["M"])
    assert M.shape == (2, 2)
    assert np.allclose(M.flatten(), np.array(expected), atol=1e-9)


def test_lineartrans_explicit_M_overrides():
    """An explicit M is returned verbatim, ignoring scale/shear/angle."""
    M = np.array([[0.0, -1.0], [1.0, 0.0]])
    pos = position_lineartrans(scale=(5, 5), shear=(9, 9), angle=123, M=M)
    out = np.asarray(pos.setup_params(pd.DataFrame({"x": [0.0]}))["M"])
    assert np.allclose(out, M)


# R fixtures: transformed [x, y] pairs flattened row-major for the unit square
# df = data.frame(x = c(0,1,1,0), y = c(0,0,1,1)).
_R_LINEARTRANS_XY = {
    "angle30": (
        {"angle": 30},
        [0, 0, 0.8660254038, -0.5, 1.3660254038, 0.3660254038, 0.5, 0.8660254038],
    ),
    "scale_shear_angle": (
        {"scale": (2, 0.5), "shear": (0.3, 0.7), "angle": 45},
        [
            0, 0,
            1.5202795796, -1.3081475452,
            2.8637824638, -1.9445436483,
            1.3435028843, -0.6363961031,
        ],
    ),
}


@pytest.mark.parametrize("name", list(_R_LINEARTRANS_XY))
def test_lineartrans_compute_layer_coords(name):
    """compute_layer transforms x/y exactly as R's t(M %*% t(coord))."""
    kwargs, expected = _R_LINEARTRANS_XY[name]
    df = pd.DataFrame({"x": [0, 1, 1, 0], "y": [0, 0, 1, 1]})
    pos = position_lineartrans(**kwargs)
    params = pos.setup_params(df)
    out = pos.compute_layer(df.copy(), params, layout=None)
    got = out[["x", "y"]].to_numpy().flatten()
    assert np.allclose(got, np.array(expected), atol=1e-9)


def test_lineartrans_custom_M_shear_then_rotate():
    """A custom M (shear then rotate) matches the R fixture coordinates."""
    theta = -30 * np.pi / 180
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    shear = np.array([[1.0, 1.0], [0.0, 1.0]])
    M = rot @ shear
    df = pd.DataFrame({"x": [0, 1, 1, 0], "y": [0, 0, 1, 1]})
    pos = position_lineartrans(M=M)
    params = pos.setup_params(df)
    out = pos.compute_layer(df.copy(), params, layout=None)
    got = out[["x", "y"]].to_numpy().flatten()
    expected = np.array(
        [
            0, 0,
            0.8660254038, -0.5,
            2.2320508076, -0.1339745962,
            1.3660254038, 0.3660254038,
        ]
    )
    assert np.allclose(got, expected, atol=1e-9)


def test_lineartrans_setup_data_identity():
    """setup_data is a no-op pass-through (no required-aes check)."""
    pos = position_lineartrans()
    df = pd.DataFrame({"foo": [1, 2, 3]})  # no x/y at all
    out = pos.setup_data(df, params={})
    assert out is df


def test_lineartrans_compute_layer_preserves_other_columns():
    """compute_layer only rewrites x/y, leaving other columns intact."""
    df = pd.DataFrame(
        {"x": [1.0, 2.0], "y": [3.0, 4.0], "PANEL": [1, 1], "group": [-1, -1]}
    )
    pos = position_lineartrans(scale=(2, 3))
    params = pos.setup_params(df)
    out = pos.compute_layer(df.copy(), params, layout=None)
    assert list(out["PANEL"]) == [1, 1]
    assert list(out["group"]) == [-1, -1]
    assert np.allclose(out["x"], [2.0, 4.0])
    assert np.allclose(out["y"], [9.0, 12.0])


def test_lineartrans_is_position_subclass():
    """The constructor returns a Position instance."""
    from ggplot2_py.position import Position

    assert isinstance(position_lineartrans(), PositionLinearTrans)
    assert isinstance(position_lineartrans(), Position)


# ===========================================================================
# PositionDisjointRanges
# ===========================================================================

def _run_disjoint(data, extend=1, stepsize=1):
    pos = position_disjoint_ranges(extend=extend, stepsize=stepsize)
    params = pos.setup_params(data)
    return pos.compute_panel(data.copy(), params, scales=None)


# Case A: all group == -1, several overlapping ranges (synthetic per-row groups).
_DATA_A = pd.DataFrame(
    {
        "xmin": [0, 0.5, 2, 2.1, 5],
        "xmax": [1, 1.5, 3, 4, 6],
        "ymin": [0, 0, 0, 0, 0],
        "ymax": [1, 1, 1, 1, 1],
        "group": [-1] * 5,
    }
)

_R_DISJOINT_A = {
    (1, 1): ([0, 1, 2, 0, 1], [1, 2, 3, 1, 2]),
    (0, 1): ([0, 1, 0, 1, 0], [1, 2, 1, 2, 1]),
    (0, 2): ([0, 2, 0, 2, 0], [1, 3, 1, 3, 1]),
    (0.1, -1): ([0, -1, 0, -1, 0], [1, 0, 1, 0, 1]),
}


@pytest.mark.parametrize("extend,stepsize", list(_R_DISJOINT_A))
def test_disjoint_all_neg1(extend, stepsize):
    """All-(-1)-group branch: one range per row, bins match R."""
    ymin, ymax = _R_DISJOINT_A[(extend, stepsize)]
    out = _run_disjoint(_DATA_A, extend=extend, stepsize=stepsize)
    assert np.allclose(out["ymin"].to_numpy(), ymin)
    assert np.allclose(out["ymax"].to_numpy(), ymax)
    # Row order and group column are preserved.
    assert list(out["group"]) == [-1] * 5


# Case B: multiple distinct groups (>1 unique) collapsed to per-group ranges.
_DATA_B = pd.DataFrame(
    {
        "xmin": [0, 0.2, 2, 2.5, 5, 5.1, 1.0],
        "xmax": [1, 0.9, 3, 4.0, 6, 7.0, 1.4],
        "ymin": [0] * 7,
        "ymax": [1] * 7,
        "group": [1, 1, 2, 2, 3, 3, 4],
    }
)

_R_DISJOINT_B = {
    (1, 1): (
        [0, 0, 2, 2, 0, 0, 1],
        [1, 1, 3, 3, 1, 1, 2],
    ),
    (0, 0.5): (
        [0, 0, 0, 0, 0, 0, 0.5],
        [1, 1, 1, 1, 1, 1, 1.5],
    ),
}


@pytest.mark.parametrize("extend,stepsize", list(_R_DISJOINT_B))
def test_disjoint_multigroup(extend, stepsize):
    """Multi-group branch: per-group ranges + match-back, bins match R."""
    ymin, ymax = _R_DISJOINT_B[(extend, stepsize)]
    out = _run_disjoint(_DATA_B, extend=extend, stepsize=stepsize)
    assert np.allclose(out["ymin"].to_numpy(), ymin)
    assert np.allclose(out["ymax"].to_numpy(), ymax)
    # Row order preserved (data never reordered, only internal ranges sorted).
    assert list(out["group"]) == [1, 1, 2, 2, 3, 3, 4]


def test_disjoint_single_group_unchanged():
    """Single non-(-1) group: data is returned unchanged (early return)."""
    data = pd.DataFrame(
        {
            "xmin": [0, 2, 5],
            "xmax": [1, 3, 6],
            "ymin": [0, 0, 0],
            "ymax": [1, 1, 1],
            "group": [2, 2, 2],
        }
    )
    out = _run_disjoint(data, extend=1, stepsize=1)
    assert np.allclose(out["ymin"].to_numpy(), [0, 0, 0])
    assert np.allclose(out["ymax"].to_numpy(), [1, 1, 1])


def test_disjoint_touching_endpoints_strict_lt():
    """Touching endpoints (xmax == xmin after extend) go to separate bins.

    Boundary semantics are R's strict ``<``: ranges [0,1] and [1,2] touch at 1
    and are therefore treated as overlapping -> bin 1 and bin 2.
    """
    data = pd.DataFrame(
        {
            "xmin": [0, 1, 2],
            "xmax": [1, 2, 3],
            "ymin": [0, 0, 0],
            "ymax": [1, 1, 1],
            "group": [-1] * 3,
        }
    )
    out = _run_disjoint(data, extend=0, stepsize=1)
    assert np.allclose(out["ymin"].to_numpy(), [0, 1, 0])
    assert np.allclose(out["ymax"].to_numpy(), [1, 2, 1])


def test_disjoint_setup_params_warns_on_missing_x():
    """setup_params warns (does not error) when xmin/xmax are absent."""
    data = pd.DataFrame({"ymin": [0], "ymax": [1], "group": [-1]})
    pos = position_disjoint_ranges()
    with pytest.warns(UserWarning, match="Undefined ranges"):
        params = pos.setup_params(data)
    assert params == {"extend": 1, "stepsize": 1}


def test_disjoint_setup_params_passthrough():
    """setup_params returns the configured extend/stepsize."""
    pos = position_disjoint_ranges(extend=0.3, stepsize=2.5)
    data = pd.DataFrame({"xmin": [0], "xmax": [1], "group": [-1]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        params = pos.setup_params(data)
    assert params == {"extend": 0.3, "stepsize": 2.5}


def test_disjoint_is_position_subclass():
    """The constructor returns a Position instance with required_aes."""
    from ggplot2_py.position import Position

    pos = position_disjoint_ranges()
    assert isinstance(pos, PositionDisjointRanges)
    assert isinstance(pos, Position)
    assert pos.required_aes == ("xmin", "xmax", "ymin", "ymax")


# ===========================================================================
# CoordAxesInside
# ===========================================================================

def test_coord_construction_and_origin():
    """Constructor stores origin as a 1-row frame and the limits/flags."""
    c = coord_axes_inside(xintercept=1, yintercept=-1, xlim=(0, 5), ylim=(-2, 2))
    assert isinstance(c, CoordAxesInside)
    assert list(c.origin["x"]) == [1]
    assert list(c.origin["y"]) == [-1]
    assert c.limits == {"x": [0, 5], "y": [-2, 2]}
    assert c.clip == "on"
    assert c.ratio is None


def test_coord_is_free():
    """is_free is True iff ratio is None."""
    assert coord_axes_inside().is_free() is True
    assert coord_axes_inside(ratio=2).is_free() is False


def test_coord_aspect():
    """aspect = diff(y)/diff(x) * ratio, or None for a free ratio."""
    assert coord_axes_inside().aspect({"x.range": [0, 4], "y.range": [0, 6]}) is None
    c = coord_axes_inside(ratio=2)
    # 6/4 * 2 == 3.0
    assert c.aspect({"x.range": [0, 4], "y.range": [0, 6]}) == pytest.approx(3.0)


@pytest.mark.parametrize(
    "labels_inside,expected",
    [
        (False, "none"),
        (True, "both"),
        ("x", "x"),
        ("y", "y"),
        ("none", "none"),
        ("both", "both"),
    ],
)
def test_coord_labels_inside_normalisation(labels_inside, expected):
    """labels_inside is normalised; bad strings raise (arg_match0)."""
    # Construction must succeed for all valid forms.
    c = coord_axes_inside(labels_inside=labels_inside)
    assert isinstance(c, CoordAxesInside)


def test_coord_labels_inside_bad_value():
    """An invalid labels_inside string is rejected like rlang::arg_match0."""
    with pytest.raises(ValueError, match="labels_inside"):
        coord_axes_inside(labels_inside="diagonal")


def test_coord_theme_objects_built():
    """outer_axes / inner_axes are Theme objects (used by render_* / render_bg)."""
    from ggplot2_py.theme import Theme

    c = coord_axes_inside()
    assert isinstance(c.outer_axes, Theme)
    assert isinstance(c.inner_axes, Theme)


def test_replace_vp_coord_mutates_axis_vp():
    """_replace_vp_coord replaces a single viewport coordinate of a grob."""
    from grid_py import Unit
    from ggplot2_py import theme_grey
    from ggplot2_py._guide_axis import draw_axis

    grob = draw_axis(np.array([0.2, 0.5]), ["a", "b"], "bottom", theme_grey())
    out = _replace_vp_coord(grob, "y", Unit(0.3, "npc"))
    assert out is grob
    assert float(out.vp.y.values[0]) == pytest.approx(0.3)


def test_replace_vp_coord_passthrough_when_no_vp():
    """A grob without a viewport is returned unchanged."""
    class _NoVp:
        vp = None

    g = _NoVp()
    from grid_py import Unit

    assert _replace_vp_coord(g, "x", Unit(0.5, "npc")) is g


def _panel_params():
    from ggplot2_py import scale_x_continuous, scale_y_continuous

    sx = scale_x_continuous()
    sy = scale_y_continuous()
    sx.train(pd.Series([-3.0, 3.0]))
    sy.train(pd.Series([-2.0, 4.0]))
    c = coord_axes_inside(xintercept=1, yintercept=-1)
    return c, c.setup_panel_params(sx, sy)


def test_coord_render_axis_h_v_keys():
    """render_axis_h/v return the expected top/bottom & left/right dicts."""
    from ggplot2_py import theme_grey

    c, pp = _panel_params()
    th = theme_grey()
    ah = c.render_axis_h(pp, th)
    av = c.render_axis_v(pp, th)
    assert set(ah) == {"top", "bottom"}
    assert set(av) == {"left", "right"}


def test_coord_render_bg_grobtree_structure():
    """render_bg composites the grid + four named, repositioned axes."""
    from ggplot2_py import theme_grey

    c, pp = _panel_params()
    bg = c.render_bg(pp, theme_grey())
    assert bg.n_children() == 5
    names = [bg.get_child(n).name for n in bg._children_order]
    assert names == ["grid", "axis-b", "axis-t", "axis-l", "axis-r"]


def test_coord_render_bg_repositions_to_origin():
    """The interior bottom axis viewport y equals the origin's transformed NPC."""
    from grid_py import Unit
    from scales import oob_squish
    from ggplot2_py import theme_grey, ggproto_parent
    from ggplot2_py.coord import CoordCartesian

    c, pp = _panel_params()
    bg = c.render_bg(pp, theme_grey())

    # Expected interior NPC y of the origin (yintercept = -1 in [-2.3, 4.3]).
    origin = c.transform(c.origin, pp)
    expected_y = oob_squish(float(origin["y"].iloc[0]))
    expected_x = oob_squish(float(origin["x"].iloc[0]))

    axis_b = bg.get_child("axis-b")
    axis_l = bg.get_child("axis-l")
    assert float(axis_b.vp.y.values[0]) == pytest.approx(expected_y)
    assert float(axis_l.vp.x.values[0]) == pytest.approx(expected_x)


def test_coord_render_bg_oob_origin_squished():
    """Out-of-bounds intercepts snap to the nearest [0, 1] edge via oob_squish."""
    from scales import oob_squish
    from ggplot2_py import scale_x_continuous, scale_y_continuous

    sx = scale_x_continuous()
    sy = scale_y_continuous()
    sx.train(pd.Series([-3.0, 3.0]))
    sy.train(pd.Series([-2.0, 4.0]))
    c = coord_axes_inside(xintercept=-100, yintercept=100)
    pp = c.setup_panel_params(sx, sy)
    origin = c.transform(c.origin, pp)
    assert oob_squish(float(origin["x"].iloc[0])) == pytest.approx(0.0)
    assert oob_squish(float(origin["y"].iloc[0])) == pytest.approx(1.0)
    # Render must still succeed with clipped origin.
    from ggplot2_py import theme_grey

    bg = c.render_bg(pp, theme_grey())
    assert bg.n_children() == 5
