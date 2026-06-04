"""Tests for the extra ggh4x geoms.

Covers the four geom modules implemented in this slice:

* ``geom_pointpath`` / ``GeomPointPath`` + the gap-segment grobs.
* ``geom_text_aimed`` / ``GeomTextAimed`` + ``compute_just`` + the aimed-text
  grob.
* ``geom_polygonraster`` / ``GeomPolygonRaster``.
* ``geom_rectmargin`` / ``geom_tilemargin`` + ``GeomRectMargin`` /
  ``GeomTileMargin``.

All numeric R-parity values were captured from a live ggrepel-dev R session
(ggplot2 4.0.2 + ggh4x), running the corresponding ggh4x internals
(``intersect_line_circle``, ``crop_segment_ends``, ``makeContext.*``,
``makeContent.aimed_text``, ``GeomPolygonRaster$setup_data``,
``GeomTileMargin$setup_data``, ggplot2-internal ``compute_just``).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ggplot2_py import aes, ggplot, ggsave
import grid_py
from grid_py import (
    CairoRenderer,
    Gpar,
    Unit,
    Viewport,
    convert_x,
    convert_y,
    get_state,
    grid_newpage,
    push_viewport,
)

from ggh4x._aimed_text_grob import (
    AimedTextGrob,
    aimed_text_grob,
    compute_aimed_angles,
    compute_just,
    just_dir,
)
from ggh4x._gap_grobs import (
    GapSegmentsChainGrob,
    GapSegmentsGrob,
    _chain_compute,
    crop_segment_ends,
    filter_gp,
    intersect_line_circle,
)
from ggh4x.geom_pointpath import GeomPointPath, GeomPointpath, geom_pointpath
from ggh4x.geom_polygonraster import GeomPolygonRaster, geom_polygonraster
from ggh4x.geom_rectrug import (
    GeomRectMargin,
    GeomTileMargin,
    geom_rectmargin,
    geom_tilemargin,
)
from ggh4x.geom_text_aimed import GeomTextAimed, geom_text_aimed


# ---------------------------------------------------------------------------
# Helpers for rendering / grobs against a 100 mm viewport
# ---------------------------------------------------------------------------
def _push_100mm_panel():
    """Set up a renderer and push a square 100 mm viewport for unit conversion."""
    r = CairoRenderer(width=100 / 25.4, height=100 / 25.4, dpi=72)
    get_state()._renderer = r
    grid_newpage(recording=False)
    push_viewport(
        Viewport(width=Unit(100, "mm"), height=Unit(100, "mm")), recording=False
    )


def _render_ok(plot) -> bool:
    """Render *plot* to a temporary PNG and report success."""
    import os
    import tempfile

    out = tempfile.mktemp(suffix=".png")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ggsave(out, plot, width=4, height=4, dpi=72)
    return os.path.exists(out) and os.path.getsize(out) > 0


# ===========================================================================
# intersect_line_circle / crop_segment_ends / filter_gp
# ===========================================================================
def test_intersect_line_circle_prio1():
    # R ggh4x:::intersect_line_circle(..., prio = 1L)
    x1 = np.array([0, -1, 2, 0.5])
    y1 = np.array([0, 0, 2, -0.5])
    x2 = np.array([2, 1, -2, 3])
    y2 = np.array([0, 0, -2, 3])
    cx = np.array([1, 0, 0, 1])
    cy = np.array([0, 0, 0, 1])
    r = np.array([0.5, 0.5, 1.0, 0.7])
    out = intersect_line_circle(x1, y1, x2, y2, cx, cy, r, prio=1)
    np.testing.assert_allclose(
        out["x"], [1.0, 0.0, 0.707106781186548, 1.074249258515489]
    )
    np.testing.assert_allclose(
        out["y"], [0.0, 0.0, 0.707106781186548, 0.303948961921685]
    )


def test_intersect_line_circle_prio2():
    x1 = np.array([0, -1, 2, 0.5])
    y1 = np.array([0, 0, 2, -0.5])
    x2 = np.array([2, 1, -2, 3])
    y2 = np.array([0, 0, -2, 3])
    cx = np.array([1, 0, 0, 1])
    cy = np.array([0, 0, 0, 1])
    r = np.array([0.5, 0.5, 1.0, 0.7])
    out = intersect_line_circle(x1, y1, x2, y2, cx, cy, r, prio=2)
    np.testing.assert_allclose(
        out["x"], [1.0, 0.0, -0.707106781186548, 1.682507498241268]
    )
    np.testing.assert_allclose(
        out["y"], [0.0, 0.0, -0.707106781186548, 1.155510497537775]
    )


def test_crop_segment_ends():
    # R ggh4x:::crop_segment_ends
    x0 = np.array([0, 0, 1, 5.0])
    x1 = np.array([10, 0.1, 1, 5.05])
    y0 = np.array([0, 0, 1, 5.0])
    y1 = np.array([0, 0.1, 11, 5.05])
    r = np.array([1, 1, 2, 0.5])
    out = crop_segment_ends(x0, x1, y0, y1, r)
    np.testing.assert_allclose(out["x0"], [1.0, 0.707106781186547, 1.0, 5.353553390593274])
    np.testing.assert_allclose(out["x1"], [9.0, -0.607106781186547, 1.0, 4.696446609406726])
    np.testing.assert_allclose(out["y0"], [0.0, 0.707106781186547, 3.0, 5.353553390593274])
    np.testing.assert_allclose(out["y1"], [0.0, -0.607106781186547, 9.0, 4.696446609406726])
    assert out["keep"].tolist() == [True, False, True, False]


def test_crop_segment_ends_zero_length_is_safe():
    # Zero-length segment: nudge is non-finite -> zeroed (ggh4x #73).
    out = crop_segment_ends(
        np.array([1.0]), np.array([1.0]), np.array([2.0]), np.array([2.0]), np.array([0.5])
    )
    assert out["x0"][0] == 1.0 and out["x1"][0] == 1.0


def test_filter_gp_subsets_vectors_only():
    from ggh4x._gap_grobs import _gpar_to_dict

    gp = Gpar(col=np.array(["red", "green", "blue"]), lwd=2.0)
    keep = np.array([True, False, True])
    out = _gpar_to_dict(filter_gp(gp, keep))
    assert [str(c) for c in out["col"]] == ["red", "blue"]
    # Scalar lwd untouched.
    assert float(np.asarray(out["lwd"])) == 2.0


def test_filter_gp_none():
    assert filter_gp(None, np.array([True])) is None


# ===========================================================================
# _chain_compute
# ===========================================================================
def test_chain_compute_unique_ids_matches_crop():
    # 5-point path -> 4 unique-id segments, radius 5; matches R CHAIN output.
    pts_x = np.array([0.0, 30, 60, 60, 30])
    pts_y = np.array([0.0, 0, 0, 30, 30])
    n = len(pts_x)
    x0, y0 = pts_x[:-1], pts_y[:-1]
    x1, y1 = pts_x[1:], pts_y[1:]
    id_vec = np.arange(1, n)
    mult = np.full(n - 1, 5.0)
    xy_x, xy_y, xy_id, keep, grp_start = _chain_compute(x0, x1, y0, y1, mult, id_vec)
    np.testing.assert_allclose(xy_x, [5, 25, 35, 55, 60, 60, 55, 35])
    np.testing.assert_allclose(xy_y, [0, 0, 0, 0, 5, 25, 30, 30])
    assert xy_id.tolist() == [1, 1, 2, 2, 3, 3, 4, 4]
    assert keep.tolist() == [True, True, True, True]


def test_chain_compute_drops_vanishing_segment():
    # Second short segment vanishes; the single survivor is handled without
    # the R rowSums edge-case crash.
    x0 = np.array([0.0, 8])
    y0 = np.array([0.0, 0])
    x1 = np.array([10.0, 12])
    y1 = np.array([0.0, 0])
    id_vec = np.array([1, 2])
    mult = np.array([3.0, 3])
    xy_x, xy_y, xy_id, keep, grp_start = _chain_compute(x0, x1, y0, y1, mult, id_vec)
    np.testing.assert_allclose(xy_x, [3.0, 7.0])
    assert keep.tolist() == [True, False]


def test_chain_compute_all_drop_returns_none():
    # A segment fully inside the gap radius -> nothing survives.
    out = _chain_compute(
        np.array([0.0]),
        np.array([1.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([10.0]),
        np.array([1]),
    )
    assert out is None


# ===========================================================================
# Gap grobs make_context (mm coordinates against a 100 mm viewport)
# ===========================================================================
def test_gapsegments_make_context_mm():
    _push_100mm_panel()
    g = GapSegmentsGrob(
        x0=Unit(np.array([0.1, 0.5]), "npc"),
        x1=Unit(np.array([0.5, 0.9]), "npc"),
        y0=Unit(np.array([0.1, 0.5]), "npc"),
        y1=Unit(np.array([0.5, 0.9]), "npc"),
        mult=np.array([5.0, 5.0]),
        id=np.array([1, 2]),
        gp=Gpar(col=np.array(["red", "blue"])),
    )
    res = g.make_context()
    assert res._grid_class == "segments"
    np.testing.assert_allclose(
        convert_x(res.x0, "mm", valueOnly=True), [13.53553391, 53.53553391], atol=1e-5
    )
    np.testing.assert_allclose(
        convert_x(res.x1, "mm", valueOnly=True), [46.46446609, 86.46446609], atol=1e-5
    )
    np.testing.assert_allclose(
        convert_y(res.y0, "mm", valueOnly=True), [13.53553391, 53.53553391], atol=1e-5
    )
    np.testing.assert_allclose(
        convert_y(res.y1, "mm", valueOnly=True), [46.46446609, 86.46446609], atol=1e-5
    )


def test_gapsegmentschain_make_context_mm():
    _push_100mm_panel()
    g = GapSegmentsChainGrob(
        x0=Unit(np.array([0.1, 0.5]), "npc"),
        x1=Unit(np.array([0.5, 0.9]), "npc"),
        y0=Unit(np.array([0.1, 0.5]), "npc"),
        y1=Unit(np.array([0.5, 0.9]), "npc"),
        mult=np.array([5.0, 5.0]),
        id=np.array([1, 2]),
        gp=Gpar(col=np.array(["red", "blue"])),
    )
    res = g.make_context()
    assert res._grid_class == "polyline"
    np.testing.assert_allclose(
        convert_x(res.x, "mm", valueOnly=True),
        [13.53553391, 46.46446609, 53.53553391, 86.46446609],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        convert_y(res.y, "mm", valueOnly=True),
        [13.53553391, 46.46446609, 53.53553391, 86.46446609],
        atol=1e-5,
    )
    assert list(np.asarray(res.id)) == [1, 1, 2, 2]


# ===========================================================================
# GeomPointPath
# ===========================================================================
def test_gecompointpath_alias_and_defaults():
    assert GeomPointpath is GeomPointPath
    aes_names = GeomPointPath().aesthetics()
    assert "mult" in aes_names
    assert "linewidth" in aes_names and "linetype" in aes_names
    assert GeomPointPath.default_aes["mult"] == 0.5


def test_geompointpath_constructor_params():
    layer = geom_pointpath(arrow=None)
    assert layer is not None


def test_geompointpath_renders():
    df = pd.DataFrame({"t": [0, 1, 2, 3, 4, 5.0], "p": [0, 1, 4, 9, 16, 25.0]})
    assert _render_ok(ggplot(df, aes("t", "p")) + geom_pointpath())


def test_geompointpath_renders_polar_chain():
    from ggplot2_py import coord_polar

    df = pd.DataFrame({"t": [0, 1, 2, 3, 4, 5.0], "p": [0, 1, 4, 9, 16, 25.0]})
    plot = ggplot(df, aes("t", "p")) + geom_pointpath() + coord_polar()
    assert _render_ok(plot)


# ===========================================================================
# compute_just / just_dir / compute_aimed_angles
# ===========================================================================
@pytest.mark.parametrize(
    "just,expected",
    [
        ("left", 0.0),
        ("right", 1.0),
        ("center", 0.5),
        ("centre", 0.5),
        ("top", 1.0),
        ("bottom", 0.0),
        ("middle", 0.5),
    ],
)
def test_compute_just_keywords(just, expected):
    out = compute_just(just, [1, 2, 3])
    assert float(out[0]) == expected


def test_compute_just_inward_outward():
    # values 1..5 are all > 0.5 -> just_dir == 3.
    inward = compute_just(["inward"] * 5, [1, 2, 3, 4, 5])
    outward = compute_just(["outward"] * 5, [1, 2, 3, 4, 5])
    assert inward.tolist() == [1.0] * 5
    assert outward.tolist() == [0.0] * 5


def test_compute_just_non_character_passthrough():
    arr = np.array([0.2, 0.8])
    out = compute_just(arr, [1, 2])
    assert out is arr


def test_just_dir():
    assert just_dir(np.array([0.1, 0.5, 0.9])).tolist() == [1, 2, 3]


def test_compute_aimed_angles_flip():
    x1 = np.array([5.0, 5, 5, 5, 0, 10])
    y1 = np.array([5.0, 5, 5, 5, 0, 10])
    x0 = np.array([10.0, 0, 5, 5, 5, 5])
    y0 = np.array([10.0, 0, 10, 0, 5, 5])
    rot = np.array([0.0, 0, 30, 30, 0, 90])
    hjust = np.array([0.0, 0, 0.5, 0.5, 0, 1])
    vjust = np.full(6, 0.5)
    r_rot, r_hjust, _ = compute_aimed_angles(x1, y1, x0, y0, rot, hjust, vjust, True)
    np.testing.assert_allclose(r_rot, [45, 45, 300, 300, 45, 315])
    np.testing.assert_allclose(r_hjust, [1.0, 0.0, 0.5, 0.5, 1.0, 0.0])


def test_compute_aimed_angles_noflip():
    x1 = np.array([5.0, 5, 5, 5, 0, 10])
    y1 = np.array([5.0, 5, 5, 5, 0, 10])
    x0 = np.array([10.0, 0, 5, 5, 5, 5])
    y0 = np.array([10.0, 0, 10, 0, 5, 5])
    rot = np.array([0.0, 0, 30, 30, 0, 90])
    hjust = np.array([0.0, 0, 0.5, 0.5, 0, 1])
    vjust = np.full(6, 0.5)
    r_rot, r_hjust, _ = compute_aimed_angles(x1, y1, x0, y0, rot, hjust, vjust, False)
    np.testing.assert_allclose(r_rot, [225, 45, 300, 120, 225, 135])
    np.testing.assert_allclose(r_hjust, [0.0, 0.0, 0.5, 0.5, 0.0, 1.0])


# ===========================================================================
# AimedTextGrob make_content (against R makeContent.aimed_text)
# ===========================================================================
def test_aimed_text_grob_make_content():
    _push_100mm_panel()
    g = aimed_text_grob(
        label=np.array(["a", "b", "c"]),
        x=Unit(np.array([0.5, 0.5, 0.2]), "npc"),
        y=Unit(np.array([0.5, 0.5, 0.8]), "npc"),
        x0=Unit(np.array([1.0, 0, 0.5]), "npc"),
        y0=Unit(np.array([1.0, 0, 0.5]), "npc"),
        hjust=np.array([0.0, 0, 0.5]),
        vjust=np.array([0.5, 0.5, 0.5]),
        rot=np.array([0.0, 0, 30]),
        flip_upsidedown=True,
        default_units="npc",
    )
    assert isinstance(g, AimedTextGrob)
    res = g.make_content()
    children = list(res.get_children())
    assert [c._grid_class for c in children] == ["text", "text", "text"]
    np.testing.assert_allclose([float(c.rot) for c in children], [45.0, 45.0, 345.0])
    np.testing.assert_allclose([float(c.hjust) for c in children], [1.0, 0.0, 0.5])


# ===========================================================================
# GeomTextAimed
# ===========================================================================
def test_geomtextaimed_defaults():
    da = GeomTextAimed.default_aes
    assert da["xend"] == -np.inf
    assert da["yend"] == -np.inf
    aes_names = GeomTextAimed().aesthetics()
    assert "xend" in aes_names and "yend" in aes_names
    assert "flip_upsidedown" in GeomTextAimed().parameters(extra=True)


def test_geomtextaimed_renders():
    df = pd.DataFrame(
        {
            "mpg": [21, 22.8, 21.4, 18.7, 18.1],
            "wt": [2.6, 2.9, 3.2, 3.4, 3.5],
            "lab": ["A", "B", "C", "D", "E"],
        }
    )
    plot = ggplot(df, aes("mpg", "wt")) + geom_text_aimed(
        aes(label="lab"), xend=np.inf, yend=np.inf
    )
    assert _render_ok(plot)


def test_geomtextaimed_nudge_position_conflict():
    with pytest.raises(ValueError):
        geom_text_aimed(nudge_x=1, position="dodge")


# ===========================================================================
# GeomPolygonRaster
# ===========================================================================
def test_geompolygonraster_setup_data_corners():
    g = GeomPolygonRaster()
    data = pd.DataFrame(
        {"x": [1, 2, 1, 2], "y": [1, 1, 2, 2], "fill": ["a", "b", "c", "d"]}
    )
    out = g.setup_data(data, {"hjust": 0.5, "vjust": 0.5})
    assert out["x"].tolist() == [0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 2.5, 2.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 2.5, 2.5]
    assert out["y"].tolist() == [0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5, 0.5, 1.5, 2.5, 2.5, 1.5, 1.5, 2.5, 2.5, 1.5]
    assert out["id"].tolist() == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    assert out["fill"].tolist() == ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4


def test_geompolygonraster_setup_data_vjust():
    g = GeomPolygonRaster()
    data = pd.DataFrame(
        {"x": [1, 2, 1, 2], "y": [1, 1, 2, 2], "fill": ["a", "b", "c", "d"]}
    )
    out = g.setup_data(data, {"hjust": 0.5, "vjust": 1})
    assert out["y"].tolist() == [1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 3.0, 2.0]


def test_geompolygonraster_constructor_validates_hjust():
    with pytest.raises(TypeError):
        geom_polygonraster(hjust="x")


def test_geompolygonraster_renders_default_lineartrans():
    grid = pd.DataFrame(
        {"x": np.repeat([1, 2, 3], 3), "y": np.tile([1, 2, 3], 3), "z": np.arange(9.0)}
    )
    assert _render_ok(ggplot(grid, aes("x", "y", fill="z")) + geom_polygonraster())


def test_geompolygonraster_renders_polar():
    from ggplot2_py import coord_polar

    grid = pd.DataFrame(
        {"x": np.repeat([1, 2, 3], 3), "y": np.tile([1, 2, 3], 3), "z": np.arange(9.0)}
    )
    plot = (
        ggplot(grid, aes("x", "y", fill="z"))
        + geom_polygonraster(position="identity")
        + coord_polar()
    )
    assert _render_ok(plot)


# ===========================================================================
# GeomTileMargin / GeomRectMargin
# ===========================================================================
def test_geomtilemargin_setup_data():
    g = GeomTileMargin()
    data = pd.DataFrame(
        {"x": [1, 4.0], "y": [1, 2.0], "width": [2, 1.0], "height": [1, 2.0], "fill": ["A", "B"]}
    )
    out = g.setup_data(data, {})
    assert out["xmin"].tolist() == [0.0, 3.5]
    assert out["xmax"].tolist() == [2.0, 4.5]
    assert out["ymin"].tolist() == [0.5, 1.0]
    assert out["ymax"].tolist() == [1.5, 3.0]
    assert "width" not in out.columns and "height" not in out.columns


def test_geomtilemargin_setup_data_resolution_fallback():
    g = GeomTileMargin()
    data = pd.DataFrame({"x": [1, 4.0], "y": [1, 2.0], "fill": ["A", "B"]})
    out = g.setup_data(data, {})
    # resolution(x, zero=False) = 3 ; resolution(y, zero=False) = 1
    assert out["xmin"].tolist() == [-0.5, 2.5]
    assert out["xmax"].tolist() == [2.5, 5.5]
    assert out["ymin"].tolist() == [0.5, 1.5]
    assert out["ymax"].tolist() == [1.5, 2.5]


def test_geomrectmargin_optional_aes_and_default():
    geom = GeomRectMargin()
    for a in ("x", "y", "xmin", "xmax", "ymin", "ymax"):
        assert a in geom.aesthetics()
    # No hard-coded required aesthetics.
    assert tuple(getattr(geom, "required_aes", ())) == ()


def test_geomrectmargin_length_must_be_unit():
    geom = GeomRectMargin()
    data = pd.DataFrame({"xmin": [1.0], "xmax": [2.0], "ymin": [1.0], "ymax": [2.0]})

    class _DummyCoord:
        def is_linear(self):
            return True

        def transform(self, d, pp):
            return d

    with pytest.raises(TypeError):
        geom.draw_panel(data, None, _DummyCoord(), length=0.03)


def test_geomrectmargin_renders():
    rect = pd.DataFrame(
        {"xmin": [1, 5.0], "xmax": [2, 7.0], "ymin": [1, 2.0], "ymax": [2, 4.0], "fill": ["A", "B"]}
    )
    plot = ggplot(
        rect, aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax", fill="fill")
    ) + geom_rectmargin()
    assert _render_ok(plot)


def test_geomrectmargin_renders_outside_tr():
    rect = pd.DataFrame(
        {"xmin": [1, 5.0], "xmax": [2, 7.0], "ymin": [1, 2.0], "ymax": [2, 4.0]}
    )
    plot = ggplot(
        rect, aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax")
    ) + geom_rectmargin(sides="tr", outside=True)
    assert _render_ok(plot)


def test_geomtilemargin_renders():
    tile = pd.DataFrame(
        {"x": [1, 4.0], "y": [1, 2.0], "width": [2, 1.0], "height": [1, 2.0], "fill": ["A", "B"]}
    )
    plot = ggplot(
        tile, aes("x", "y", width="width", height="height", fill="fill")
    ) + geom_tilemargin()
    assert _render_ok(plot)


def test_geomtilemargin_inherits_rectmargin_draw_panel():
    assert GeomTileMargin.draw_panel is GeomRectMargin.draw_panel
    assert issubclass(GeomTileMargin, GeomRectMargin)
