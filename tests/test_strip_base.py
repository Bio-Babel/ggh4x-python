"""Tests for the ggh4x Strip base class and helpers (``strip_vanilla.py``).

These verify the pure pieces against values dumped from R (ggplot2 4.0.2 + ggh4x
0.3.1.9000 via the ggrepel-dev Rscript):

* ``resolve_strip`` dispatch (string -> constructor, instance passthrough,
  callable, bad arg -> error);
* ``validate_element_list`` (None, single-wrap, mixed-valid, invalid-abort);
* ``inherit_element`` (early returns + None-prop fill + S7 rel non-multiply);
* ``setup_elements`` precedence (the strip.placement ``inside`` gotcha, padding
  cm, element resolution);
* column-major ``col(labels)`` index + flatten;
* a vanilla strip built on a 1x2 grid layout vs the R gtable structure;
* ``incorporate_grid`` weaving (inside + outside placement).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggh4x.strip_vanilla import (
    Strip,
    assert_strip,
    inherit_element,
    resolve_strip,
    strip_vanilla,
    validate_element_list,
)
from ggh4x.strip_vanilla import (
    _col_index,
    _flatten_col_major,
    _placement_inside,
)

from ggplot2_py import element_blank, element_rect, element_text
from ggplot2_py.theme_defaults import theme_grey
from grid_py import convert_height, convert_width, convert_unit
import gtable_py


# ---------------------------------------------------------------------------
# Constructor + resolve_strip
# ---------------------------------------------------------------------------
def test_strip_vanilla_is_instance():
    s = strip_vanilla()
    assert isinstance(s, Strip)
    assert s.params == {"clip": "inherit", "size": "constant"}


def test_strip_vanilla_params():
    s = strip_vanilla(clip="on", size="variable")
    assert s.params == {"clip": "on", "size": "variable"}


def test_strip_vanilla_bad_clip():
    with pytest.raises(ValueError):
        strip_vanilla(clip="bogus")


def test_strip_vanilla_bad_size():
    with pytest.raises(ValueError):
        strip_vanilla(size="bogus")


def test_resolve_strip_string():
    s = resolve_strip("vanilla")
    assert isinstance(s, Strip)


def test_resolve_strip_instance_passthrough():
    s = strip_vanilla()
    assert resolve_strip(s) is s


def test_resolve_strip_callable():
    s = resolve_strip(strip_vanilla)
    assert isinstance(s, Strip)


def test_resolve_strip_bad_arg():
    with pytest.raises(ValueError):
        resolve_strip(42)


def test_assert_strip_is_resolve_strip():
    # deeptime fallback alias.
    assert assert_strip is resolve_strip


# ---------------------------------------------------------------------------
# validate_element_list (R-dumped expectations)
# ---------------------------------------------------------------------------
def test_validate_element_list_none():
    assert validate_element_list(None, "element_text") is None


def test_validate_element_list_single_wrap():
    out = validate_element_list(element_text(colour="red"), "element_text")
    assert isinstance(out, list)
    assert len(out) == 1


def test_validate_element_list_mixed_valid():
    out = validate_element_list(
        [element_blank(), None, element_text(colour="x")], "element_text"
    )
    assert len(out) == 3


def test_validate_element_list_invalid():
    with pytest.raises(ValueError):
        validate_element_list([element_rect(fill="red")], "element_text")


def test_validate_element_list_rect_prototype():
    out = validate_element_list([element_rect(fill="red")], "element_rect")
    assert len(out) == 1


# ---------------------------------------------------------------------------
# inherit_element (R-dumped expectations)
# ---------------------------------------------------------------------------
def test_inherit_element_child_blank():
    r = inherit_element(element_blank(), element_text(colour="red"))
    assert isinstance(r, type(element_blank()))


def test_inherit_element_parent_none():
    r = inherit_element(element_text(colour="blue"), None)
    assert r.colour == "blue"


def test_inherit_element_child_none():
    r = inherit_element(None, element_text(colour="green"))
    assert r.colour == "green"


def test_inherit_element_parent_blank_inherit_true():
    child = element_text(colour="red", inherit_blank=True)
    r = inherit_element(child, element_blank())
    assert isinstance(r, type(element_blank()))


def test_inherit_element_parent_blank_inherit_false():
    child = element_text(colour="red", inherit_blank=False)
    r = inherit_element(child, element_blank())
    assert not isinstance(r, type(element_blank()))
    assert r.colour == "red"


def test_inherit_element_fill_none_props():
    child = element_text(colour="purple")
    parent = element_text(colour="black", size=14, family="serif")
    r = inherit_element(child, parent)
    assert r.colour == "purple"      # not overwritten
    assert r.size == 14              # filled from parent
    assert r.family == "serif"      # filled from parent


def test_inherit_element_rel_not_multiplied_s7():
    # R gold standard (ggplot2 4.0.2, S7 elements): rel(2) against parent size
    # 10 stays at 2 (the rel-multiply branch is in the non-S7 path).
    from ggplot2_py.theme_elements import Rel

    child = element_text(size=Rel(2))
    parent = element_text(size=10)
    r = inherit_element(child, parent)
    assert isinstance(r.size, Rel)
    assert r.size.value == 2


# ---------------------------------------------------------------------------
# placement precedence gotcha (R parse-tree verified)
# ---------------------------------------------------------------------------
def test_placement_inside_none():
    assert _placement_inside(None) is True


def test_placement_inside_inside():
    assert _placement_inside("inside") is True


def test_placement_inside_outside():
    assert _placement_inside("outside") is False


# ---------------------------------------------------------------------------
# column-major index + flatten
# ---------------------------------------------------------------------------
def test_col_index_column_major():
    labels = np.array(
        [["a", "d"], ["b", "e"], ["c", "f"]], dtype=object
    )  # 3x2
    # R as.vector(col(labels)) == c(1,1,1,2,2,2)
    assert _col_index(labels) == [1, 1, 1, 2, 2, 2]


def test_flatten_col_major():
    labels = np.array([["a", "d"], ["b", "e"], ["c", "f"]], dtype=object)
    # column-major flatten == a,b,c,d,e,f
    assert _flatten_col_major(labels) == ["a", "b", "c", "d", "e", "f"]


# ---------------------------------------------------------------------------
# setup_elements (R-dumped expectations)
# ---------------------------------------------------------------------------
def test_setup_elements_keys_and_inside():
    th = theme_grey()
    se = strip_vanilla().setup_elements(th, "grid")
    assert set(se.keys()) == {"padding", "background", "text", "inside"}
    # Default theme: strip.placement resolves to "inside" -> both True.
    assert se["inside"]["x"] is True
    assert se["inside"]["y"] is True


def test_setup_elements_padding_cm():
    th = theme_grey()
    se = strip_vanilla().setup_elements(th, "grid")
    pad = float(np.atleast_1d(convert_unit(se["padding"], "cm", valueOnly=True))[0])
    # R: convertUnit(strip.switch.pad.grid, "cm") == 0.09665145
    assert pad == pytest.approx(0.09665145, abs=1e-6)


def test_setup_elements_wrap_vs_grid_padding():
    th = theme_grey()
    se_grid = strip_vanilla().setup_elements(th, "grid")
    se_wrap = strip_vanilla().setup_elements(th, "wrap")
    g = float(np.atleast_1d(convert_unit(se_grid["padding"], "cm", valueOnly=True))[0])
    w = float(np.atleast_1d(convert_unit(se_wrap["padding"], "cm", valueOnly=True))[0])
    # In theme_grey both are 2.75 pt -> identical (matches R).
    assert g == pytest.approx(w, abs=1e-9)


def test_setup_elements_element_types():
    from ggplot2_py.theme_elements import ElementText

    th = theme_grey()
    se = strip_vanilla().setup_elements(th, "grid")
    # background rendered to a grob; text resolved to an element.
    assert se["background"]["x"] is not None
    assert isinstance(se["text"]["x"]["top"], ElementText)


def test_setup_elements_placement_outside():
    import ggplot2_py as gg

    th = theme_grey() + gg.theme(strip_placement="outside")
    se = strip_vanilla().setup_elements(th, "grid")
    assert se["inside"]["x"] is False
    assert se["inside"]["y"] is False


# ---------------------------------------------------------------------------
# Build a vanilla strip on a 1x2 grid layout (vs R gtable structure)
# ---------------------------------------------------------------------------
def _setup_1x2():
    th = theme_grey()
    strip = strip_vanilla()
    layout = pd.DataFrame(
        {"PANEL": [1, 2], "ROW": [1, 1], "COL": [1, 2], "g": ["A", "B"]}
    )
    params = {"cols": {"g": "g"}, "rows": {}, "labeller": "label_value"}
    strip.setup(layout, params, th, "grid")
    return strip


def test_build_strip_structure():
    strip = _setup_1x2()
    assert set(strip.strips.keys()) == {"x", "y"}
    assert set(strip.strips["x"].keys()) == {"top", "bottom"}
    assert set(strip.strips["y"].keys()) == {"left", "right"}


def test_build_strip_x_top_placement():
    strip = _setup_1x2()
    top = strip.strips["x"]["top"]
    # R: t=l=b=r=PANEL -> (1,2) per panel.
    assert list(top["t"]) == [1, 2]
    assert list(top["l"]) == [1, 2]
    assert list(top["b"]) == [1, 2]
    assert list(top["r"]) == [1, 2]
    assert len(top["grobs"]) == 2


def test_build_strip_x_top_grob_is_1x1_gtable():
    strip = _setup_1x2()
    g1 = strip.strips["x"]["top"]["grobs"][0]
    assert gtable_py.is_gtable(g1)
    # R: dim 1x1 (1 row=1 layer, 1 col). width = 1null, height ~ 0.607cm.
    assert len(g1.heights) == 1
    assert len(g1.widths) == 1
    h = float(np.atleast_1d(convert_height(g1.heights, "cm", valueOnly=True))[0])
    # R height 0.6071cm; font-metric backend variance ~1%.
    assert h == pytest.approx(0.607, abs=0.03)


def test_build_strip_y_left_none_when_no_rows():
    strip = _setup_1x2()
    assert strip.strips["y"]["left"] is None
    assert strip.strips["y"]["right"] is None


def test_build_strip_two_layer_x_top_dims():
    th = theme_grey()
    layout = pd.DataFrame(
        {"PANEL": [1], "ROW": [1], "COL": [1], "short": ["X"], "long": ["Long"]}
    )
    params = {
        "cols": {"short": "short", "long": "long"},
        "rows": {},
        "labeller": "label_value",
    }
    strip = strip_vanilla(size="constant")
    strip.setup(layout, params, th, "grid")
    g = strip.strips["x"]["top"]["grobs"][0]
    # R: 2 rows (2 layers) x 1 col.
    assert len(g.heights) == 2
    assert len(g.widths) == 1


def test_build_strip_two_layer_y_left_dims():
    th = theme_grey()
    layout = pd.DataFrame(
        {"PANEL": [1], "ROW": [1], "COL": [1], "short": ["X"], "long": ["Long"]}
    )
    params = {
        "cols": {},
        "rows": {"short": "short", "long": "long"},
        "labeller": "label_value",
    }
    strip = strip_vanilla()
    strip.setup(layout, params, th, "grid")
    g = strip.strips["y"]["left"]["grobs"][0]
    # R: 1 row x 2 cols (2 layers).
    assert len(g.heights) == 1
    assert len(g.widths) == 2


# ---------------------------------------------------------------------------
# Self-less method binding (HIGH-RISK)
# ---------------------------------------------------------------------------
def test_self_less_methods_unbound():
    import types

    s = strip_vanilla()
    # draw_labels / init_strip / finish_strip must NOT be bound methods.
    assert isinstance(s.draw_labels, types.FunctionType)
    assert isinstance(s.init_strip, types.FunctionType)
    assert isinstance(s.finish_strip, types.FunctionType)
    # setup_elements / assemble_strip / build_strip ARE bound (have self).
    assert isinstance(s.setup_elements, types.MethodType)
    assert isinstance(s.assemble_strip, types.MethodType)


def test_init_strip_rep_len():
    # base Strip has no by_layer -> rep_len recycling of single el/bg.
    s = strip_vanilla()
    elements = {
        "text": {"x": {"top": "T"}, "y": {}},
        "background": {"x": "BG"},
    }
    out = s.init_strip(elements, "top", [1, 1, 2])
    assert out["el"] == ["T", "T", "T"]
    assert out["bg"] == ["BG", "BG", "BG"]


# ---------------------------------------------------------------------------
# incorporate_grid weaving
# ---------------------------------------------------------------------------
def _minimal_panel_gtable():
    from gtable_py import Gtable, gtable_add_grob
    from grid_py import Unit, null_grob

    gt = Gtable(
        widths=Unit([1, 0.1, 1], ["null", "cm", "null"]),
        heights=Unit([1], ["null"]),
    )
    gt = gtable_add_grob(gt, null_grob(), t=1, l=1, name="panel-1-1")
    gt = gtable_add_grob(gt, null_grob(), t=1, l=3, name="panel-2-1")
    return gt


def _strip_names(gt):
    lay = gt.layout
    names = list(lay["name"])
    return [
        (str(n), int(lay["t"][i]), int(lay["l"][i]), float(lay["z"][i]))
        for i, n in enumerate(names)
        if "strip" in str(n)
    ]


def test_incorporate_grid_inside_top():
    strip = _setup_1x2()
    gt = _minimal_panel_gtable()
    out = strip.incorporate_grid(gt, switch=None)
    # inside.x True, switch_x False -> 1 new row (strip row), no padding.
    assert len(out.heights) == 2
    sn = _strip_names(out)
    assert len(sn) == 2
    # strip placed above panel (t=2 = where+1), z=2, panel columns 1 and 3.
    assert all(t == 2 and z == 2 for (_, t, _l, z) in sn)
    assert sorted(l for (_, _t, l, _z) in sn) == [1, 3]


def test_incorporate_grid_outside_adds_padding():
    import ggplot2_py as gg

    th = theme_grey() + gg.theme(strip_placement="outside")
    strip = strip_vanilla()
    layout = pd.DataFrame(
        {"PANEL": [1, 2], "ROW": [1, 1], "COL": [1, 2], "g": ["A", "B"]}
    )
    params = {"cols": {"g": "g"}, "rows": {}, "labeller": "label_value"}
    strip.setup(layout, params, th, "grid")
    gt = _minimal_panel_gtable()
    out = strip.incorporate_grid(gt, switch=None)
    # outside -> padding row + strip row added (nrow 1 -> 3).
    assert len(out.heights) == 3
    sn = _strip_names(out)
    assert len(sn) == 2
    assert all(t == 1 for (_, t, _l, _z) in sn)


def test_incorporate_grid_no_strips_when_empty_y():
    # y has no rows -> y$left/right are None -> no y strips, no error.
    strip = _setup_1x2()
    gt = _minimal_panel_gtable()
    out = strip.incorporate_grid(gt, switch=None)
    sn = _strip_names(out)
    # Only x (top) strips, no left/right.
    assert all(n.startswith("strip-t-") for (n, _t, _l, _z) in sn)
