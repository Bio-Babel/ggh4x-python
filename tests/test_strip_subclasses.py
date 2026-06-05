"""Tests for the ggh4x strip subclasses (themed / nested / split / tag).

These exercise the four ``Strip`` subclasses ported in:

* :mod:`ggh4x.strip_themed`  -- :class:`StripThemed` / :func:`strip_themed`
* :mod:`ggh4x.strip_nested`  -- :class:`StripNested` / :func:`strip_nested`
* :mod:`ggh4x.strip_split`   -- :class:`StripSplit`  / :func:`strip_split`
* :mod:`ggh4x.strip_tag`     -- :class:`StripTag`    / :func:`strip_tag`

The load-bearing cases (the nested RLE merge runs and the split ``id()`` spans)
carry their expected values from a gold-standard ``Rscript`` run of ggh4x
0.x on the ``mpg`` ``facet_wrap2(vars(cyl, drv))`` layout (see the test
docstrings for the exact R-dumped numbers).
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

import gtable_py
from ggplot2_py import element_rect, element_text, theme_grey

from ggh4x.strip_nested import StripNested, _rle_lengths, strip_nested
from ggh4x.strip_split import (
    StripSplit,
    _arg_match_multiple,
    _match_first,
    _split_max,
    strip_split,
)
from ggh4x.strip_tag import StripTag, strip_tag
from ggh4x.strip_themed import StripThemed, strip_themed
from ggh4x.strip_vanilla import Strip, _col_index


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
def _mpg_layout() -> pd.DataFrame:
    """The de-duplicated ``facet_wrap2(vars(cyl, drv))`` layout for ``mpg``.

    Matches the R ``ggplot_build`` layout (9 panels, ROW = 1:3 each x 3 cols).
    """
    return pd.DataFrame(
        {
            "PANEL": list(range(1, 10)),
            "ROW": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "COL": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "cyl": [4, 4, 5, 6, 6, 6, 8, 8, 8],
            "drv": ["4", "f", "f", "4", "f", "r", "4", "f", "r"],
        }
    )


def _label_matrix(layout: pd.DataFrame) -> np.ndarray:
    """A 9x2 column-major label matrix ``[cyl, drv]`` (rows = panels)."""
    mat = np.empty((len(layout), 2), dtype=object)
    for i in range(len(layout)):
        mat[i, 0] = str(layout["cyl"].iloc[i])
        mat[i, 1] = str(layout["drv"].iloc[i])
    return mat


def _panel_3x3():
    """A minimal 3x3 panel gtable named ``panel-<col>-<row>``."""
    from grid_py import Unit, null_grob
    from gtable_py import Gtable, gtable_add_grob

    gt = Gtable(
        widths=Unit([1, 0.1, 1, 0.1, 1], ["null", "cm", "null", "cm", "null"]),
        heights=Unit([1, 0.1, 1, 0.1, 1], ["null", "cm", "null", "cm", "null"]),
    )
    for r, row in enumerate([1, 3, 5]):
        for c, col in enumerate([1, 3, 5]):
            gt = gtable_add_grob(
                gt, null_grob(), t=row, l=col, name=f"panel-{c + 1}-{r + 1}"
            )
    return gt


# ---------------------------------------------------------------------------
# Construction + inheritance
# ---------------------------------------------------------------------------
def test_constructors_return_correct_types():
    assert isinstance(strip_themed(), StripThemed)
    assert isinstance(strip_nested(), StripNested)
    assert isinstance(strip_split(), StripSplit)
    assert isinstance(strip_tag(), StripTag)


def test_inheritance_chain():
    # StripThemed(Strip) -> StripNested(StripThemed) -> StripSplit(StripNested);
    # StripTag(StripThemed) is a sibling of Nested.
    assert issubclass(StripThemed, Strip)
    assert issubclass(StripNested, StripThemed)
    assert issubclass(StripSplit, StripNested)
    assert issubclass(StripTag, StripThemed)
    assert not issubclass(StripTag, StripNested)


def test_all_are_strip_instances():
    for ctor in (strip_themed, strip_nested, strip_split, strip_tag):
        assert isinstance(ctor(), Strip)


def test_strip_themed_class_name_is_stripelemental():
    # R class string is "StripElemental" even though the object is StripThemed.
    assert StripThemed._class_name == "StripElemental"


def test_params_carry_subclass_fields():
    assert strip_nested(bleed=True).params["bleed"] is True
    assert strip_split(["bottom", "right"]).params["position"] == ["bottom", "right"]
    g = strip_tag(order=("y", "x"), just=(1, 0))
    assert g.params["order"] == ["y", "x"]
    assert g.params["just"] == [1, 0]


def test_given_elements_present_on_themed():
    s = strip_themed(by_layer_x=True)
    assert set(s.given_elements) == {
        "text_x",
        "text_y",
        "background_x",
        "background_y",
        "by_layer_x",
        "by_layer_y",
    }
    assert s.given_elements["by_layer_x"] is True


def test_bad_clip_rejected():
    with pytest.raises(Exception):
        strip_nested(clip="bogus")


def test_split_bad_position_rejected():
    with pytest.raises(Exception):
        strip_split(["top", "diagonal"])


# ---------------------------------------------------------------------------
# StripThemed.setup_elements + by_layer
# ---------------------------------------------------------------------------
def test_themed_setup_elements_keys_and_by_layer():
    s = strip_themed()
    el = s.setup_elements(theme_grey(), "wrap")
    assert set(el) == {"padding", "background", "text", "inside", "by_layer"}
    assert el["by_layer"] == {"x": False, "y": False}


def test_themed_setup_elements_lists_when_given():
    s = strip_themed(
        text_x=[element_text(colour="red"), element_text(face="bold")],
        background_x=[element_rect(fill="white"), element_rect(fill="grey80")],
    )
    el = s.setup_elements(theme_grey(), "wrap")
    # user lists -> per-element lists (top + bottom independently inherited).
    assert isinstance(el["text"]["x"]["top"], list) and len(el["text"]["x"]["top"]) == 2
    assert isinstance(el["text"]["x"]["bottom"], list) and len(el["text"]["x"]["bottom"]) == 2
    assert isinstance(el["background"]["x"], list) and len(el["background"]["x"]) == 2
    # y untouched -> single grob / element.
    assert not isinstance(el["background"]["y"], list)


def test_themed_by_layer_maps_elements_to_layers():
    # by_layer_x=True -> text[0] on layer 1, text[1] on layer 2.
    s = strip_themed(
        text_x=[element_text(colour="red"), element_text(face="bold")],
        by_layer_x=True,
    )
    el = s.setup_elements(theme_grey(), "wrap")
    labels = np.array([["4", "4"], ["4", "f"], ["5", "f"]], dtype=object)
    index = _col_index(labels)  # [1,1,1,2,2,2]
    out = s.init_strip(el, "top", index)
    assert len(out["el"]) == 6
    assert getattr(out["el"][0], "colour", None) == "red"  # layer 1
    assert getattr(out["el"][3], "face", None) == "bold"  # layer 2


def test_themed_by_layer_false_recycles():
    s = strip_themed(
        text_x=[element_text(colour="red"), element_text(colour="blue")],
        by_layer_x=False,
    )
    el = s.setup_elements(theme_grey(), "wrap")
    # 3 cells in layer 1 only -> rep_len recycles red, blue, red.
    out = s.init_strip(el, "top", [1, 1, 1])
    cols = [getattr(e, "colour", None) for e in out["el"]]
    assert cols == ["red", "blue", "red"]


# ---------------------------------------------------------------------------
# StripNested -- RLE merge (LOAD-BEARING, gold standard from Rscript)
# ---------------------------------------------------------------------------
def test_rle_lengths():
    assert _rle_lengths(["a", "a", "b", "c", "c", "c"]) == [2, 1, 3]
    assert _rle_lengths([]) == []
    assert _rle_lengths(["x"]) == [1]


def _nested_assemble(position, bleed=False):
    layout = _mpg_layout()
    labels = _label_matrix(layout)
    s = strip_nested(bleed=bleed)
    s._set(elements=s.setup_elements(theme_grey(), "wrap"))
    return s.assemble_strip(labels, position, s.elements, s.params, layout)


def test_nested_top_merge_runs():
    """R gold standard (facet_wrap2(vars(cyl,drv)), top):

    layer 1 (cyl) runs -> (t,b) = (1,2),(3,3),(4,6),(7,9);
    layer 2 (drv) -> 9 single-panel cells; layer col = [1]*4 + [2]*9.
    """
    out = _nested_assemble("top")
    assert list(out["layer"]) == [1, 1, 1, 1] + [2] * 9
    l1 = out[out["layer"] == 1]
    assert list(zip(l1["t"], l1["b"])) == [(1, 2), (3, 3), (4, 6), (7, 9)]
    l2 = out[out["layer"] == 2]
    assert list(l2["t"]) == list(range(1, 10))
    assert list(l2["b"]) == list(range(1, 10))
    assert len(out["grobs"]) == 13


def test_nested_bottom_reversed_index_same_runs():
    """Bottom shares the top's run structure (R gold standard)."""
    out = _nested_assemble("bottom")
    assert list(out["layer"]) == [1, 1, 1, 1] + [2] * 9
    l1 = out[out["layer"] == 1]
    assert list(zip(l1["t"], l1["b"])) == [(1, 2), (3, 3), (4, 6), (7, 9)]


def test_nested_right_merge_runs():
    """R gold standard (right): layer 1 spans (t,b) = (1,7),(2,8),(3,3),(6,9)."""
    out = _nested_assemble("right")
    l1 = out[out["layer"] == 1]
    assert list(zip(l1["t"], l1["b"])) == [(1, 7), (2, 8), (3, 3), (6, 9)]


def test_nested_left_no_merge_single_cells():
    """R gold standard (left): layer 1 cyl ordered by (COL,ROW), all single."""
    out = _nested_assemble("left")
    l1 = out[out["layer"] == 1]
    # ordered by COL then ROW -> t = 1,4,7,2,5,8,3,6,9.
    assert list(l1["t"]) == [1, 4, 7, 2, 5, 8, 3, 6, 9]
    assert list(l1["b"]) == list(l1["t"])  # no merge


def test_nested_bleed_true_merges_inner_layer():
    """R gold standard (bleed=True, top): drv panels 2,3 merge to (t=2,b=3)."""
    out = _nested_assemble("top", bleed=True)
    l2 = out[out["layer"] == 2]
    assert list(zip(l2["t"], l2["b"]))[:3] == [(1, 1), (2, 3), (4, 4)]


def test_nested_monolayer_fast_path_delegates_to_base():
    # single column -> base assembly -> no 'layer' col, t=l=b=r=PANEL.
    layout = pd.DataFrame(
        {"PANEL": [1, 2, 3], "ROW": [1, 1, 1], "COL": [1, 2, 3], "cyl": [4, 5, 6]}
    )
    labels = np.array([["4"], ["5"], ["6"]], dtype=object)
    s = strip_nested()
    s._set(elements=s.setup_elements(theme_grey(), "wrap"))
    out = s.assemble_strip(labels, "top", s.elements, s.params, layout)
    assert "layer" not in out.columns
    assert list(out["t"]) == [1, 2, 3]
    assert list(out["l"]) == list(out["r"]) == [1, 2, 3]


def test_nested_merged_grob_is_gtable():
    out = _nested_assemble("top")
    g = out["grobs"].iloc[0]
    assert gtable_py.is_gtable(g)


def test_nested_finish_strip_delegates_when_no_layer_col():
    # finish_strip on a plain (no 'layer') layout -> base behaviour: an empty
    # grob list returns the raw (zero) strip as the grobs column, t=l=b=r=PANEL.
    s = strip_nested()
    s._set(elements=s.setup_elements(theme_grey(), "wrap"))
    layout = pd.DataFrame({"PANEL": [1]})
    out = s.finish_strip([None], None, None, "top", layout, (1, 1), "inherit")
    assert "layer" not in out.columns
    assert list(out["t"]) == [1]
    assert list(out["b"]) == list(out["l"]) == list(out["r"]) == [1]


def test_nested_full_setup_wrap_placement():
    layout = _mpg_layout()
    s = strip_nested()
    s.setup(layout, {"facets": ["cyl", "drv"], "labeller": None}, theme_grey(), "wrap")
    top = s.strips["x"]["top"]
    assert list(top["layer"]) == [1, 1, 1, 1] + [2] * 9
    l1 = top[top["layer"] == 1]
    assert list(zip(l1["t"], l1["b"])) == [(1, 2), (3, 3), (4, 6), (7, 9)]


# ---------------------------------------------------------------------------
# StripSplit -- id() spans (LOAD-BEARING, gold standard from Rscript)
# ---------------------------------------------------------------------------
def test_split_arg_match_multiple_valid():
    assert _arg_match_multiple(["top", "left"], ["top", "bottom", "left", "right"]) == [
        "top",
        "left",
    ]


def test_split_arg_match_multiple_invalid():
    with pytest.raises(Exception):
        _arg_match_multiple(["top", "nope"], ["top", "bottom", "left", "right"])


def test_split_helpers_split_max_and_match():
    # split(layout$COL, ids) max + match-first, from the R cyl example.
    key = [1, 1, 2, 3, 3, 3, 4, 4, 4]
    col = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    assert _split_max(col, key) == [2, 3, 3, 3]
    assert _match_first([2, 3, 3, 3], col) == [2, 3, 3, 3]


def test_split_setup_default_top_left():
    """R gold standard (strip_split() default, facet_wrap2(vars(cyl,drv))).

    Verified by end-to-end ggplotGrob render parity with R: top(cyl) = 4 strips,
    left(drv) = 9 strips (one per panel).  The secondary strip uses R's
    CUMULATIVE id ``id(vars[, 1:k])`` so it is NOT merged across the primary
    variable.  (The earlier ``[1,2,6]`` left expectation encoded the pre-fix
    single-column id and was never actually checked against R.)

    top (cyl):  dedup PANELs [1,3,4,7], strp.b = [1,1,4,7], strp.r = [2,3,3,3];
    left (drv): one per panel, strp.b = [1,1,1,4,4,4,7,7,7], strp.r = [1,2,3]x3.
    bottom / right unused -> None.
    """
    layout = _mpg_layout()
    s = strip_split()  # ("top", "left")
    s.setup(layout, {"facets": ["cyl", "drv"], "labeller": None}, theme_grey(), "wrap")

    top = s.strips["x"]["top"]
    assert list(top["t"]) == [1, 3, 4, 7]
    assert list(top["b"]) == [1, 1, 4, 7]
    assert list(top["l"]) == [1, 3, 4, 7]
    assert list(top["r"]) == [2, 3, 3, 3]

    left = s.strips["y"]["left"]
    assert list(left["t"]) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert list(left["b"]) == [1, 1, 1, 4, 4, 4, 7, 7, 7]
    assert list(left["l"]) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert list(left["r"]) == [1, 2, 3, 1, 2, 3, 1, 2, 3]

    assert s.strips["x"]["bottom"] is None
    assert s.strips["y"]["right"] is None


def test_split_position_recycle_warns():
    layout = _mpg_layout()
    # single position for 2 vars -> recycled with a warning.
    s = strip_split(["bottom"])
    with pytest.warns(UserWarning):
        s.setup(
            layout, {"facets": ["cyl", "drv"], "labeller": None}, theme_grey(), "wrap"
        )
    # both vars on bottom -> top/left/right None.
    assert s.strips["x"]["top"] is None
    assert s.strips["x"]["bottom"] is not None


def test_split_incorporate_grid_runs():
    layout = _mpg_layout()
    s = strip_split()
    s.setup(layout, {"facets": ["cyl", "drv"], "labeller": None}, theme_grey(), "wrap")
    out = s.incorporate_grid(_panel_3x3(), False)
    names = [str(n) for n in out.layout["name"] if str(n).startswith("strip-")]
    # R gold standard (cumulative id, ggplotGrob-verified): 4 top (cyl) +
    # 9 left (drv, one per panel -- not merged across cyl).
    assert sum(n.startswith("strip-t-") for n in names) == 4
    assert sum(n.startswith("strip-l-") for n in names) == 9


def test_split_incorporate_wrap_delegates_to_grid():
    layout = _mpg_layout()
    s = strip_split()
    s.setup(layout, {"facets": ["cyl", "drv"], "labeller": None}, theme_grey(), "wrap")
    out = s.incorporate_wrap(_panel_3x3(), "top", clip="off", sizes=None)
    names = [str(n) for n in out.layout["name"] if str(n).startswith("strip-")]
    assert sum(n.startswith("strip-t-") for n in names) == 4


def test_split_empty_vars_returns_none_nested():
    # wrap with no facets -> "(all)" single strip; grid with empty -> all None.
    s = strip_split()
    s.setup(
        pd.DataFrame({"PANEL": [1], "ROW": [1], "COL": [1]}),
        {"cols": {}, "rows": {}, "labeller": None},
        theme_grey(),
        "grid",
    )
    # union(rows, cols) empty -> empty vars -> all sides None.
    assert s.strips["x"]["top"] is None
    assert s.strips["y"]["left"] is None


# ---------------------------------------------------------------------------
# StripTag -- fitted box / viewport
# ---------------------------------------------------------------------------
def test_tag_setup_per_panel_no_dedup():
    # tags are per-panel: 3 distinct class values -> 3 tags (no de-dup).
    layout = pd.DataFrame(
        {"PANEL": [1, 2, 3], "ROW": [1, 1, 1], "COL": [1, 2, 3], "class": ["a", "b", "c"]}
    )
    s = strip_tag()
    s.setup(layout, {"facets": ["class"], "labeller": None}, theme_grey(), "wrap")
    top = s.strips["x"]["top"]
    # placement t=l=b=r=PANEL (tag sits on the panel cell).
    assert list(top["t"]) == [1, 2, 3]
    assert list(top["l"]) == list(top["b"]) == list(top["r"]) == [1, 2, 3]
    assert len(top["grobs"]) == 3


def test_tag_finish_strip_has_self():
    # StripTag.finish_strip has self (unlike base self-less finish_strip).
    s = strip_tag()
    assert isinstance(s.finish_strip, types.MethodType)
    # draw_labels remains self-less.
    assert isinstance(s.draw_labels, types.FunctionType)


def test_tag_fitted_box_viewport():
    layout = pd.DataFrame(
        {"PANEL": [1, 2, 3], "ROW": [1, 1, 1], "COL": [1, 2, 3], "class": ["a", "b", "c"]}
    )
    s = strip_tag(just=(0, 1))
    s.setup(layout, {"facets": ["class"], "labeller": None}, theme_grey(), "wrap")
    g = s.strips["x"]["top"]["grobs"].iloc[0]
    assert gtable_py.is_gtable(g)
    # viewport anchored at just=(0,1); npc-fraction widths inside a cm box.
    assert tuple(g.vp.just) == (0.0, 1.0)
    w = g.widths
    # single layer -> one npc width summing to 1.
    from grid_py import convert_unit

    frac = float(np.atleast_1d(convert_unit(w, "npc", valueOnly=True))[0])
    assert frac == pytest.approx(1.0, abs=1e-6)


def test_tag_clip_on_clamps_viewport_to_npc():
    layout = pd.DataFrame(
        {"PANEL": [1], "ROW": [1], "COL": [1], "class": ["averylonglabelXXXX"]}
    )
    s = strip_tag(clip="on")
    s.setup(layout, {"facets": ["class"], "labeller": None}, theme_grey(), "wrap")
    g = s.strips["x"]["top"]["grobs"].iloc[0]
    # clip='on' -> vp built; width is a unit (clamp applied via unit_pmin).
    assert g.vp is not None
    assert g.vp.clip in ("on", True)


def test_tag_incorporate_wrap_places_on_panel():
    layout = pd.DataFrame(
        {"PANEL": [1, 2, 3], "ROW": [1, 1, 1], "COL": [1, 2, 3], "class": ["a", "b", "c"]}
    )
    s = strip_tag()
    s.setup(layout, {"facets": ["class"], "labeller": None}, theme_grey(), "wrap")
    out = s.incorporate_wrap(_panel_3x3(), "top", clip="off")
    names = [str(n) for n in out.layout["name"] if str(n).startswith("strip-")]
    assert len(names) == 3
    # no new rows/cols added (tags sit on panel cells).
    assert len(out.heights) == 5
    assert len(out.widths) == 5


def test_tag_incorporate_grid_combines_x_and_y():
    # grid layout: cyl on x (top), drv on y (left) -> rbind into one tag/panel.
    layout = _mpg_layout()
    s = strip_tag()
    s.setup(
        layout,
        {"cols": {"cyl": "cyl"}, "rows": {"drv": "drv"}, "labeller": None},
        theme_grey(),
        "grid",
    )
    out = s.incorporate_grid(_panel_3x3(), None)
    names = [str(n) for n in out.layout["name"] if str(n).startswith("strip-")]
    # one combined (x+y rbind) tag per panel.
    assert len(names) == 9


def test_tag_incorporate_grid_order_reversed():
    layout = _mpg_layout()
    s = strip_tag(order=("y", "x"))
    s.setup(
        layout,
        {"cols": {"cyl": "cyl"}, "rows": {"drv": "drv"}, "labeller": None},
        theme_grey(),
        "grid",
    )
    # just exercise the reversed-order rbind path without error.
    out = s.incorporate_grid(_panel_3x3(), None)
    names = [str(n) for n in out.layout["name"] if str(n).startswith("strip-")]
    assert len(names) == 9
