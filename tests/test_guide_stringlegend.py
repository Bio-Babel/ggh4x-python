"""R-parity tests for ``ggh4x.guide_stringlegend`` (GuideStringlegend).

Expected values were produced by running the R ``ggh4x`` package live with
``ggplot2`` 4.0.2 (``guide_stringlegend`` structure, ``GuideStringlegend$setup_params``
nrow/ncol/sizes, and the coloured-text ``build_labels`` behaviour) and pasted here
as fixtures.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import ggplot2_py as gg
from ggplot2_py import aes, ggplot

from ggh4x.guide_stringlegend import GuideStringlegend, guide_stringlegend


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _label_colour(grob):
    """Return the colour applied to a string-legend label grob (from Gpar repr)."""
    txt = list(grob._children.values())[0]
    rep = repr(txt._gp)
    # Gpar repr: "Gpar(..., col='#RRGGBB', ...)"
    if "col='" in rep:
        return rep.split("col='", 1)[1].split("'", 1)[0]
    return None


def _setup(guide, key, direction="vertical"):
    params = dict(guide.params)
    params["key"] = key
    params["n_breaks"] = len(key)
    params["direction"] = direction
    params = type(guide).setup_params(params)
    elems = guide.setup_elements(params, dict(guide.elements), gg.theme_get())
    return params, elems


# ==========================================================================
# Constructor  (guide_stringlegend.R:22-44)
# ==========================================================================
class TestConstructor:
    def test_class_and_inheritance(self):
        g = guide_stringlegend(ncol=2)
        assert isinstance(g, GuideStringlegend)
        mro = [c.__name__ for c in type(g).__mro__]
        assert mro[:3] == ["GuideStringlegend", "GuideLegend", "Guide"]

    def test_available_aes(self):
        # R: available_aes == c("colour","fill","family","fontface").
        g = guide_stringlegend()
        assert g.available_aes == ["colour", "fill", "family", "fontface"]

    def test_name(self):
        g = guide_stringlegend()
        name = getattr(g, "name", None) or g.params.get("name")
        assert name == "stringlegend"

    def test_params_passthrough(self):
        g = guide_stringlegend(ncol=2, reverse=True, order=3)
        assert g.params.get("ncol") == 2
        assert g.params.get("reverse") is True
        assert g.params.get("order") == 3


# ==========================================================================
# get_layer_key  (guide_stringlegend.R:55-57)
# ==========================================================================
class TestGetLayerKey:
    def test_identity_passthrough(self):
        g = guide_stringlegend()
        params = {"foo": 1, "bar": 2}
        assert g.get_layer_key(params, layers=[], data=None) is params

    def test_arity_accepts_layers_and_data(self):
        g = guide_stringlegend()
        params = {"x": 1}
        # Must accept layers/data positionally so process_layers still dispatches.
        assert g.get_layer_key(params, [], None) is params


# ==========================================================================
# setup_params  (guide_stringlegend.R:59-63)
# ==========================================================================
class TestSetupParams:
    def test_zero_sizes(self):
        g = guide_stringlegend(ncol=2)
        params = dict(g.params)
        params["key"] = pd.DataFrame({".label": ["x", "y", "z"], "colour": ["r", "g", "b"]})
        params["n_breaks"] = 3
        params["direction"] = "vertical"
        out = type(g).setup_params(params)
        # R: nrow=2, ncol=2, sizes=list(widths=0, heights=0).
        assert out["nrow"] == 2
        assert out["ncol"] == 2
        assert out["sizes"] == {"widths": 0, "heights": 0}

    def test_is_staticmethod(self):
        # Base GuideLegend.setup_params is a @staticmethod(params); the override
        # must keep that surface so MRO dispatch from Guide.draw resolves it.
        import inspect

        static = inspect.getattr_static(GuideStringlegend, "setup_params")
        assert isinstance(static, staticmethod)


# ==========================================================================
# setup_elements  (guide_stringlegend.R:65-73)
# ==========================================================================
class TestSetupElements:
    def test_zero_key_size(self):
        from grid_py import convert_height, convert_width

        g = guide_stringlegend()
        key = pd.DataFrame({".label": ["4", "f", "r"], "colour": ["#F8766D", "#00BA38", "#619CFF"]})
        _, elems = _setup(g, key)
        # R: elements$key_height <- elements$key_width <- unit(0, "cm").
        assert float(np.sum(convert_width(elems["key_width"], "cm", valueOnly=True))) == 0.0
        assert float(np.sum(convert_height(elems["key_height"], "cm", valueOnly=True))) == 0.0

    def test_spacing_y_set(self):
        g = guide_stringlegend()
        key = pd.DataFrame({".label": ["a"], "colour": ["#000000"]})
        _, elems = _setup(g, key)
        assert elems.get("spacing_y") is not None

    def test_text_element_has_margin(self):
        g = guide_stringlegend()
        key = pd.DataFrame({".label": ["a"], "colour": ["#000000"]})
        _, elems = _setup(g, key)
        # legend.text margin pulled onto the resolved text element.
        assert hasattr(elems.get("text"), "margin")


# ==========================================================================
# build_labels  (guide_stringlegend.R:75-95)
# ==========================================================================
class TestBuildLabels:
    def test_one_grob_per_key_row(self):
        g = guide_stringlegend()
        key = pd.DataFrame({".label": ["4", "f", "r"], "colour": ["#F8766D", "#00BA38", "#619CFF"]})
        params, elems = _setup(g, key)
        labels = type(g).build_labels(key, elems, params)
        assert len(labels) == 3

    def test_labels_coloured_by_colour_column(self):
        g = guide_stringlegend()
        key = pd.DataFrame({".label": ["4", "f", "r"], "colour": ["#F8766D", "#00BA38", "#619CFF"]})
        params, elems = _setup(g, key)
        labels = type(g).build_labels(key, elems, params)
        assert [_label_colour(l) for l in labels] == ["#F8766D", "#00BA38", "#619CFF"]

    def test_labels_fall_back_to_fill(self):
        # R: colour <- key$colour %||% key$fill (whole-column coalesce).
        g = guide_stringlegend()
        key = pd.DataFrame({".label": ["a", "b"], "fill": ["#112233", "#445566"]})
        params, elems = _setup(g, key)
        labels = type(g).build_labels(key, elems, params)
        assert [_label_colour(l) for l in labels] == ["#112233", "#445566"]

    def test_empty_label_branch_returns_null_grobs(self):
        # R: n_labels < 1 -> rep(list(zeroGrob()), nrow(key)).
        g = guide_stringlegend()
        key = pd.DataFrame({"colour": ["red", "green"]})  # no .label column
        params, elems = _setup(g, key)
        labels = type(g).build_labels(key, elems, params)
        assert len(labels) == 2

    def test_missing_family_fontface_columns_tolerated(self):
        # family/fontface columns may be absent; build_labels must not KeyError.
        g = guide_stringlegend()
        key = pd.DataFrame({".label": ["a", "b"], "colour": ["#111111", "#222222"]})
        params, elems = _setup(g, key)
        labels = type(g).build_labels(key, elems, params)
        assert len(labels) == 2


# ==========================================================================
# build_decor  (guide_stringlegend.R:97)
# ==========================================================================
class TestBuildDecor:
    def test_returns_empty_grobs_shaped_to_n_breaks(self):
        # R returns a single zeroGrob; ggplot2_py's procedural assembly iterates
        # decor by index, so we return one null_grob per key row.
        g = guide_stringlegend()
        key = pd.DataFrame({".label": ["a", "b", "c"], "colour": ["#1", "#2", "#3"]})
        params, elems = _setup(g, key)
        decor = type(g).build_decor(None, None, elems, params)
        assert len(decor) == 3
        # Every decor grob is empty (no key swatch).
        for d in decor:
            assert type(d).__name__ in ("Grob", "NullGrob", "ZeroGrob")

    def test_absorbs_positional_args(self):
        g = guide_stringlegend()
        # Guide.draw calls build_decor(decor, grobs, elems, params) positionally.
        out = type(g).build_decor("decor", "grobs", {}, {"n_breaks": 0, "key": None})
        assert out == []


# ==========================================================================
# Full render  (via the working scale guide= path)
# ==========================================================================
class TestFullRender:
    def test_string_legend_renders_via_scale_guide(self):
        rng = np.random.RandomState(1)
        mpg = pd.DataFrame(
            {
                "displ": rng.rand(30) * 5 + 1,
                "hwy": rng.rand(30) * 20 + 15,
                "drv": rng.choice(["4", "f", "r"], 30),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = (
                ggplot(mpg, aes("displ", "hwy"))
                + gg.geom_point(aes(colour="drv"))
                + gg.scale_colour_discrete(guide=guide_stringlegend())
            )
            b = gg.ggplot_build(p)
            gt = gg.ggplot_gtable(b)
        assert type(gt).__name__ == "Gtable"
