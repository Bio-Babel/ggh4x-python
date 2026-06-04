"""R-parity tests for ``ggh4x.multiscale`` (scale_multi / scale_listed / scale_manual).

All expected values were produced by running the R ``ggh4x`` package live with
``ggplot2`` 4.0.2 (``ggh4x:::pickvalue``, ``ScaleManualPosition$train``/``$map``,
``scale_fill_multi`` / ``scale_listed`` container structure, and the default
``white``/``black`` gradient palette midpoint ``#777777``) and pasted here as
fixtures.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import ggplot2_py as gg
from ggplot2_py import aes, ggplot

import ggh4x.multiscale  # noqa: F401  (registers the MultiScale add handler)
from ggh4x.multiscale import (
    MultiScale,
    ScaleManualPosition,
    scale_color_multi,
    scale_colour_multi,
    scale_fill_multi,
    scale_listed,
    scale_x_manual,
    scale_y_manual,
    sep_discrete,
)
from ggh4x.multiscale.scale_multi import _is_per_aesthetic_list, _pickvalue


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _factor(vals, levels):
    return pd.Categorical(vals, categories=levels)


def _label_to_colour(grob):
    """Extract the colour from a string-legend label grob's ``Gpar`` repr."""
    txt = list(grob._children.values())[0]
    return getattr(txt._gp, "_col", None) if hasattr(txt._gp, "_col") else repr(txt._gp)


# ==========================================================================
# pickvalue  (scale_multi.R:174-183)
# ==========================================================================
class TestPickValue:
    def test_per_aesthetic_nested_list(self):
        # R: pickvalue(list(c("white","red"), c("black","blue")), i)
        x = [["white", "red"], ["black", "blue"]]
        assert _pickvalue(x, 0) == ["white", "red"]
        assert _pickvalue(x, 1) == ["black", "blue"]

    def test_per_aesthetic_wraps_when_index_exceeds(self):
        # R: i <- if (i > length(x)) 1 else i  -> first element
        assert _pickvalue([["only"]], 5) == ["only"]

    def test_flat_colour_vector_is_broadcast(self):
        # R: c("white","black") is a *vector* -> returned whole for every i.
        assert _pickvalue(["white", "black"], 0) == ["white", "black"]
        assert _pickvalue(["white", "black"], 1) == ["white", "black"]

    def test_scalar_is_broadcast(self):
        assert _pickvalue("transparent", 4) == "transparent"

    def test_tuple_and_ndarray_are_broadcast(self):
        assert _pickvalue(("white", "red"), 1) == ("white", "red")
        arr = np.array([0.0, 1.0])
        assert _pickvalue(arr, 1) is arr

    def test_classification(self):
        assert _is_per_aesthetic_list([["a", "b"], ["c"]]) is True
        assert _is_per_aesthetic_list(["a", "b"]) is False
        assert _is_per_aesthetic_list("a") is False
        assert _is_per_aesthetic_list(("a", "b")) is False


# ==========================================================================
# scale_fill_multi / scale_colour_multi  (scale_multi.R:48-110)
# ==========================================================================
class TestScaleMulti:
    def test_fill_multi_container_structure(self):
        sm = scale_fill_multi(
            aesthetics=["fill1", "fill2", "fill3"],
            colours=[["white", "red"], ["black", "blue"], ["grey50", "green"]],
        )
        assert isinstance(sm, MultiScale)
        assert sm.aes == ["fill1", "fill2", "fill3"]
        assert sm.replaced_aes == "fill"
        assert len(sm.scales) == 3
        assert sm.scales[0].aesthetics == ["fill1"]

    def test_fill_multi_guide_is_colourbar_with_single_available_aes(self):
        sm = scale_fill_multi(aesthetics=["fill1", "fill2"])
        # R: guide class GuideColourbar, available_aes == "fill1".
        assert type(sm.scales[0].guide).__name__ == "GuideColourbar"
        assert sm.scales[0].guide.available_aes == ["fill1"]
        assert sm.scales[1].guide.available_aes == ["fill2"]

    def test_fill_multi_default_palette_midpoint(self):
        # R: default colours white/black -> palette(0.5) == "#777777".
        sm = scale_fill_multi(aesthetics=["fill1"])
        assert list(sm.scales[0].palette(0.5)) == ["#777777"]

    def test_fill_multi_na_value_default_transparent(self):
        sm = scale_fill_multi(aesthetics=["fill1"])
        assert sm.scales[0].na_value == "transparent"

    def test_per_aesthetic_palettes_match_r(self):
        sm = scale_fill_multi(
            aesthetics=["fill1", "fill2", "fill3"],
            colours=[["white", "red"], ["black", "blue"], ["grey50", "green"]],
        )
        # R gradient midpoints (scales::gradient_n_pal in Lab space).
        assert list(sm.scales[0].palette(0.5)) == ["#ff9e81"]
        assert list(sm.scales[1].palette(0.5)) == ["#241178"]
        assert list(sm.scales[2].palette(0.5)) == ["#70bf5d"]

    def test_colour_multi_replaced_aes(self):
        sm = scale_colour_multi(aesthetics=["colour1"])
        assert sm.replaced_aes == "colour"

    def test_colors_american_alias(self):
        sm = scale_fill_multi(aesthetics=["fill1"], colors=["white", "red"])
        assert list(sm.scales[0].palette(0.5)) == ["#ff9e81"]

    def test_scale_color_multi_alias(self):
        sm = scale_color_multi(aesthetics=["colour1"])
        assert sm.replaced_aes == "colour"
        assert isinstance(sm, MultiScale)

    def test_invalid_guide_raises(self):
        with pytest.raises(ValueError):
            scale_fill_multi(aesthetics=["fill1"], guide="not_a_guide")

    def test_full_plot_build_rewrites_geoms(self):
        df = pd.concat(
            [
                pd.DataFrame({"x": [1, 2, 3], "y": 1, "v": [np.nan] * 3, "w": [1, 2, 3], "z": [np.nan] * 3}),
                pd.DataFrame({"x": [1, 2, 3], "y": 2, "v": [1, 2, 3], "w": [np.nan] * 3, "z": [np.nan] * 3}),
                pd.DataFrame({"x": [1, 2, 3], "y": 3, "v": [np.nan] * 3, "w": [np.nan] * 3, "z": [1, 2, 3]}),
            ],
            ignore_index=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = (
                ggplot(df, aes("x", "y"))
                + gg.geom_raster(aes(fill1="v"))
                + gg.geom_raster(aes(fill2="w"))
                + gg.geom_raster(aes(fill3="z"))
                + scale_fill_multi(
                    aesthetics=["fill1", "fill2", "fill3"],
                    colours=[["white", "red"], ["black", "blue"], ["grey50", "green"]],
                )
            )
            # Geom rewrite: each layer's geom default_aes uses the non-standard aes.
            assert "fill1" in p.layers[0].geom.default_aes
            assert "fill2" in p.layers[1].geom.default_aes
            assert "fill3" in p.layers[2].geom.default_aes
            assert len(p.scales.scales) == 3
            # Full render goes through the handle_na / draw_key column-rename shims.
            b = gg.ggplot_build(p)
            gt = gg.ggplot_gtable(b)
        assert type(gt).__name__ == "Gtable"


# ==========================================================================
# scale_listed  (scale_listed.R:56-133)
# ==========================================================================
class TestScaleListed:
    def test_single_group_structure(self):
        sl = scale_listed(
            [
                gg.scale_fill_brewer(palette="Set1", aesthetics="spec"),
                gg.scale_fill_brewer(palette="Dark2", aesthetics="leav"),
            ],
            replaces=["fill", "fill"],
        )
        # R: a single MultiScale (both replace 'fill').
        assert len(sl) == 1
        assert isinstance(sl[0], MultiScale)
        assert sl[0].aes == ["spec", "leav"]
        assert sl[0].replaced_aes == "fill"
        assert len(sl[0].scales) == 2

    def test_guide_available_aes_union_with_any(self):
        # guide_legend default available_aes == ['any'] -> union appends aes.
        sl = scale_listed(
            [gg.scale_fill_brewer(aesthetics="spec")], replaces=["fill"]
        )
        assert sl[0].scales[0].guide.available_aes == ["any", "spec"]

    def test_split_by_replaces_alphabetical(self):
        # R split() orders groups by factor level (alphabetical): colour, fill.
        sl = scale_listed(
            [gg.scale_fill_brewer(aesthetics="a"), gg.scale_colour_brewer(aesthetics="b")],
            replaces=["fill", "colour"],
        )
        assert [m.replaced_aes for m in sl] == ["colour", "fill"]
        assert [m.aes for m in sl] == [["b"], ["a"]]

    def test_replaces_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            scale_listed([gg.scale_fill_brewer(aesthetics="a")], replaces=["fill", "colour"])

    def test_invalid_replaces_aesthetic_raises(self):
        with pytest.raises(ValueError):
            scale_listed([gg.scale_fill_brewer(aesthetics="a")], replaces=["not_an_aes"])

    def test_non_scale_element_raises(self):
        with pytest.raises(ValueError):
            scale_listed(["not a scale"], replaces=["fill"])

    def test_multiple_aesthetics_per_scale_raises(self):
        s = gg.scale_fill_brewer(aesthetics=["a", "b"])
        with pytest.raises(ValueError):
            scale_listed([s], replaces=["fill"])

    def test_full_plot_build(self):
        df = pd.DataFrame(
            {
                "x": np.repeat(np.arange(1, 6), 5),
                "y": np.tile(np.arange(1, 6), 5),
                "value": np.random.RandomState(0).rand(25),
            }
        )
        ann = pd.DataFrame(
            {"z": np.arange(1, 6), "Species": ["s"] * 3 + ["v"] * 2, "Leaves": ["Short"] * 2 + ["Long"] * 3}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = (
                ggplot(df, aes("x", "y"))
                + gg.geom_raster(aes(fill="value"))
                + gg.geom_tile(data=ann, mapping=aes(x="z", y=-5, spec="Species"), height=5)
                + gg.geom_tile(data=ann, mapping=aes(y="z", x=-5, leav="Leaves"), width=5)
                + scale_listed(
                    [
                        gg.scale_fill_brewer(palette="Set1", aesthetics="spec"),
                        gg.scale_fill_brewer(palette="Dark2", aesthetics="leav"),
                    ],
                    replaces=["fill", "fill"],
                )
            )
            b = gg.ggplot_build(p)
        assert len(b.data) == 3


# ==========================================================================
# ScaleManualPosition  (scale_manual.R:48-206)
# ==========================================================================
class TestScaleManual:
    def test_x_constructor_aesthetics(self):
        sc = scale_x_manual(values=[1, 2, 4])
        assert isinstance(sc, ScaleManualPosition)
        assert sc.aesthetics == ["x", "xmin", "xmax", "xend"]

    def test_y_constructor_aesthetics(self):
        sc = scale_y_manual(values=[1, 2, 4])
        assert sc.aesthetics == ["y", "ymin", "ymax", "yend"]

    def test_train_plain_values_bakes_expansion(self):
        # R: values c(1,2,4,6,7,9,10), train factor a/b/c -> range(1,2,4)=[1,4]
        #    + expansion(add=0.6) -> range_c = [0.4, 4.6].
        sc = scale_x_manual(values=[1, 2, 4, 6, 7, 9, 10])
        sc.train(_factor(["a", "b", "c"], ["a", "b", "c"]))
        assert sc.range_c.range == pytest.approx((0.4, 4.6))

    def test_map_plain_values(self):
        sc = scale_x_manual(values=[1, 2, 4, 6, 7, 9, 10])
        sc.train(_factor(["a", "b", "c"], ["a", "b", "c"]))
        m = sc.map(["a", "b", "c"])
        assert type(m).__name__ == "_MappedDiscrete"
        assert list(np.asarray(m)) == pytest.approx([1.0, 2.0, 4.0])

    def test_c_limits_na_aware_override(self):
        # R: values c(1,2,3) -> range [1,3]; c_limits c(NA,15) -> [1,15]
        #    + expansion -> range_c = [0.4, 15.6].
        sc = scale_x_manual(values=[1, 2, 3], c_limits=[np.nan, 15])
        sc.train(_factor(["a", "b", "c"], ["a", "b", "c"]))
        assert sc.range_c.range == pytest.approx((0.4, 15.6))

    def test_named_values_map_and_range(self):
        # R: values c(a=10,b=20,c=30); map(a,c,b) -> 10,30,20; range_c [9.4,30.6].
        sc = scale_x_manual(values={"a": 10, "b": 20, "c": 30})
        sc.train(_factor(["a", "b", "c"], ["a", "b", "c"]))
        assert list(sc.get_limits()) == ["a", "b", "c"]
        m = sc.map(["a", "c", "b"])
        assert list(np.asarray(m)) == pytest.approx([10.0, 30.0, 20.0])
        assert sc.range_c.range == pytest.approx((9.4, 30.6))

    def test_unnamed_values_named_by_breaks(self):
        # R: unnamed values + breaks -> name values by breaks; map(c,a,b)=30,10,20.
        sc = scale_x_manual(values=[10, 20, 30], breaks=["a", "b", "c"])
        sc.train(_factor(["a", "b", "c"], ["a", "b", "c"]))
        m = sc.map(["c", "a", "b"])
        assert list(np.asarray(m)) == pytest.approx([30.0, 10.0, 20.0])

    def test_full_boxplot_render(self):
        df = pd.DataFrame({"g": ["a", "a", "b", "b", "c", "c"], "v": [1.0, 2, 3, 4, 5, 6]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = ggplot(df, aes("g", "v")) + gg.geom_boxplot() + scale_x_manual(values=[1, 2, 4])
            gt = gg.ggplot_gtable(gg.ggplot_build(p))
        assert type(gt).__name__ == "Gtable"

    def test_sep_discrete_values(self):
        # R: sep_discrete()(c("0.4","1.4","1.5")) -> 1,2,4; range_c [0.4,4.6].
        sc = scale_x_manual(values=sep_discrete())
        sc.train(_factor(["0.4", "1.4", "1.5"], ["0.4", "1.4", "1.5"]))
        assert sc.range_c.range == pytest.approx((0.4, 4.6))
        m = sc.map(["0.4", "1.4", "1.5"])
        assert list(np.asarray(m)) == pytest.approx([1.0, 2.0, 4.0])

    def test_non_numeric_values_raises(self):
        with pytest.raises(TypeError):
            scale_x_manual(values=["a", "b", "c"])

    def test_c_limits_wrong_length_raises(self):
        with pytest.raises(ValueError):
            scale_x_manual(values=[1, 2, 3], c_limits=[1, 2, 3])

    def test_insufficient_values_raises_on_map(self):
        sc = scale_x_manual(values=[1, 2])
        with pytest.raises(ValueError):
            sc.train(_factor(["a", "b", "c"], ["a", "b", "c"]))


# ==========================================================================
# sep_discrete  (scale_manual.R:230-257) -- reused from conveniences
# ==========================================================================
class TestSepDiscrete:
    def test_default(self):
        # R: sep_discrete()(c("foo.bar","bar.bar","bar.qux")) -> 1 2 4
        out = sep_discrete()(["foo.bar", "bar.bar", "bar.qux"])
        assert list(out) == pytest.approx([1.0, 2.0, 4.0])

    def test_inv(self):
        # R: sep_discrete(inv=TRUE)(...) -> 1 3 4
        out = sep_discrete(inv=True)(["foo.bar", "bar.bar", "bar.qux"])
        assert list(out) == pytest.approx([1.0, 3.0, 4.0])

    def test_single_level(self):
        out = sep_discrete()(["a", "b", "c"])
        assert list(out) == pytest.approx([1.0, 2.0, 3.0])

    def test_ragged(self):
        # R: sep_discrete()(c("a.b.c","a.b","a")) -> 1 3 5
        out = sep_discrete()(["a.b.c", "a.b", "a"])
        assert list(out) == pytest.approx([1.0, 3.0, 5.0])


# ==========================================================================
# MultiScale dispatch  (scale_listed.R:142-203 via update_ggplot)
# ==========================================================================
class TestMultiScaleDispatch:
    def test_multiscale_is_distinct_class_for_singledispatch(self):
        from ggplot2_py.plot import update_ggplot

        # The handler must be registered specifically for MultiScale (not falling
        # through to the list/Mapping handlers).
        assert update_ggplot.dispatch(MultiScale) is not update_ggplot.dispatch(list)

    def test_list_of_multiscale_flows_through_list_handler(self):
        # scale_listed returns a list; the list handler dispatches each element.
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "g": ["a", "b", "c"]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = (
                ggplot(df, aes("x", "y"))
                + gg.geom_point(aes(spec="g"))
                + scale_listed([gg.scale_colour_brewer(aesthetics="spec")], replaces=["colour"])
            )
            assert len(p.scales.scales) == 1
            assert "spec" in p.layers[0].geom.default_aes
