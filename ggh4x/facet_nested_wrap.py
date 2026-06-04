"""Nested-strip wrapped facets (port of ggh4x ``R/facet_nested_wrap.R``).

``facet_nested_wrap()`` wraps a sequence of panels onto a 2-D layout (like
:func:`ggh4x.facet_wrap2`) and merges grouped strips where they fall in the same
row / column, drawing hierarchy ``nest_line``s between strip layers.

The :class:`FacetNestedWrap` ggproto subclasses
:class:`ggh4x.facet_wrap2.FacetWrap2` and overrides a single seam --
:meth:`FacetNestedWrap.finish_panels` -- which delegates to the shared
:func:`ggh4x.facet_nested.add_nest_indicator` helper (identical to
``FacetNested``).  Everything else (layout, axes, the label-merging strip) is
inherited.

The default strip is :func:`ggh4x.strip_nested.strip_nested`, matching R's
``strip = "nested"``.

R source: ``ggh4x/R/facet_nested_wrap.R``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import pandas as pd

from grid_py import Unit

from ggh4x.facet_nested import _coerce_nest_line, add_nest_indicator
from ggh4x.facet_wrap2 import FacetWrap2, new_wrap_facets
from ggh4x.strip_nested import strip_nested

# Importing this module registers the ``ggh4x.facet.nestline`` theme element.
import ggh4x.themes_ggh4x  # noqa: F401

__all__ = [
    "facet_nested_wrap",
    "FacetNestedWrap",
]


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------
def facet_nested_wrap(
    facets: Any,
    nrow: Optional[int] = None,
    ncol: Optional[int] = None,
    scales: Any = "fixed",
    axes: Any = "margins",
    remove_labels: Any = "none",
    shrink: bool = True,
    labeller: Any = "label_value",
    as_table: bool = True,
    drop: bool = True,
    dir: str = "h",
    strip_position: str = "top",
    nest_line: Any = None,
    solo_line: bool = False,
    resect: Any = None,
    trim_blank: bool = True,
    strip: Any = strip_nested,
    bleed: Optional[bool] = None,
) -> "FacetNestedWrap":
    """Ribbon of panels with nested strips.

    Port of ggh4x's ``facet_nested_wrap()`` (``R/facet_nested_wrap.R:49-101``).
    Inherits the capabilities of :func:`ggh4x.facet_wrap2` and adds label-merged
    nested strips (merged only within the same wrap row / column) plus hierarchy
    ``nest_line``s.

    Parameters
    ----------
    facets : formula / list / dict
        Faceting variables.  Variable order encodes the hierarchy (first =
        outermost).
    nrow, ncol : int or None
    scales : {"fixed", "free_x", "free_y", "free"} or bool, default "fixed"
    axes : {"margins", "x", "y", "all"} or bool, default "margins"
    remove_labels : {"none", "x", "y", "all"} or bool, default "none"
    shrink : bool, default True
    labeller : callable or str, default "label_value"
    as_table : bool, default True
    drop : bool, default True
    dir : {"h", "v"}, default "h"
    strip_position : {"top", "bottom", "left", "right"}, default "top"
    nest_line : ElementLine / ElementBlank / bool / None, default None
        Hierarchy line element; see :func:`ggh4x.facet_nested.facet_nested`.
    solo_line : bool, default False
        Draw nest lines on single-child parent strips too.
    resect : Unit or None, default None
        How much to shorten each nest line at both ends (``None`` -> ``0 mm``).
    trim_blank : bool, default True
        When ``False``, ``nrow``/``ncol`` are taken literally.
    strip : Strip or callable or str, default :func:`ggh4x.strip_nested.strip_nested`
    bleed : bool or None, default None
        Deprecated; forwards to ``strip_nested(bleed=...)`` with a warning.

    Returns
    -------
    FacetNestedWrap
        A ggproto facet object that can be added to a plot.
    """
    from ggh4x.strip_vanilla import resolve_strip

    strip = resolve_strip(strip)
    if bleed is not None:
        warnings.warn(
            "The `bleed` argument of `facet_nested_wrap()` is deprecated as of "
            "ggh4x 0.2.0. The `bleed` argument should be set in the "
            "`strip_nested()` function instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        strip.params["bleed"] = bool(bleed)

    nest_line = _coerce_nest_line(nest_line)
    if resect is None:
        resect = Unit(0, "mm")

    params = {
        "nest_line": nest_line,
        "solo_line": bool(solo_line),
        "resect": resect,
    }

    return new_wrap_facets(
        facets,
        nrow,
        ncol,
        scales,
        axes,
        remove_labels,
        shrink,
        labeller,
        as_table,
        drop,
        dir,
        strip_position,
        strip,
        trim_blank,
        params=params,
        super_=FacetNestedWrap,
    )


# ---------------------------------------------------------------------------
# ggproto
# ---------------------------------------------------------------------------
class FacetNestedWrap(FacetWrap2):
    """Nested-strip wrapped facet ggproto (port of R ``FacetNestedWrap``).

    Subclasses :class:`ggh4x.facet_wrap2.FacetWrap2`.  Only overrides
    ``finish_panels`` to draw the nest indicator lines via
    :func:`ggh4x.facet_nested.add_nest_indicator`; everything else is inherited.

    Attributes
    ----------
    shrink : bool
    strip : Strip
        Defaults to a :class:`ggh4x.strip_nested.StripNested`.
    params : dict
        Adds ``nest_line``, ``solo_line``, ``resect`` to the ``FacetWrap2`` params.
    """

    _class_name = "FacetNestedWrap"

    def finish_panels(
        self,
        panels: Any,
        layout: pd.DataFrame,
        params: Dict[str, Any],
        theme: Any,
    ) -> Any:
        """Draw the nest indicator lines onto the assembled panel table.

        Port of R ``FacetNestedWrap$finish_panels`` (``facet_nested_wrap.R:111-113``);
        delegates to :func:`ggh4x.facet_nested.add_nest_indicator`.

        Parameters
        ----------
        panels : Gtable
            The assembled panel gtable.
        layout : pandas.DataFrame
        params : dict
        theme : Theme

        Returns
        -------
        Gtable
            The panel gtable with nest-line ``"nester"`` grobs added.
        """
        return add_nest_indicator(panels, params, theme)
