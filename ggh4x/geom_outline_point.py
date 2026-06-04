"""Points with a shared outline.

Python port of ``geom_outline_point.R`` from the R package **ggh4x**.

This is a variant of the point geom in which overlapping points share a
common outline.  It works by drawing an additional layer of points
*below* a regular layer of points, with a thicker stroke.  The colour of
the lower (outline) layer is controlled by the new ``stroke_colour``
aesthetic, which can be mapped to a scale via
``scale_colour_hue(aesthetics="stroke_colour")``.

R source
--------
``ggh4x/R/geom_outline_point.R`` -- :func:`geom_outline_point`,
:class:`GeomOutlinePoint`, :func:`draw_key_outline_point`.

Notes
-----
Because of the two-layer implementation, the ``alpha`` aesthetic is
handled rather ungracefully (it is applied to *both* layers, so the
outline shows through semi-transparent fills).  This mirrors the R
package behaviour exactly.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping as TMapping, Optional

import numpy as np
import pandas as pd

from ggplot2_py import GeomPoint
from ggplot2_py.geom import (
    PT,
    STROKE,
    FromTheme,
    Gpar,
    Mapping,
    _coord_transform,
    _fill_alpha,
    grob_tree,
    points_grob,
    scales_alpha,
)

from ._rlang import value_or

__all__ = [
    "GeomOutlinePoint",
    "geom_outline_point",
    "draw_key_outline_point",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mix_ink_paper_half(geom_el: Any) -> Any:
    """Compute ``col_mix(ink, paper)`` for the ``colour`` fallback.

    R ``geom_outline_point.R:103``::

        colour = from_theme(colour %||% col_mix(ink, paper))

    ``col_mix`` with no explicit ratio defaults to ``0.5`` (an equal
    blend of the ink and paper colours).

    Parameters
    ----------
    geom_el : element_geom
        The resolved ``element_geom`` carrying ``ink`` / ``paper``.

    Returns
    -------
    str
        The mixed colour as a hex string.
    """
    from scales import col_mix

    return col_mix(geom_el.ink, geom_el.paper, 0.5)


def _to_float_array(values: Any) -> np.ndarray:
    """Coerce a column/scalar to a float ``ndarray`` (NA-preserving).

    Parameters
    ----------
    values : Any
        A pandas Series, numpy array, list or scalar.

    Returns
    -------
    numpy.ndarray
        1-D float array; non-finite entries become ``nan``.
    """
    arr = np.asarray(values, dtype="float64")
    return np.atleast_1d(arr)


def _outline_point_gpars(
    shape: Any,
    size: Any,
    stroke: Any,
    colour: Any,
    fill: Any,
    stroke_colour: Any,
    alpha: Any,
) -> tuple:
    """Compute the foreground and background ``gpar`` arguments.

    This is the shared core of :meth:`GeomOutlinePoint.draw_panel` and
    :func:`draw_key_outline_point`, ported verbatim from
    ``geom_outline_point.R:60-91`` / ``:117-148``.

    Parameters
    ----------
    shape : array-like
        ``pch`` values (numeric).
    size : array-like
        Point sizes (mm).
    stroke : array-like
        Stroke widths; ``NA`` becomes ``0``.
    colour, fill, stroke_colour : array-like
        Colour specifications for the foreground colour, foreground fill
        and background (outline) colour respectively.
    alpha : array-like
        Alpha values applied to all colours.

    Returns
    -------
    tuple of (dict, dict)
        ``(foreground_gp, background_gp)`` keyword dictionaries suitable
        for :class:`grid_py.Gpar`.
    """
    shape_arr = _to_float_array(shape)

    is_solid = shape_arr > 14
    has_fill = shape_arr > 20

    stroke_size = _to_float_array(stroke).astype("float64").copy()
    stroke_size[np.isnan(stroke_size)] = 0.0

    size_arr = _to_float_array(size)

    # R: lwd <- ifelse(is_solid & !has_fill, 0, stroke_size * .stroke / 2)
    lwd = np.where(is_solid & ~has_fill, 0.0, stroke_size * STROKE / 2.0)

    foreground_gp: Dict[str, Any] = dict(
        col=scales_alpha(colour, alpha),
        fill=_fill_alpha(fill, alpha),
        fontsize=size_arr * PT,
        lwd=lwd,
    )

    # R: size <- coords$size * .pt + ifelse(is_solid, stroke_size * .stroke, 0)
    bg_fontsize = size_arr * PT + np.where(is_solid, stroke_size * STROKE, 0.0)
    # R: lwd  <- lwd + ifelse(is_solid, 0, stroke_size * .stroke)
    bg_lwd = lwd + np.where(is_solid, 0.0, stroke_size * STROKE)

    background_gp: Dict[str, Any] = dict(
        col=scales_alpha(stroke_colour, alpha),
        fill=scales_alpha(stroke_colour, alpha),
        lwd=bg_lwd,
        fontsize=bg_fontsize,
    )

    return foreground_gp, background_gp


# ---------------------------------------------------------------------------
# Legend key
# ---------------------------------------------------------------------------
def draw_key_outline_point(
    data: Any,
    params: Dict[str, Any],
    size: Any = None,
) -> Any:
    """Draw a legend key for :class:`GeomOutlinePoint`.

    Port of R ``draw_key_outline_point`` (``geom_outline_point.R:59-94``).
    Replicates the two-layer point glyph: a thick ``stroke_colour``
    background under a normal foreground point, both centred at
    ``(0.5, 0.5)`` in the key viewport.

    Parameters
    ----------
    data : dict or DataFrame
        Scaled aesthetics for a single legend entry.  Must carry
        ``shape``, ``size``, ``stroke``, ``colour``, ``fill``,
        ``stroke_colour`` and ``alpha``.
    params : dict
        Extra layer parameters (unused, accepted for signature parity).
    size : optional
        Key dimensions (unused).

    Returns
    -------
    grob
        A :class:`grid_py.GTree` with the background drawn first and the
        foreground on top.
    """
    shape = _key_get(data, "shape", 19)
    pt_size = _key_get(data, "size", 1.5)
    stroke = _key_get(data, "stroke", 0.5)
    colour = _key_get(data, "colour", "black")
    fill = _key_get(data, "fill", None)
    stroke_colour = _key_get(data, "stroke_colour", "black")
    alpha = _key_get(data, "alpha", None)

    fg_gp, bg_gp = _outline_point_gpars(
        shape=shape,
        size=pt_size,
        stroke=stroke,
        colour=colour,
        fill=fill,
        stroke_colour=stroke_colour,
        alpha=alpha,
    )

    foreground = points_grob(
        x=0.5,
        y=0.5,
        pch=np.asarray(shape),
        gp=Gpar(**fg_gp),
    )
    background = points_grob(
        x=0.5,
        y=0.5,
        pch=np.asarray(shape),
        gp=Gpar(**bg_gp),
    )

    return grob_tree(background, foreground)


def _key_get(data: Any, key: str, default: Any = None) -> Any:
    """Fetch ``key`` from a dict- or DataFrame-like legend ``data``.

    Mirrors ggplot2_py's ``draw_key`` accessor but is column-safe for
    DataFrames (``getattr(df, "shape")`` would return the DataFrame's
    ``.shape`` tuple, so column access must be explicit).

    Parameters
    ----------
    data : dict or DataFrame
        The legend entry data.
    key : str
        Aesthetic name.
    default : Any
        Value returned when ``key`` is absent.

    Returns
    -------
    Any
        The stored value (scalar for dicts; the first element for
        DataFrame columns, matching a single-row legend key).
    """
    if isinstance(data, pd.DataFrame):
        if key in data.columns:
            col = data[key]
            return col.iloc[0] if len(col) else default
        return default
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


# ---------------------------------------------------------------------------
# ggproto class
# ---------------------------------------------------------------------------
class GeomOutlinePoint(GeomPoint):
    """Point geom that draws a shared outline beneath the points.

    Two stacked :func:`grid_py.points_grob` layers are emitted per panel:
    a thicker *background* stroke layer (coloured by ``stroke_colour``)
    and, on top of it, a normal *foreground* point layer.  Overlapping
    points therefore appear to share a single outline.

    Subclasses :class:`ggplot2_py.GeomPoint`.  Unlike its parent,
    :meth:`draw_panel` does **not** call ``ggproto_parent`` -- it builds
    both grobs directly.
    """

    # R geom_outline_point.R:101-109.  Adds the brand-new ``stroke_colour``
    # aesthetic and overrides the ``colour`` fallback to col_mix(ink, paper).
    default_aes: Mapping = Mapping(
        shape=FromTheme("pointshape"),
        colour=FromTheme("colour", fallback=_mix_ink_paper_half),
        size=FromTheme("pointsize"),
        fill=FromTheme("fill"),
        alpha=None,
        stroke=FromTheme("borderwidth"),
        stroke_colour=FromTheme("ink"),
    )

    draw_key = staticmethod(draw_key_outline_point)

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        na_rm: bool = True,
        **params: Any,
    ) -> Any:
        """Draw the outline + point layers for one panel.

        Port of R ``GeomOutlinePoint$draw_panel``
        (``geom_outline_point.R:113-155``).  Note the R default
        ``na.rm = TRUE`` (the parent :class:`GeomPoint` defaults to
        ``FALSE``).

        Parameters
        ----------
        data : DataFrame
            Layer data with at least ``x``, ``y``, ``shape``, ``size``,
            ``stroke``, ``colour``, ``fill``, ``stroke_colour``,
            ``alpha``.
        panel_params : Any
            Panel parameters (ranges, etc.).
        coord : Coord
            The active coordinate system.
        na_rm : bool, default True
            Whether missing values are silently removed upstream.
        **params : Any
            Ignored extra parameters.

        Returns
        -------
        grob
            A :class:`grid_py.GTree` named ``outline_points`` with the
            background layer drawn first and the foreground on top.
        """
        coords = _coord_transform(coord, data, panel_params)

        def _col(name: str, default: Any) -> Any:
            return coords[name].values if name in coords.columns else default

        fg_gp, bg_gp = _outline_point_gpars(
            shape=_col("shape", 19),
            size=_col("size", 1.5),
            stroke=_col("stroke", 0.5),
            colour=_col("colour", "black"),
            fill=_col("fill", None),
            stroke_colour=_col("stroke_colour", "black"),
            alpha=_col("alpha", None),
        )

        x = coords["x"].values
        y = coords["y"].values
        shape_vals = np.asarray(_col("shape", 19))

        foreground = points_grob(
            x=x,
            y=y,
            pch=shape_vals,
            gp=Gpar(**fg_gp),
        )

        background = points_grob(
            x=x,
            y=y,
            pch=shape_vals,
            gp=Gpar(**bg_gp),
        )

        # R: grob <- grobTree(background, foreground); grob$name <- ...
        grob = grob_tree(background, foreground)
        grob.name = "outline_points"
        return grob


# Make the class-level ``draw_key`` resolvable both as an unbound function
# (legend machinery calls ``draw_key_fn(data, params, size)``) and as an
# attribute lookup.
GeomOutlinePoint.draw_key = draw_key_outline_point


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------
def geom_outline_point(
    mapping: Optional[TMapping] = None,
    data: Any = None,
    stat: Any = "identity",
    position: Any = "identity",
    *,
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Points with a shared outline.

    Port of R ``geom_outline_point`` (``geom_outline_point.R:32-55``).  A
    variant of :func:`ggplot2_py.geom_point` in which overlapping points
    are given a shared outline by drawing a thicker stroke layer beneath
    the regular points.

    The outline colour can be mapped to a scale by setting the aesthetic
    to ``"stroke_colour"`` and supplying e.g.
    ``scale_colour_hue(aesthetics="stroke_colour")``.

    Parameters
    ----------
    mapping : Mapping, optional
        Aesthetic mapping (see :func:`ggplot2_py.aes`).
    data : DataFrame or callable, optional
        Layer data.
    stat : str or Stat, default ``"identity"``
        Statistical transformation.
    position : str or Position, default ``"identity"``
        Position adjustment.
    na_rm : bool, default False
        If ``False``, missing values are removed with a warning.
    show_legend : bool, optional
        Whether to include this layer in the legend.  ``None`` mirrors
        R's ``NA`` (include only mapped aesthetics).
    inherit_aes : bool, default True
        Whether to inherit the plot-level mapping.
    **kwargs : Any
        Other arguments passed on to the layer (e.g. ``size``,
        ``stroke``, fixed aesthetics).

    Returns
    -------
    ggplot2_py.Layer
        A layer backed by :class:`GeomOutlinePoint`.
    """
    from ggplot2_py.layer import layer

    return layer(
        data=data,
        mapping=mapping,
        stat=stat,
        geom=GeomOutlinePoint,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "na_rm": na_rm,
            **kwargs,
        },
    )
