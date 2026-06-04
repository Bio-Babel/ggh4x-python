"""Rectangular rugs in the margins (port of ggh4x ``geom_rectrug.R``).

Where rug plots show 1-D data as ticks in the plot margins,
``geom_rectmargin()`` and ``geom_tilemargin()`` draw *rectangles* in the
margins.  They are convenient for one-dimensional, ranged annotations on a 2-D
plot.  ``geom_rectmargin()`` is parameterised by ``xmin``/``xmax`` and/or
``ymin``/``ymax``; ``geom_tilemargin()`` is parameterised by centre ``x``/``y``
and ``width``/``height``.

R source: ``ggh4x/R/geom_rectrug.R``.

Notes
-----
* :meth:`GeomRectMargin.draw_panel` validates that ``length`` is a grid unit,
  coord-transforms the data, remaps ``sides`` via ``chartr("tblr", "rlbt")``
  under :class:`ggplot2_py.CoordFlip`, and builds a thin rectangle band on each
  requested side using only the ``min`` of the ``rug_length`` band.
* :class:`GeomTileMargin` inherits :meth:`GeomRectMargin.draw_panel` unchanged
  and only overrides :meth:`setup_data` (centre+size -> min/max, with
  ``width``/``height`` falling back to ``resolution(x/y, zero=False)``) and the
  default aesthetics.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ggplot2_py import CoordFlip
from ggplot2_py.geom import (
    GeomRug,
    FromTheme,
    Gpar,
    GList,
    GTree,
    Mapping,
    PT,
    _coord_transform,
    _mix_ink_paper,
    draw_key_polygon,
    rect_grob,
    scales_alpha,
)
from ggplot2_py._utils import resolution
from grid_py import Unit, is_unit

from ._cli import cli_abort

__all__ = [
    "geom_rectmargin",
    "geom_tilemargin",
    "GeomRectMargin",
    "GeomTileMargin",
]


def _alpha_na(colour: Any, alpha: Any) -> Any:
    """Apply alpha to a colour spec, passing ``NA`` (``None``/``NaN``) through.

    Port of R's ``scales::alpha(colour, alpha)`` ``NA``-handling: an ``NA``
    colour stays ``NA`` (drawn as no-paint) instead of erroring.  Scalars and
    arrays are both supported; ``None`` entries inside an array are preserved.

    Parameters
    ----------
    colour : Any
        A colour scalar, array (possibly with ``None``/``NaN`` entries) or
        ``None``.
    alpha : Any
        Alpha scalar or array.

    Returns
    -------
    Any
        The alpha-applied colour(s), with ``NA`` preserved.
    """
    if colour is None:
        return None
    arr = np.atleast_1d(np.asarray(colour, dtype=object))
    if arr.shape == (1,) and (arr[0] is None or (isinstance(arr[0], float) and np.isnan(arr[0]))):
        return None
    is_na = np.array(
        [c is None or (isinstance(c, float) and np.isnan(c)) for c in arr]
    )
    if not is_na.any():
        return scales_alpha(colour, alpha)
    if is_na.all():
        return None
    # Mixed: apply alpha only to the valid colours, keep None elsewhere.
    out = np.empty(arr.shape, dtype=object)
    valid_idx = np.flatnonzero(~is_na)
    alpha_arr = np.atleast_1d(np.asarray(alpha, dtype=object))
    applied = scales_alpha(
        np.array([arr[i] for i in valid_idx], dtype=object),
        np.array(
            [alpha_arr[i % len(alpha_arr)] for i in valid_idx], dtype=object
        )
        if alpha is not None
        else None,
    )
    out[:] = None
    out[valid_idx] = applied
    return out


class GeomRectMargin(GeomRug):
    """Rectangular rug geom parameterised by ``(x/y)min``/``(x/y)max``.

    Subclass of :class:`ggplot2_py.GeomRug` ported from R ``GeomRectMargin``
    (``geom_rectrug.R:189-280``).  It completely replaces the rug's
    ``draw_panel``/``default_aes``/``draw_key`` and only inherits the missing
    value / setup plumbing.
    """

    optional_aes = ("x", "y", "xmin", "xmax", "ymin", "ymax")

    # R geom_rectrug.R:272-278.
    default_aes: Mapping = Mapping(
        colour=FromTheme("colour", fallback=lambda g: None),
        fill=FromTheme("fill", fallback=_mix_ink_paper(0.35)),
        linewidth=FromTheme("borderwidth"),
        linetype=FromTheme("bordertype"),
        alpha=None,
    )

    draw_key = draw_key_polygon

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        sides: str = "bl",
        outside: bool = False,
        length: Any = None,
        linejoin: str = "mitre",
        **params: Any,
    ) -> Any:
        """Build the margin rectangles for one panel.

        Port of R ``GeomRectMargin$draw_panel`` (``geom_rectrug.R:191-270``).

        Parameters
        ----------
        data : pandas.DataFrame
            Layer data (``xmin``/``xmax`` and/or ``ymin``/``ymax``).
        panel_params : Any
            Panel scales / ranges.
        coord : Any
            Active coordinate system.
        sides : str, default ``"bl"``
            Which margins to draw on (any of ``"trbl"``).
        outside : bool, default ``False``
            Whether to move the rectangles outside the plot area.
        length : grid_py.Unit, optional
            Rectangle thickness (defaults to ``Unit(0.03, "npc")``).
        linejoin : str, default ``"mitre"``
            Line join style.
        **params : Any
            Ignored extra parameters.

        Returns
        -------
        grid_py.GTree
            A tree of the per-side rectangle grobs.

        Raises
        ------
        TypeError
            If ``length`` is not a grid unit.
        """
        if length is None:
            length = Unit(0.03, "npc")
        if not is_unit(length):
            cli_abort(
                "The `length` argument must be a `unit` object.",
                error_class=TypeError,
            )

        coords = _coord_transform(coord, data, panel_params)
        if isinstance(coord, CoordFlip):
            sides = sides.translate(str.maketrans("tblr", "rlbt"))

        if not outside:
            rug_min = length
        else:
            rug_min = -1 * length

        colour = coords["colour"].to_numpy() if "colour" in coords.columns else None
        fill = coords["fill"].to_numpy() if "fill" in coords.columns else "grey35"
        alpha = coords["alpha"].to_numpy() if "alpha" in coords.columns else None
        linetype = coords["linetype"].to_numpy() if "linetype" in coords.columns else 1
        linewidth = (
            coords["linewidth"].to_numpy(dtype="float64")
            if "linewidth" in coords.columns
            else np.full(len(coords), 0.5)
        )

        gp = Gpar(
            col=_alpha_na(colour, alpha),
            fill=_alpha_na(fill, alpha),
            linejoin=linejoin,
            lty=linetype,
            lwd=linewidth * PT,
            lineend="round" if linejoin == "round" else "square",
        )

        rugs = []
        have_x = "xmin" in coords.columns and "xmax" in coords.columns
        have_y = "ymin" in coords.columns and "ymax" in coords.columns

        if have_x:
            xmin = coords["xmin"].to_numpy(dtype="float64")
            xmax = coords["xmax"].to_numpy(dtype="float64")
            xwidth = xmax - xmin
            if "b" in sides:
                rugs.append(
                    rect_grob(
                        x=Unit(xmin, "native"),
                        y=Unit(0.0, "npc"),
                        width=Unit(xwidth, "native"),
                        height=rug_min,
                        just=("left", "bottom"),
                        gp=gp,
                    )
                )
            if "t" in sides:
                rugs.append(
                    rect_grob(
                        x=Unit(xmin, "native"),
                        y=Unit(1.0, "npc"),
                        width=Unit(xwidth, "native"),
                        height=rug_min,
                        just=("left", "top"),
                        gp=gp,
                    )
                )

        if have_y:
            ymin = coords["ymin"].to_numpy(dtype="float64")
            ymax = coords["ymax"].to_numpy(dtype="float64")
            yheight = ymax - ymin
            if "l" in sides:
                rugs.append(
                    rect_grob(
                        x=Unit(0.0, "npc"),
                        y=Unit(ymax, "native"),
                        width=rug_min,
                        height=Unit(yheight, "native"),
                        just=("left", "top"),
                        gp=gp,
                    )
                )
            if "r" in sides:
                rugs.append(
                    rect_grob(
                        x=Unit(1.0, "npc"),
                        y=Unit(ymax, "native"),
                        width=rug_min,
                        height=Unit(yheight, "native"),
                        just=("right", "top"),
                        gp=gp,
                    )
                )

        return GTree(children=GList(*rugs))


class GeomTileMargin(GeomRectMargin):
    """Rectangular rug geom parameterised by centre + size.

    Subclass of :class:`GeomRectMargin` ported from R ``GeomTileMargin``
    (``geom_rectrug.R:286-309``).  Inherits :meth:`GeomRectMargin.draw_panel`
    unchanged.
    """

    extra_params = ("na_rm",)

    # R geom_rectrug.R:299-307.
    default_aes: Mapping = Mapping(
        fill=FromTheme("fill", fallback=_mix_ink_paper(0.2)),
        colour=FromTheme("colour", fallback=lambda g: None),
        linewidth=FromTheme("borderwidth", fallback=lambda g: g.borderwidth / 5),
        linetype=FromTheme("bordertype"),
        alpha=None,
        width=None,
        height=None,
    )

    def setup_data(self, data: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Derive ``xmin``/``xmax``/``ymin``/``ymax`` from centre + size.

        Port of R ``GeomTileMargin$setup_data`` (``geom_rectrug.R:290-297``).
        ``width``/``height`` fall back to ``params`` then
        ``resolution(x/y, zero=False)``.

        Parameters
        ----------
        data : pandas.DataFrame
            Layer data with centre ``x``/``y`` and optional ``width``/
            ``height``.
        params : dict
            Layer parameters.

        Returns
        -------
        pandas.DataFrame
            Data with the four corner columns set and ``width``/``height``
            dropped.
        """
        data = data.copy()

        def _resolved(col: str, axis: str) -> Optional[np.ndarray]:
            if col in data.columns and not data[col].isna().all():
                return data[col].to_numpy(dtype="float64")
            pval = params.get(col)
            if pval is not None:
                return np.full(len(data), float(pval))
            if axis in data.columns:
                return np.full(len(data), resolution(data[axis].to_numpy(dtype="float64"), zero=False))
            return None

        if "x" in data.columns:
            width = _resolved("width", "x")
            x = data["x"].to_numpy(dtype="float64")
            data["xmin"] = x - width / 2
            data["xmax"] = x + width / 2
        if "y" in data.columns:
            height = _resolved("height", "y")
            y = data["y"].to_numpy(dtype="float64")
            data["ymin"] = y - height / 2
            data["ymax"] = y + height / 2

        for drop in ("width", "height"):
            if drop in data.columns:
                data = data.drop(columns=drop)
        return data

    draw_key = draw_key_polygon


def geom_rectmargin(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    outside: bool = False,
    sides: str = "bl",
    length: Any = None,
    linejoin: str = "mitre",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a rectangular margin-rug layer parameterised by min/max.

    Port of R ``geom_rectmargin()`` (``geom_rectrug.R:117-147``).

    Parameters
    ----------
    mapping : Mapping, optional
        Aesthetic mapping created by :func:`ggplot2_py.aes`.
    data : Any, optional
        Layer data.
    stat : str, default ``"identity"``
        Statistical transformation.
    position : str, default ``"identity"``
        Position adjustment.
    outside : bool, default ``False``
        Whether to move the rectangles outside the plot area.
    sides : str, default ``"bl"``
        Which margins to draw on (any of ``"trbl"``).
    length : grid_py.Unit, optional
        Rectangle thickness (defaults to ``Unit(0.03, "npc")``).
    linejoin : str, default ``"mitre"``
        Line join style.
    na_rm : bool, default ``False``
        If ``True``, silently remove missing values.
    show_legend : bool or None, default ``None``
        Whether to show a legend for this layer.
    inherit_aes : bool, default ``True``
        Whether to inherit the plot's default aesthetics.
    **kwargs : Any
        Additional aesthetic parameters passed to the layer.

    Returns
    -------
    ggplot2_py.Layer
        A layer object that can be added to a plot.
    """
    from ggplot2_py.layer import layer

    if length is None:
        length = Unit(0.03, "npc")

    return layer(
        data=data,
        mapping=mapping,
        stat=stat,
        geom=GeomRectMargin,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "outside": outside,
            "sides": sides,
            "length": length,
            "linejoin": linejoin,
            "na_rm": na_rm,
            **kwargs,
        },
    )


def geom_tilemargin(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    outside: bool = False,
    sides: str = "bl",
    length: Any = None,
    linejoin: str = "mitre",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a rectangular margin-rug layer parameterised by centre + size.

    Port of R ``geom_tilemargin()`` (``geom_rectrug.R:151-181``).

    Parameters
    ----------
    mapping : Mapping, optional
        Aesthetic mapping created by :func:`ggplot2_py.aes`.
    data : Any, optional
        Layer data.
    stat : str, default ``"identity"``
        Statistical transformation.
    position : str, default ``"identity"``
        Position adjustment.
    outside : bool, default ``False``
        Whether to move the rectangles outside the plot area.
    sides : str, default ``"bl"``
        Which margins to draw on (any of ``"trbl"``).
    length : grid_py.Unit, optional
        Rectangle thickness (defaults to ``Unit(0.03, "npc")``).
    linejoin : str, default ``"mitre"``
        Line join style.
    na_rm : bool, default ``False``
        If ``True``, silently remove missing values.
    show_legend : bool or None, default ``None``
        Whether to show a legend for this layer.
    inherit_aes : bool, default ``True``
        Whether to inherit the plot's default aesthetics.
    **kwargs : Any
        Additional aesthetic parameters passed to the layer.

    Returns
    -------
    ggplot2_py.Layer
        A layer object that can be added to a plot.
    """
    from ggplot2_py.layer import layer

    if length is None:
        length = Unit(0.03, "npc")

    return layer(
        data=data,
        mapping=mapping,
        stat=stat,
        geom=GeomTileMargin,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "outside": outside,
            "sides": sides,
            "length": length,
            "linejoin": linejoin,
            "na_rm": na_rm,
            **kwargs,
        },
    )
