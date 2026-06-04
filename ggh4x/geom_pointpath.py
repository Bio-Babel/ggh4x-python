"""Point paths (port of ggh4x ``geom_pointpath.R``).

``geom_pointpath()`` makes a scatterplot in which the points are connected by
line segments in data order, mimicking base R's ``type = "b"`` line plots.  The
inter-point segments are *interrupted* by a gap around every point; crucially,
the gap is sized in absolute units at draw time so it does not deform under
different aspect ratios or device sizes.

R source: ``ggh4x/R/geom_pointpath.R``.

Notes
-----
* :meth:`GeomPointPath.draw_panel` first draws the underlying points via
  ``ggproto_parent(GeomPoint, self).draw_panel`` (it keeps ``self`` in its
  formals), then builds the inter-point segments, attaches a custom gap grob
  and returns ``grob_tree(gap_grob, point_grob)`` so the **points sit on top**.
* The custom grob class is :class:`~ggh4x._gap_grobs.GapSegmentsGrob` under
  linear coordinates and :class:`~ggh4x._gap_grobs.GapSegmentsChainGrob`
  otherwise, exactly mirroring R's ``cl = if (coord$is_linear())`` switch.
* ``mult`` is a non-standard *mappable* aesthetic (default ``0.5``) that scales
  the gap radius; it is declared in :attr:`GeomPointPath.default_aes` so
  ``use_defaults`` broadcasts it.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ggplot2_py import ggproto_parent
from ggplot2_py.geom import (
    GeomPoint,
    FromTheme,
    Gpar,
    Mapping,
    PT,
    STROKE,
    _ggname,
    grob_tree,
    scales_alpha,
)
from ggplot2_py.coord import coord_munch
from grid_py import Unit

from ._gap_grobs import GapSegmentsChainGrob, GapSegmentsGrob

__all__ = [
    "geom_pointpath",
    "GeomPointPath",
    "GeomPointpath",
]


class GeomPointPath(GeomPoint):
    """Point geom whose points are connected by gapped line segments.

    Subclass of :class:`ggplot2_py.GeomPoint` ported from R ``GeomPointPath``
    (``geom_pointpath.R:71-140``).  Adds the ``linewidth``, ``linetype`` and
    ``mult`` aesthetics on top of the point defaults and overrides
    :meth:`draw_panel` to emit the interrupted path beneath the points.
    """

    # R geom_pointpath.R:128-138.  GeomPoint defaults + line aesthetics + the
    # non-standard ``mult`` gap-scaling aesthetic (default 0.5).
    default_aes: Mapping = Mapping(
        shape=FromTheme("pointshape"),
        colour=FromTheme("colour", fallback="ink"),
        size=FromTheme("pointsize"),
        fill=FromTheme("fill"),
        alpha=None,
        stroke=FromTheme("borderwidth"),
        linewidth=FromTheme("linewidth"),
        linetype=FromTheme("linetype"),
        mult=0.5,
    )
    non_missing_aes = ("size", "colour")

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        arrow: Any = None,
        na_rm: bool = False,
        **params: Any,
    ) -> Any:
        """Draw the points and the gapped inter-point path for one panel.

        Port of R ``GeomPointPath$draw_panel`` (``geom_pointpath.R:73-126``).

        Parameters
        ----------
        data : pandas.DataFrame
            Layer data for one panel.
        panel_params : Any
            Panel scales / ranges.
        coord : Any
            Active coordinate system.
        arrow : grid_py.Arrow or None, default ``None``
            Optional arrow specification for the path ends.
        na_rm : bool, default ``False``
            Whether missing values are silently removed.
        **params : Any
            Ignored extra parameters.

        Returns
        -------
        grid_py.Grob
            ``grob_tree(gap_path, point_grob)`` with the points drawn on top,
            or ``grob_tree(point_grob)`` when no inter-point segment survives.
        """
        # Default geom_point behaviour for the points themselves.
        pointgrob = ggproto_parent(GeomPoint, self).draw_panel(
            data, panel_params, coord, na_rm=na_rm
        )

        data = data.copy()
        data["id"] = np.arange(1, len(data) + 1)
        # order(group) is a stable sort in R.
        data = data.sort_values("group", kind="stable").reset_index(drop=True)
        data = coord_munch(coord, data, panel_params)
        data = data.reset_index(drop=True)

        x = data["x"].to_numpy(dtype="float64")
        y = data["y"].to_numpy(dtype="float64")
        group = data["group"].to_numpy()
        n = len(data)

        # transform: xend = c(tail(x, -1), NA); yend likewise;
        #            keep  = c(group[-1] == head(group, -1), FALSE)
        xend = np.concatenate([x[1:], [np.nan]]) if n > 0 else np.array([])
        yend = np.concatenate([y[1:], [np.nan]]) if n > 0 else np.array([])
        if n > 1:
            keep = np.concatenate([group[1:] == group[:-1], [False]])
        elif n == 1:
            keep = np.array([False])
        else:
            keep = np.array([], dtype=bool)

        data["xend"] = xend
        data["yend"] = yend
        sub = data.loc[keep].reset_index(drop=True)

        if len(sub) < 1:
            return _ggname("geom_pointpath", grob_tree(pointgrob))

        size = sub["size"].to_numpy(dtype="float64") if "size" in sub else np.full(len(sub), 1.5)
        stroke = (
            sub["stroke"].to_numpy(dtype="float64")
            if "stroke" in sub
            else np.full(len(sub), 0.5)
        )
        mult_aes = (
            sub["mult"].to_numpy(dtype="float64")
            if "mult" in sub
            else np.full(len(sub), 0.5)
        )
        mult = (size * PT + stroke * STROKE / 2.0) * mult_aes

        colour = sub["colour"].to_numpy() if "colour" in sub else "black"
        alpha = sub["alpha"].to_numpy() if "alpha" in sub else None
        linewidth = (
            sub["linewidth"].to_numpy(dtype="float64")
            if "linewidth" in sub
            else np.full(len(sub), 0.5)
        )
        linetype = sub["linetype"].to_numpy() if "linetype" in sub else 1

        gp = Gpar(
            col=scales_alpha(colour, alpha),
            fill=scales_alpha(colour, alpha),
            lwd=linewidth * PT,
            lty=linetype,
            lineend="butt",
            linejoin="round",
            linemitre=10,
        )

        grob_cls = GapSegmentsGrob if coord.is_linear() else GapSegmentsChainGrob
        my_path = grob_cls(
            x0=Unit(sub["x"].to_numpy(dtype="float64"), "npc"),
            x1=Unit(sub["xend"].to_numpy(dtype="float64"), "npc"),
            y0=Unit(sub["y"].to_numpy(dtype="float64"), "npc"),
            y1=Unit(sub["yend"].to_numpy(dtype="float64"), "npc"),
            mult=mult,
            id=sub["id"].to_numpy(),
            arrow=arrow,
            gp=gp,
            name="pointpath",
        )

        return _ggname("geom_pointpath", grob_tree(my_path, pointgrob))


# R geom_pointpath.R:146 ``GeomPointpath <- GeomPointPath``.
GeomPointpath = GeomPointPath


def geom_pointpath(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    arrow: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a point-path layer.

    Port of R ``geom_pointpath()`` (``geom_pointpath.R:43-63``).  Connects
    points with gapped line segments in data order, à la base R's
    ``type = "b"``.

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
    na_rm : bool, default ``False``
        If ``True``, silently remove missing values.
    show_legend : bool or None, default ``None``
        Whether to show a legend for this layer.
    arrow : grid_py.Arrow or None, default ``None``
        Arrow specification (see :func:`grid_py.arrow`) for the path ends.
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

    return layer(
        data=data,
        mapping=mapping,
        stat=stat,
        geom=GeomPointPath,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "na_rm": na_rm,
            "arrow": arrow,
            **kwargs,
        },
    )
