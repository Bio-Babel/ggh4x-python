"""Cartesian coordinates with interior axes (port of ggh4x ``coord_axes_inside.R``).

This module ports the R ggh4x ``coord_axes_inside()`` constructor, the
``CoordAxesInside`` ggproto class and the ``replace_vp_coord()`` helper.  The
coordinate system places the plot axes at interior positions (controlled by
``xintercept``/``yintercept``); otherwise it behaves like
:func:`ggplot2_py.coord_cartesian` (or :func:`ggplot2_py.coord_fixed` when
``ratio`` is set).

R source: ``ggh4x/R/coord_axes_inside.R``.

Notes
-----
* The constructor builds two :class:`~ggplot2_py.theme.Theme` objects:
  ``outer_axes`` blanks the axis lines/ticks (and, when labels go inside, the
  axis text + tick length) in the panel gutters, while ``inner_axes`` blanks the
  axis text that should *not* be drawn inside the panel.
* :meth:`CoordAxesInside.render_bg` is the load-bearing override: it renders the
  *inner* axes (with un-blanked content) and re-positions their viewports to the
  interior NPC coordinate of the origin, then composites them into the panel
  background grob.  The *outer* (blanked) axes still occupy the gutters via the
  :meth:`render_axis_h` / :meth:`render_axis_v` overrides.
* :func:`_replace_vp_coord` mirrors R's ``replace_vp_coord``: when a grob carries
  no viewport it is returned unchanged; otherwise the single coordinate
  (``x`` or ``y``) of its viewport is replaced.  The axis grobs returned by
  :func:`ggplot2_py._guide_axis.draw_axis` already carry a viewport, so the
  coordinate replacement applies directly (via :func:`grid_py.edit_viewport`,
  since :class:`grid_py.Viewport` coordinates are immutable).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ggplot2_py import ggproto_parent
from ggplot2_py.coord import CoordCartesian
from ggplot2_py.theme import theme
from ggplot2_py.theme_elements import element_blank

from grid_py import Unit, edit_viewport, grob_tree

from scales import oob_squish

from ggh4x._rlang import arg_match0

__all__ = [
    "coord_axes_inside",
    "CoordAxesInside",
    "_replace_vp_coord",
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _replace_vp_coord(grob: Any, param: str = "x", value: Any = None) -> Any:
    """Replace a single coordinate of a grob's viewport.

    Port of R ``replace_vp_coord`` (``coord_axes_inside.R:164-170``).  When the
    grob carries no viewport (``grob.vp is None``) it is returned unchanged;
    otherwise the ``param`` coordinate (``"x"`` or ``"y"``) of its viewport is
    replaced with *value*.

    Because :class:`grid_py.Viewport` coordinates are immutable, the replacement
    is performed by :func:`grid_py.edit_viewport`, which returns an edited copy
    of the viewport.  The edited viewport is then re-attached to *grob*.

    Parameters
    ----------
    grob : grid_py.Grob
        A grob, typically an axis grob from
        :func:`ggplot2_py._guide_axis.draw_axis`.
    param : {"x", "y"}, default "x"
        Which viewport coordinate to replace.
    value : grid_py.Unit
        The replacement coordinate (an NPC :class:`~grid_py.Unit`).

    Returns
    -------
    grid_py.Grob
        *grob* with its viewport's ``param`` coordinate replaced, or *grob*
        unchanged when it carries no viewport.
    """
    vp = getattr(grob, "vp", None)
    if vp is None:
        return grob
    grob.vp = edit_viewport(vp, **{param: value})
    return grob


# ---------------------------------------------------------------------------
# CoordAxesInside ggproto class
# ---------------------------------------------------------------------------
class CoordAxesInside(CoordCartesian):
    """Cartesian coordinate system with interior axes.

    Subclass of :class:`ggplot2_py.coord.CoordCartesian` ported from R
    ``CoordAxesInside`` (``coord_axes_inside.R:122-160``).

    Attributes
    ----------
    origin : pandas.DataFrame
        A 1-row frame ``{"x": [xintercept], "y": [yintercept]}`` giving the
        interior position where the axes meet.
    outer_axes : ggplot2_py.theme.Theme
        Theme additions blanking the gutter axis lines/ticks (and text/tick
        length when labels are drawn inside).
    inner_axes : ggplot2_py.theme.Theme
        Theme additions blanking the axis text that should not appear inside the
        panel.
    """

    def render_axis_h(self, panel_params: Dict[str, Any], theme: Any) -> Dict[str, Any]:
        """Render the horizontal (top/bottom) gutter axes, blanked.

        Port of R ``CoordAxesInside$render_axis_h``
        (``coord_axes_inside.R:124-126``): delegate to
        ``CoordCartesian$render_axis_h`` after adding ``self.outer_axes`` to the
        theme so the gutter axis lines/ticks (and text, if labels are inside)
        are blank.

        Parameters
        ----------
        panel_params : dict
            Panel parameters from :meth:`setup_panel_params`.
        theme : ggplot2_py.theme.Theme
            The active theme.

        Returns
        -------
        dict
            ``{"top": grob, "bottom": grob}``.
        """
        return ggproto_parent(CoordCartesian, self).render_axis_h(
            panel_params, theme + self.outer_axes
        )

    def render_axis_v(self, panel_params: Dict[str, Any], theme: Any) -> Dict[str, Any]:
        """Render the vertical (left/right) gutter axes, blanked.

        Port of R ``CoordAxesInside$render_axis_v``
        (``coord_axes_inside.R:127-129``): as :meth:`render_axis_h` but for the
        vertical axes.

        Parameters
        ----------
        panel_params : dict
            Panel parameters from :meth:`setup_panel_params`.
        theme : ggplot2_py.theme.Theme
            The active theme.

        Returns
        -------
        dict
            ``{"left": grob, "right": grob}``.
        """
        return ggproto_parent(CoordCartesian, self).render_axis_v(
            panel_params, theme + self.outer_axes
        )

    def render_bg(self, panel_params: Dict[str, Any], theme: Any) -> Any:
        """Render the panel background with the interior axes injected.

        Port of R ``CoordAxesInside$render_bg``
        (``coord_axes_inside.R:130-152``).  The *inner* axes are rendered here
        (so their un-blanked content lands inside the panel background) and each
        axis viewport is re-positioned to the interior NPC coordinate of the
        transformed origin.  The repositioned axes are composited into the
        background grob with the grid/axis-b/axis-t/axis-l/axis-r names.

        Parameters
        ----------
        panel_params : dict
            Panel parameters from :meth:`setup_panel_params`.
        theme : ggplot2_py.theme.Theme
            The active theme.

        Returns
        -------
        grid_py.Grob
            A grob tree containing the grid background and the four repositioned
            interior axes.
        """
        theme = theme + self.inner_axes
        grid_grob = ggproto_parent(CoordCartesian, self).render_bg(panel_params, theme)
        xaxes = ggproto_parent(CoordCartesian, self).render_axis_h(panel_params, theme)
        yaxes = ggproto_parent(CoordCartesian, self).render_axis_v(panel_params, theme)

        origin = self.transform(self.origin, panel_params)

        x = Unit(oob_squish(float(origin["x"].iloc[0])), "npc")
        y = Unit(oob_squish(float(origin["y"].iloc[0])), "npc")

        xaxes["bottom"] = _replace_vp_coord(xaxes["bottom"], "y", y)
        xaxes["top"] = _replace_vp_coord(xaxes["top"], "y", y)
        yaxes["left"] = _replace_vp_coord(yaxes["left"], "x", x)
        yaxes["right"] = _replace_vp_coord(yaxes["right"], "x", x)

        # R grobTree names each child for later grid-path edits; mirror that by
        # setting each grob's .name before assembling the tree (draw order:
        # grid, axis-b, axis-t, axis-l, axis-r).
        children = [
            ("grid", grid_grob),
            ("axis-b", xaxes["bottom"]),
            ("axis-t", xaxes["top"]),
            ("axis-l", yaxes["left"]),
            ("axis-r", yaxes["right"]),
        ]
        for name, child in children:
            if hasattr(child, "name"):
                child.name = name

        return grob_tree(*(child for _, child in children))

    def is_free(self) -> bool:
        """Whether the aspect ratio is free.

        Port of R ``CoordAxesInside$is_free`` (``coord_axes_inside.R:154``):
        ``self.ratio is None``.  (Identical to
        :meth:`CoordCartesian.is_free`; re-declared for parity.)

        Returns
        -------
        bool
            ``True`` when ``self.ratio`` is ``None``.
        """
        return self.ratio is None

    def aspect(self, ranges: Any) -> Optional[float]:
        """Compute the fixed aspect ratio, if any.

        Port of R ``CoordAxesInside$aspect`` (``coord_axes_inside.R:156-159``):
        ``None`` when ``ratio`` is ``None``, else
        ``diff(y.range) / diff(x.range) * ratio``.

        Parameters
        ----------
        ranges : dict
            Must expose ``y.range`` / ``x.range`` (or ``y_range`` / ``x_range``).

        Returns
        -------
        float or None
            The aspect ratio, or ``None`` when ``ratio`` is unset.
        """
        if self.ratio is None:
            return None
        y_range = ranges.get("y.range") or ranges.get("y_range", [0, 1])
        x_range = ranges.get("x.range") or ranges.get("x_range", [0, 1])
        return (y_range[1] - y_range[0]) / (x_range[1] - x_range[0]) * self.ratio


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------
def coord_axes_inside(
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    xintercept: float = 0,
    yintercept: float = 0,
    labels_inside: Union[bool, str] = False,
    ratio: Optional[float] = None,
    expand: bool = True,
    default: bool = False,
    clip: str = "on",
) -> CoordAxesInside:
    """Create a Cartesian coordinate system with interior axes.

    Port of R ``coord_axes_inside()`` (``coord_axes_inside.R:49-114``).  Other
    than placing the axes at interior positions, this behaves like
    :func:`ggplot2_py.coord_cartesian` (or :func:`ggplot2_py.coord_fixed` when
    ``ratio`` is set).

    Parameters
    ----------
    xlim, ylim : sequence of float, optional
        Coordinate limits (zoom, does not filter data).
    xintercept, yintercept : float, optional
        Positions where the orthogonal axes should be placed.  When outside the
        limits, the axes snap to the nearest extreme.  Default ``0``.
    labels_inside : bool or {"x", "y", "both", "none"}, optional
        The axes whose labels are placed inside the panel.  ``True`` maps to
        ``"both"`` and ``False`` (default) maps to ``"none"``.
    ratio : float, optional
        Fixed aspect ratio expressed as ``y / x``, or ``None`` for a free ratio.
    expand : bool, default True
        Whether to expand limits to avoid data/axis overlap.
    default : bool, default False
        Whether this is the plot's default coordinate system.
    clip : str, default "on"
        Clipping: ``"on"`` or ``"off"``.

    Returns
    -------
    CoordAxesInside
        A coordinate object that can be added to a plot.

    Examples
    --------
    >>> isinstance(coord_axes_inside(xintercept=1), CoordAxesInside)
    True
    """
    if isinstance(labels_inside, str):
        labels_inside = arg_match0(
            labels_inside, ("x", "y", "none", "both"), arg_name="labels_inside"
        )
    else:
        labels_inside = "both" if labels_inside is True else "none"

    inner_axes = theme()
    outer_axes = theme(
        **{
            "axis.line.x.bottom": element_blank(),
            "axis.line.x.top": element_blank(),
            "axis.ticks.x.bottom": element_blank(),
            "axis.ticks.x.top": element_blank(),
            "axis.line.y.left": element_blank(),
            "axis.line.y.right": element_blank(),
            "axis.ticks.y.left": element_blank(),
            "axis.ticks.y.right": element_blank(),
        }
    )

    if labels_inside in ("x", "both"):
        outer_axes = outer_axes + theme(
            **{
                "axis.text.x.bottom": element_blank(),
                "axis.text.x.top": element_blank(),
                "axis.ticks.length.x.bottom": Unit(0, "pt"),
                "axis.ticks.length.x.top": Unit(0, "pt"),
            }
        )
    else:
        inner_axes = inner_axes + theme(
            **{
                "axis.text.x.bottom": element_blank(),
                "axis.text.x.top": element_blank(),
            }
        )

    if labels_inside in ("y", "both"):
        outer_axes = outer_axes + theme(
            **{
                "axis.text.y.left": element_blank(),
                "axis.text.y.right": element_blank(),
                "axis.ticks.length.y.left": Unit(0, "pt"),
                "axis.ticks.length.y.right": Unit(0, "pt"),
            }
        )
    else:
        inner_axes = inner_axes + theme(
            **{
                "axis.text.y.left": element_blank(),
                "axis.text.y.right": element_blank(),
            }
        )

    return CoordAxesInside(
        limits={"x": list(xlim) if xlim is not None else None,
                "y": list(ylim) if ylim is not None else None},
        expand=expand,
        default=default,
        clip=clip,
        ratio=ratio,
        # R: data_frame0(x = xintercept[1], y = yintercept[1]) takes the FIRST
        # element of a (possibly vector) intercept; np.atleast_1d handles both a
        # scalar and a vector uniformly (previously a vector was stored verbatim
        # and crashed downstream in float(origin["x"])).
        origin=pd.DataFrame(
            {
                "x": [np.atleast_1d(xintercept)[0]],
                "y": [np.atleast_1d(yintercept)[0]],
            }
        ),
        outer_axes=outer_axes,
        inner_axes=inner_axes,
    )
