"""Aimed text (port of ggh4x ``geom_text_aimed.R``).

Like :func:`ggplot2_py.geom_text`, ``geom_text_aimed()`` draws text, but it
rotates each label so it appears *aimed* towards a point defined by the
``xend``/``yend`` aesthetics.  The computed angle is added to the ``angle``
aesthetic and is evaluated in absolute coordinates, so resizing the plot keeps
the same appearance.

R source: ``ggh4x/R/geom_text_aimed.R``.

Notes
-----
* :meth:`GeomTextAimed.draw_panel` builds a second data frame ``aim`` from the
  ``xend``/``yend`` aesthetics (renamed to ``x``/``y``) and coord-transforms it
  separately so the aim point lands in the same transformed space as the text
  anchor.  Character ``hjust``/``vjust`` are resolved with :func:`compute_just`.
* ``xend``/``yend`` default to ``-Inf`` (the lower-left corner), so unaimed text
  points to the lower-left.  These are real, mappable default aesthetics.
* ``parse=True`` (R plotmath) has no engine in :mod:`grid_py`; this port falls
  back to plain-string labels (documented deviation).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ggplot2_py.geom import (
    GeomText,
    FromTheme,
    Gpar,
    Mapping,
    PT,
    _coord_transform,
    scales_alpha,
)

from ._aimed_text_grob import aimed_text_grob, compute_just

__all__ = [
    "geom_text_aimed",
    "GeomTextAimed",
]


class GeomTextAimed(GeomText):
    """Text geom that aims each label towards an ``(xend, yend)`` point.

    Subclass of :class:`ggplot2_py.GeomText` ported from R ``GeomTextAimed``
    (``geom_text_aimed.R:93-156``).
    """

    # R geom_text_aimed.R:95-107.  GeomText-like defaults plus the aim-target
    # aesthetics ``xend``/``yend`` (default -Inf) and ``angle``.
    default_aes: Mapping = Mapping(
        colour=FromTheme("colour", fallback="ink"),
        size=FromTheme("fontsize"),
        family=FromTheme("family", fallback=""),
        angle=0,
        xend=-np.inf,
        yend=-np.inf,
        hjust=0.5,
        vjust=0.5,
        alpha=None,
        fontface=1,
        lineheight=1.2,
    )

    extra_params = ("na_rm", "flip_upsidedown")

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        parse: bool = False,
        na_rm: bool = False,
        check_overlap: bool = False,
        flip_upsidedown: bool = True,
        **params: Any,
    ) -> Any:
        """Build an :class:`~ggh4x._aimed_text_grob.AimedTextGrob` for one panel.

        Port of R ``GeomTextAimed$draw_panel`` (``geom_text_aimed.R:108-154``).

        Parameters
        ----------
        data : pandas.DataFrame
            Layer data for one panel (carries ``xend``/``yend``).
        panel_params : Any
            Panel scales / ranges.
        coord : Any
            Active coordinate system.
        parse : bool, default ``False``
            R plotmath parsing.  Not supported; treated as a plain-label
            passthrough (documented deviation).
        na_rm : bool, default ``False``
            Whether missing values are silently removed.
        check_overlap : bool, default ``False``
            Whether overlapping labels are suppressed.
        flip_upsidedown : bool, default ``True``
            Whether labels rotated into ``(90, 270)`` are flipped for
            readability.
        **params : Any
            Ignored extra parameters.

        Returns
        -------
        grid_py.Grob
            An :class:`~ggh4x._aimed_text_grob.AimedTextGrob`.
        """
        data = data.copy()
        lab = data["label"].to_numpy() if "label" in data.columns else np.array([])

        # parse=True would build R plotmath expressions; no engine exists in
        # grid_py, so fall back to plain string labels.
        # (R raises if label is not character; we simply pass through.)

        # Build the aim frame from xend/yend and coord-transform separately so
        # the aim point ends up in the same transformed space.
        aim = pd.DataFrame(
            {
                "x": data["xend"].to_numpy(dtype="float64"),
                "y": data["yend"].to_numpy(dtype="float64"),
            }
        )
        data = _coord_transform(coord, data, panel_params)
        aim = _coord_transform(coord, aim, panel_params)

        # Resolve character justifications.
        hjust = data["hjust"].to_numpy() if "hjust" in data.columns else np.full(len(data), 0.5)
        vjust = data["vjust"].to_numpy() if "vjust" in data.columns else np.full(len(data), 0.5)
        if _is_character(vjust):
            vjust = compute_just(vjust, data["y"].to_numpy(dtype="float64"))
        else:
            vjust = np.asarray(vjust, dtype="float64")
        if _is_character(hjust):
            hjust = compute_just(hjust, data["x"].to_numpy(dtype="float64"))
        else:
            hjust = np.asarray(hjust, dtype="float64")

        size = data["size"].to_numpy(dtype="float64") if "size" in data.columns else np.full(len(data), 3.88)
        colour = data["colour"].to_numpy() if "colour" in data.columns else "black"
        alpha = data["alpha"].to_numpy() if "alpha" in data.columns else None
        family = data["family"].to_numpy() if "family" in data.columns else ""
        fontface = data["fontface"].to_numpy() if "fontface" in data.columns else 1
        lineheight = data["lineheight"].to_numpy() if "lineheight" in data.columns else 1.2
        angle = data["angle"].to_numpy(dtype="float64") if "angle" in data.columns else np.zeros(len(data))

        gp = Gpar(
            col=scales_alpha(colour, alpha),
            fontsize=size * PT,
            fontfamily=family,
            fontface=fontface,
            lineheight=lineheight,
        )

        return aimed_text_grob(
            label=lab,
            x=data["x"].to_numpy(dtype="float64"),
            y=data["y"].to_numpy(dtype="float64"),
            x0=aim["x"].to_numpy(dtype="float64"),
            y0=aim["y"].to_numpy(dtype="float64"),
            default_units="native",
            hjust=hjust,
            vjust=vjust,
            rot=angle,
            gp=gp,
            flip_upsidedown=flip_upsidedown,
            check_overlap=check_overlap,
        )


def _is_character(values: Any) -> bool:
    """Return ``True`` when *values* contains (any) string entries.

    Parameters
    ----------
    values : Any
        A scalar or array of justification values.

    Returns
    -------
    bool
        Whether the input is character-like (R ``is.character``).
    """
    arr = np.atleast_1d(np.asarray(values, dtype=object))
    return any(isinstance(v, str) for v in arr)


def geom_text_aimed(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    parse: bool = False,
    nudge_x: float = 0,
    nudge_y: float = 0,
    flip_upsidedown: bool = True,
    check_overlap: bool = False,
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Create an aimed-text layer.

    Port of R ``geom_text_aimed()`` (``geom_text_aimed.R:45-85``).  Draws text
    rotated to point at the ``xend``/``yend`` aim target.

    Parameters
    ----------
    mapping : Mapping, optional
        Aesthetic mapping created by :func:`ggplot2_py.aes`.
    data : Any, optional
        Layer data.
    stat : str, default ``"identity"``
        Statistical transformation.
    position : str, default ``"identity"``
        Position adjustment.  Cannot be combined with ``nudge_x``/``nudge_y``.
    parse : bool, default ``False``
        R plotmath parsing.  Not supported (plain-label fallback).
    nudge_x, nudge_y : float, default ``0``
        Horizontal / vertical nudge offsets (translated to a
        :func:`ggplot2_py.position_nudge`).
    flip_upsidedown : bool, default ``True``
        Whether labels rotated into ``(90, 270)`` are flipped for readability.
    check_overlap : bool, default ``False``
        Whether overlapping labels are suppressed.
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

    Raises
    ------
    ValueError
        If both ``position`` and ``nudge_x``/``nudge_y`` are specified.
    """
    from ggplot2_py.layer import layer

    if nudge_x != 0 or nudge_y != 0:
        if position != "identity":
            from ggh4x._cli import cli_abort

            cli_abort(
                "Specify either `position` or `nudge_x`/`nudge_y`, not both."
            )
        from ggplot2_py.position import position_nudge

        position = position_nudge(nudge_x, nudge_y)

    return layer(
        data=data,
        mapping=mapping,
        stat=stat,
        geom=GeomTextAimed,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "parse": parse,
            "check_overlap": check_overlap,
            "na_rm": na_rm,
            "flip_upsidedown": flip_upsidedown,
            **kwargs,
        },
    )
