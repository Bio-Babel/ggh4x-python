"""Partial rectangle theme element.

Port of ggh4x R source ``element_part_rect.R``.

The :func:`element_part_rect` factory draws individual sides of a rectangle as a
theme element, substituting :func:`ggplot2_py.theme_elements.element_rect`.  A
``side`` string built from the letters ``"t"`` (top), ``"l"`` (left), ``"b"``
(bottom) and ``"r"`` (right) selects which borders are drawn.

Components ported from the R file
---------------------------------
* :class:`ElementPartRect` -- subclass of ``ElementRect`` carrying a ``side`` slot
  (the R S3 class vector ``c("element_part_rect", "element_rect", "element")``).
* :func:`element_part_rect` -- polymorphic factory (R L29-78).  All four sides ->
  plain ``element_rect``; no sides -> ``element_rect(colour=NA)``; otherwise an
  ``ElementPartRect``.
* :func:`_grob_from_part_rect` -- the ``element_grob.element_part_rect`` S3 method
  (R L82-109): merges caller graphical parameters over the element's own and
  delegates to :func:`part_rect_grob`.
* :func:`part_rect_grob` -- ``partrectGrob`` (R L111-243): a grob tree of a
  fill-only rectangle plus a segments grob drawing the requested sides.
* :func:`element_grob` -- a ggh4x-local wrapper around
  ``ggplot2_py.theme_elements.element_grob`` that intercepts ``ElementPartRect``
  (which is-a ``ElementRect``) *before* the upstream ``ElementRect`` branch.

Notes
-----
``ggplot2_py.theme_elements.element_grob`` is a hard ``isinstance`` chain rather
than R-style S3 dispatch, so the wrapper here adds the missing
``ElementPartRect`` branch.  Importing this module also monkeypatches the
upstream ``element_grob`` so any code path that dispatches an ``ElementPartRect``
through the original function still renders partial borders.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Union

from grid_py import (
    Gpar,
    Unit,
    grob_tree,
    is_unit,
    rect_grob,
    segments_grob,
    unit_c,
    unit_rep,
)

import ggplot2_py.theme_elements as _te
from ggplot2_py._compat import NA, is_na
from ggplot2_py.theme_elements import ElementRect, element_rect
from ggplot2_py.theme_elements import _PT  # 72.27 / 25.4 == R's .pt

__all__ = [
    "ElementPartRect",
    "element_part_rect",
    "part_rect_grob",
    "element_grob",
]


# ---------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------


class ElementPartRect(ElementRect):
    """Theme element drawing partial rectangle borders.

    Subclasses :class:`ggplot2_py.theme_elements.ElementRect` so existing theme
    inheritance (``combine_elements`` / ``merge_element``) treats it like a rect
    for ``strip.background`` / ``panel.background``.  Adds a ``side`` slot
    selecting which borders to draw.

    Parameters
    ----------
    fill : str or None
        Fill colour.
    colour : str or None
        Border colour.
    linewidth : float or None
        Border width in millimetres.
    linetype : int, str, or None
        Border line type.
    inherit_blank : bool
        Whether to inherit ``element_blank`` from parents.
    side : str
        Any combination of ``"t"``, ``"l"``, ``"b"``, ``"r"`` selecting the
        top, left, bottom and right borders respectively.
    """

    def __init__(
        self,
        fill: Optional[str] = None,
        colour: Optional[str] = None,
        linewidth: Optional[float] = None,
        linetype: Optional[Union[int, str]] = None,
        inherit_blank: bool = False,
        side: str = "tlbr",
    ) -> None:
        super().__init__(
            fill=fill,
            colour=colour,
            linewidth=linewidth,
            linetype=linetype,
            inherit_blank=inherit_blank,
        )
        self.side = side

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        parts = []
        for attr in ("fill", "colour", "linewidth", "linetype", "side", "inherit_blank"):
            val = getattr(self, attr)
            if val is not None and val is not False:
                parts.append(f"{attr}={val!r}")
        return f"element_part_rect({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Factory (R L29-78)
# ---------------------------------------------------------------------------


def element_part_rect(
    side: str = "tlbr",
    fill: Optional[str] = None,
    colour: Optional[str] = None,
    linewidth: Optional[float] = None,
    linetype: Optional[Union[int, str]] = None,
    color: Optional[str] = None,
    inherit_blank: bool = False,
) -> ElementRect:
    """Construct a partial-rectangle theme element.

    Polymorphic factory mirroring R ``element_part_rect`` (L29-78):

    * If ``side`` contains all of ``t``, ``l``, ``b`` and ``r`` the element
      simplifies to a plain :func:`ggplot2_py.theme_elements.element_rect`.
    * If ``side`` contains none of those letters it simplifies to
      ``element_rect(colour=NA, ...)`` (a borderless rectangle).
    * Otherwise an :class:`ElementPartRect` is returned.

    Parameters
    ----------
    side : str
        Any combination of ``"t"``, ``"l"``, ``"b"``, ``"r"``.  Including all
        or none of these letters defaults to a regular ``element_rect()``.
    fill : str or None
        Fill colour.
    colour : str or None
        Border colour.
    linewidth : float or None
        Line/border size in millimetres.
    linetype : int, str, or None
        Line type: an integer (0--8), a name, or a hex dash string.
    color : str or None
        American-spelling alias for ``colour``; overrides ``colour`` when given.
    inherit_blank : bool
        Whether to inherit ``element_blank`` from parents.

    Returns
    -------
    ElementRect or ElementPartRect
        A plain ``ElementRect`` for the all-sides / no-sides cases, otherwise
        an ``ElementPartRect``.
    """
    if color is not None:
        colour = color

    # R: grepl("(?=.*t)(?=.*l)(?=.*r)(?=.*b)", side, perl = TRUE)
    all_sides = re.search(r"(?=.*t)(?=.*l)(?=.*r)(?=.*b)", side) is not None
    if all_sides:
        # Simplifies to regular rectangle.
        return element_rect(
            fill=fill,
            colour=colour,
            linewidth=linewidth,
            linetype=linetype,
            inherit_blank=inherit_blank,
        )

    # R: !grepl("t|l|r|b", side, perl = TRUE)
    no_sides = re.search(r"[tlrb]", side) is None
    if no_sides:
        # Also simplifies to a regular rectangle, but with no colour.
        return element_rect(
            fill=fill,
            colour=NA,
            linewidth=linewidth,
            linetype=linetype,
            inherit_blank=inherit_blank,
        )

    return ElementPartRect(
        fill=fill,
        colour=colour,
        linewidth=linewidth,
        linetype=linetype,
        inherit_blank=inherit_blank,
        side=side,
    )


# ---------------------------------------------------------------------------
# gpar helpers
# ---------------------------------------------------------------------------


def _gpar_from_fields(
    lwd: Any = None,
    col: Any = None,
    fill: Any = None,
    lty: Any = None,
) -> dict:
    """Build a gpar-parameter dict applying R ``gpar(...)`` NULL-dropping.

    R ``gpar(lwd = NULL, col = colour, ...)`` silently drops any ``NULL``
    argument.  ``NA`` is retained (e.g. ``gpar(col = NA)``) and forwarded as
    grid_py's ``None`` NA sentinel.  This helper returns a plain ``dict`` so
    callers can further merge/strip fields before constructing a
    :class:`grid_py.Gpar`.

    Parameters
    ----------
    lwd, col, fill, lty : Any
        Candidate graphical-parameter values.  ``None`` mirrors R ``NULL``
        (drop the field); :data:`ggplot2_py._compat.NA` maps to grid_py's
        ``None`` NA sentinel (retain the field, draw nothing).

    Returns
    -------
    dict
        Parameter dict with ``None`` (NULL) entries omitted.  ``NA`` entries
        are stored as Python ``None`` (grid_py NA sentinel).
    """
    out: dict = {}
    for key, value in (("lwd", lwd), ("col", col), ("fill", fill), ("lty", lty)):
        if value is None:
            continue  # R NULL -> drop entirely
        if is_na(value):
            out[key] = None  # R NA -> grid_py NA sentinel
        else:
            out[key] = value
    return out


def _strip_field(params: dict, field: str) -> Gpar:
    """Return a :class:`grid_py.Gpar` with ``field`` forced to the NA sentinel.

    Mirrors R ``do.call(gpar, within(gp, col <- NA))`` (and the ``fill`` twin):
    the named field is *set* to ``NA`` (kept, value NA) while every other field
    is preserved unchanged.

    Parameters
    ----------
    params : dict
        Source parameter dict (already NULL-dropped).
    field : str
        Field to force to ``NA`` -- ``"col"`` or ``"fill"``.

    Returns
    -------
    grid_py.Gpar
        The graphical parameters with ``field`` set to grid_py's ``None`` NA
        sentinel (renders nothing for that aesthetic).
    """
    merged = dict(params)
    merged[field] = None  # grid_py NA sentinel (== R NA)
    return Gpar(**merged)


# ---------------------------------------------------------------------------
# element_grob.element_part_rect (R L82-109)
# ---------------------------------------------------------------------------


def _grob_from_part_rect(
    element: ElementPartRect,
    x: Any = 0.5,
    y: Any = 0.5,
    width: Any = 1,
    height: Any = 1,
    fill: Optional[str] = None,
    colour: Optional[str] = None,
    linewidth: Optional[float] = None,
    linetype: Optional[Union[int, str]] = None,
    **kwargs: Any,
) -> Any:
    """Render an :class:`ElementPartRect` as a grob.

    Port of R ``element_grob.element_part_rect`` (L82-109).  Builds a caller
    graphical-parameter set and an element graphical-parameter set, then lets
    the caller values override the element values field-by-field before calling
    :func:`part_rect_grob`.

    Parameters
    ----------
    element : ElementPartRect
        The element being rendered (supplies ``side`` plus default
        fill/colour/linewidth/linetype).
    x, y : Unit or numeric
        Rectangle centre.  Default ``0.5`` npc.
    width, height : Unit or numeric
        Rectangle size.  Default ``1`` npc.
    fill : str or None
        Caller fill override (millimetre-independent).
    colour : str or None
        Caller border-colour override.
    linewidth : float or None
        Caller border width in millimetres (converted to ``lwd`` via ``_PT``).
    linetype : int, str, or None
        Caller line-type override.
    **kwargs
        Forwarded to :func:`part_rect_grob` (e.g. ``name``, ``vp``,
        ``default_units``).

    Returns
    -------
    grid_py.GTree
        A grob tree drawing the requested borders.
    """
    # R: caller gp = gpar(lwd = linewidth * .pt or NULL, col, fill, lty)
    caller = _gpar_from_fields(
        lwd=(linewidth * _PT) if linewidth is not None else None,
        col=colour,
        fill=fill,
        lty=linetype,
    )

    # R: element_gp = gpar(lwd = element$linewidth * .pt or NULL, ...)
    el_lw = element.linewidth
    element_gp = _gpar_from_fields(
        lwd=(el_lw * _PT) if el_lw is not None else None,
        col=element.colour,
        fill=element.fill,
        lty=element.linetype,
    )

    # R: for (i in names(gp)) element_gp[[i]] <- gp[[i]]  (caller overrides)
    for key, value in caller.items():
        element_gp[key] = value

    return part_rect_grob(
        x,
        y,
        width,
        height,
        gp=element_gp,
        sides=element.side,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# partrectGrob (R L111-243)
# ---------------------------------------------------------------------------


def _as_unit(value: Any, default_units: str) -> Unit:
    """Coerce *value* to a :class:`grid_py.Unit` (R ``if (!is.unit(x)) unit(...)``)."""
    if is_unit(value):
        return value
    return Unit(value, default_units)


# Side geometry: factors of (x, width) for x0/x1 and (y, height) for y0/y1.
# Each tuple is (wx0, wx1, hy0, hy1) where the endpoint is
#   x0 = x + wx0 * width, x1 = x + wx1 * width,
#   y0 = y + hy0 * height, y1 = y + hy1 * height.
# Order of evaluation is fixed t -> b -> l -> r regardless of the input string
# (mirrors R's sequence of grepl blocks, L144-230).
_SIDE_SPECS = (
    ("t", (-0.5, 0.5, 0.5, 0.5)),
    ("b", (-0.5, 0.5, -0.5, -0.5)),
    ("l", (-0.5, -0.5, -0.5, 0.5)),
    ("r", (0.5, 0.5, -0.5, 0.5)),
)


def part_rect_grob(
    x: Any = None,
    y: Any = None,
    width: Any = None,
    height: Any = None,
    default_units: str = "npc",
    name: Optional[str] = None,
    gp: Optional[Union[Gpar, dict]] = None,
    vp: Optional[Any] = None,
    sides: str = "tlbr",
) -> Any:
    """Build a grob tree drawing selected rectangle borders.

    Port of R ``partrectGrob`` (L111-243).  Produces a :class:`grid_py.GTree`
    containing (1) a fill-only rectangle (border stripped) drawn first, and (2)
    a segments grob (fill stripped) drawing only the requested sides, drawn on
    top.

    Parameters
    ----------
    x, y : Unit or numeric, optional
        Rectangle centre.  Defaults to ``0.5`` npc.
    width, height : Unit or numeric, optional
        Rectangle size.  Defaults to ``1`` npc.
    default_units : str
        Unit type applied to bare numerics.  Default ``"npc"``.
    name : str or None
        Name for the returned grob tree.
    gp : grid_py.Gpar or dict or None
        Graphical parameters.  A dict is accepted and treated like R's
        unclassed gpar list.
    vp : object or None
        Optional viewport (applied to the children and the tree, matching R).
    sides : str
        Any combination of ``"t"``, ``"l"``, ``"b"``, ``"r"`` selecting which
        borders to draw.

    Returns
    -------
    grid_py.GTree
        ``grob_tree(fillgrob, sidegrob, name=name, vp=vp)``.
    """
    if x is None:
        x = Unit(0.5, "npc")
    if y is None:
        y = Unit(0.5, "npc")
    if width is None:
        width = Unit(1, "npc")
    if height is None:
        height = Unit(1, "npc")

    x = _as_unit(x, default_units)
    y = _as_unit(y, default_units)
    width = _as_unit(width, default_units)
    height = _as_unit(height, default_units)

    # R: gp <- unclass(gp)  -- normalise to a plain param dict.
    if gp is None:
        params: dict = {}
    elif isinstance(gp, Gpar):
        params = gp.params
    elif isinstance(gp, dict):
        params = dict(gp)
    else:  # pragma: no cover - defensive
        raise TypeError(f"gp must be a Gpar, dict, or None, got {type(gp)!r}")

    # R: rectfill = rectGrob(..., gp = do.call(gpar, within(gp, col <- NA)))
    rectfill = rect_grob(
        x=x,
        y=y,
        width=width,
        height=height,
        default_units=default_units,
        name="fillgrob",
        vp=vp,
        gp=_strip_field(params, "col"),
    )

    # n = max(length(x), length(width), length(y), length(height))
    n = max(len(x), len(width), len(y), len(height))

    x0: Optional[Unit] = None
    x1: Optional[Unit] = None
    y0: Optional[Unit] = None
    y1: Optional[Unit] = None

    for letter, (wx0, wx1, hy0, hy1) in _SIDE_SPECS:
        # R: grepl("(?=.*t)", sides, perl = TRUE) etc. -- letter present?
        if letter not in sides:
            continue
        seg_x0 = unit_rep(x + width * wx0, length_out=n)
        seg_x1 = unit_rep(x + width * wx1, length_out=n)
        seg_y0 = unit_rep(y + height * hy0, length_out=n)
        seg_y1 = unit_rep(y + height * hy1, length_out=n)
        if x0 is None:
            x0, x1, y0, y1 = seg_x0, seg_x1, seg_y0, seg_y1
        else:
            x0 = unit_c(x0, seg_x0)
            x1 = unit_c(x1, seg_x1)
            y0 = unit_c(y0, seg_y0)
            y1 = unit_c(y1, seg_y1)

    # R: sidegrob = segmentsGrob(..., gp = do.call(gpar, within(gp, fill <- NA)))
    sidegrob = segments_grob(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        name="sidegrob",
        gp=_strip_field(params, "fill"),
        vp=vp,
    )

    # R: grobTree(rectfill, sidegrob, name = name, vp = vp)  -- fill first.
    return grob_tree(rectfill, sidegrob, name=name, vp=vp)


# ---------------------------------------------------------------------------
# element_grob dispatch wrapper
# ---------------------------------------------------------------------------

# Capture the upstream (possibly already-patched) implementation exactly once.
_gg_element_grob = getattr(_te, "_ggh4x_orig_element_grob", _te.element_grob)


def element_grob(element: Any, **kwargs: Any) -> Any:
    """Dispatch a theme element to its grob, handling :class:`ElementPartRect`.

    ``ggplot2_py.theme_elements.element_grob`` is a hard ``isinstance`` chain,
    not S3 dispatch.  Because :class:`ElementPartRect` is-a ``ElementRect``, the
    upstream ``ElementRect`` branch would wrongly draw a full rectangle, so this
    wrapper intercepts ``ElementPartRect`` *first* and otherwise defers to the
    upstream function.

    Parameters
    ----------
    element : Element
        Any theme element.
    **kwargs
        Forwarded to the appropriate renderer.

    Returns
    -------
    Grob
        A grid grob.
    """
    if isinstance(element, ElementPartRect):
        return _grob_from_part_rect(element, **kwargs)
    return _gg_element_grob(element, **kwargs)


def _install_element_grob_patch() -> None:
    """Monkeypatch ``ggplot2_py.theme_elements.element_grob`` with the wrapper.

    Stores the original under ``_ggh4x_orig_element_grob`` so re-imports stay
    idempotent, then replaces the module attribute so any code dispatching an
    ``ElementPartRect`` through the upstream function renders partial borders.
    """
    if getattr(_te, "_ggh4x_orig_element_grob", None) is None:
        _te._ggh4x_orig_element_grob = _te.element_grob  # type: ignore[attr-defined]
    _te.element_grob = element_grob  # type: ignore[assignment]


_install_element_grob_patch()
