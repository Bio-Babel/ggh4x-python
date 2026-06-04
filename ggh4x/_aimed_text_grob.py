"""Aimed-text grob (port of the grid layer of ggh4x ``geom_text_aimed.R``).

This module ports ``aimed_textGrob()`` and its ``makeContent.aimed_text`` S3
method, plus the ggplot2-internal ``compute_just`` helper that
``GeomTextAimed`` relies on.

R source: ``ggh4x/R/geom_text_aimed.R`` (``aimed_textGrob``,
``makeContent.aimed_text``) and ggplot2-internal ``compute_just`` / ``just_dir``.

Notes
-----
* R's ``makeContent.aimed_text`` *reclasses* the grob to a ``"text"`` grob in
  place and returns it.  :mod:`grid_py`'s ``text`` renderer reads a scalar
  ``rot``/``hjust``/``vjust`` per call, so per-row angle/justification cannot
  travel on a single text grob.  :class:`AimedTextGrob` therefore subclasses
  :class:`grid_py.GTree`: its :meth:`make_content` performs the absolute-unit
  angle computation (after the panel viewport is active) and rebuilds its
  children as one scalar :func:`grid_py.text_grob` per label.  This is
  behaviourally identical to R while staying renderable.
* Angles are computed in millimetres (``convert_x``/``convert_y``) so they are
  invariant under resizing, matching R's design intent.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

import numpy as np

from grid_py import (
    GList,
    GTree,
    Gpar,
    Unit,
    convert_x,
    convert_y,
    is_unit,
    text_grob,
)

__all__ = [
    "compute_just",
    "just_dir",
    "aimed_text_grob",
    "AimedTextGrob",
]


# ---------------------------------------------------------------------------
# compute_just (ggplot2 internal)
# ---------------------------------------------------------------------------
def just_dir(x: np.ndarray, tol: float = 0.001) -> np.ndarray:
    """Map relative positions to justification directions ``1``/``2``/``3``.

    Port of ggplot2-internal ``just_dir``.  ``1`` (left/bottom) for values
    below ``0.5 - tol``, ``3`` (right/top) above ``0.5 + tol``, else ``2``
    (centre/middle).

    Parameters
    ----------
    x : numpy.ndarray
        Relative positions (typically in ``[0, 1]``).
    tol : float, default ``0.001``
        Half-width of the central dead zone.

    Returns
    -------
    numpy.ndarray
        Integer direction codes (1-based, matching R).
    """
    x = np.asarray(x, dtype="float64")
    out = np.full(x.shape, 2, dtype=int)
    out[x < 0.5 - tol] = 1
    out[x > 0.5 + tol] = 3
    return out


def compute_just(
    just: Union[str, Sequence[str], np.ndarray, Any],
    a: Any = 0.5,
    b: Any = None,
    angle: Any = 0,
) -> Any:
    """Resolve character justifications to numeric ``hjust``/``vjust``.

    Port of ggplot2-internal ``compute_just(just, a, b, angle)``.  Plain
    keywords (``"left"``/``"right"``/``"center"``/``"bottom"``/``"middle"``/
    ``"top"``) map to ``0``/``1``/``0.5``; ``"inward"``/``"outward"`` resolve
    relative to ``a`` (and ``b`` after rotation), honouring ``angle``.

    Parameters
    ----------
    just : str or sequence of str or array
        The justification specification.  Non-character input is returned
        unchanged (matching R's early return).
    a : array-like, default ``0.5``
        Primary position values (e.g. data ``x`` for ``hjust``).
    b : array-like, optional
        Secondary position values; defaults to *a* (used when rotation swaps
        the relevant axis).
    angle : array-like, default ``0``
        Text rotation in degrees.

    Returns
    -------
    numpy.ndarray
        Numeric justification values.
    """
    # R: if (!is.character(just)) return(just)
    arr = np.atleast_1d(np.asarray(just, dtype=object))
    if not all(isinstance(v, str) for v in arr):
        return just

    just_chr = np.array([str(v) for v in arr], dtype=object)
    a = np.atleast_1d(np.asarray(a, dtype="float64"))
    if b is None:
        b = a
    b = np.atleast_1d(np.asarray(b, dtype="float64"))

    has_io = np.array(["outward" in v or "inward" in v for v in just_chr])
    if np.any(has_io):
        ang = np.atleast_1d(np.asarray(angle, dtype="float64")) % 360
        ang = np.where(ang > 180, ang - 360, ang)
        ang = np.where(ang < -180, ang + 360, ang)
        # Broadcast angle to the length of just.
        ang = np.broadcast_to(ang, just_chr.shape).astype("float64").copy()

        rotated_forward = has_io & (ang > 45) & (ang < 135)
        rotated_backwards = has_io & (ang < -45) & (ang > -135)
        # ab <- ifelse(rotated_forward | rotated_backwards, b, a)
        a_b = np.broadcast_to(a, just_chr.shape).astype("float64").copy()
        b_b = np.broadcast_to(b, just_chr.shape).astype("float64").copy()
        ab = np.where(rotated_forward | rotated_backwards, b_b, a_b)

        just_swap = rotated_backwards | (np.abs(ang) > 135)

        inward = ((just_chr == "inward") & ~just_swap) | (
            (just_chr == "outward") & just_swap
        )
        if np.any(inward):
            dir_codes = just_dir(ab[inward])
            mapping = np.array(["left", "middle", "right"], dtype=object)
            just_chr[inward] = mapping[dir_codes - 1]

        outward = ((just_chr == "outward") & ~just_swap) | (
            (just_chr == "inward") & just_swap
        )
        if np.any(outward):
            dir_codes = just_dir(ab[outward])
            mapping = np.array(["right", "middle", "left"], dtype=object)
            just_chr[outward] = mapping[dir_codes - 1]

    lookup = {
        "left": 0.0,
        "center": 0.5,
        "centre": 0.5,
        "right": 1.0,
        "bottom": 0.0,
        "middle": 0.5,
        "top": 1.0,
    }
    out = np.array([lookup.get(str(v), 0.5) for v in just_chr], dtype="float64")
    return out


# ---------------------------------------------------------------------------
# aimed_text_grob constructor
# ---------------------------------------------------------------------------
def aimed_text_grob(
    label: Any,
    x: Any = 0.5,
    y: Any = 0.5,
    x0: Any = 0.0,
    y0: Any = 0.0,
    just: Any = "centre",
    hjust: Any = None,
    vjust: Any = None,
    rot: Any = 0,
    check_overlap: bool = False,
    default_units: str = "npc",
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
    flip_upsidedown: bool = True,
) -> "AimedTextGrob":
    """Construct an :class:`AimedTextGrob`.

    Port of R ``aimed_textGrob()`` (``geom_text_aimed.R:161-183``).  Bare
    numeric coordinates are wrapped in *default_units*.

    Parameters
    ----------
    label : array-like
        Text labels.
    x, y : array-like or grid unit
        Text anchor positions.
    x0, y0 : array-like or grid unit
        The aim-target positions.
    just : str or sequence of str, default ``"centre"``
        Default justification.
    hjust, vjust : array-like, optional
        Horizontal / vertical justification.
    rot : array-like, default ``0``
        Base rotation (the aim angle is added on top at draw time).
    check_overlap : bool, default ``False``
        Whether overlapping labels are suppressed.
    default_units : str, default ``"npc"``
        Units for bare numeric coordinates.
    name : str, optional
        Grob name.
    gp : grid_py.Gpar, optional
        Graphical parameters.
    vp : Any, optional
        Viewport.
    flip_upsidedown : bool, default ``True``
        If ``True``, labels rotated into ``(90, 270)`` are flipped 180 degrees
        for readability (with mirrored ``hjust``).

    Returns
    -------
    AimedTextGrob
        The custom grob.
    """
    x = x if is_unit(x) else Unit(np.atleast_1d(np.asarray(x, dtype="float64")), default_units)
    y = y if is_unit(y) else Unit(np.atleast_1d(np.asarray(y, dtype="float64")), default_units)
    x0 = x0 if is_unit(x0) else Unit(np.atleast_1d(np.asarray(x0, dtype="float64")), default_units)
    y0 = y0 if is_unit(y0) else Unit(np.atleast_1d(np.asarray(y0, dtype="float64")), default_units)

    return AimedTextGrob(
        label=label,
        x=x,
        y=y,
        x0=x0,
        y0=y0,
        just=just,
        hjust=hjust,
        vjust=vjust,
        rot=rot,
        check_overlap=check_overlap,
        flip_upsidedown=flip_upsidedown,
        name=name,
        gp=gp,
        vp=vp,
    )


# ---------------------------------------------------------------------------
# AimedTextGrob
# ---------------------------------------------------------------------------
class AimedTextGrob(GTree):
    """Text grob that rotates each label towards a per-label aim point.

    Subclass of :class:`grid_py.GTree`.  Stored ``x``/``y``/``x0``/``y0`` are
    grid units; :meth:`make_content` converts them to millimetres against the
    live panel viewport, computes ``ang = atan2(y1 - y0, x1 - x0)``, adds it to
    the base ``rot`` and rebuilds the children as one scalar
    :func:`grid_py.text_grob` per label.

    Port of the grob carrying ``cl = "aimed_text"`` and its S3 method
    ``makeContent.aimed_text`` (``geom_text_aimed.R:188-214``).
    """

    def __init__(
        self,
        label: Any,
        x: Unit,
        y: Unit,
        x0: Unit,
        y0: Unit,
        just: Any = "centre",
        hjust: Any = None,
        vjust: Any = None,
        rot: Any = 0,
        check_overlap: bool = False,
        flip_upsidedown: bool = True,
        name: Optional[str] = None,
        gp: Optional[Gpar] = None,
        vp: Optional[Any] = None,
    ) -> None:
        super().__init__(
            children=None,
            name=name,
            gp=gp,
            vp=vp,
            _grid_class="aimed_text",
            label=label,
            x=x,
            y=y,
            x0=x0,
            y0=y0,
            just=just,
            hjust=hjust,
            vjust=vjust,
            rot=rot,
            check_overlap=check_overlap,
            flip_upsidedown=flip_upsidedown,
        )

    def make_content(self) -> "AimedTextGrob":
        """Compute aim angles in mm and rebuild the per-label text children.

        Returns
        -------
        AimedTextGrob
            ``self``, with its children replaced by the rotated text grobs.
        """
        x1 = np.atleast_1d(np.asarray(convert_x(self.x, "mm", valueOnly=True), dtype="float64"))
        y1 = np.atleast_1d(np.asarray(convert_y(self.y, "mm", valueOnly=True), dtype="float64"))
        x0 = np.atleast_1d(np.asarray(convert_x(self.x0, "mm", valueOnly=True), dtype="float64"))
        y0 = np.atleast_1d(np.asarray(convert_y(self.y0, "mm", valueOnly=True), dtype="float64"))

        rot, hjust, vjust = compute_aimed_angles(
            x1, y1, x0, y0, self.rot, self.hjust, self.vjust, self.flip_upsidedown
        )

        labels = self.label
        if isinstance(labels, str):
            labels = [labels]
        labels = list(labels)
        n = len(labels)

        # Native anchor positions for the text grobs (drawing happens in the
        # panel vp, so the original native/npc x/y are reused directly).
        xs = np.atleast_1d(np.asarray(convert_x(self.x, "native", valueOnly=True), dtype="float64"))
        ys = np.atleast_1d(np.asarray(convert_y(self.y, "native", valueOnly=True), dtype="float64"))

        gp = self.gp
        children: List[Any] = []
        for i in range(n):
            gp_i = _subset_gpar(gp, i, n)
            children.append(
                text_grob(
                    label=str(labels[i]),
                    x=Unit(float(xs[i % len(xs)]), "native"),
                    y=Unit(float(ys[i % len(ys)]), "native"),
                    hjust=float(hjust[i % len(hjust)]),
                    vjust=float(vjust[i % len(vjust)]),
                    rot=float(rot[i % len(rot)]),
                    check_overlap=bool(self.check_overlap),
                    gp=gp_i,
                    name=f"aimed_text.{i}",
                )
            )

        self.set_children(GList(*children))
        return self


def compute_aimed_angles(
    x1: np.ndarray,
    y1: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    rot: Any,
    hjust: Any,
    vjust: Any,
    flip_upsidedown: bool,
) -> tuple:
    """Compute rotated text angles and (possibly mirrored) justifications.

    Pure-math core of :meth:`AimedTextGrob.make_content`, split out for
    verification.  Port of ``makeContent.aimed_text``
    (``geom_text_aimed.R:195-204``).

    Parameters
    ----------
    x1, y1 : numpy.ndarray
        Text anchor positions (mm).
    x0, y0 : numpy.ndarray
        Aim-target positions (mm).
    rot : array-like
        Base rotation (degrees).
    hjust, vjust : array-like
        Justification values.
    flip_upsidedown : bool
        Whether to flip labels rotated into ``(90, 270)``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(rot, hjust, vjust)`` after the aim rotation (and optional flip).
    """
    rot = np.atleast_1d(np.asarray(rot, dtype="float64"))
    hjust = np.atleast_1d(np.asarray(hjust, dtype="float64"))
    vjust = np.atleast_1d(np.asarray(vjust, dtype="float64"))

    ang = np.arctan2(y1 - y0, x1 - x0)
    rot = (rot + ang * (180.0 / np.pi)) % 360.0

    if flip_upsidedown:
        upsidedown = (rot > 90) & (rot < 270)
        rot = np.where(upsidedown, rot + 180, rot) % 360.0
        # hjust may be a scalar; broadcast before mirroring.
        hjust = np.broadcast_to(hjust, upsidedown.shape).astype("float64").copy()
        hjust = np.where(upsidedown, 1.0 - hjust, hjust)

    return rot, hjust, vjust


def _subset_gpar(gp: Optional[Gpar], i: int, n: int) -> Optional[Gpar]:
    """Return the ``i``-th element of every length>1 :class:`grid_py.Gpar` field.

    Parameters
    ----------
    gp : grid_py.Gpar or None
        Graphical parameters that may carry per-label vectors.
    i : int
        Row index.
    n : int
        Total number of labels.

    Returns
    -------
    grid_py.Gpar or None
        A per-label :class:`grid_py.Gpar`, or ``None`` if *gp* is ``None``.
    """
    if gp is None:
        return None
    from ._gap_grobs import _gpar_to_dict

    params = _gpar_to_dict(gp)
    out = {}
    for key, value in params.items():
        if value is None:
            out[key] = value
            continue
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == n and n > 1:
            out[key] = arr[i]
        else:
            out[key] = value
    return Gpar(**out)
