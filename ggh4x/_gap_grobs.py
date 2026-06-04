"""Custom grobs and geometry helpers for :mod:`ggh4x.geom_pointpath`.

This module ports the grid-level machinery of R ggh4x ``geom_pointpath.R``:
the two custom grob classes that interrupt a path around its points, plus the
three pure-geometry helpers they rely on.

R source: ``ggh4x/R/geom_pointpath.R`` (``makeContext.gapsegments``,
``makeContext.gapsegmentschain``, ``intersect_line_circle``,
``crop_segment_ends``, ``filter_gp``).

Notes
-----
* In R the ``makeContext`` methods *reclass* the grob in place
  (``class(x)[1] <- "segments"``) and return it.  :mod:`grid_py` dispatches
  rendering on a fixed ``_grid_class`` registry and treats an unknown class as
  a silent no-op, so the Python ports instead **construct and return a freshly
  built renderable grob** (:func:`grid_py.segments_grob` /
  :func:`grid_py.polyline_grob`) from :meth:`Grob.make_context`.
* All trimming happens in absolute millimetres: the stored ``x0``/``y0``/
  ``x1``/``y1`` are npc :class:`grid_py.Unit` objects that are converted to mm
  against the live panel viewport inside ``make_context`` (so the gaps are
  resize-invariant, matching R).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from grid_py import (
    Gpar,
    Grob,
    Unit,
    convert_x,
    convert_y,
    null_grob,
    polyline_grob,
    segments_grob,
)

__all__ = [
    "intersect_line_circle",
    "crop_segment_ends",
    "filter_gp",
    "GapSegmentsGrob",
    "GapSegmentsChainGrob",
    "_chain_compute",
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def intersect_line_circle(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    r: np.ndarray,
    prio: int = 1,
) -> Dict[str, np.ndarray]:
    """Intersect a circle with a line, returning one intersection per element.

    Port of R ``intersect_line_circle`` (``geom_pointpath.R:322-363``).  The
    circle is parameterised by centre ``(cx, cy)`` and radius ``r``; the line
    passes through ``(x1, y1)`` and ``(x2, y2)``.  The implementation follows
    the Wolfram MathWorld *Circle-Line Intersection* formula.

    Parameters
    ----------
    x1, y1 : numpy.ndarray
        Coordinates of the first point on each line.
    x2, y2 : numpy.ndarray
        Coordinates of the second point on each line.
    cx, cy : numpy.ndarray
        Coordinates of each circle centre.
    r : numpy.ndarray
        Radius of each circle.
    prio : int, default ``1``
        Which intersection to return: ``1`` selects the intersection closer to
        ``(x1, y1)``, ``2`` selects the one closer to ``(x2, y2)``.

    Returns
    -------
    dict of numpy.ndarray
        ``{"x": ndarray, "y": ndarray}`` of the chosen intersection points
        (``NaN`` where the line misses the circle).
    """
    x1 = np.asarray(x1, dtype="float64")
    y1 = np.asarray(y1, dtype="float64")
    x2 = np.asarray(x2, dtype="float64")
    y2 = np.asarray(y2, dtype="float64")
    cx = np.asarray(cx, dtype="float64")
    cy = np.asarray(cy, dtype="float64")
    r = np.asarray(r, dtype="float64")

    # Centre the circle at (0, 0).
    x1 = x1 - cx
    x2 = x2 - cx
    y1 = y1 - cy
    y2 = y2 - cy

    dx = x2 - x1
    dy = y2 - y1
    dr2 = dx ** 2 + dy ** 2
    det = x1 * y2 - x2 * y1

    # Discriminant: <0 no intersection, 0 tangent, >0 two intersections.
    with np.errstate(invalid="ignore"):
        dis = r ** 2 * dr2 - det ** 2
        dis = np.where(dis < 0, np.nan, dis)
        dis = np.sqrt(dis)

    # R uses sign(dy); note sign(0) == 0 (matches numpy).
    sign_dy = np.sign(dy)
    abs_dy = np.abs(dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        x_1 = (det * dy + sign_dy * dx * dis) / dr2
        x_2 = (det * dy - sign_dy * dx * dis) / dr2
        y_1 = (-det * dx + abs_dy * dis) / dr2
        y_2 = (-det * dx - abs_dy * dis) / dr2

    if prio == 1:
        dist1 = np.sqrt((x1 - x_1) ** 2 + (y1 - y_1) ** 2)
        dist2 = np.sqrt((x1 - x_2) ** 2 + (y1 - y_2) ** 2)
    else:
        dist1 = np.sqrt((x2 - x_1) ** 2 + (y2 - y_1) ** 2)
        dist2 = np.sqrt((x2 - x_2) ** 2 + (y2 - y_2) ** 2)

    # R: ifelse(test, x_2, x_1); a NaN test propagates NaN (ifelse keeps NA).
    test = dist2 < dist1
    new_x = np.where(test, x_2, x_1) + cx
    new_y = np.where(test, y_2, y_1) + cy
    # Preserve R's NA propagation when the comparison itself is NA.
    nan_test = np.isnan(dist1) | np.isnan(dist2)
    new_x = np.where(nan_test, np.nan, new_x)
    new_y = np.where(nan_test, np.nan, new_y)
    return {"x": new_x, "y": new_y}


def crop_segment_ends(
    x0: np.ndarray,
    x1: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    r: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Shorten both ends of each segment by ``r``.

    Port of R ``crop_segment_ends`` (``geom_pointpath.R:365-386``).  Each
    segment ``(x0, y0) -> (x1, y1)`` is nudged inward at both ends by radius
    ``r`` (the gap around each point).  Non-finite nudges (zero-length
    segments) are replaced by zero, mirroring ggh4x issue #73.

    Parameters
    ----------
    x0, x1, y0, y1 : numpy.ndarray
        Segment endpoint coordinates.
    r : numpy.ndarray
        Per-segment crop radius.

    Returns
    -------
    dict of numpy.ndarray
        ``{"x0", "x1", "y0", "y1", "keep"}`` with the cropped coordinates and
        a boolean ``keep`` flagging segments that did not over-shoot (i.e. the
        crop did not reverse their orientation).
    """
    x0 = np.asarray(x0, dtype="float64").copy()
    x1 = np.asarray(x1, dtype="float64").copy()
    y0 = np.asarray(y0, dtype="float64").copy()
    y1 = np.asarray(y1, dtype="float64").copy()
    r = np.asarray(r, dtype="float64")

    dx = x1 - x0
    dy = y1 - y0
    hyp = np.sqrt(dx ** 2 + dy ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        nudge_y = (dy / hyp) * r
        nudge_x = (dx / hyp) * r

    nudge_y = np.where(np.isfinite(nudge_y), nudge_y, 0.0)
    nudge_x = np.where(np.isfinite(nudge_x), nudge_x, 0.0)

    new_x0 = x0 + nudge_x
    new_x1 = x1 - nudge_x
    new_y0 = y0 + nudge_y
    new_y1 = y1 - nudge_y

    keep = (np.sign(dx) == np.sign(new_x1 - new_x0)) & (
        np.sign(dy) == np.sign(new_y1 - new_y0)
    )
    return {
        "x0": new_x0,
        "x1": new_x1,
        "y0": new_y0,
        "y1": new_y1,
        "keep": keep,
    }


def filter_gp(gp: Optional[Gpar], keep: np.ndarray) -> Optional[Gpar]:
    """Subset every length>1 entry of a :class:`grid_py.Gpar` by ``keep``.

    Port of R ``filter_gp`` (``geom_pointpath.R:388-392``).  Scalar (length-1)
    graphical parameters are passed through unchanged; per-point vectors are
    subset to the kept positions so the gp stays aligned with the surviving
    segments.

    Parameters
    ----------
    gp : grid_py.Gpar or None
        The graphical parameters to filter.
    keep : numpy.ndarray
        Boolean mask (or integer index) selecting which positions to keep.

    Returns
    -------
    grid_py.Gpar or None
        A new :class:`grid_py.Gpar` with vector entries filtered, or ``None``
        if *gp* is ``None``.
    """
    if gp is None:
        return None
    keep = np.asarray(keep)
    params = _gpar_to_dict(gp)
    out: Dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            out[key] = value
            continue
        arr = np.asarray(value)
        # R: consider <- lengths(gp) > 1L
        if arr.ndim >= 1 and arr.shape[0] > 1:
            out[key] = arr[keep]
        else:
            out[key] = value
    return Gpar(**out)


def _gpar_to_dict(gp: Gpar) -> Dict[str, Any]:
    """Return the populated fields of a :class:`grid_py.Gpar` as a dict.

    Parameters
    ----------
    gp : grid_py.Gpar
        Graphical parameters.

    Returns
    -------
    dict
        Mapping of populated gp field names to their values.
    """
    # Gpar stores its values either in a dict-like ``_params`` mapping or as
    # plain attributes; handle both defensively.
    for attr in ("_params", "params"):
        store = getattr(gp, attr, None)
        if isinstance(store, dict):
            return {k: v for k, v in store.items() if v is not None}
    if hasattr(gp, "to_dict"):
        try:
            return {k: v for k, v in gp.to_dict().items() if v is not None}
        except Exception:
            pass
    # Fallback: scrape known gpar slots.
    known = (
        "col",
        "fill",
        "alpha",
        "lty",
        "lwd",
        "lex",
        "lineend",
        "linejoin",
        "linemitre",
        "fontsize",
        "cex",
        "fontfamily",
        "fontface",
        "lineheight",
        "font",
    )
    return {k: getattr(gp, k) for k in known if getattr(gp, k, None) is not None}


# ---------------------------------------------------------------------------
# Custom grobs
# ---------------------------------------------------------------------------
class GapSegmentsGrob(Grob):
    """Path-interrupting segments grob for *linear* coordinates.

    Stores the inter-point segments of a :class:`~ggh4x.geom_pointpath.GeomPointPath`
    as npc-unit coordinates plus the per-point crop radius ``mult``.  At draw
    time, :meth:`make_context` converts to millimetres, crops the segment ends
    and emits a plain :func:`grid_py.segments_grob`.

    Port of the grob carrying ``cl = "gapsegments"`` and its S3 method
    ``makeContext.gapsegments`` (``geom_pointpath.R:272-298``).
    """

    def __init__(
        self,
        x0: Unit,
        x1: Unit,
        y0: Unit,
        y1: Unit,
        mult: np.ndarray,
        id: np.ndarray,
        arrow: Any = None,
        name: Optional[str] = None,
        gp: Optional[Gpar] = None,
        vp: Optional[Any] = None,
    ) -> None:
        super().__init__(
            name=name,
            gp=gp,
            vp=vp,
            _grid_class="gapsegments",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            mult=np.asarray(mult, dtype="float64"),
            id=np.asarray(id),
            arrow=arrow,
        )

    def make_context(self) -> Grob:
        """Crop the segment ends in mm and return a renderable segments grob.

        Returns
        -------
        grid_py.Grob
            A :func:`grid_py.segments_grob` with cropped millimetre coordinates
            and the filtered ``gp``/``arrow``, or :func:`grid_py.null_grob`
            when every segment over-shoots.
        """
        x0 = np.asarray(convert_x(self.x0, "mm", valueOnly=True), dtype="float64")
        y0 = np.asarray(convert_y(self.y0, "mm", valueOnly=True), dtype="float64")
        x1 = np.asarray(convert_x(self.x1, "mm", valueOnly=True), dtype="float64")
        y1 = np.asarray(convert_y(self.y1, "mm", valueOnly=True), dtype="float64")

        cut = crop_segment_ends(x0, x1, y0, y1, self.mult)
        keep = cut["keep"]
        if not np.any(keep):
            return null_grob()

        gp = filter_gp(self.gp, keep)
        return segments_grob(
            x0=Unit(cut["x0"][keep], "mm"),
            x1=Unit(cut["x1"][keep], "mm"),
            y0=Unit(cut["y0"][keep], "mm"),
            y1=Unit(cut["y1"][keep], "mm"),
            arrow=self.arrow,
            gp=gp,
            name=self.name,
        )


class GapSegmentsChainGrob(Grob):
    """Path-interrupting polyline grob for *non-linear* coordinates.

    A much more involved version of :class:`GapSegmentsGrob`: it deletes
    segments whose start *and* end fall within the gap radius of a point, trims
    the partial-overlap edge cases with a circle-line intersection, re-stitches
    the surviving segments into a polyline and returns it.

    Port of the grob carrying ``cl = "gapsegmentschain"`` and its S3 method
    ``makeContext.gapsegmentschain`` (``geom_pointpath.R:156-264``).
    """

    def __init__(
        self,
        x0: Unit,
        x1: Unit,
        y0: Unit,
        y1: Unit,
        mult: np.ndarray,
        id: np.ndarray,
        arrow: Any = None,
        name: Optional[str] = None,
        gp: Optional[Gpar] = None,
        vp: Optional[Any] = None,
    ) -> None:
        super().__init__(
            name=name,
            gp=gp,
            vp=vp,
            _grid_class="gapsegmentschain",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            mult=np.asarray(mult, dtype="float64"),
            id=np.asarray(id),
            arrow=arrow,
        )

    def make_context(self) -> Grob:
        """Trim, re-stitch and return a renderable polyline grob.

        Returns
        -------
        grid_py.Grob
            A :func:`grid_py.polyline_grob` with millimetre coordinates, the
            re-grouped ``id`` and one ``gp`` entry per group, or
            :func:`grid_py.null_grob` when nothing survives.
        """
        x0 = np.asarray(convert_x(self.x0, "mm", valueOnly=True), dtype="float64")
        y0 = np.asarray(convert_y(self.y0, "mm", valueOnly=True), dtype="float64")
        x1 = np.asarray(convert_x(self.x1, "mm", valueOnly=True), dtype="float64")
        y1 = np.asarray(convert_y(self.y1, "mm", valueOnly=True), dtype="float64")

        result = _chain_compute(x0, x1, y0, y1, self.mult, self.id)
        if result is None:
            return null_grob()
        xy_x, xy_y, xy_id, keep, grp_start = result

        gp = filter_gp(self.gp, keep)
        gp = filter_gp(gp, grp_start)

        return polyline_grob(
            x=Unit(xy_x, "mm"),
            y=Unit(xy_y, "mm"),
            id=xy_id,
            gp=gp,
            arrow=self.arrow,
            name=self.name,
        )


def _chain_compute(
    x0: np.ndarray,
    x1: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    mult: np.ndarray,
    id_vec: np.ndarray,
) -> Optional[tuple]:
    """Run the gap-segments-chain trimming and re-stitching algorithm.

    Pure-geometry core of :meth:`GapSegmentsChainGrob.make_context`, split out
    so it can be verified against R without a live viewport.  Coordinates are
    in millimetres.  Port of ``makeContext.gapsegmentschain``
    (``geom_pointpath.R:156-256``).

    Parameters
    ----------
    x0, x1, y0, y1 : numpy.ndarray
        Segment endpoint coordinates (mm).
    mult : numpy.ndarray
        Per-segment gap radius (mm).
    id_vec : numpy.ndarray
        Per-segment group identifier.

    Returns
    -------
    tuple or None
        ``(x, y, id, keep, grp_start)`` where ``x``/``y``/``id`` are the
        polyline vertex arrays, ``keep`` is the per-segment survival mask and
        ``grp_start`` flags the first kept segment of each group.  Returns
        ``None`` when nothing survives.
    """
    x0 = np.asarray(x0, dtype="float64").copy()
    x1 = np.asarray(x1, dtype="float64").copy()
    y0 = np.asarray(y0, dtype="float64").copy()
    y1 = np.asarray(y1, dtype="float64").copy()
    mult = np.asarray(mult, dtype="float64")
    id_vec = np.asarray(id_vec)
    n = len(x0)

    # rle(id) -> per-element group start/end (0-based indices).
    start, end = _rle_start_end(id_vec)

    keep = np.ones(n, dtype=bool)

    # Distances to the group's start point.
    dist0_start = np.sqrt((x0 - x0[start]) ** 2 + (y0 - y0[start]) ** 2)
    dist1_start = np.sqrt((x1 - x0[start]) ** 2 + (y1 - y0[start]) ** 2)
    keep = keep & ((dist0_start > mult) | (dist1_start > mult))
    left = np.flatnonzero((dist1_start > mult) & ~(dist0_start > mult))

    # Distances to the group's end point.
    dist0 = np.sqrt((x0 - x1[end]) ** 2 + (y0 - y1[end]) ** 2)
    dist1 = np.sqrt((x1 - x1[end]) ** 2 + (y1 - y1[end]) ** 2)
    keep = keep & ((dist0 > mult) | (dist1 > mult))
    right = np.flatnonzero((dist0 > mult) != (dist1 > mult))

    # Edge cases that are both left and right need special handling.
    isect = np.intersect1d(left, right)
    if isect.size > 0:
        cut = crop_segment_ends(
            x0[isect], x1[isect], y0[isect], y1[isect], mult[isect]
        )
        x0[isect] = cut["x0"]
        x1[isect] = cut["x1"]
        y0[isect] = cut["y0"]
        y1[isect] = cut["y1"]
        keep[isect] = cut["keep"]
        left = np.setdiff1d(left, isect)
        right = np.setdiff1d(right, isect)

    if keep.sum() == 0:
        return None

    # Handle left edge cases.
    if left.size > 0:
        xy = intersect_line_circle(
            x1=x0[left],
            y1=y0[left],
            x2=x1[left],
            y2=y1[left],
            cx=x0[start[left]],
            cy=y0[start[left]],
            r=mult[left],
            prio=2,
        )
        x0[left] = xy["x"]
        y0[left] = xy["y"]

    # Handle right edge cases.
    if right.size > 0:
        xy = intersect_line_circle(
            x1=x1[right],
            y1=y1[right],
            x2=x0[right],
            y2=y0[right],
            cx=x1[end[right]],
            cy=y1[end[right]],
            r=mult[right],
            prio=2,
        )
        x1[right] = xy["x"]
        y1[right] = xy["y"]

    # Index that interleaves (start, end) of each kept segment, so the
    # segment list becomes a polyline vertex list.
    kept = np.flatnonzero(keep)
    idx = np.empty(2 * kept.size, dtype=int)
    idx[0::2] = kept
    idx[1::2] = kept + n

    cat_x = np.concatenate([x0, x1])
    cat_y = np.concatenate([y0, y1])
    cat_id = np.concatenate([id_vec, id_vec])
    xy_x = cat_x[idx]
    xy_y = cat_y[idx]
    xy_id = cat_id[idx]

    # Deduplicate consecutive identical (x, y, id) rows.
    m = len(xy_x)
    if m >= 2:
        same = (
            (xy_x[1:] == xy_x[:-1])
            & (xy_y[1:] == xy_y[:-1])
            & (xy_id[1:] == xy_id[:-1])
        )
        dup = np.concatenate([[False], same])
        keep_rows = ~dup
        xy_x = xy_x[keep_rows]
        xy_y = xy_y[keep_rows]
        xy_id = xy_id[keep_rows]

    # First kept segment of each group (one gp entry per group).
    id_kept = id_vec[keep]
    if id_kept.size > 0:
        grp_start = np.concatenate([[True], id_kept[1:] != id_kept[:-1]])
    else:
        grp_start = np.array([], dtype=bool)

    return xy_x, xy_y, xy_id, keep, grp_start


def _rle_start_end(id_vec: np.ndarray) -> tuple:
    """Compute per-element run start/end indices (0-based), mirroring R's ``rle``.

    For each element, ``start`` is the index of the first element of its
    run and ``end`` is the index of the last element of its run.  This is the
    0-based translation of the ``rep.int(start, lengths)`` /
    ``rep.int(end, lengths)`` idiom in ``makeContext.gapsegmentschain``.

    Parameters
    ----------
    id_vec : numpy.ndarray
        Run-length-encodable identifier vector.

    Returns
    -------
    tuple of numpy.ndarray
        ``(start, end)`` index arrays the same length as *id_vec*.
    """
    n = len(id_vec)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = id_vec[1:] != id_vec[:-1]
    run_starts = np.flatnonzero(change)  # index of first element of each run
    lengths = np.diff(np.append(run_starts, n))
    run_ends = run_starts + lengths - 1
    start = np.repeat(run_starts, lengths)
    end = np.repeat(run_ends, lengths)
    return start, end
