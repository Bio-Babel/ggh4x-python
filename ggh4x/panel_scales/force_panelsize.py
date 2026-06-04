"""Force facet panel sizes (port of ggh4x ``R/force_panelsize.R``).

:func:`force_panelsizes` returns a :class:`ForcedSize` add-on object.  When added
to a plot with ``+``, its handler (registered on
:func:`ggplot2_py.plot.update_ggplot`) clones the plot's live facet into a
dynamic ``Forced<FacetClass>`` subclass whose ``draw_panels`` calls the original
facet's ``draw_panels`` and then overwrites the resulting gtable's panel row
heights / column widths with the user-forced ``"null"`` units (and sets
``respect``).

This mirrors R, where ``ggplot_add.forcedsize`` wraps ``old.facet$draw_panels``
in a new function that mutates the panel gtable's ``widths`` / ``heights`` after
the parent produced it.  Because the rewrite happens *inside* ``draw_panels``
(post-parent), it overrules the theme ``panel.widths`` / ``panel.heights``
applied by ``Facet.set_panel_size`` afterwards, exactly as the R ``space``
argument is overruled.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ggplot2_py import ggproto
from ggplot2_py.ggproto import ggproto_parent
from ggplot2_py.plot import update_ggplot
from grid_py import Unit, convert_height, convert_width, is_unit, unit_type

from ggh4x._facet_utils import panel_cols, panel_rows

__all__ = [
    "force_panelsizes",
    "ForcedSize",
    "is_null_unit",
]


# ---------------------------------------------------------------------------
# is_null_unit (R force_panelsize.R:197-202)
# ---------------------------------------------------------------------------
def is_null_unit(x: Any) -> bool:
    """Return whether *x* is a unit composed entirely of ``"null"`` units.

    Faithful port of ggh4x's ``is_null_unit`` (``R/force_panelsize.R:197-202``):
    a non-unit returns ``False``; otherwise every element's :func:`unit_type`
    must equal ``"null"``.

    Parameters
    ----------
    x : Any
        Candidate object.

    Returns
    -------
    bool
        ``True`` when *x* is a unit whose every element is ``"null"``.
    """
    if not is_unit(x):
        return False
    types = unit_type(x)
    if isinstance(types, str):
        types = [types]
    return all(t == "null" for t in types)


_ABSOLUTE_TOTAL_UNITS = ("cm", "mm", "inches", "points", "bigpts")


# ---------------------------------------------------------------------------
# ForcedSize container (R: structure(list(...), class = "forcedsize"))
# ---------------------------------------------------------------------------
class ForcedSize:
    """Deferred container of forced panel sizes (R ``forcedsize`` S3 object).

    Holds the five fields produced by :func:`force_panelsizes`.  It is *not* a
    facet; it is consumed by :func:`_update_forcedsize` at ``+``-time, which
    rewrites the plot's facet.

    Attributes
    ----------
    rows, cols : grid_py.Unit or None
        Forced panel heights (rows) / widths (cols), as ``"null"`` units (or
        absolute when supplied directly).
    respect : bool or None
        Forced ``respect`` flag for the panel gtable, or ``None`` to inherit.
    total_width, total_height : grid_py.Unit or None
        Absolute total width / height of all panels plus inter-panel decoration.
    """

    def __init__(
        self,
        rows: Optional[Unit] = None,
        cols: Optional[Unit] = None,
        respect: Optional[bool] = None,
        total_width: Optional[Unit] = None,
        total_height: Optional[Unit] = None,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.respect = respect
        self.total_width = total_width
        self.total_height = total_height

    def _lengths_sum(self) -> int:
        """Return ``sum(lengths(object))`` (R ``force_panelsize.R:96``)."""
        total = 0
        for field in (self.rows, self.cols, self.respect,
                      self.total_width, self.total_height):
            if field is None:
                continue
            if is_unit(field):
                total += len(field)
            else:
                total += 1
        return total


# ---------------------------------------------------------------------------
# force_panelsizes constructor (R force_panelsize.R:51-85)
# ---------------------------------------------------------------------------
def force_panelsizes(
    rows: Any = None,
    cols: Any = None,
    respect: Optional[bool] = None,
    total_width: Any = None,
    total_height: Any = None,
) -> ForcedSize:
    """Force a facetted plot to have specified panel sizes.

    Faithful port of ggh4x's ``force_panelsizes`` (``R/force_panelsize.R:51-85``).
    ``rows`` / ``cols`` set panel heights / widths; bare numerics become relative
    ``"null"`` units (ratios), recycled / shortened to the number of panel rows /
    columns.  ``total_width`` / ``total_height`` set the absolute total of all
    panels plus the decoration between them, and require the corresponding
    ``rows`` / ``cols`` to be relative (numeric or ``"null"`` units).

    Parameters
    ----------
    rows, cols : numeric or grid_py.Unit or None, default None
        Panel heights (rows) / widths (cols).  ``None`` leaves that direction
        unchanged.
    respect : bool or None, default None
        When ``True``, ``"null"`` widths and heights are proportional.  ``None``
        inherits the behaviour specified elsewhere.
    total_width, total_height : grid_py.Unit or None, default None
        Absolute total width / height (length-1 unit) of all panels plus the
        decoration between panels.

    Returns
    -------
    ForcedSize
        An add-on object that can be added to a plot with ``+``.

    Raises
    ------
    ValueError
        When ``total_width`` is set but ``cols`` is a non-relative unit (or vice
        versa for ``total_height`` / ``rows``), or when a ``total_*`` argument is
        not an absolute unit of the allowed types.
    """
    if rows is not None and not is_unit(rows):
        rows = Unit(rows, "null")
    if cols is not None and not is_unit(cols):
        cols = Unit(cols, "null")

    if total_width is not None:
        if is_unit(cols) and not is_null_unit(cols):
            raise ValueError(
                "Cannot set `total_width` when `cols` is not relative."
            )
        if not is_unit(total_width):
            raise ValueError("`total_width` must be a unit object.")
        _arg_match_unit(total_width, "total_width")
    if total_height is not None:
        if is_unit(rows) and not is_null_unit(rows):
            raise ValueError(
                "Cannot set `total_height` when `rows` is not relative."
            )
        if not is_unit(total_height):
            raise ValueError("`total_height` must be a unit object.")
        _arg_match_unit(total_height, "total_height")

    return ForcedSize(
        rows=rows,
        cols=cols,
        respect=respect,
        total_width=total_width,
        total_height=total_height,
    )


def _arg_match_unit(u: Unit, nm: str) -> None:
    """Validate that *u*'s unit type is one of the allowed absolute units.

    Mirrors R's ``arg_match0(unitType(total_width), c("cm","mm",...))``.
    """
    t = unit_type(u)
    if isinstance(t, (list, tuple)):
        t = t[0] if t else ""
    if t not in _ABSOLUTE_TOTAL_UNITS:
        allowed = ", ".join(_ABSOLUTE_TOTAL_UNITS)
        raise ValueError(
            f"`{nm}` unit type {t!r} must be one of: {allowed}."
        )


# ---------------------------------------------------------------------------
# draw_panels override builder
# ---------------------------------------------------------------------------
def _seq_range_int(values: List[int]) -> List[int]:
    """Return the contiguous integer range ``min:max`` over *values* (R ``seq_range``)."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    return list(range(lo, hi + 1))


def _make_forced_draw_panels(parent_cls: Any) -> Any:
    """Build the ``draw_panels`` override for a ``Forced<FacetClass>`` clone.

    Closes over the *concrete parent class* so the override can dispatch to the
    original facet's ``draw_panels`` via :func:`ggproto_parent`.  Faithful port
    of R's ``new.fun`` (``R/force_panelsize.R:107-179``).

    Parameters
    ----------
    parent_cls : type
        The concrete facet class the forced facet was cloned from.

    Returns
    -------
    callable
        A ``draw_panels(self, ...)`` method.
    """

    def draw_panels(
        self: Any,
        panels: list,
        layout: Any,
        x_scales: list,
        y_scales: list,
        ranges: list,
        coord: Any,
        data: Any,
        theme: Any,
        params: Dict[str, Any],
    ) -> Any:
        # Call the original facet's draw_panels to build the panel gtable.
        panel_table = ggproto_parent(parent_cls, self).draw_panels(
            panels, layout, x_scales, y_scales, ranges, coord, data, theme, params
        )

        force = self.params
        prows = panel_rows(panel_table)
        pcols = panel_cols(panel_table)
        t_pos = [int(v) for v in prows["t"]]
        l_pos = [int(v) for v in pcols["l"]]
        n_rows = len(t_pos)
        n_cols = len(l_pos)
        # seq_range over ALL pcols / prows values (l & r, t & b) per R's
        # `seq_range(pcols)` / `seq_range(prows)`.
        all_col = [int(v) for v in pcols["l"]] + [int(v) for v in pcols["r"]]
        all_row = [int(v) for v in prows["t"]] + [int(v) for v in prows["b"]]

        force_rows = force.get("force_rows")
        force_cols = force.get("force_cols")
        force_respect = force.get("force_respect")
        total_width = force.get("force_total_width")
        total_height = force.get("force_total_height")

        # --- total_width branch (R:122-141) -------------------------------
        if total_width is not None:
            if force_cols is None:
                colwidths = [
                    float(panel_table.widths[p - 1]._values[0])
                    for p in l_pos
                ]
            else:
                colwidths = [
                    float(_recycle(force_cols, i)) for i in range(n_cols)
                ]
            # extra_width = columns between panel cells (the decoration):
            # setdiff(seq_range(pcols), unique(unlist(pcols))).
            uniq_cols = set(all_col)
            extra_idx = [i for i in _seq_range_int(all_col) if i not in uniq_cols]
            if len(extra_idx) > 1:
                extra_width = sum(
                    _to_scalar(convert_width(panel_table.widths[i - 1], "cm", valueOnly=True))
                    for i in extra_idx
                )
            else:
                extra_width = 0.0
            tw = _to_scalar(convert_width(total_width, "cm", valueOnly=True))
            avail = tw - extra_width
            denom = sum(colwidths)
            new_widths = [avail * c / denom for c in colwidths]
            for pos, w in zip(l_pos, new_widths):
                panel_table.widths[pos - 1] = Unit(w, "cm")
            force_cols = None

        # --- total_height branch (R:143-161) ------------------------------
        if total_height is not None:
            if force_rows is None:
                rowheights = [
                    float(panel_table.heights[p - 1]._values[0])
                    for p in t_pos
                ]
            else:
                rowheights = [
                    float(_recycle(force_rows, i)) for i in range(n_rows)
                ]
            uniq_rows = set(all_row)
            extra_idx = [i for i in _seq_range_int(all_row) if i not in uniq_rows]
            if len(extra_idx) > 1:
                extra_height = sum(
                    _to_scalar(convert_height(panel_table.heights[i - 1], "cm", valueOnly=True))
                    for i in extra_idx
                )
            else:
                extra_height = 0.0
            th = _to_scalar(convert_height(total_height, "cm", valueOnly=True))
            avail = th - extra_height
            denom = sum(rowheights)
            new_heights = [avail * r / denom for r in rowheights]
            for pos, h in zip(t_pos, new_heights):
                panel_table.heights[pos - 1] = Unit(h, "cm")
            force_rows = None

        # --- plain override (R:163-176) -----------------------------------
        if force_rows is not None:
            for i, pos in enumerate(t_pos):
                panel_table.heights[pos - 1] = _recycle_unit(force_rows, i)
        if force_cols is not None:
            for i, pos in enumerate(l_pos):
                panel_table.widths[pos - 1] = _recycle_unit(force_cols, i)
        if force_respect is not None:
            panel_table.respect = force_respect

        return panel_table

    return draw_panels


def _to_scalar(x: Any) -> float:
    """Coerce a ``convert_*`` result (scalar or length-1 array) to a float."""
    import numpy as np

    arr = np.asarray(x).ravel()
    return float(arr[0]) if arr.size else 0.0


def _recycle(u: Unit, i: int) -> float:
    """Return the numeric value of element ``i mod len`` of unit *u*."""
    return float(u._values[i % len(u)])


def _recycle_unit(u: Unit, i: int) -> Unit:
    """Return element ``i mod len`` of unit *u* as a length-1 Unit (R ``rep(..., length.out)``)."""
    return u[i % len(u)]


# ---------------------------------------------------------------------------
# ggplot_add.forcedsize (R force_panelsize.R:94-195)
# ---------------------------------------------------------------------------
@update_ggplot.register(ForcedSize)
def _update_forcedsize(obj: ForcedSize, plot: Any, object_name: str = "") -> Any:
    """Add a :class:`ForcedSize` to *plot* (R ``ggplot_add.forcedsize``).

    Clones the plot's current facet into a dynamic ``Forced<FacetClass>``
    subclass whose ``draw_panels`` mutates the panel gtable's panel-row heights /
    panel-column widths to the forced sizes.  The forced parameters are merged
    into a *copied* params dict under ``force_rows`` / ``force_cols`` /
    ``force_respect`` / ``force_total_width`` / ``force_total_height``.

    Parameters
    ----------
    obj : ForcedSize
        The container produced by :func:`force_panelsizes`.
    plot : ggplot2_py.plot.GGPlot
        The plot to mutate.
    object_name : str, optional
        Unused (kept for the ``update_ggplot`` dispatch signature).

    Returns
    -------
    ggplot2_py.plot.GGPlot
        The mutated plot (returned unchanged when *obj* carries no sizes).
    """
    if obj._lengths_sum() < 1:
        return plot

    old_facet = plot.facet
    parent_cls = type(old_facet)

    # Merge force params into a copy of the facet's params (R:184).
    old_params = dict(old_facet.params) if old_facet.params else {}
    old_params["force_rows"] = obj.rows
    old_params["force_cols"] = obj.cols
    old_params["force_respect"] = obj.respect
    old_params["force_total_width"] = obj.total_width
    old_params["force_total_height"] = obj.total_height

    new_facet = ggproto(
        f"Forced{parent_cls.__name__}",
        old_facet,
        draw_panels=_make_forced_draw_panels(parent_cls),
        params=old_params,
    )

    plot.facet = new_facet
    return plot
