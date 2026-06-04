"""Extended wrapped facets (port of ggh4x ``R/facet_wrap2.R``).

``facet_wrap2`` behaves like :func:`ggplot2_py.facet_wrap` but adds inner-axis
drawing (``axes``), inner-axis label removal (``remove_labels``) and a literal
``trim_blank=False`` layout (``nrow`` / ``ncol`` taken verbatim).

The :class:`FacetWrap2` ggproto subclasses :class:`ggplot2_py.facet.FacetWrap` and
*fully replaces* ``draw_panels`` with a decomposed, strip-pluggable pipeline.  Its
``setup_axes`` is substantially larger than ``FacetGrid2``'s: it masks interior
axes, measures them after deletion, then re-places marginal axes bordering empty
cells so dangling panels keep their axis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ggplot2_py import calc_element, ggproto
from ggplot2_py.coord import CoordFlip
from ggplot2_py.facet import FacetWrap, _resolve_facet_vars, facet_wrap
from grid_py import GList, GTree, Unit, null_grob
from gtable_py import (
    Gtable,
    gtable_add_col_space,
    gtable_add_grob,
    gtable_add_row_space,
)

from ggh4x._borrowed_ggplot2 import is_zero, snake_class
from ggh4x._cli import cli_abort, cli_warn
from ggh4x._facet_helpers import AspectRatio, _match_facet_arg
from ggh4x._facet_utils import render_axes, weave_tables_col, weave_tables_row
from ggh4x.facet_grid2 import _decorate_panels, _measure_axes, purge_guide_labels
from ggh4x.strip_vanilla import resolve_strip

__all__ = [
    "facet_wrap2",
    "FacetWrap2",
    "new_wrap_facets",
    "purge_guide_labels",
    "_measure_axes",
]


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------
def facet_wrap2(
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
    trim_blank: bool = True,
    strip: Any = "vanilla",
) -> "FacetWrap2":
    """Extended wrapped facets.

    Port of ggh4x's ``facet_wrap2()`` (``R/facet_wrap2.R:67-86``).  Like
    :func:`ggplot2_py.facet_wrap` but can draw / label-purge inner axes when
    scales are fixed, and can honour a literal ``nrow``/``ncol`` via
    ``trim_blank=False``.

    Parameters
    ----------
    facets : formula / list / dict
        Faceting variables.
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
    trim_blank : bool, default True
        When ``False``, ``nrow``/``ncol`` are taken literally.
    strip : Strip or callable or str, default "vanilla"

    Returns
    -------
    FacetWrap2
    """
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
        super_=FacetWrap2,
    )


def new_wrap_facets(
    facets: Any,
    nrow: Optional[int],
    ncol: Optional[int],
    scales: Any,
    axes: Any,
    rmlab: Any,
    shrink: bool,
    labeller: Any,
    as_table: bool,
    drop: bool,
    dir: str,
    strip_position: str,
    strip: Any,
    trim_blank: bool,
    params: Optional[Dict[str, Any]] = None,
    super_: Any = None,
) -> "FacetWrap2":
    """Build a :class:`FacetWrap2` instance from raw arguments.

    Port of ggh4x's ``new_wrap_facets()`` (``R/facet_wrap2.R:90-120``).  Obtains
    the full prototype param dict from :func:`ggplot2_py.facet_wrap`, normalises
    ``axes`` / ``remove_labels``, resolves the strip, computes the ``dim`` for the
    non-trimmed case and assembles the params.

    Parameters
    ----------
    facets : Any
    nrow, ncol : int or None
    scales, axes, rmlab : Any
    shrink : bool
    labeller : Any
    as_table : bool
    drop : bool
    dir : str
    strip_position : str
    strip : Any
    trim_blank : bool
    params : dict, optional
    super_ : type, optional

    Returns
    -------
    FacetWrap2
    """
    if super_ is None:
        super_ = FacetWrap2
    params = dict(params or {})

    prototype = facet_wrap(
        facets=facets,
        nrow=nrow,
        ncol=ncol,
        scales=scales,
        shrink=shrink,
        labeller=labeller,
        as_table=as_table,
        drop=drop,
        dir=dir,
        strip_position=strip_position,
    ).params

    axes = _match_facet_arg(axes, ["margins", "x", "y", "all"], nm="axes")
    rmlab = _match_facet_arg(rmlab, ["none", "x", "y", "all"], nm="remove_labels")
    strip = resolve_strip(strip)

    if trim_blank:
        dim = None
    else:
        dim = [
            prototype.get("nrow") if prototype.get("nrow") is not None else np.nan,
            prototype.get("ncol") if prototype.get("ncol") is not None else np.nan,
        ]

    merged = dict(prototype)
    merged.update(params)
    # Store ``facets`` as a name list (R keeps a named quosure list; the strip /
    # layout read the names) so the strip subsystem agrees with the layout.
    merged["facets"] = _resolve_facet_vars(prototype.get("facets"))
    merged.update({"dim": dim, "axes": axes, "rmlab": rmlab})
    # Ensure a strip.position alias exists for the R-style param name used in
    # setup_axes warnings / strip incorporation.
    merged.setdefault("strip.position", merged.get("strip_position", strip_position))

    obj = super_()
    obj._set(shrink=shrink, strip=strip, params=merged)
    return obj


# ---------------------------------------------------------------------------
# ggproto
# ---------------------------------------------------------------------------
class FacetWrap2(FacetWrap):
    """Extended wrapped facet ggproto (port of R ``FacetWrap2``).

    Subclasses :class:`ggplot2_py.facet.FacetWrap`.  Replaces ``draw_panels`` with
    a decomposed, strip-pluggable pipeline; all sub-steps are overridable.

    Attributes
    ----------
    shrink : bool
    strip : Strip
    params : dict
    """

    _class_name = "FacetWrap2"

    shrink: bool = True
    strip: Any = None

    # -- setup_aspect_ratio (identical to FacetGrid2) -----------------------
    def setup_aspect_ratio(
        self,
        coord: Any,
        free: Dict[str, bool],
        theme: Any,
        ranges: Sequence[Any],
    ) -> AspectRatio:
        """Resolve the panel aspect ratio + ``respect`` flag.

        Port of ggh4x's ``FacetWrap2$setup_aspect_ratio``
        (``R/facet_wrap2.R:134-148``); identical to ``FacetGrid2``'s.

        Parameters
        ----------
        coord : Coord
        free : dict
        theme : Theme
        ranges : sequence

        Returns
        -------
        AspectRatio
        """
        aspect_ratio = calc_element("aspect.ratio", theme) if theme is not None else None
        if aspect_ratio is None and not free["x"] and not free["y"] and ranges:
            aspect_ratio = coord.aspect(ranges[0])
        if aspect_ratio is None:
            return AspectRatio(1.0, False)
        return AspectRatio(float(aspect_ratio), True)

    # -- setup_layout (CoordFlip scale swap) --------------------------------
    def setup_layout(
        self,
        layout: pd.DataFrame,
        coord: Any,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """Swap ``SCALE_X`` / ``SCALE_Y`` assignment under ``CoordFlip``.

        Port of ggh4x's ``FacetWrap2$setup_layout`` (``R/facet_wrap2.R:149-166``).
        A no-op for non-flipped coords; under :class:`ggplot2_py.coord.CoordFlip`
        each free dimension's ``SCALE_*`` becomes per-panel (``seq_len(nrow)``).

        Parameters
        ----------
        layout : pandas.DataFrame
        coord : Coord
        params : dict

        Returns
        -------
        pandas.DataFrame
        """
        if isinstance(coord, CoordFlip):
            layout = layout.copy()
            n = len(layout)
            layout["SCALE_X"] = np.arange(1, n + 1) if params["free"]["x"] else 1
            layout["SCALE_Y"] = np.arange(1, n + 1) if params["free"]["y"] else 1
        return layout

    # -- setup_panel_table (self-first R signature) -------------------------
    def setup_panel_table(
        self,
        panels: List[Any],
        layout: pd.DataFrame,
        theme: Any,
        coord: Any,
        ranges: Sequence[Any],
        params: Dict[str, Any],
    ) -> Gtable:
        """Build the wrap panel gtable (panels may span multiple cells).

        Port of ggh4x's ``FacetWrap2$setup_panel_table``
        (``R/facet_wrap2.R:167-196``).  ``ncol``/``nrow`` come from ``params.dim``
        or the spanning ``.LEFT``/``.RIGHT``/``.TOP``/``.BOTTOM`` columns; panels
        are placed at ``t=.TOP, b=.BOTTOM, l=.LEFT, r=.RIGHT`` (``z=1``) so a
        single panel can span empty cells.

        Parameters
        ----------
        panels : list of grob
            One decorated panel grob per PANEL.
        layout : pandas.DataFrame
            Carries ``.TOP``/``.BOTTOM``/``.LEFT``/``.RIGHT``.
        theme : Theme
        coord : Coord
        ranges : sequence
        params : dict

        Returns
        -------
        Gtable
        """
        dim = params.get("dim")
        if dim is not None and not _is_na(dim[1]):
            ncol = int(dim[1])
        else:
            ncol = int(max(layout[".LEFT"].max(), layout[".RIGHT"].max()))
        if dim is not None and not _is_na(dim[0]):
            nrow = int(dim[0])
        else:
            nrow = int(max(layout[".TOP"].max(), layout[".BOTTOM"].max()))

        aspect = self.setup_aspect_ratio(coord, params["free"], theme, ranges)

        respect = params.get("respect")
        if respect is None:
            respect = aspect.respect
        widths = params.get("widths")
        if widths is None:
            widths = Unit(1, "null")
        heights = params.get("heights")
        if heights is None:
            heights = Unit(abs(aspect.value), "null")

        panel_table = Gtable(
            widths=_rep_unit(widths, ncol),
            heights=_rep_unit(heights, nrow),
            respect=respect,
        )

        t = [int(v) for v in layout[".TOP"]]
        b = [int(v) for v in layout[".BOTTOM"]]
        l = [int(v) for v in layout[".LEFT"]]
        r = [int(v) for v in layout[".RIGHT"]]
        names = [f"panel-{i + 1}" for i in range(len(panels))]
        panel_table = gtable_add_grob(
            panel_table,
            list(panels),
            t=t,
            b=b,
            l=l,
            r=r,
            z=1,
            clip=coord.clip,
            name=names,
        )
        panel_table = gtable_add_col_space(
            panel_table, calc_element("panel.spacing.x", theme)
        )
        panel_table = gtable_add_row_space(
            panel_table, calc_element("panel.spacing.y", theme)
        )
        return panel_table

    # -- setup_axes (the large one) -----------------------------------------
    def setup_axes(
        self,
        axes: Dict[str, Any],
        layout: pd.DataFrame,
        params: Dict[str, Any],
        theme: Any,
    ) -> Dict[str, Any]:
        """Fill / blank / measure / re-place the axis matrices for wrap facets.

        Port of ggh4x's ``FacetWrap2$setup_axes`` (``R/facet_wrap2.R:197-335``).
        Fills the four ``nrow x ncol`` grob matrices by ``SCALE_X`` / ``SCALE_Y``
        at ``cbind(ROW, COL)``, blanks interior axes unless they repeat
        (``free`` | ``axes``), purges labels when requested, **measures the axes
        after deletion** (so dangling-panel gaps are not over-sized), then
        re-places marginal axes bordering empty cells using ``diff()`` of the
        ``empties`` mask.

        Parameters
        ----------
        axes : dict
            Transposed batch axes ``{"x": {"top", "bottom"},
            "y": {"left", "right"}}`` from :func:`render_axes` (indexed by
            ``SCALE_*``).
        layout : pandas.DataFrame
        params : dict
        theme : Theme

        Returns
        -------
        dict
            ``{"top", "bottom", "left", "right", "measurements"}``.
        """
        nrow = int(layout["ROW"].max())
        ncol = int(layout["COL"].max())

        rows = [int(v) for v in layout["ROW"]]
        cols = [int(v) for v in layout["COL"]]
        scale_x = [int(v) for v in layout["SCALE_X"]]
        scale_y = [int(v) for v in layout["SCALE_Y"]]

        # empties[r, c] = TRUE unless a panel occupies (r, c).
        empties = np.ones((nrow, ncol), dtype=bool)
        for rr, cc in zip(rows, cols):
            empties[rr - 1, cc - 1] = False

        def _new() -> List[List[Any]]:
            return [[null_grob() for _ in range(ncol)] for _ in range(nrow)]

        top = _new()
        bottom = _new()
        left = _new()
        right = _new()

        x_top = axes["x"]["top"]
        x_bottom = axes["x"]["bottom"]
        y_left = axes["y"]["left"]
        y_right = axes["y"]["right"]

        # Fill by SCALE id at each panel's (ROW, COL).
        for i in range(len(rows)):
            r0 = rows[i] - 1
            c0 = cols[i] - 1
            top[r0][c0] = x_top[scale_x[i] - 1]
            bottom[r0][c0] = x_bottom[scale_x[i] - 1]
            left[r0][c0] = y_left[scale_y[i] - 1]
            right[r0][c0] = y_right[scale_y[i] - 1]

        repeat_x = params["free"]["x"] or params["axes"]["x"]
        repeat_y = params["free"]["y"] or params["axes"]["y"]

        if not repeat_x:
            for r in range(1, nrow):  # top[-1, ]
                for c in range(ncol):
                    top[r][c] = null_grob()
            for r in range(nrow - 1):  # bottom[-nrow, ]
                for c in range(ncol):
                    bottom[r][c] = null_grob()
        if not repeat_y:
            for r in range(nrow):  # left[, -1]
                for c in range(1, ncol):
                    left[r][c] = null_grob()
            for r in range(nrow):  # right[, -ncol]
                for c in range(ncol - 1):
                    right[r][c] = null_grob()

        if params["axes"]["x"] and params["rmlab"]["x"] and not params["free"]["x"]:
            for r in range(1, nrow):
                for c in range(ncol):
                    top[r][c] = purge_guide_labels(top[r][c])
            for r in range(nrow - 1):
                for c in range(ncol):
                    bottom[r][c] = purge_guide_labels(bottom[r][c])
        if params["axes"]["y"] and params["rmlab"]["y"] and not params["free"]["y"]:
            for r in range(nrow):
                for c in range(1, ncol):
                    left[r][c] = purge_guide_labels(left[r][c])
            for r in range(nrow):
                for c in range(ncol - 1):
                    right[r][c] = purge_guide_labels(right[r][c])

        # Measure AFTER deletion, BEFORE re-placement.
        measurements = _measure_axes(
            {"top": top, "bottom": bottom, "left": left, "right": right}
        )

        if not empties.any():
            return {
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
                "measurements": measurements,
            }

        inside = (
            (calc_element("strip.placement", theme) if theme is not None else None)
            or "inside"
        ) == "inside"
        strip_pos = params.get("strip.position", params.get("strip_position", "top"))

        rc_to_panel = {(rows[i], cols[i]): i for i in range(len(rows))}

        # bottom_empty: per column, c(diff(empties)==1, FALSE) — a panel whose
        # cell below is empty (transition non-empty -> empty going down).
        bottom_empty = _diff_down(empties, target=1, append_last=False)
        if bottom_empty.any():
            replace_warn = _gather_replace(x_bottom, scale=None, empties_pos=bottom_empty,
                                           rc_to_panel=rc_to_panel, scale_ids=scale_x,
                                           rows=rows, cols=cols, side_list=x_bottom)
            if (
                strip_pos == "bottom"
                and not inside
                and any(not is_zero(g) for g in replace_warn)
                and not params["free"]["x"]
            ):
                cli_warn(
                    'Suppressing axis rendering when `strip.position = "bottom"` '
                    'and `strip.placement == "outside"`'
                )
            else:
                _place_back(bottom, bottom_empty, x_bottom, scale_x, rc_to_panel,
                            rows, cols)

        # top_empty: per column, c(FALSE, diff(empties)==-1) — a panel whose cell
        # above is empty.
        top_empty = _diff_down(empties, target=-1, append_last=True)
        if top_empty.any():
            replace_warn = _gather_replace(x_top, scale=None, empties_pos=top_empty,
                                           rc_to_panel=rc_to_panel, scale_ids=scale_x,
                                           rows=rows, cols=cols, side_list=x_top)
            if (
                strip_pos == "top"
                and not inside
                and any(not is_zero(g) for g in replace_warn)
                and not params["free"]["x"]
            ):
                cli_warn(
                    'Suppressing axis rendering when `strip.position = "top"` '
                    'and `strip.placement == "outside"`'
                )
            else:
                _place_back(top, top_empty, x_top, scale_x, rc_to_panel, rows, cols)

        # right_empty: per row, c(diff(empties)==1, FALSE).
        right_empty = _diff_right(empties, target=1, append_last=False)
        if right_empty.any():
            replace_warn = _gather_replace(y_right, scale=None, empties_pos=right_empty,
                                           rc_to_panel=rc_to_panel, scale_ids=scale_y,
                                           rows=rows, cols=cols, side_list=y_right)
            if (
                strip_pos == "right"
                and not inside
                and any(not is_zero(g) for g in replace_warn)
                and not params["free"]["y"]
            ):
                cli_warn(
                    'Suppressing axis rendering when `strip.position = "right"` '
                    'and `strip.placement == "outside"`'
                )
            # R places back unconditionally for right/left (no else guard).
            _place_back(right, right_empty, y_right, scale_y, rc_to_panel, rows, cols)

        # left_empty: per row, c(FALSE, diff(empties)==-1).
        left_empty = _diff_right(empties, target=-1, append_last=True)
        if left_empty.any():
            replace_warn = _gather_replace(y_left, scale=None, empties_pos=left_empty,
                                           rc_to_panel=rc_to_panel, scale_ids=scale_y,
                                           rows=rows, cols=cols, side_list=y_left)
            if (
                strip_pos == "left"
                and not inside
                and any(not is_zero(g) for g in replace_warn)
                and not params["free"]["y"]
            ):
                cli_warn(
                    'Suppressing axis rendering when `strip.position = "left"` '
                    'and `strip.placement == "outside"`'
                )
            _place_back(left, left_empty, y_left, scale_y, rc_to_panel, rows, cols)

        return {
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
            "measurements": measurements,
        }

    # -- attach_axes (explicit sizes) ---------------------------------------
    def attach_axes(
        self,
        panel_table: Gtable,
        axes: Dict[str, Any],
        sizes: Dict[str, Unit],
    ) -> Gtable:
        """Weave the axis bands using pre-computed sizes.

        Port of ggh4x's ``FacetWrap2$attach_axes`` (``R/facet_wrap2.R:336-346``).
        Differs from ``FacetGrid2`` by taking an explicit ``sizes`` argument (the
        post-deletion measurements) instead of measuring internally.

        Parameters
        ----------
        panel_table : Gtable
        axes : dict
            ``{"top", "bottom", "left", "right"}`` grob matrices.
        sizes : dict
            ``{"top", "bottom", "left", "right"}`` size unit vectors.

        Returns
        -------
        Gtable
        """
        panel_table = weave_tables_row(
            panel_table, axes["top"], -1, sizes["top"], "axis-t", 3
        )
        panel_table = weave_tables_row(
            panel_table, axes["bottom"], 0, sizes["bottom"], "axis-b", 3
        )
        panel_table = weave_tables_col(
            panel_table, axes["left"], -1, sizes["left"], "axis-l", 3
        )
        panel_table = weave_tables_col(
            panel_table, axes["right"], 0, sizes["right"], "axis-r", 3
        )
        return panel_table

    # -- finish_panels (identity seam) --------------------------------------
    def finish_panels(
        self,
        panels: Any,
        layout: pd.DataFrame,
        params: Dict[str, Any],
        theme: Any,
    ) -> Any:
        """Identity post-processing hook (extension seam).

        Port of ggh4x's ``FacetWrap2$finish_panels`` (``R/facet_wrap2.R:347-349``).
        """
        return panels

    # -- draw_panels (full override) ----------------------------------------
    def draw_panels(
        self,
        panels: list,
        layout: pd.DataFrame,
        x_scales: list,
        y_scales: list,
        ranges: list,
        coord: Any,
        data: Any,
        theme: Any,
        params: Dict[str, Any],
    ) -> Gtable:
        """Assemble the wrap panel gtable (full replacement of the base pipeline).

        Port of ggh4x's ``FacetWrap2$draw_panels`` (``R/facet_wrap2.R:350-391``).
        Runs ``setup_layout``; sets the spanning ``.TOP``/``.BOTTOM``/``.LEFT``/
        ``.RIGHT = ROW/COL``; resolves ``params.dim`` NA -> nrow/ncol; then
        ``setup_panel_table`` -> ``render_axes`` -> ``setup_axes`` ->
        ``attach_axes(measurements)`` -> strip ``setup`` / ``incorporate_wrap`` ->
        ``finish_panels``.

        Parameters
        ----------
        panels : list
        layout : pandas.DataFrame
        x_scales, y_scales : list
        ranges : list
        coord : Coord
        data : Any
        theme : Theme
        params : dict

        Returns
        -------
        Gtable
        """
        if (params["free"]["x"] or params["free"]["y"]) and not coord.is_free():
            cli_abort(f"`{snake_class(coord)}` doesn't support free scales.")

        strip = self.strip
        layout = self.setup_layout(layout, coord, params)

        ncol = int(layout["COL"].max())
        nrow = int(layout["ROW"].max())
        layout = layout.copy()
        layout[".TOP"] = layout["ROW"].to_numpy()
        layout[".BOTTOM"] = layout["ROW"].to_numpy()
        layout[".LEFT"] = layout["COL"].to_numpy()
        layout[".RIGHT"] = layout["COL"].to_numpy()

        params = dict(params)
        dim = params.get("dim")
        if dim is not None:
            dim = list(dim)
            if _is_na(dim[0]):
                dim[0] = nrow
            if _is_na(dim[1]):
                dim[1] = ncol
            params["dim"] = dim

        # Decorate per-layer grobs into one panel grob per PANEL.
        panel_grobs = _decorate_panels(panels, layout, ranges, coord, theme)

        panel_table = self.setup_panel_table(
            panel_grobs, layout, theme, coord, ranges, params
        )

        axes = render_axes(ranges, ranges, coord, theme, transpose=True)
        axes = self.setup_axes(axes, layout, params, theme)
        panel_table = self.attach_axes(panel_table, axes, axes["measurements"])

        strip.setup(layout, params, theme, type="wrap")
        panel_table = strip.incorporate_wrap(
            panel_table,
            params.get("strip.position", params.get("strip_position", "top")),
            clip=coord.clip,
            sizes=axes["measurements"],
        )

        return self.finish_panels(
            panels=panel_table, layout=layout, params=params, theme=theme
        )


# ---------------------------------------------------------------------------
# Module-private helpers (empties re-placement)
# ---------------------------------------------------------------------------
def _is_na(x: Any) -> bool:
    """Return ``True`` when *x* is ``None`` or NaN (R ``is.na``)."""
    if x is None:
        return True
    try:
        return bool(np.isnan(x))
    except (TypeError, ValueError):
        return False


def _rep_unit(u: Unit, length_out: int) -> Unit:
    """Recycle a (scalar) unit to ``length_out`` (R ``rep(u, length.out=n)``)."""
    from grid_py import unit_rep

    return unit_rep(u, length_out=length_out)


def _diff_down(empties: np.ndarray, target: int, append_last: bool) -> np.ndarray:
    """Port of ``apply(empties, 2, function(x) c(diff(x)==target, FALSE/...))``.

    Computes, per column, ``diff`` of the boolean mask (as ints) compared to
    *target*, then pads.  When ``append_last`` is ``False`` the result is
    ``c(diff==target, FALSE)`` (length nrow, last row FALSE); when ``True`` it is
    ``c(FALSE, diff==target)`` (first row FALSE).

    Parameters
    ----------
    empties : numpy.ndarray
        ``nrow x ncol`` boolean mask.
    target : int
        ``1`` (non-empty -> empty going down) or ``-1`` (empty -> non-empty).
    append_last : bool
        ``False`` -> pad FALSE at the bottom; ``True`` -> pad FALSE at the top.

    Returns
    -------
    numpy.ndarray
        ``nrow x ncol`` boolean mask of panels bordering a vertical hole.
    """
    nrow, ncol = empties.shape
    ints = empties.astype(int)
    out = np.zeros((nrow, ncol), dtype=bool)
    if nrow < 2:
        return out
    d = (ints[1:, :] - ints[:-1, :]) == target  # (nrow-1, ncol)
    if append_last:
        out[1:, :] = d
    else:
        out[:-1, :] = d
    return out


def _diff_right(empties: np.ndarray, target: int, append_last: bool) -> np.ndarray:
    """Port of ``t(apply(empties, 1, function(x) c(diff(x)==target, ...)))``.

    Row-wise counterpart of :func:`_diff_down`: detects panels bordering a
    horizontal hole.  ``append_last=False`` -> ``c(diff==target, FALSE)`` per row
    (last col FALSE); ``append_last=True`` -> ``c(FALSE, diff==target)`` (first
    col FALSE).

    Parameters
    ----------
    empties : numpy.ndarray
    target : int
        ``1`` (right neighbour empty) or ``-1`` (left neighbour empty).
    append_last : bool

    Returns
    -------
    numpy.ndarray
    """
    nrow, ncol = empties.shape
    ints = empties.astype(int)
    out = np.zeros((nrow, ncol), dtype=bool)
    if ncol < 2:
        return out
    d = (ints[:, 1:] - ints[:, :-1]) == target  # (nrow, ncol-1)
    if append_last:
        out[:, 1:] = d
    else:
        out[:, :-1] = d
    return out


def _gather_replace(
    side_axes: Sequence[Any],
    scale: Any,
    empties_pos: np.ndarray,
    rc_to_panel: Dict[tuple, int],
    scale_ids: Sequence[int],
    rows: Sequence[int],
    cols: Sequence[int],
    side_list: Sequence[Any],
) -> List[Any]:
    """Collect the would-be replacement axis grobs for the warning check.

    Mirrors R's ``replace <- axes$x$bottom[panels]`` where ``panels`` are the
    panel indices at the hole-bordering positions.  Returns the rendered axis
    grobs (by ``SCALE_*``) for each flagged ``(row, col)`` so the caller can test
    whether any is non-zero (the strip-placement="outside" suppression warning).

    Parameters
    ----------
    side_axes, side_list : sequence
        The rendered per-scale axis list (e.g. ``axes$x$bottom``).
    scale : Any
        Unused (kept for signature clarity).
    empties_pos : numpy.ndarray
        Boolean mask of flagged positions.
    rc_to_panel : dict
        ``(row, col) -> panel-layout-index`` (0-based).
    scale_ids : sequence of int
        Per-panel ``SCALE_X`` / ``SCALE_Y`` (1-based).
    rows, cols : sequence of int
        Per-panel ``ROW`` / ``COL`` (1-based).

    Returns
    -------
    list of grob
    """
    out: List[Any] = []
    nrow, ncol = empties_pos.shape
    for r in range(nrow):
        for c in range(ncol):
            if not empties_pos[r, c]:
                continue
            panel_idx = rc_to_panel.get((r + 1, c + 1))
            if panel_idx is None:
                out.append(null_grob())
                continue
            sid = scale_ids[panel_idx]
            out.append(side_list[sid - 1])
    return out


def _place_back(
    matrix: List[List[Any]],
    empties_pos: np.ndarray,
    side_list: Sequence[Any],
    scale_ids: Sequence[int],
    rc_to_panel: Dict[tuple, int],
    rows: Sequence[int],
    cols: Sequence[int],
) -> None:
    """Re-insert marginal axes at hole-bordering panels (in place).

    Mirrors R's ``bottom[pos] <- axes$x$bottom[panels]`` block: at each flagged
    position ``(r, c)`` look up the panel that lives there, and write its
    ``SCALE``-indexed rendered axis back into the matrix so the dangling panel
    keeps its marginal axis.

    Parameters
    ----------
    matrix : list of list
        The side grob matrix being mutated.
    empties_pos : numpy.ndarray
        Boolean mask of positions to re-place.
    side_list : sequence
        The rendered per-scale axis list.
    scale_ids : sequence of int
        Per-panel ``SCALE_*`` (1-based).
    rc_to_panel : dict
        ``(row, col) -> panel-layout-index`` (0-based).
    rows, cols : sequence of int
        Per-panel ``ROW`` / ``COL`` (unused; kept for parity).
    """
    nrow, ncol = empties_pos.shape
    for r in range(nrow):
        for c in range(ncol):
            if not empties_pos[r, c]:
                continue
            panel_idx = rc_to_panel.get((r + 1, c + 1))
            if panel_idx is None:
                continue
            sid = scale_ids[panel_idx]
            matrix[r][c] = side_list[sid - 1]
