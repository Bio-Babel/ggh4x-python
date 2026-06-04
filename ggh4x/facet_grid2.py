"""Extended grid facets (port of ggh4x ``R/facet_grid2.R``).

``facet_grid2`` behaves like :func:`ggplot2_py.facet_grid` but adds three things:

* axes may be drawn (and optionally label-purged) at inner panels (``axes`` /
  ``remove_labels``),
* position scales may be *independent* within a row or column (``independent``),
* empty panels may be rendered as blanks (``render_empty``).

The :class:`FacetGrid2` ggproto subclasses :class:`ggplot2_py.facet.FacetGrid` and
*fully replaces* its ``draw_panels`` with a decomposed, strip-pluggable pipeline
(the explicit reason ggh4x exists -- to give ``facet_nested`` clean override
seams).  Every step (``setup_axes``, ``setup_aspect_ratio``, ``setup_panel_table``,
``attach_axes``, ``finish_panels``) is an overridable method.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ggplot2_py import calc_element, ggproto
from ggplot2_py.facet import FacetGrid, _combine_vars, _resolve_facet_vars, facet_grid
from grid_py import GList, GTree, Unit, edit_viewport, null_grob
from gtable_py import (
    Gtable,
    gtable_add_col_space,
    gtable_add_grob,
    gtable_add_row_space,
    gtable_filter,
    gtable_height,
    gtable_trim,
    gtable_width,
)

from ggh4x._borrowed_ggplot2 import id, is_zero, snake_class, ulevels
from ggh4x._cli import cli_abort
from ggh4x._facet_helpers import (
    AspectRatio,
    _match_facet_arg,
    _validate_independent,
    reshape_add_margins,
)
from ggh4x._facet_utils import (
    df_grid,
    render_axes,
    weave_tables_col,
    weave_tables_row,
)
from ggh4x._rlang import arg_match0
from ggh4x.strip_vanilla import resolve_strip

__all__ = [
    "facet_grid2",
    "FacetGrid2",
    "new_grid_facets",
    "purge_guide_labels",
    "_measure_axes",
]


# ---------------------------------------------------------------------------
# Shared axis helpers (R: facet_wrap2.R purge_guide_labels / .measure_axes)
# ---------------------------------------------------------------------------
def purge_guide_labels(guide: Any) -> Any:
    """Strip the label grobs from an axis guide, keeping line + ticks.

    Faithful port of ggh4x's ``purge_guide_labels`` (``R/facet_wrap2.R:396-416``)
    adapted to the ``ggplot2_py`` axis grob layout.  R reaches into
    ``guide$children$axis$grobs`` and drops every grob that is a ``titleGrob`` /
    ``richtext_grob`` / ``zeroGrob`` (i.e. the labels), trims the inner axis
    gtable, then resets the guide's reported width/height.

    In ``ggplot2_py`` :func:`ggplot2_py._guide_axis.draw_axis` returns an
    ``_AbsoluteAxisGrob`` whose children are ``[axis_line, inner_gtable]`` and the
    inner gtable carries the ``"axis.labels"`` cells.  This port locates that
    inner gtable, deletes its ``"axis.labels"`` grobs via :func:`gtable_filter`,
    trims it, and refreshes the wrapper's measured width / height.

    Parameters
    ----------
    guide : grob
        An axis grob (``_AbsoluteAxisGrob``) or a zero grob.

    Returns
    -------
    grob
        The label-purged axis grob (zero grobs pass through unchanged).
    """
    if is_zero(guide):
        return guide

    children = list(guide.get_children()) if hasattr(guide, "get_children") else []
    inner_idx = None
    inner_gt = None
    for i, child in enumerate(children):
        if isinstance(child, Gtable):
            inner_idx = i
            inner_gt = child
            break
    if inner_gt is None:
        return guide

    # Drop the label cells, keep everything else (line, ticks).
    purged = gtable_filter(inner_gt, "axis.labels", trim=False, invert=True)
    purged = gtable_trim(purged)

    # Replace the inner gtable child in place.
    order = list(guide._children_order) if hasattr(guide, "_children_order") else None
    if order is not None and inner_idx < len(order):
        key = order[inner_idx]
        guide._children[key] = purged
    else:
        # Fallback: rebuild children list.
        new_children = list(children)
        new_children[inner_idx] = purged
        guide.set_children(GList(*new_children))

    # Reset reported width/height (R: guide$width/height <- sum(axis$...)).
    new_w = gtable_width(purged)
    new_h = gtable_height(purged)
    if hasattr(guide, "_abs_width"):
        guide._abs_width = new_w
        guide._abs_height = new_h
    vp = getattr(guide, "vp", None)
    if vp is not None:
        edits = {}
        if getattr(vp, "width", None) is not None:
            edits["width"] = new_w
        if getattr(vp, "height", None) is not None:
            edits["height"] = new_h
        if edits:
            # grid_py Viewport is immutable (read-only width/height); rebuild it
            # via edit_viewport rather than assigning in place.
            guide.vp = edit_viewport(vp, **edits)
    return guide


def _measure_axes(axes: Dict[str, List[List[Any]]]) -> Dict[str, Unit]:
    """Measure per-row / per-column axis bands in centimetres.

    Faithful port of ggh4x's ``.measure_axes`` (``R/facet_wrap2.R:435-441``).
    For the ``top`` / ``bottom`` grob matrices the maximum *height* (cm) is taken
    over each matrix **row**; for ``left`` / ``right`` the maximum *width* (cm)
    over each matrix **column**.

    Parameters
    ----------
    axes : dict
        ``{"top", "bottom", "left", "right"}`` -- each a row-major grob matrix
        (list of lists), one row per panel row and one column per panel column.

    Returns
    -------
    dict
        ``{"top", "bottom", "left", "right"}`` -- each a ``"cm"`` :class:`Unit`
        of per-band sizes (length = nrow for top/bottom, ncol for left/right).
    """
    from ggplot2_py.facet import _axis_height_cm, _axis_width_cm

    def _h(g: Any) -> float:
        return 0.0 if is_zero(g) else _axis_height_cm(g)

    def _w(g: Any) -> float:
        return 0.0 if is_zero(g) else _axis_width_cm(g)

    top_m = axes["top"]
    bottom_m = axes["bottom"]
    left_m = axes["left"]
    right_m = axes["right"]
    nrow = len(top_m)
    ncol = len(top_m[0]) if nrow else 0

    top = [max((_h(top_m[r][c]) for c in range(ncol)), default=0.0) for r in range(nrow)]
    bottom = [
        max((_h(bottom_m[r][c]) for c in range(ncol)), default=0.0) for r in range(nrow)
    ]
    left = [max((_w(left_m[r][c]) for r in range(nrow)), default=0.0) for c in range(ncol)]
    right = [
        max((_w(right_m[r][c]) for r in range(nrow)), default=0.0) for c in range(ncol)
    ]
    return {
        "top": Unit(top, "cm"),
        "bottom": Unit(bottom, "cm"),
        "left": Unit(left, "cm"),
        "right": Unit(right, "cm"),
    }


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------
def facet_grid2(
    rows: Any = None,
    cols: Any = None,
    scales: Any = "fixed",
    space: Any = "fixed",
    axes: Any = "margins",
    remove_labels: Any = "none",
    independent: Any = "none",
    shrink: bool = True,
    labeller: Any = "label_value",
    as_table: bool = True,
    switch: Optional[str] = None,
    drop: bool = True,
    margins: Any = False,
    render_empty: bool = True,
    strip: Any = "vanilla",
) -> "FacetGrid2":
    """Extended grid facets.

    Port of ggh4x's ``facet_grid2()`` (``R/facet_grid2.R:100-124``).  Like
    :func:`ggplot2_py.facet_grid` but can draw partial / full axis guides at inner
    panels and supports independent position scales.

    Parameters
    ----------
    rows, cols : formula / list / dict / None
        Faceting variables for rows and columns (same spec as ``facet_grid``).
    scales : {"fixed", "free_x", "free_y", "free"} or bool, default "fixed"
        Whether scales are shared or free across facets.
    space : {"fixed", "free_x", "free_y", "free"} or bool, default "fixed"
        Whether panel sizes are proportional to the scales.
    axes : {"margins", "x", "y", "all"} or bool, default "margins"
        Where inner axes are drawn.
    remove_labels : {"none", "x", "y", "all"} or bool, default "none"
        Whether inner-axis text is removed.
    independent : {"none", "x", "y", "all"} or bool, default "none"
        Whether scales vary within a row / column.
    shrink : bool, default True
        Shrink scales to fit stat output.
    labeller : callable or str, default "label_value"
        Strip labeller.
    as_table : bool, default True
        When ``False``, reverse the row factor level order.
    switch : {"x", "y", "both", None}, default None
        Which strips switch sides.
    drop : bool, default True
        Drop unused factor levels.
    margins : bool or list of str, default False
        Add marginal panels.
    render_empty : bool, default True
        Draw data-less panels (``True``) or blank them (``False``).
    strip : Strip or callable or str, default "vanilla"
        Strip specification.

    Returns
    -------
    FacetGrid2
        A ggproto facet object that can be added to a plot.
    """
    return new_grid_facets(
        rows,
        cols,
        scales,
        space,
        axes,
        remove_labels,
        independent,
        shrink,
        labeller,
        as_table,
        switch,
        drop,
        margins,
        render_empty,
        strip,
        super_=FacetGrid2,
    )


def new_grid_facets(
    rows: Any,
    cols: Any,
    scales: Any,
    space: Any,
    axes: Any,
    rmlab: Any,
    indy: Any,
    shrink: bool,
    labeller: Any,
    as_table: bool,
    switch: Optional[str],
    drop: bool,
    margins: Any,
    render_empty: bool,
    strip: Any,
    params: Optional[Dict[str, Any]] = None,
    super_: Any = None,
) -> "FacetGrid2":
    """Build a :class:`FacetGrid2` instance from raw arguments.

    Port of ggh4x's ``new_grid_facets()`` (``R/facet_grid2.R:128-171``).
    Normalises the option arguments, validates the ``independent`` interactions,
    resolves the formula spec via :func:`ggplot2_py.facet_grid`, resolves the
    strip, assembles the parameter dict and instantiates the ggproto.

    Parameters
    ----------
    rows, cols : Any
        Faceting variable specs.
    scales, space, axes, rmlab, indy : Any
        Option arguments (normalised via :func:`_match_facet_arg`).
    shrink : bool
    labeller : Any
    as_table : bool
    switch : str or None
    drop : bool
    margins : Any
    render_empty : bool
    strip : Any
    params : dict, optional
        Extra params merged in (used by ``facet_nested``).
    super_ : type, optional
        The ggproto class to instantiate (default :class:`FacetGrid2`).

    Returns
    -------
    FacetGrid2
    """
    if super_ is None:
        super_ = FacetGrid2
    params = dict(params or {})

    switch = switch if switch is not None else "none"
    switch = arg_match0(switch, ["none", "both", "x", "y"], arg_name="switch")
    switch_param = None if switch == "none" else switch

    axes = _match_facet_arg(axes, ["margins", "x", "y", "all"], nm="axes")
    free = _match_facet_arg(scales, ["fixed", "free_x", "free_y", "free"], nm="scales")
    space = _match_facet_arg(space, ["fixed", "free_x", "free_y", "free"], nm="space")
    rmlab = _match_facet_arg(rmlab, ["none", "x", "y", "all"], nm="remove_labels")
    indy = _match_facet_arg(indy, ["none", "x", "y", "all"], nm="independent")
    strip = resolve_strip(strip)

    axis_params = _validate_independent(indy, free, space, rmlab)

    # Resolve the formula -> rows / cols var lists via the base facet_grid.
    # Store as name lists (R keeps a named quosure list; the strip / layout read
    # the names) so both the strip subsystem and ``compute_layout`` agree.
    facets = facet_grid(rows=rows, cols=cols).params
    proto_rows = _resolve_facet_vars(facets["rows"])
    proto_cols = _resolve_facet_vars(facets["cols"])

    params.update(axis_params)
    params.update(
        {
            "rows": proto_rows,
            "cols": proto_cols,
            "margins": margins,
            "labeller": labeller,
            "as_table": as_table,
            "switch": switch_param,
            "drop": drop,
            "axes": axes,
            "render_empty": render_empty is not False,
        }
    )

    obj = super_()
    obj._set(shrink=shrink, strip=strip, params=params)
    return obj


# ---------------------------------------------------------------------------
# ggproto
# ---------------------------------------------------------------------------
def _as_name_list(spec: Any) -> List[str]:
    """Return the faceting-variable names from a ``rows``/``cols`` spec."""
    if spec is None:
        return []
    if isinstance(spec, dict):
        return list(spec.keys())
    if isinstance(spec, (list, tuple)):
        return [str(s) for s in spec]
    if isinstance(spec, str):
        return [spec]
    return []


class FacetGrid2(FacetGrid):
    """Extended grid facet ggproto (port of R ``FacetGrid2``).

    Subclasses :class:`ggplot2_py.facet.FacetGrid`.  Replaces ``compute_layout``
    (adds a ``_render`` column, ``id()``-stable ordering and independent-axis
    ``SCALE_X``/``SCALE_Y`` assignment) and ``draw_panels`` (a decomposed,
    strip-pluggable pipeline).  All drawing sub-steps are overridable methods so
    ``facet_nested`` can hook them.

    Attributes
    ----------
    shrink : bool
    strip : Strip
        The pluggable strip instance.
    params : dict
        Facet parameters (carries ``independent``, ``free``, ``space_free``,
        ``rmlab``, ``axes``, ``render_empty``, ``rows``, ``cols``, ...).
    """

    _class_name = "FacetGrid2"

    shrink: bool = True
    strip: Any = None

    # -- vars_combine (extension seam) --------------------------------------
    def vars_combine(
        self,
        data: List[pd.DataFrame],
        env: Any,
        vars_: Any,
        drop: bool = True,
    ) -> pd.DataFrame:
        """Combine the faceting variables across datasets.

        Extension seam mirroring ggh4x's ``FacetGrid2$vars_combine``
        (``R/facet_grid2.R:190-192``) which delegates to ggplot2's
        ``combine_vars``.  ``facet_nested`` overrides this to substitute its own
        variable-combination logic, so it must remain a real method called via
        ``self.vars_combine(...)``.

        Parameters
        ----------
        data : list of DataFrame
            The plot + layer data frames.
        env : Any
            The plot environment (unused -- kept for R signature parity).
        vars_ : dict or list of str
            The variable names (the ``rows`` / ``cols`` spec).
        drop : bool, default True
            Drop unused factor combinations.

        Returns
        -------
        pandas.DataFrame
            Unique combinations of the requested variables.
        """
        names = _as_name_list(vars_)
        return _combine_vars(data, names, drop=drop)

    # -- compute_layout -----------------------------------------------------
    def compute_layout(
        self,
        data: List[pd.DataFrame],
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """Build the panel layout with ``_render`` + independent-scale columns.

        Port of ggh4x's ``FacetGrid2$compute_layout`` (``R/facet_grid2.R:193-283``).

        Parameters
        ----------
        data : list of DataFrame
        params : dict

        Returns
        -------
        pandas.DataFrame
            Layout with ``PANEL``, ``ROW``, ``COL``, ``SCALE_X``, ``SCALE_Y``, a
            ggh4x-specific boolean ``_render`` column, and the faceting-var
            columns.
        """
        rows = params["rows"]
        cols = params["cols"]
        row_names = _as_name_list(rows)
        col_names = _as_name_list(cols)

        dups = [d for d in row_names if d in col_names]
        if dups:
            cli_abort(
                "Facetting variables can only appear in `rows` or `cols`, not "
                f"both. Duplicated variables: {dups}"
            )

        drop = params.get("drop", True)
        env = params.get("plot_env")

        base_rows = self.vars_combine(data, env, rows, drop=drop)
        if not params.get("as_table", True):
            base_rows = base_rows.copy()
            for c in base_rows.columns:
                levels = list(ulevels(base_rows[c]))[::-1]
                base_rows[c] = pd.Categorical(base_rows[c], categories=levels)

        base_cols = self.vars_combine(data, env, cols, drop=drop)
        base = df_grid(base_rows, base_cols)

        if base is None or len(base) == 0:
            return pd.DataFrame(
                {
                    "PANEL": pd.Categorical([1]),
                    "ROW": [1],
                    "COL": [1],
                    "SCALE_X": [1],
                    "SCALE_Y": [1],
                    "_render": [True],
                }
            )

        base = reshape_add_margins(
            base, [row_names, col_names], params.get("margins", False)
        )
        base = base.drop_duplicates().reset_index(drop=True)

        if not params.get("render_empty", True):
            both = {**(rows or {}), **(cols or {})} if isinstance(rows, dict) else None
            if both is not None:
                universe = self.vars_combine(data, env, both, drop=drop)
            else:
                universe = self.vars_combine(
                    data, env, row_names + col_names, drop=drop
                )
            keys = [c for c in base.columns if c in universe.columns]
            if keys:
                uni_rows = set(
                    tuple(r) for r in universe[keys].itertuples(index=False, name=None)
                )
                render = [
                    tuple(r) in uni_rows
                    for r in base[keys].itertuples(index=False, name=None)
                ]
            else:
                render = [True] * len(base)
        else:
            render = [True] * len(base)

        # PANEL / ROW / COL via id() (R radix ordering, NOT Categorical.codes).
        panel = id(base, drop=True)
        n_panel = int(panel.n)

        if not row_names:
            row_ids = np.array([1] * len(base), dtype=int)
        else:
            row_ids = np.asarray(id(base[row_names], drop=True), dtype=int)
        if not col_names:
            col_ids = np.array([1] * len(base), dtype=int)
        else:
            col_ids = np.asarray(id(base[col_names], drop=True), dtype=int)

        panel_int = np.asarray(panel, dtype=int)
        panels = base.copy()
        panels.insert(0, "PANEL", pd.Categorical(panel_int, categories=range(1, n_panel + 1)))
        panels.insert(1, "ROW", row_ids)
        panels.insert(2, "COL", col_ids)
        panels["_render"] = render

        # order(PANEL)
        order = np.argsort(panel_int, kind="mergesort")
        panels = panels.iloc[order].reset_index(drop=True)

        free = params["free"]
        independent = params["independent"]
        n = len(panels)

        if free["x"]:
            if independent["x"]:
                panels["SCALE_X"] = np.arange(1, n + 1)
            else:
                panels["SCALE_X"] = panels["COL"].to_numpy()
        else:
            panels["SCALE_X"] = 1

        if free["y"]:
            if independent["y"]:
                panels["SCALE_Y"] = np.arange(1, n + 1)
            else:
                panels["SCALE_Y"] = panels["ROW"].to_numpy()
        else:
            panels["SCALE_Y"] = 1

        return panels

    # -- setup_aspect_ratio -------------------------------------------------
    def setup_aspect_ratio(
        self,
        coord: Any,
        free: Dict[str, bool],
        theme: Any,
        ranges: Sequence[Any],
    ) -> AspectRatio:
        """Resolve the panel aspect ratio + ``respect`` flag.

        Port of ggh4x's ``FacetGrid2$setup_aspect_ratio``
        (``R/facet_grid2.R:284-296``).  Uses ``theme$aspect.ratio`` if set; else
        ``coord$aspect(ranges[[1]])`` when neither dimension is free; else ``1``
        with ``respect=False``.

        Parameters
        ----------
        coord : Coord
        free : dict
            ``{"x": bool, "y": bool}``.
        theme : Theme
        ranges : sequence
            Per-panel ``panel_params`` dicts.

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

    # -- setup_panel_table --------------------------------------------------
    def setup_panel_table(
        self,
        panels: List[Any],
        layout: pd.DataFrame,
        space: Dict[str, bool],
        ranges: Sequence[Any],
        aspect: AspectRatio,
        clip: str,
        theme: Any,
    ) -> Gtable:
        """Build the panel gtable (byrow matrix, null-unit / proportional sizes).

        Port of ggh4x's ``FacetGrid2$setup_panel_table``
        (``R/facet_grid2.R:297-340``).  Blanks non-rendered panels, lays panels
        into a ``nrow x ncol`` gtable at ``(ROW, COL)`` with ``z=1``, sizes
        columns / rows as ``"null"`` units (proportional to ``diff(range)`` when
        ``space`` is free, heights scaled by ``abs(aspect)`` otherwise), then adds
        panel spacing.

        Parameters
        ----------
        panels : list of grob
            One decorated panel grob per PANEL (PANEL-ordered).
        layout : pandas.DataFrame
            The facet layout (carries ``ROW``, ``COL``, ``_render``).
        space : dict
            ``{"x": bool, "y": bool}`` -- ``space_free`` flags.
        ranges : sequence
            Per-panel ``panel_params`` dicts.
        aspect : AspectRatio
        clip : str
            Panel clip setting (``coord.clip``).
        theme : Theme

        Returns
        -------
        Gtable
        """
        panels = list(panels)
        render = list(layout["_render"]) if "_render" in layout.columns else [True] * len(panels)
        panels = [p if render[i] else null_grob() for i, p in enumerate(panels)]

        ncol = int(layout["COL"].max())
        nrow = int(layout["ROW"].max())

        panel_arr = np.asarray(layout["PANEL"]).astype(int)

        if space["x"]:
            row1 = layout[layout["ROW"] == 1]
            ps = [int(p) for p in row1["PANEL"]]
            widths_vals = [
                _range_diff(ranges[i - 1], "x") for i in ps
            ]
            widths = Unit(widths_vals, "null")
        else:
            widths = Unit([1.0] * ncol, "null")

        if space["y"]:
            col1 = layout[layout["COL"] == 1]
            ps = [int(p) for p in col1["PANEL"]]
            heights_vals = [
                _range_diff(ranges[i - 1], "y") for i in ps
            ]
            heights = Unit(heights_vals, "null")
        else:
            heights = Unit([1.0 * abs(aspect.value)] * nrow, "null")

        panel_table = Gtable(widths=widths, heights=heights, respect=aspect.respect)

        rows_idx = [int(v) for v in layout["ROW"]]
        cols_idx = [int(v) for v in layout["COL"]]
        # R: paste0("panel-", rep(seq_len(nrow), ncol), "-", rep(seq_len(ncol), each = nrow))
        # This is a fixed *positional* name vector applied to the PANEL-ordered
        # ``panels`` list -- the suffixes do NOT track each panel's actual
        # ROW/COL, they enumerate row-fastest, column-slowest.
        name_i = [((k % nrow) + 1) for k in range(nrow * ncol)]
        name_j = [((k // nrow) + 1) for k in range(nrow * ncol)]
        names = [f"panel-{name_i[k]}-{name_j[k]}" for k in range(len(panels))]
        panel_table = gtable_add_grob(
            panel_table,
            panels,
            t=rows_idx,
            l=cols_idx,
            z=1,
            clip=clip,
            name=names,
        )
        panel_table = gtable_add_col_space(
            panel_table, calc_element("panel.spacing.x", theme)
        )
        panel_table = gtable_add_row_space(
            panel_table, calc_element("panel.spacing.y", theme)
        )
        return panel_table

    # -- attach_axes --------------------------------------------------------
    def attach_axes(self, panel_table: Gtable, axes: Dict[str, Any]) -> Gtable:
        """Weave the four axis bands into the panel gtable.

        Port of ggh4x's ``FacetGrid2$attach_axes`` (``R/facet_grid2.R:341-356``).
        Measures the axes (:func:`_measure_axes`) then weaves the top (shift -1),
        bottom (shift 0), left (shift -1) and right (shift 0) bands at ``z=3``.

        Parameters
        ----------
        panel_table : Gtable
        axes : dict
            ``{"top", "bottom", "left", "right"}`` grob matrices from
            :meth:`setup_axes`.

        Returns
        -------
        Gtable
        """
        sizes = _measure_axes(axes)
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

    # -- setup_axes ---------------------------------------------------------
    def setup_axes(
        self,
        axes: Dict[str, Any],
        empty: List[List[Any]],
        position: Sequence[int],
        layout: pd.DataFrame,
        params: Dict[str, Any],
    ) -> Dict[str, List[List[Any]]]:
        """Fill the 4 axis grob matrices by scale-id and blank interior axes.

        Port of ggh4x's ``FacetGrid2$setup_axes`` (``R/facet_grid2.R:358-394``).
        Fills the top/bottom/left/right ``nrow x ncol`` matrices by the per-cell
        ``position`` (panel) index, blanks redundant interior axes unless they
        must repeat (``independent`` | ``axes``), and purges labels from interior
        axes when ``axes & rmlab & !independent``.

        Parameters
        ----------
        axes : dict
            The transposed batch axes ``{"x": {"top", "bottom"},
            "y": {"left", "right"}}`` from :func:`render_axes`.
        empty : list of list
            A ``nrow x ncol`` zero-grob matrix template.
        position : sequence of int
            The per-cell panel index (row-major / byrow), 1-based.
        layout : pandas.DataFrame
        params : dict

        Returns
        -------
        dict
            ``{"top", "bottom", "left", "right"}`` grob matrices (list of lists).
        """
        nrow = len(empty)
        ncol = len(empty[0]) if nrow else 0

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

        # position is row-major: position[r*ncol + c] = panel index (already the
        # PANEL-ordered axis index because axes were rendered over ranges[panel_pos]).
        k = 0
        for r in range(nrow):
            for c in range(ncol):
                # axes$x$top[position] where the rendered list is itself indexed
                # by the same row-major order, so cell (r,c) takes element k.
                top[r][c] = x_top[k]
                bottom[r][c] = x_bottom[k]
                left[r][c] = y_left[k]
                right[r][c] = y_right[k]
                k += 1

        independent = params["independent"]
        axes_p = params["axes"]
        rmlab = params["rmlab"]
        repeat_x = independent["x"] or axes_p["x"]
        repeat_y = independent["y"] or axes_p["y"]

        if not repeat_x:
            # top[-1, ]: blank all rows except first.
            for r in range(1, nrow):
                for c in range(ncol):
                    top[r][c] = null_grob()
            # bottom[-nrow, ]: blank all rows except last.
            for r in range(nrow - 1):
                for c in range(ncol):
                    bottom[r][c] = null_grob()
        if not repeat_y:
            # left[, -1]: blank all cols except first.
            for r in range(nrow):
                for c in range(1, ncol):
                    left[r][c] = null_grob()
            # right[, -ncol]: blank all cols except last.
            for r in range(nrow):
                for c in range(ncol - 1):
                    right[r][c] = null_grob()

        if axes_p["x"] and rmlab["x"] and not independent["x"]:
            for r in range(1, nrow):
                for c in range(ncol):
                    top[r][c] = purge_guide_labels(top[r][c])
            for r in range(nrow - 1):
                for c in range(ncol):
                    bottom[r][c] = purge_guide_labels(bottom[r][c])
        if axes_p["y"] and rmlab["y"] and not independent["y"]:
            for r in range(nrow):
                for c in range(1, ncol):
                    left[r][c] = purge_guide_labels(left[r][c])
            for r in range(nrow):
                for c in range(ncol - 1):
                    right[r][c] = purge_guide_labels(right[r][c])

        return {"top": top, "bottom": bottom, "left": left, "right": right}

    # -- finish_panels (identity seam) --------------------------------------
    def finish_panels(
        self,
        panels: Any,
        layout: pd.DataFrame,
        params: Dict[str, Any],
        theme: Any,
    ) -> Any:
        """Identity post-processing hook (extension seam).

        Port of ggh4x's ``FacetGrid2$finish_panels`` (``R/facet_grid2.R:395-397``).
        Returns *panels* unchanged; subclasses (``facet_nested``) override it.
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
        """Assemble the grid panel gtable (full replacement of the base pipeline).

        Port of ggh4x's ``FacetGrid2$draw_panels`` (``R/facet_grid2.R:398-433``).
        Builds the byrow ``panel_pos`` vector, batch-renders the axes
        (``transpose=True``), then runs ``setup_axes`` -> ``setup_aspect_ratio``
        -> ``setup_panel_table`` -> ``attach_axes`` -> strip ``setup`` /
        ``incorporate_grid`` -> ``finish_panels``.  Does **not** chain to the base
        ``draw_panels``.

        Parameters
        ----------
        panels : list
            Per-layer lists of per-panel geom grobs (``ggplot2_py`` shape), or an
            already-flat list of decorated panel grobs.
        layout : pandas.DataFrame
        x_scales, y_scales : list
        ranges : list
            Per-panel ``panel_params`` dicts.
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
        ncol = int(layout["COL"].max())
        nrow = int(layout["ROW"].max())
        empty_table = [[null_grob() for _ in range(ncol)] for _ in range(nrow)]

        # Decorate per-layer grobs into one panel grob per PANEL.
        panel_grobs = _decorate_panels(panels, layout, ranges, coord, theme)

        # panel_pos: byrow reshape of as.integer(PANEL).
        panel_int = [int(p) for p in layout["PANEL"]]
        # Map (ROW,COL) -> PANEL for byrow ordering.
        rc_to_panel: Dict[tuple, int] = {}
        for _, row in layout.iterrows():
            rc_to_panel[(int(row["ROW"]), int(row["COL"]))] = int(row["PANEL"])
        panel_pos: List[int] = []
        for r in range(1, nrow + 1):
            for c in range(1, ncol + 1):
                panel_pos.append(rc_to_panel.get((r, c), 1))

        ranges_pos = [ranges[p - 1] for p in panel_pos]
        axes = render_axes(ranges_pos, ranges_pos, coord, theme, transpose=True)
        axes = self.setup_axes(axes, empty_table, panel_pos, layout, params)

        aspect_ratio = self.setup_aspect_ratio(coord, params["free"], theme, ranges)

        panel_table = self.setup_panel_table(
            panel_grobs, layout, params["space_free"], ranges, aspect_ratio,
            coord.clip, theme,
        )
        panel_table = self.attach_axes(panel_table, axes)

        strip.setup(layout, params, theme, type="grid")
        panel_table = strip.incorporate_grid(panel_table, params["switch"])

        return self.finish_panels(
            panels=panel_table, layout=layout, params=params, theme=theme
        )


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------
def _range_diff(panel_params: Any, axis: str) -> float:
    """Return ``diff(range)`` for the x or y axis of a ``panel_params`` dict.

    ``ggplot2_py`` exposes both ``"x_range"`` / ``"y_range"`` and the R-style
    ``"x.range"`` / ``"y.range"`` keys; prefer the R-style key to match the
    gold-standard, falling back to the underscore variant.
    """
    if panel_params is None:
        return 1.0
    key_dot = f"{axis}.range"
    key_us = f"{axis}_range"
    rng = panel_params.get(key_dot)
    if rng is None:
        rng = panel_params.get(key_us)
    if rng is None:
        return 1.0
    return float(rng[1] - rng[0])


def _decorate_panels(
    panels: list,
    layout: pd.DataFrame,
    ranges: list,
    coord: Any,
    theme: Any,
) -> List[Any]:
    """Compose per-layer geom grobs into one decorated panel grob per PANEL.

    ``ggplot2_py`` passes ``panels`` to ``draw_panels`` as a list-of-layers (each
    a list of per-panel grobs); R passes a flat list of already-decorated panel
    grobs (one per PANEL).  This bridges the two by running
    ``coord.draw_panel(layer_grobs_for_panel, pp, theme)`` per panel, matching the
    base ``Facet.draw_panels`` decoration (facet.py:766-782).  When *panels* is
    already flat (one grob per panel) it is returned as-is.

    Parameters
    ----------
    panels : list
    layout : pandas.DataFrame
    ranges : list
    coord : Coord
    theme : Theme

    Returns
    -------
    list of grob
        One decorated panel grob per PANEL, PANEL-ordered.
    """
    n_panel = len(layout)
    # Detect the already-flat case: a list of length n_panel of non-list grobs.
    if (
        len(panels) == n_panel
        and n_panel > 0
        and not isinstance(panels[0], list)
    ):
        return list(panels)

    out: List[Any] = []
    panel_order = sorted(int(p) for p in layout["PANEL"])
    for panel_id in panel_order:
        panel_idx = panel_id - 1
        pp = ranges[panel_idx] if panel_idx < len(ranges) else {}
        layer_grobs: List[Any] = []
        for layer in panels:
            if isinstance(layer, list):
                if panel_idx < len(layer):
                    layer_grobs.append(layer[panel_idx])
            elif layer is not None:
                layer_grobs.append(layer)
        if hasattr(coord, "draw_panel"):
            decorated = coord.draw_panel(layer_grobs, pp, theme)
        else:
            decorated = GTree(children=GList(*layer_grobs), name=f"panel-{panel_id}")
        out.append(decorated)
    return out
