"""Manual panel layout facets (port of ggh4x ``R/facet_manual.R``).

``facet_manual`` lays panels out according to a user *design* (a character
art-string or an integer matrix), letting panels span arbitrary rectangles of a
cell grid.  :class:`FacetManual` subclasses the ggh4x :class:`~ggh4x.facet_wrap2.FacetWrap2`
sibling (transitively :class:`ggplot2_py.facet.FacetWrap`), inheriting its
``setup_panel_table`` / ``finish_panels`` but fully overriding ``compute_layout``
(producing a span layout with ``.TOP``/``.RIGHT``/``.BOTTOM``/``.LEFT`` instead
of ``ROW``/``COL``), ``map_data``, ``setup_aspect_ratio``, ``setup_axes``,
``attach_axes`` and ``draw_panels``.

The design matrix is a 2-D NumPy integer array of 1-based panel ids with
``np.nan`` for blank cells, carrying a sidecar ``design_names`` list when the
design was character-coded with non-numeric labels.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ggplot2_py import calc_element, ggproto
from ggplot2_py.facet import _combine_vars, _resolve_facet_vars, facet_null, facet_wrap
from grid_py import Unit, is_unit, null_grob, unit_rep

from ggh4x._borrowed_ggplot2 import empty, id, snake_class
from ggh4x._cli import cli_abort, cli_warn
from ggh4x._facet_helpers import AspectRatio, _match_facet_arg
from ggh4x._facet_utils import (
    render_axes,
    split_heights_cm,
    split_widths_cm,
    weave_panel_cols,
    weave_panel_rows,
)
from ggh4x._rlang import arg_match0
from ggh4x.facet_wrap2 import FacetWrap2, purge_guide_labels
from ggh4x.strip_vanilla import resolve_strip

__all__ = [
    "facet_manual",
    "FacetManual",
    "_validate_design",
    "_restrict_axes",
    "_do_purge",
]


# ---------------------------------------------------------------------------
# validate_design (R facet_manual.R:381-437)
# ---------------------------------------------------------------------------
class _Design:
    """A validated design: an integer matrix plus optional ``design_names``.

    Stands in for R's ``matrix`` with ``attr(., "design_names")``.  Blank cells
    are ``np.nan`` (the matrix is stored as float to admit ``nan``); ``names`` is
    the sorted unique character labels when the design was character-coded, else
    ``None``.

    Attributes
    ----------
    matrix : numpy.ndarray
        2-D float array of 1-based panel ids (``nan`` = blank).
    names : list or None
        The ``design_names`` sidecar.
    """

    def __init__(self, matrix: np.ndarray, names: Optional[List[Any]] = None) -> None:
        self.matrix = matrix
        self.names = names

    @property
    def shape(self) -> Any:
        """Return the design matrix shape ``(nrow, ncol)``."""
        return self.matrix.shape


def _validate_design(design: Any = None, trim: bool = True) -> _Design:
    """Validate / normalise a *design* into an integer :class:`_Design`.

    Faithful port of ggh4x's ``validate_design`` (``R/facet_manual.R:381-437``).
    Character designs (patchwork-style art-strings) are split on newlines,
    trimmed, and split per character (``'#'`` -> blank).  Matrices use ``np.nan``
    for blanks.  The unique non-blank values are sorted and renumbered to
    ``1..k``; non-numeric labels are kept as ``design_names``.  When ``trim`` is
    ``True`` the design is cropped to its non-empty rows / columns.

    Parameters
    ----------
    design : str or array-like or None
        The design specification.
    trim : bool, default True
        Whether to trim empty rows / columns.

    Returns
    -------
    _Design
        The validated design.

    Raises
    ------
    ValueError
        When *design* is ``None``, non-rectangular, of invalid dimensions, or
        not interpretable as a matrix.
    """
    if design is None:
        cli_abort("The `design` argument cannot be `None`.")

    names: Optional[List[Any]] = None

    # --- character art-string path (patchwork::as_areas) --------------------
    if isinstance(design, str):
        lines = design.split("\n")
        lines = [s.strip() for s in lines]
        lines = [s for s in lines if len(s) > 0]
        rows = [list(s) for s in lines]
        ncols = [len(r) for r in rows]
        if len(set(ncols)) != 1:
            cli_abort("The `design` argument must be rectangular.")
        mat = np.array(rows, dtype=object)
    else:
        mat = np.asarray(design)
        if mat.ndim != 2:
            # Force to a column matrix (R as.matrix on an atomic vector).
            mat = mat.reshape(-1, 1)

    if mat.ndim != 2 or any(d < 1 for d in mat.shape):
        cli_abort("The `design` argument has invalid dimensions.")

    dim = mat.shape

    # Character matrices: '#' -> blank (NA).
    is_char = mat.dtype.kind in ("U", "S", "O") and any(
        isinstance(v, str) for v in mat.flatten()
    )
    flat = mat.flatten()
    if is_char:
        flat = np.array(
            [np.nan if (isinstance(v, str) and v == "#") else v for v in flat],
            dtype=object,
        )

    # uniq = sort(unique(design)) over non-blank values.
    def _is_blank(v: Any) -> bool:
        if v is None:
            return True
        try:
            return bool(np.isnan(v))
        except (TypeError, ValueError):
            return False

    non_blank = [v for v in flat if not _is_blank(v)]
    uniq = sorted(set(non_blank), key=lambda z: (str(type(z)), z))
    # design = match(design, uniq) -> 1-based ids, nan for blanks.
    lookup = {v: i + 1 for i, v in enumerate(uniq)}
    matched = np.array(
        [float(lookup[v]) if not _is_blank(v) else np.nan for v in flat],
        dtype=float,
    ).reshape(dim)

    if trim:
        non_empty = ~np.isnan(matched)
        row_any = np.where(non_empty.any(axis=1))[0]
        col_any = np.where(non_empty.any(axis=0))[0]
        if len(row_any):
            keep_row = list(range(int(row_any.min()), int(row_any.max()) + 1))
        else:
            keep_row = list(range(dim[0]))
        if len(col_any):
            keep_col = list(range(int(col_any.min()), int(col_any.max()) + 1))
        else:
            keep_col = list(range(dim[1]))
        matched = matched[np.ix_(keep_row, keep_col)]

    # Non-numeric uniques become design_names.
    numeric_uniq = all(
        isinstance(v, (int, float, np.integer, np.floating)) for v in uniq
    )
    if not numeric_uniq:
        names = list(uniq)

    return _Design(matched, names=names)


# ---------------------------------------------------------------------------
# restrict_axes (R facet_manual.R:442-453)
# ---------------------------------------------------------------------------
def _restrict_axes(
    axes: List[Any],
    position: Sequence[int],
    by: Sequence[int],
    which_fun: Any = min,
    restrictor: Any = null_grob,
) -> List[Any]:
    """Keep only the edge-most axis per group, blanking / purging the rest.

    Faithful port of ggh4x's ``restrict_axes`` (``R/facet_manual.R:442-453``).
    Groups *axes* by *by*; within each group only the grob whose *position*
    equals ``which_fun(group)`` is kept.  Non-kept grobs are replaced: if
    *restrictor* is callable it is applied per grob (e.g. ``purge_guide_labels``);
    otherwise the value (e.g. ``null_grob()``) is assigned.

    Parameters
    ----------
    axes : list of grob
        One rendered axis per panel.
    position : sequence of int
        Per-panel ``.TOP`` / ``.BOTTOM`` / ``.LEFT`` / ``.RIGHT`` span coordinate.
    by : sequence of int
        Grouping coordinate (the orthogonal span coordinate).
    which_fun : callable, default min
        ``min`` (keep the smallest position) or ``max`` (keep the largest).
    restrictor : callable or grob, default null_grob
        Per-grob replacement function or replacement value.

    Returns
    -------
    list of grob
        The restricted axis list.
    """
    axes = list(axes)
    position = list(position)
    by = list(by)
    n = len(axes)

    # keep[i] = (position[i] == which_fun(positions in same group))
    group_target: Dict[Any, Any] = {}
    grouped: Dict[Any, List[int]] = {}
    for i, g in enumerate(by):
        grouped.setdefault(g, []).append(i)
    for g, idxs in grouped.items():
        group_target[g] = which_fun([position[i] for i in idxs])

    keep = [position[i] == group_target[by[i]] for i in range(n)]

    is_callable = callable(restrictor)
    for i in range(n):
        if not keep[i]:
            if is_callable:
                axes[i] = restrictor(axes[i])
            else:
                axes[i] = restrictor() if callable(restrictor) else restrictor
    return axes


# ---------------------------------------------------------------------------
# do_purge (R facet_manual.R:455-466)
# ---------------------------------------------------------------------------
def _do_purge(a: Sequence[Any], b: Sequence[Any], check_disjoint: bool = False) -> bool:
    """Decide whether axes can be purged across a span layout.

    Faithful port of ggh4x's ``do_purge`` (``R/facet_manual.R:455-466``).  Takes
    the unique ``(a, b)`` pairs; returns ``True`` when those pairs are 1:1 with
    both ``a`` and ``b`` (each appears once).  When *check_disjoint* and more than
    one pair exists, additionally requires the spans (ordered by ``(a, b)``) to be
    non-overlapping (``cummax([0, b[:-1]]) < a``).

    Parameters
    ----------
    a, b : sequence
        Paired span coordinates.
    check_disjoint : bool, default False
        Whether to additionally enforce non-overlapping spans.

    Returns
    -------
    bool
    """
    df = pd.DataFrame({"a": list(a), "b": list(b)}).drop_duplicates().reset_index(drop=True)
    aa = df["a"].to_numpy()
    bb = df["b"].to_numpy()
    n = len(df)
    ans = n == len(pd.unique(aa)) and n == len(pd.unique(bb))
    if not check_disjoint or n == 1:
        return bool(ans)
    order = np.lexsort((bb, aa))  # order by a, then b
    a_o = aa[order]
    b_o = bb[order]
    cum = np.maximum.accumulate(np.concatenate([[0], b_o[:-1]]))
    return bool(ans and np.all(cum < a_o))


# ---------------------------------------------------------------------------
# Constructor (R facet_manual.R:62-130)
# ---------------------------------------------------------------------------
def facet_manual(
    facets: Any,
    design: Any = None,
    widths: Any = None,
    heights: Any = None,
    respect: bool = False,
    drop: bool = True,
    strip_position: str = "top",
    scales: Any = "fixed",
    axes: Any = "margins",
    remove_labels: Any = "none",
    labeller: Any = "label_value",
    trim_blank: bool = True,
    strip: Any = "vanilla",
) -> Any:
    """Manual layout for panels.

    Faithful port of ggh4x's ``facet_manual`` (``R/facet_manual.R:62-130``).
    Panels are placed according to *design* (a character art-string or integer
    matrix), each panel spanning the bounding rectangle of its design cells.

    Parameters
    ----------
    facets : formula / list / dict / str
        Faceting variables.
    design : str or array-like
        Panel-area specification (``'#'`` / ``NA`` mark blank cells).
    widths, heights : numeric or grid_py.Unit or None, default None
        Cell sizes; numerics become relative ``"null"`` units.
    respect : bool, default False
        Whether ``"null"`` widths / heights are proportional.
    drop : bool, default True
        Drop unused factor combinations.
    strip_position : {"top", "bottom", "left", "right"}, default "top"
    scales : {"fixed", "free_x", "free_y", "free"} or bool, default "fixed"
    axes : {"margins", "x", "y", "all"} or bool, default "margins"
    remove_labels : {"none", "x", "y", "all"} or bool, default "none"
    labeller : callable or str, default "label_value"
    trim_blank : bool, default True
        Trim empty design rows / columns.
    strip : Strip or callable or str, default "vanilla"

    Returns
    -------
    FacetManual or FacetNull
        A facet ggproto object; :func:`ggplot2_py.facet.facet_null` when *facets*
        is empty.
    """
    strip_position = arg_match0(
        strip_position, ["top", "bottom", "left", "right"], arg_name="strip_position"
    )

    design = _validate_design(design, trim_blank)

    facets_resolved = _resolve_facet_vars(facet_wrap(facets=facets).params.get("facets"))
    if len(facets_resolved) == 0:
        return facet_null()

    if widths is not None and not is_unit(widths):
        widths = Unit(widths, "null")
    if heights is not None and not is_unit(heights):
        heights = Unit(heights, "null")

    dim = design.shape  # (nrow, ncol)

    if widths is not None:
        widths = unit_rep(widths, length_out=dim[1])
    if heights is not None:
        heights = unit_rep(heights, length_out=dim[0])

    free = _match_facet_arg(scales, ["fixed", "free_x", "free_y", "free"], nm="scales")
    axes = _match_facet_arg(axes, ["margins", "x", "y", "all"], nm="axes")
    rmlab = _match_facet_arg(remove_labels, ["none", "x", "y", "all"], nm="remove_labels")
    strip = resolve_strip(strip)

    params: Dict[str, Any] = {
        "design": design,
        "facets": {name: name for name in facets_resolved},
        "widths": widths,
        "heights": heights,
        "respect": respect,
        "strip.position": strip_position,
        "strip_position": strip_position,
        "labeller": labeller,
        "drop": drop,
        "nrow": dim[0],
        "ncol": dim[1],
        "free": free,
        "axes": axes,
        "rmlab": rmlab,
        "dim": [dim[0], dim[1]],
    }

    obj = FacetManual()
    obj._set(shrink=True, strip=strip, params=params)
    return obj


# ---------------------------------------------------------------------------
# ggproto class (R facet_manual.R:138-377)
# ---------------------------------------------------------------------------
class FacetManual(FacetWrap2):
    """Manual-layout facet ggproto (port of R ``FacetManual``).

    Subclasses the ggh4x :class:`~ggh4x.facet_wrap2.FacetWrap2` sibling.  Produces
    a *span* layout (``.TOP``/``.RIGHT``/``.BOTTOM``/``.LEFT``, one row per unique
    panel) rather than a ``ROW``/``COL`` grid, then weaves per-panel axes and
    strips honouring those spans.

    Attributes
    ----------
    shrink : bool
    strip : Strip
    params : dict
    """

    _class_name = "FacetManual"

    shrink: bool = True
    strip: Any = None

    # -- compute_layout (R:141-201) -----------------------------------------
    def compute_layout(self, data: List[pd.DataFrame], params: Dict[str, Any]) -> pd.DataFrame:
        """Translate the design matrix into a span layout.

        Port of ggh4x's ``FacetManual$compute_layout`` (``R/facet_manual.R:141-201``).

        Parameters
        ----------
        data : list of DataFrame
        params : dict

        Returns
        -------
        pandas.DataFrame
            Span layout with ``.TOP``/``.RIGHT``/``.BOTTOM``/``.LEFT``, ``PANEL``,
            ``SCALE_X``, ``SCALE_Y`` and the faceting-var columns.
        """
        vars_ = _resolve_facet_vars(params.get("facets"))
        if len(vars_) == 0:
            return pd.DataFrame(
                {
                    ".TOP": [1],
                    ".RIGHT": [1],
                    ".BOTTOM": [1],
                    ".LEFT": [1],
                    "PANEL": pd.Categorical([1]),
                    "SCALE_X": [1],
                    "SCALE_Y": [1],
                }
            )

        design = params["design"]
        mat = design.matrix
        nrow, ncol = mat.shape

        # split(row(design), design) -> per-id row range; same for cols.
        row_idx = np.repeat(np.arange(1, nrow + 1)[:, None], ncol, axis=1)
        col_idx = np.repeat(np.arange(1, ncol + 1)[None, :], nrow, axis=0)
        flat_design = mat.flatten(order="F")  # R fills column-major
        flat_row = row_idx.flatten(order="F")
        flat_col = col_idx.flatten(order="F")

        # ids in first-occurrence order matching R's split() (sorted by id value).
        valid = ~np.isnan(flat_design)
        ids_present = sorted(set(int(v) for v in flat_design[valid]))

        tops, rights, bottoms, lefts, panel_ids = [], [], [], [], []
        for pid in ids_present:
            mask = (flat_design == pid)
            rows_g = flat_row[mask]
            cols_g = flat_col[mask]
            tops.append(int(rows_g.min()))
            bottoms.append(int(rows_g.max()))
            lefts.append(int(cols_g.min()))
            rights.append(int(cols_g.max()))
            panel_ids.append(pid)

        layout = pd.DataFrame(
            {
                ".TOP": tops,
                ".RIGHT": rights,
                ".BOTTOM": bottoms,
                ".LEFT": lefts,
                "PANEL": pd.Categorical(panel_ids, categories=panel_ids),
            }
        )

        base = _combine_vars(data, vars_, drop=params.get("drop", True))
        base = base.reset_index(drop=True)
        id_arr = id(base, drop=True)
        n = int(id_arr.n)

        if n > len(layout):
            n = len(layout)
            id_arr = np.asarray(id_arr)[:n]
            dropped = base.apply(
                lambda r: ":".join(str(x) for x in r), axis=1
            ).tolist()[n:]
            cli_warn(
                "Found more facetting levels than designed. The following levels "
                "are dropped: " + ", ".join(dropped)
            )

        # R: lnames <- attr(layout, "design_names").  The design_names attribute
        # is set on `design` (validate_design), NOT on `layout`, so in R this is
        # always NULL and the partial-match warning + reorder below is dead code.
        # Mirror R exactly by reading it from `layout` (which carries no such
        # attr) so the block never fires -- reading `design.names` here produced
        # a spurious "partial match" warning that R never emits.
        lnames = getattr(layout, "attrs", {}).get("design_names")
        if lnames is not None and len(base.columns) > 0:
            first_col = base.iloc[:, 0]
            isect = [v for v in lnames if v in set(first_col)]
            if len(isect) != 0 and len(isect) != len(base):
                cli_warn(
                    "Only partial match found between facetting levels and design levels."
                )
            elif len(isect) > 0:
                # base <- base[match(base[[1]], isect), ]
                order_map = {v: i for i, v in enumerate(isect)}
                new_order = [order_map.get(v, len(isect)) for v in first_col]
                base = base.iloc[np.argsort(np.argsort(new_order, kind="mergesort"))].reset_index(drop=True)

        if n < len(layout):
            panel_int = np.asarray(layout["PANEL"].astype(int))
            keep = panel_int <= n
            layout = layout.loc[keep].reset_index(drop=True)
            kept_ids = [p for p in panel_ids if p <= n]
            layout["PANEL"] = pd.Categorical(
                layout["PANEL"].astype(int), categories=kept_ids
            )

        id_int = np.asarray(id_arr, dtype=int)
        order = np.argsort(id_int, kind="mergesort")  # base[order(id), ]
        base_ordered = base.iloc[order].reset_index(drop=True)

        panels = pd.concat(
            [layout.reset_index(drop=True), base_ordered], axis=1
        )
        panels["SCALE_X"] = np.arange(1, n + 1) if params["free"]["x"] else 1
        panels["SCALE_Y"] = np.arange(1, n + 1) if params["free"]["y"] else 1

        # order(PANEL)
        panel_int = np.asarray(panels["PANEL"].astype(int))
        panels = panels.iloc[np.argsort(panel_int, kind="mergesort")].reset_index(drop=True)
        return panels

    # -- map_data (R:203-240) -----------------------------------------------
    def map_data(self, data: pd.DataFrame, layout: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Assign ``PANEL`` to layer data by matching facet values to the layout.

        Port of ggh4x's ``FacetManual$map_data`` (``R/facet_manual.R:203-240``).

        Parameters
        ----------
        data : pandas.DataFrame
        layout : pandas.DataFrame
        params : dict

        Returns
        -------
        pandas.DataFrame
            *data* with a ``PANEL`` column; rows not matching any panel dropped.
        """
        if empty(data):
            out = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()
            out["PANEL"] = pd.Categorical([])
            return out

        vars_ = _resolve_facet_vars(params.get("facets"))
        if len(vars_) == 0:
            data = data.copy()
            data["PANEL"] = list(layout["PANEL"])
            return data

        data = data.copy()
        facet_vals = data[[v for v in vars_ if v in data.columns]].copy()
        # Coerce facet vals + layout keys to factor (string) for matching.
        for c in facet_vals.columns:
            facet_vals[c] = facet_vals[c].astype(str)
        layout = layout.copy()
        lkeys = [v for v in vars_ if v in layout.columns]

        missing_facets = [v for v in vars_ if v not in facet_vals.columns]
        if missing_facets:
            to_add = layout[missing_facets].drop_duplicates().reset_index(drop=True)
            data_rep = np.repeat(np.arange(len(data)), len(to_add))
            facet_rep = np.tile(np.arange(len(to_add)), len(data))
            data = data.iloc[data_rep].reset_index(drop=True)
            facet_vals = pd.concat(
                [
                    facet_vals.iloc[data_rep].reset_index(drop=True),
                    to_add.iloc[facet_rep].reset_index(drop=True),
                ],
                axis=1,
            )

        # join_keys: match facet_vals to layout on the faceting vars.
        layout_keys = layout[lkeys].copy()
        for c in layout_keys.columns:
            layout_keys[c] = layout_keys[c].astype(str)
        for c in facet_vals.columns:
            facet_vals[c] = facet_vals[c].astype(str)

        layout_keys = layout_keys.assign(_PANEL=list(layout["PANEL"]))
        merged = facet_vals.merge(layout_keys, on=lkeys, how="left")
        data["PANEL"] = pd.Categorical(
            merged["_PANEL"].to_numpy(),
            categories=list(layout["PANEL"].cat.categories)
            if isinstance(layout["PANEL"].dtype, pd.CategoricalDtype)
            else None,
        )
        data = data.loc[~data["PANEL"].isna()].reset_index(drop=True)
        return data

    # -- setup_aspect_ratio (R:242-252) -------------------------------------
    def setup_aspect_ratio(
        self,
        coord: Any,
        free: Dict[str, bool],
        theme: Any,
        ranges: Sequence[Any],
    ) -> AspectRatio:
        """Resolve the aspect ratio + ``respect`` flag.

        Port of ggh4x's ``FacetManual$setup_aspect_ratio``
        (``R/facet_manual.R:242-252``).  Unlike :class:`FacetWrap2`, R returns
        ``NULL`` (not 1) in the free-scales / no-theme-aspect case, relying on the
        inherited ``setup_panel_table`` to apply ``respect <- params$respect %||%
        attr(aspect, "respect") %||% FALSE`` and ``heights <- params$heights %||%
        unit(abs(aspect %||% 1), "null")``.

        Because the Python :class:`FacetWrap2` ``setup_panel_table`` consumes an
        :class:`AspectRatio` carrier (it reads ``aspect.value`` / ``aspect.respect``
        and cannot accept ``None``), the ``NULL`` case is modelled here as
        ``AspectRatio(1.0, False)`` -- exactly R's ``aspect %||% 1`` (heights fall
        back to ``unit(1, "null")``) and ``attr(NULL, "respect") %||% FALSE``
        (``respect`` defers to ``params['respect']``, which ``facet_manual``
        always sets to a concrete bool).  The non-``NULL`` case carries
        ``respect=True`` like ``FacetWrap2``.

        Parameters
        ----------
        coord : Coord
        free : dict
        theme : Theme
        ranges : sequence

        Returns
        -------
        AspectRatio
            ``AspectRatio(1.0, False)`` for the R ``NULL`` case; otherwise the
            theme/coord aspect with ``respect=True``.
        """
        aspect_ratio = calc_element("aspect.ratio", theme) if theme is not None else None
        if aspect_ratio is None and not free["x"] and not free["y"] and ranges:
            aspect_ratio = coord.aspect(ranges[0])
        if aspect_ratio is None:
            # R: returns NULL -> setup_panel_table uses `abs(aspect %||% 1)` = 1
            # and `params$respect %||% attr(NULL,"respect") %||% FALSE`.
            return AspectRatio(1.0, False)
        return AspectRatio(float(aspect_ratio), True)

    # -- setup_axes (R:254-294) ---------------------------------------------
    def setup_axes(
        self,
        axes: Dict[str, Any],
        layout: pd.DataFrame,
        params: Dict[str, Any],
        theme: Any,
    ) -> pd.DataFrame:
        """Pick per-panel axis grobs and decide span-aware purging.

        Port of ggh4x's ``FacetManual$setup_axes`` (``R/facet_manual.R:254-294``).
        Returns a DataFrame of per-panel axis grobs keyed by ``t``/``b``/``l``/``r``
        = PANEL index (NOT the matrix + measurements list of :class:`FacetWrap2`).

        Parameters
        ----------
        axes : dict
            Transposed batch axes from :func:`render_axes`.
        layout : pandas.DataFrame
        params : dict
        theme : Theme

        Returns
        -------
        pandas.DataFrame
            ``{t, b, l, r, axes_top, axes_bottom, axes_left, axes_right}``.
        """
        panel = [int(v) for v in layout["PANEL"].astype(int)]
        scale_x = [int(v) for v in layout["SCALE_X"]]
        scale_y = [int(v) for v in layout["SCALE_Y"]]

        x_top = axes["x"]["top"]
        x_bottom = axes["x"]["bottom"]
        y_left = axes["y"]["left"]
        y_right = axes["y"]["right"]

        top = [x_top[i - 1] for i in scale_x]
        bottom = [x_bottom[i - 1] for i in scale_x]
        left = [y_left[i - 1] for i in scale_y]
        right = [y_right[i - 1] for i in scale_y]

        dot_top = list(layout[".TOP"])
        dot_bottom = list(layout[".BOTTOM"])
        dot_left = list(layout[".LEFT"])
        dot_right = list(layout[".RIGHT"])

        purge_x = (not params["free"]["x"]) and (params["rmlab"]["x"] or not params["axes"]["x"])
        purge_y = (not params["free"]["y"]) and (params["rmlab"]["y"] or not params["axes"]["y"])

        purge_x = purge_x and _do_purge(dot_left, dot_right)
        purge_y = purge_y and _do_purge(dot_top, dot_bottom)

        if purge_x:
            purger = purge_guide_labels if params["rmlab"]["x"] else null_grob()
            top = _restrict_axes(top, dot_top, dot_left, min, purger)
            bottom = _restrict_axes(bottom, dot_bottom, dot_left, max, purger)

        if purge_y:
            purger = purge_guide_labels if params["rmlab"]["y"] else null_grob()
            left = _restrict_axes(left, dot_left, dot_top, min, purger)
            right = _restrict_axes(right, dot_right, dot_top, max, purger)

        return pd.DataFrame(
            {
                "t": panel,
                "b": panel,
                "l": panel,
                "r": panel,
                "axes_top": top,
                "axes_bottom": bottom,
                "axes_left": left,
                "axes_right": right,
            }
        )

    # -- attach_axes (R:296-327) --------------------------------------------
    def attach_axes(
        self,
        panels: Any,
        axes: pd.DataFrame,
        sizes: Dict[str, Unit],
        params: Dict[str, Any],
        inside: bool = True,
    ) -> Any:
        """Weave the four per-panel axis sides into the panel gtable.

        Port of ggh4x's ``FacetManual$attach_axes`` (``R/facet_manual.R:296-327``).
        Zeroes interior strip-side gaps when scales are fixed and the spans are
        disjoint, then weaves each axis side via :func:`weave_panel_rows` /
        :func:`weave_panel_cols`.

        Parameters
        ----------
        panels : Gtable
        axes : pandas.DataFrame
            The per-panel grob frame from :meth:`setup_axes`.
        sizes : dict
            ``{top, bottom, left, right}`` size unit vectors.
        params : dict
        inside : bool, default True
            Whether ``strip.placement`` is ``"inside"``.

        Returns
        -------
        Gtable
        """
        panel_layout = _panel_layout(panels)
        strip_pos = params.get("strip.position", params.get("strip_position", "top"))

        if (not params["free"]["y"]) and _do_purge(panel_layout["t"], panel_layout["b"], True):
            if inside or strip_pos != "left":
                # sizes$left[-1] <- 0  (drop first element)
                _zero_unit_slice(sizes, "left", drop="first")
            if inside or strip_pos != "right":
                _zero_unit_slice(sizes, "right", drop="last")
        if (not params["free"]["x"]) and _do_purge(panel_layout["l"], panel_layout["r"], True):
            # NOTE: R has a missing brace here so only the first line is the
            # conditional body; the second `if` always runs. Reproduce faithfully.
            if inside or strip_pos != "bottom":
                _zero_unit_slice(sizes, "bottom", drop="last")
            if inside or strip_pos != "top":
                _zero_unit_slice(sizes, "top", drop="first")

        panels = weave_panel_rows(
            panels, axes, -1, sizes["top"], "axis-t", 3, "off", "t", "axes_top"
        )
        panels = weave_panel_rows(
            panels, axes, 0, sizes["bottom"], "axis-b", 3, "off", "b", "axes_bottom"
        )
        panels = weave_panel_cols(
            panels, axes, -1, sizes["left"], "axis-l", 3, "off", "l", "axes_left"
        )
        panels = weave_panel_cols(
            panels, axes, 0, sizes["right"], "axis-r", 3, "off", "r", "axes_right"
        )
        return panels

    # -- draw_panels (R:329-376) --------------------------------------------
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
    ) -> Any:
        """Assemble the manual panel gtable (full replacement of the base pipeline).

        Port of ggh4x's ``FacetManual$draw_panels`` (``R/facet_manual.R:329-376``).

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

        # Decorate per-layer grobs into one panel grob per PANEL.
        from ggh4x.facet_grid2 import _decorate_panels

        panel_grobs = _decorate_panels(panels, layout, ranges, coord, theme)

        # Inherited FacetWrap2.setup_panel_table reads .TOP/.BOTTOM/.LEFT/.RIGHT.
        panel_table = self.setup_panel_table(
            panel_grobs, layout, theme, coord, ranges, params
        )

        axes = render_axes(ranges, ranges, coord, theme, transpose=True)
        axes = self.setup_axes(axes, layout, params, theme)

        panel_pos = _panel_layout(panel_table)
        sizes = {
            "top": split_heights_cm(list(axes["axes_top"]), split=panel_pos["t"]),
            "bottom": split_heights_cm(list(axes["axes_bottom"]), split=panel_pos["b"]),
            "left": split_widths_cm(list(axes["axes_left"]), split=panel_pos["l"]),
            "right": split_widths_cm(list(axes["axes_right"]), split=panel_pos["r"]),
        }

        strip_placement = calc_element("strip.placement", theme) if theme is not None else None
        inside = (strip_placement if strip_placement is not None else "inside") == "inside"
        panel_table = self.attach_axes(panel_table, axes, sizes, params, inside=inside)

        # Synthesize ROW/COL late so Strip.setup(type="wrap") can treat the span
        # layout like a wrap layout.
        strip_pos = params.get("strip.position", params.get("strip_position", "top"))
        simplify = {
            "top": (".TOP", ".LEFT"),
            "bottom": (".BOTTOM", ".LEFT"),
            "left": (".TOP", ".LEFT"),
            "right": (".TOP", ".RIGHT"),
        }[strip_pos]
        layout = layout.copy()
        layout["ROW"] = layout[simplify[0]].to_numpy()
        layout["COL"] = layout[simplify[1]].to_numpy()

        strip.setup(layout, params, theme, type="wrap")
        panel_table = strip.incorporate_wrap(
            panel_table, strip_pos, clip=coord.clip, sizes=sizes
        )

        return self.finish_panels(
            panels=panel_table, layout=layout, params=params, theme=theme
        )


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------
def _panel_layout(table: Any) -> pd.DataFrame:
    """Return the ``"panel-*"`` rows of *table*'s layout as a DataFrame (t/b/l/r)."""
    lay = table.layout
    if not isinstance(lay, pd.DataFrame):
        lay = pd.DataFrame({k: list(v) for k, v in lay.items()})
    mask = lay["name"].astype(str).str.match(r"^panel")
    return lay.loc[mask, ["t", "b", "l", "r"]].reset_index(drop=True)


def _zero_unit_slice(sizes: Dict[str, Unit], key: str, drop: str) -> None:
    """Zero all-but-one elements of ``sizes[key]`` in place (R negative-index drop).

    ``drop="first"`` zeroes elements ``[1:]`` (R ``x[-1] <- 0``); ``drop="last"``
    zeroes elements ``[:-1]`` (R ``x[-length(x)] <- 0``).
    """
    u = sizes[key]
    n = len(u)
    if n <= 1:
        return
    if drop == "first":
        for i in range(1, n):
            u[i] = Unit(0, "cm")
    else:  # last
        for i in range(0, n - 1):
            u[i] = Unit(0, "cm")
