"""Nested-strip grid facets (port of ggh4x ``R/facet_nested.R``).

``facet_nested()`` behaves like :func:`ggh4x.facet_grid2` but merges adjacent
strips that share an (outer) faceting-variable value into a single spanning
strip, and -- unlike ``facet_grid()`` -- only auto-expands a missing faceting
variable when there is **no** variable in that direction at all (so partially
faceted layers are allowed, the defining feature of nesting).  Hierarchy lines
(``nest_line``) are drawn between strip layers to indicate the grouping.

The :class:`FacetNested` ggproto subclasses :class:`ggh4x.facet_grid2.FacetGrid2`
and overrides three seams:

* :meth:`FacetNested.map_data` -- assign data rows to panels, treating a
  faceting variable as "missing" (and force-expanding) only when *no* variable
  in its direction is present in the layer (R: ``facet_nested.R:130-187``).
* :meth:`FacetNested.vars_combine` -- build the cross-product base of facet
  values, permitting layers missing some vars by blank-filling (``""``) the
  absent columns rather than erroring (R: ``facet_nested.R:188-234``).
* :meth:`FacetNested.finish_panels` -- draw the nest indicator lines via
  :func:`add_nest_indicator` (R: ``facet_nested.R:236-238``).

The default strip is :func:`ggh4x.strip_nested.strip_nested` (the label-merging
nested strip), matching R's ``strip = "nested"``.

R source: ``ggh4x/R/facet_nested.R``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ggplot2_py.theme_elements import (
    ElementBlank,
    calc_element,
    combine_elements,
    element_blank,
    element_grob,
    element_line,
    is_theme_element,
)
from grid_py import Unit, unit_c
from gtable_py import gtable_add_grob

from ggh4x._borrowed_ggplot2 import empty, id, unique_combs
from ggh4x._cli import cli_abort
from ggh4x._facet_helpers import reshape_add_margins
from ggh4x._facet_utils import df_grid
from ggh4x.facet_grid2 import FacetGrid2, _as_name_list, new_grid_facets
from ggh4x.strip_nested import strip_nested

# Importing this module registers the ``ggh4x.facet.nestline`` theme element.
import ggh4x.themes_ggh4x  # noqa: F401

__all__ = [
    "facet_nested",
    "FacetNested",
    "add_nest_indicator",
]


# ---------------------------------------------------------------------------
# Borrowed ggplot2 internals (eval_facets / join_keys) -- not exported by the
# sibling ports, so reimplemented here on resolved column-name lists.
# ---------------------------------------------------------------------------
def _eval_facets(
    facets: Sequence[str],
    data: pd.DataFrame,
    possible_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Evaluate the faceting variables against a layer's data.

    Port of ggplot2's ``eval_facets`` (``borrowed_ggplot2.R:297-327``) reduced to
    the resolved-column-name model: each name in *facets* that names a column of
    *data* yields that column; names absent from *data* are dropped (so a layer
    missing some faceting variables simply contributes fewer columns).

    Parameters
    ----------
    facets : sequence of str
        Resolved faceting-variable names (e.g. ``["vs", "cyl"]``).
    data : pandas.DataFrame
        A single layer's data frame.
    possible_columns : sequence of str, optional
        The union of all layers' column names (R ``.possible_columns``); used only
        to mirror the R signature -- evaluation here is purely by membership in
        ``data.columns``.

    Returns
    -------
    pandas.DataFrame
        One column per evaluated facet, in *facets* order, length ``len(data)``.
    """
    cols: Dict[str, Any] = {}
    for f in facets:
        if isinstance(data, pd.DataFrame) and f in data.columns:
            cols[f] = data[f].to_numpy()
    return pd.DataFrame(cols, index=range(len(data)) if isinstance(data, pd.DataFrame) else None)


def _join_keys(x: pd.DataFrame, y: pd.DataFrame, by: Sequence[str]) -> Dict[str, np.ndarray]:
    """Compute matchable integer keys for two frames over shared columns.

    Port of ggplot2's ``join_keys`` (``borrowed_ggplot2.R:429-437``): row-bind
    ``x[by]`` and ``y[by]``, assign a single stable id over the combined frame
    (via :func:`ggh4x._borrowed_ggplot2.id`), then split back into the ``x`` and
    ``y`` id vectors so that equal key-combinations share an id.

    Parameters
    ----------
    x, y : pandas.DataFrame
        Frames sharing the *by* columns (already factor-coerced by the caller).
    by : sequence of str
        The columns to key on.

    Returns
    -------
    dict
        ``{"x": ndarray, "y": ndarray, "n": int}`` -- the per-row ids of ``x`` and
        ``y`` and the total number of distinct combinations.
    """
    by = list(by)
    nx = len(x)
    joint = pd.concat(
        [x[by].reset_index(drop=True), y[by].reset_index(drop=True)],
        ignore_index=True,
    )
    keys = id(joint, drop=True)
    n = int(keys.n)
    return {"x": np.asarray(keys[:nx]), "y": np.asarray(keys[nx:]), "n": n}


def _as_r_character(value: Any) -> Any:
    """Stringify a value the way R's ``as.character`` would for facet labels.

    R coerces a numeric like ``6`` (stored as a double) to ``"6"`` -- not
    ``"6.0"``.  Mirror that: whole-valued floats lose the trailing ``.0``; ``NaN``
    / ``None`` pass through unchanged so downstream factor coercion still sees a
    missing value.

    Parameters
    ----------
    value : Any
        A scalar facet value.

    Returns
    -------
    Any
        The R-style character representation (or the original value if NA).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return str(int(f)) if f.is_integer() else repr(f)
    return str(value)


def _as_factor_addNA(series: Any) -> pd.Categorical:
    """Coerce a column to a factor (a present NA handled downstream by ``id``).

    Mirrors R ``addNA(as.factor(x), ifany = TRUE)`` for key matching.  The
    :func:`ggh4x._borrowed_ggplot2.id` used by :func:`_join_keys` already ports
    ``addNA(ifany = TRUE)`` (it gives a present NA its own extra level), so this
    only needs to produce a clean :class:`pandas.Categorical`, preserving an
    existing categorical's level order.

    Parameters
    ----------
    series : pandas.Series or array-like
        A facet-value column.

    Returns
    -------
    pandas.Categorical
        The factor; a present NA becomes a matchable extra level via ``id``.
    """
    if isinstance(series, pd.Categorical):
        return series
    if isinstance(series, pd.Series) and isinstance(series.dtype, pd.CategoricalDtype):
        return pd.Categorical(series)
    return pd.Categorical(series)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------
def facet_nested(
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
    nest_line: Any = None,
    solo_line: bool = False,
    resect: Any = None,
    render_empty: bool = True,
    strip: Any = strip_nested,
    bleed: Optional[bool] = None,
) -> "FacetNested":
    """Layout panels in a grid with nested strips.

    Port of ggh4x's ``facet_nested()`` (``R/facet_nested.R:60-120``).  Inherits the
    capabilities of :func:`ggh4x.facet_grid2` and adds label-merged nested strips
    plus hierarchy ``nest_line``s.  Unlike ``facet_grid()`` it only auto-expands a
    missing faceting variable when there is no variable in that direction
    (allowing partially faceted layers); at least one layer must still contain all
    faceting variables.

    Parameters
    ----------
    rows, cols : formula / list / dict / None
        Faceting variables for rows and columns.  Variable order encodes the
        hierarchy: the first is the outermost (furthest from the panels).
    scales : {"fixed", "free_x", "free_y", "free"} or bool, default "fixed"
    space : {"fixed", "free_x", "free_y", "free"} or bool, default "fixed"
    axes : {"margins", "x", "y", "all"} or bool, default "margins"
    remove_labels : {"none", "x", "y", "all"} or bool, default "none"
    independent : {"none", "x", "y", "all"} or bool, default "none"
    shrink : bool, default True
    labeller : callable or str, default "label_value"
    as_table : bool, default True
    switch : {"x", "y", "both", None}, default None
    drop : bool, default True
    margins : bool or list of str, default False
    nest_line : ElementLine / ElementBlank / bool / None, default None
        The hierarchy line element.  ``None`` is treated as R's default
        ``element_line(inherit_blank=True)`` (so it inherits the (blank) theme
        element ``ggh4x.facet.nestline`` and draws nothing unless the theme turns
        it on).  ``True`` -> ``element_line()``; ``False`` -> ``element_blank()``.
    solo_line : bool, default False
        Draw nest lines on single-child parent strips too (``True``) or only on
        multi-child parents (``False``).
    resect : Unit or None, default None
        How much to shorten each nest line at both ends.  ``None`` -> ``0 mm``.
    render_empty : bool, default True
    strip : Strip or callable or str, default :func:`ggh4x.strip_nested.strip_nested`
        The strip specification (defaults to the nested label-merging strip).
    bleed : bool or None, default None
        Deprecated.  When given, emits a ``DeprecationWarning`` and forwards to the
        resolved strip's ``bleed`` param (set it via ``strip_nested(bleed=...)``).

    Returns
    -------
    FacetNested
        A ggproto facet object that can be added to a plot.
    """
    from ggh4x.strip_vanilla import resolve_strip

    strip = resolve_strip(strip)
    if bleed is not None:
        warnings.warn(
            "The `bleed` argument of `facet_nested()` is deprecated as of ggh4x "
            "0.2.0. The `bleed` argument should be set in the `strip_nested()` "
            "function instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        strip.params["bleed"] = bool(bleed)

    nest_line = _coerce_nest_line(nest_line)
    if resect is None:
        resect = Unit(0, "mm")

    params = {
        "nest_line": nest_line,
        "solo_line": bool(solo_line),
        "resect": resect,
    }

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
        params=params,
        super_=FacetNested,
    )


def _coerce_nest_line(nest_line: Any) -> Any:
    """Normalise the ``nest_line`` argument to an element (R ``facet_nested.R:91-104``).

    ``None`` reproduces R's constructor default ``element_line(inherit_blank =
    TRUE)``; ``True`` -> ``element_line()``; ``False`` -> ``element_blank()``.  Any
    other value must be an :class:`ElementLine` or :class:`ElementBlank`, else an
    error is raised.

    Parameters
    ----------
    nest_line : ElementLine / ElementBlank / bool / None
        The user-supplied nest-line argument.

    Returns
    -------
    Element
        A validated line / blank element.
    """
    if nest_line is None:
        return element_line(inherit_blank=True)
    if nest_line is True:
        return element_line()
    if nest_line is False:
        return element_blank()
    if not (
        is_theme_element(nest_line, "line") or is_theme_element(nest_line, "blank")
    ):
        cli_abort(
            "The `nest_line` argument must be `element_blank` or inherit from "
            "`element_line`."
        )
    return nest_line


# ---------------------------------------------------------------------------
# ggproto
# ---------------------------------------------------------------------------
class FacetNested(FacetGrid2):
    """Nested-strip grid facet ggproto (port of R ``FacetNested``).

    Subclasses :class:`ggh4x.facet_grid2.FacetGrid2`.  Overrides ``map_data`` (the
    per-direction "missing only when no var in that direction" rule),
    ``vars_combine`` (blank-fill absent columns instead of erroring) and
    ``finish_panels`` (draw nest indicator lines).

    Attributes
    ----------
    shrink : bool
    strip : Strip
        Defaults to a :class:`ggh4x.strip_nested.StripNested`.
    params : dict
        Adds ``nest_line``, ``solo_line``, ``resect`` to the ``FacetGrid2`` params.
    """

    _class_name = "FacetNested"

    # -- map_data -----------------------------------------------------------
    def map_data(
        self,
        data: pd.DataFrame,
        layout: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """Assign data rows to panels with the nesting missing-var rule.

        Port of R ``FacetNested$map_data`` (``facet_nested.R:130-187``).  Differs
        from the stock :meth:`ggplot2_py.facet.FacetGrid.map_data`: a faceting
        variable is only treated as missing (and force-expanded across panels)
        when *none* of the variables in its direction (rows / cols) are present in
        the layer, which is what lets a partially faceted layer nest.

        Parameters
        ----------
        data : pandas.DataFrame
            A single layer's data.
        layout : pandas.DataFrame
            The panel layout (carries ``PANEL`` + the faceting-var columns).
        params : dict
            Facet params (``rows``, ``cols``, ``margins``, ``_possible_columns``).

        Returns
        -------
        pandas.DataFrame
            *data* with an integer/categorical ``PANEL`` column.
        """
        if empty(data):
            out = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()
            out["PANEL"] = pd.Series([], dtype="int64")
            return out

        rows = params.get("rows")
        cols = params.get("cols")
        row_names = _as_name_list(rows)
        col_names = _as_name_list(cols)
        vars_ = row_names + col_names

        if len(vars_) == 0:
            data = data.copy()
            # R: data$PANEL <- layout$PANEL (recycles the single layout PANEL
            # across all data rows when there are no faceting variables).
            panel_vals = list(layout["PANEL"])
            n = len(data)
            if len(panel_vals) and n:
                data["PANEL"] = [panel_vals[i % len(panel_vals)] for i in range(n)]
            else:
                data["PANEL"] = panel_vals
            return data

        possible_columns = params.get("_possible_columns")
        margin_vars = [
            [c for c in row_names if c in data.columns],
            [c for c in col_names if c in data.columns],
        ]

        data = reshape_add_margins(data, margin_vars, params.get("margins", False))
        facet_vals = _eval_facets(vars_, data, possible_columns)

        # Only set as missing if it has no variable in that direction.
        missing_facets: List[str] = []
        if not any(r in facet_vals.columns for r in row_names):
            missing_facets += [r for r in row_names if r not in facet_vals.columns]
        if not any(c in facet_vals.columns for c in col_names):
            missing_facets += [c for c in col_names if c not in facet_vals.columns]

        if len(missing_facets) > 0:
            to_add = layout[missing_facets].drop_duplicates().reset_index(drop=True)
            n_data = len(data)
            n_add = len(to_add)
            # data_rep = rep.int(1:nrow(data), nrow(to_add))  (data fastest)
            data_rep = np.tile(np.arange(n_data), n_add)
            # facet_rep = rep(1:nrow(to_add), each = nrow(data))
            facet_rep = np.repeat(np.arange(n_add), n_data)
            data = data.iloc[data_rep].reset_index(drop=True)
            facet_vals = facet_vals.iloc[data_rep].reset_index(drop=True)
            add_block = to_add.iloc[facet_rep].reset_index(drop=True)
            facet_vals = pd.concat([facet_vals, add_block], axis=1)

        data = data.copy()
        if len(facet_vals) == 0:
            data["PANEL"] = -1
            return data

        # ``by`` = vars that appear in facet_vals (intersection, in vars order).
        by = [v for v in vars_ if v in facet_vals.columns and v in layout.columns]
        # Factor-coerce + addNA on both sides (R: lapply(.., as.factor) then
        # addNA(ifany=TRUE)).  R's ``as.factor`` stringifies levels, so numeric
        # facet values (``6``) on one side match the blank-filled character
        # values (``"6"`` / ``""``) on the other -- normalise via
        # :func:`_as_r_character` before factoring so both key columns align.
        fv = facet_vals.copy()
        lay = layout.copy()
        for c in by:
            fv[c] = _as_factor_addNA(fv[c].map(_as_r_character))
            lay[c] = _as_factor_addNA(lay[c].map(_as_r_character))
        keys = _join_keys(fv, lay, by=by)
        # PANEL = layout$PANEL[match(keys$x, keys$y)]
        panel_lookup: Dict[int, Any] = {}
        layout_panel = list(layout["PANEL"])
        for j, ky in enumerate(keys["y"]):
            panel_lookup.setdefault(int(ky), layout_panel[j])
        matched = [panel_lookup.get(int(kx)) for kx in keys["x"]]
        data["PANEL"] = matched
        return data

    # -- vars_combine -------------------------------------------------------
    def vars_combine(
        self,
        data: List[pd.DataFrame],
        env: Any = None,
        vars_: Any = None,
        drop: bool = True,
    ) -> pd.DataFrame:
        """Combine faceting variables, blank-filling layers missing some vars.

        Port of R ``FacetNested$vars_combine`` (``facet_nested.R:188-234``).
        Builds the cross-product base from layers that provide *all* requested
        variables (at least one must), then for each partial layer appends the
        grid of its present-variable values against the base's other columns with
        those absent columns set to the empty string ``""`` (line 227 -- the
        divergence from vanilla ``combine_vars``).

        Parameters
        ----------
        data : list of DataFrame
            The plot + layer data frames.
        env : Any, optional
            Unused (kept for R signature parity).
        vars_ : dict or list of str
            The faceting-variable names for this direction.
        drop : bool, default True
            Drop unused factor combinations.

        Returns
        -------
        pandas.DataFrame
            Unique combinations (with blank-filled partial-layer rows).
        """
        names = _as_name_list(vars_)
        if len(names) == 0:
            return pd.DataFrame()

        possible_columns = sorted(
            {c for df in data if isinstance(df, pd.DataFrame) for c in df.columns}
        )

        values: List[pd.DataFrame] = []
        for df in data:
            v = _eval_facets(names, df, possible_columns)
            if v.shape[1] > 0:
                values.append(v)

        has_all = [v.shape[1] == len(names) for v in values]
        if not any(has_all):
            missing_per = []
            for v in values:
                missing_per.append([n for n in names if n not in v.columns])
            detail = "; ".join(
                f"layer {i} is missing {m}" for i, m in enumerate(missing_per)
            )
            cli_abort(
                "At least one layer must contain all faceting variables: "
                f"{names}. {detail}"
            )

        base = (
            pd.concat([v for v, h in zip(values, has_all) if h], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        base = base[names]
        if not drop:
            base = unique_combs(base)

        for v, h in zip(values, has_all):
            if h or empty(v):
                continue
            present = [c for c in base.columns if c in v.columns]
            absent = [c for c in base.columns if c not in v.columns]
            # new = unique(value[intersect(names(base), names(value))]) (R:222).
            new = v[present].drop_duplicates().reset_index(drop=True)
            if drop:
                new = unique_combs(new)
            # old = base[setdiff(names(base), names(value))] -- ALL base rows of
            # the absent columns (NOT deduplicated, R:221), blank-filled to ""
            # (R:227).  R's assignment also coerces the whole column to character;
            # cast the absent columns of ``base`` to object so the rbind below
            # yields a uniform string column (matching R's coercion).
            old = base[absent].reset_index(drop=True)
            for c in absent:
                old[c] = ""
                # R's ``old[...] <- ""`` coerces the whole column to character,
                # which then propagates to ``base`` through ``rbind``; mirror that
                # by stringifying the existing (e.g. numeric) ``base`` values so
                # the concatenated column is a uniform character vector.
                base[c] = base[c].map(_as_r_character)
            grid = df_grid(old, new)
            base = pd.concat([base, grid[base.columns]], ignore_index=True)

        # R does NOT deduplicate after the partial-layer rbind (facet_nested.R:228);
        # the downstream ``compute_layout`` collapses duplicates.  Keep parity.
        base = base.reset_index(drop=True)
        if empty(base):
            cli_abort("Facetting variables must have at least one value.")
        return base

    # -- finish_panels ------------------------------------------------------
    def finish_panels(
        self,
        panels: Any,
        layout: pd.DataFrame,
        params: Dict[str, Any],
        theme: Any,
    ) -> Any:
        """Draw the nest indicator lines onto the assembled panel table.

        Port of R ``FacetNested$finish_panels`` (``facet_nested.R:236-238``);
        delegates to :func:`add_nest_indicator`.

        Parameters
        ----------
        panels : Gtable
            The assembled panel gtable.
        layout : pandas.DataFrame
        params : dict
        theme : Theme

        Returns
        -------
        Gtable
            The panel gtable with nest-line ``"nester"`` grobs added.
        """
        return add_nest_indicator(panels, params, theme)


# ---------------------------------------------------------------------------
# Nest-indicator helper (shared by FacetNested + FacetNestedWrap)
# ---------------------------------------------------------------------------
def _layout_df(table: Any) -> pd.DataFrame:
    """Return a gtable layout (dict-of-lists) as a DataFrame with a 1-based index.

    Adds an ``index`` column equal to the 1-based position of each layout entry
    (R ``layout$index <- seq_len(nrow(layout))``), so the positional index can be
    carried through the ``startswith("strip-")`` filter back to ``panels.grobs`` /
    ``panels.layout`` (which remain dict-of-lists / list and are mutated in place).

    Parameters
    ----------
    table : Gtable

    Returns
    -------
    pandas.DataFrame
        The layout with an added 1-based integer ``index`` column.
    """
    lay = table.layout
    df = pd.DataFrame({k: list(v) for k, v in lay.items()})
    df["index"] = np.arange(1, len(df) + 1)
    return df


def add_nest_indicator(panels: Any, params: Dict[str, Any], theme: Any) -> Any:
    """Draw hierarchy nest lines between strip layers of an assembled panel table.

    Faithful port of ggh4x's ``add_nest_indicator`` (``facet_nested.R:243-354``).
    Resolves the ``nest_line`` element (inheriting from the theme's
    ``ggh4x.facet.nestline``); returns *panels* unchanged when it is ``None`` /
    ``False`` / blank.  Otherwise, for the horizontal (top/bottom) and vertical
    (left/right) strips it draws a shortened polyline (``resect``) along the inner
    edge of every *parent* (multi-child, ``l != r`` / ``t != b``) strip -- or, when
    ``solo_line`` is set, every strip except the innermost layer -- by adding a
    ``"nester"`` grob onto that strip's sub-gtable.  It then bumps
    ``panels.layout['z']`` by the exact z-offset (R lines 300-307 / 343-350) so the
    line-carrying strip layer paints above the lower strip layers.

    Parameters
    ----------
    panels : Gtable
        The assembled panel gtable; its ``strip-*`` cells must be per-layer
        sub-gtables (as produced by the nested strip subsystem).
    params : dict
        Facet params; reads ``nest_line``, ``solo_line``, ``resect``.
    theme : Theme
        The resolved plot theme (for ``ggh4x.facet.nestline``).

    Returns
    -------
    Gtable
        *panels* with the nest-line grobs added and z-order adjusted.
    """
    nest_line = params.get("nest_line")
    if nest_line is None or nest_line is False:
        return panels
    nest_line = combine_elements(nest_line, calc_element("ggh4x.facet.nestline", theme))
    if is_theme_element(nest_line, "blank") or isinstance(nest_line, ElementBlank):
        return panels
    solo = bool(params.get("solo_line"))

    # Locate strips (1-based ``index`` carried through the filter).
    layout = _layout_df(panels)
    names = layout["name"].astype(str)
    is_strip = names.str.startswith("strip-")
    layout = layout[is_strip].reset_index(drop=True)

    resect = params.get("resect")
    if resect is None:
        resect = Unit(0, "mm")
    # active = unit(c(0, 1), "npc") + c(1, -1) * resect
    active = Unit([0, 1], "npc") + unit_c(1.0 * resect, -1.0 * resect)

    # -- Horizontal (top/bottom) strips -------------------------------------
    h_strip = layout
    if not solo:
        h_strip = h_strip[h_strip["l"] != h_strip["r"]]
    else:
        hn = h_strip["name"].astype(str)
        h_strip = h_strip[hn.str.startswith("strip-b") | hn.str.startswith("strip-t")]
    if len(h_strip) > 0:
        index = [int(i) for i in h_strip["index"]]
        is_secondary = bool(
            h_strip["name"].astype(str).str.startswith("strip-b").any()
        )
        passive = [float(is_secondary), float(is_secondary)]
        indicator = element_grob(
            nest_line, x=active, y=passive, default_units="npc"
        )

        if solo:
            kept: List[int] = []
            for idx in index:
                gt = panels.grobs[idx - 1]
                pos = 1 if is_secondary else int(gt.shape[0])
                if int(gt.layout["t"][0]) != pos:
                    kept.append(idx)
            index = kept

        for idx in index:
            gt = panels.grobs[idx - 1]
            s = {k: gt.layout[k][0] for k in ("t", "l", "r", "b", "z")}
            gt = gtable_add_grob(
                gt,
                indicator,
                t=int(s["t"]),
                l=int(s["l"]),
                b=int(s["b"]),
                r=int(s["r"]),
                z=s["z"],
                name="nester",
                clip="off",
            )
            panels.grobs[idx - 1] = gt

        if index:
            offset = [int(panels.grobs[idx - 1].layout["t"][0]) for idx in index]
            if not is_secondary:
                nlevels = int(panels.grobs[index[0] - 1].shape[0])
                offset = [nlevels - o for o in offset]
            z_list = panels.layout["z"]
            for idx, off in zip(index, offset):
                z_list[idx - 1] = z_list[idx - 1] + off

    # -- Vertical (left/right) strips ---------------------------------------
    v_strip = layout
    if not solo:
        v_strip = v_strip[v_strip["t"] != v_strip["b"]]
    else:
        vn = v_strip["name"].astype(str)
        v_strip = v_strip[vn.str.startswith("strip-r") | vn.str.startswith("strip-l")]
    if len(v_strip) > 0:
        index = [int(i) for i in v_strip["index"]]
        is_secondary = bool(
            v_strip["name"].astype(str).str.startswith("strip-r").any()
        )
        passive = [float(not is_secondary), float(not is_secondary)]
        indicator = element_grob(
            nest_line, x=passive, y=active, default_units="npc"
        )

        if solo:
            kept = []
            for idx in index:
                gt = panels.grobs[idx - 1]
                pos = 1 if is_secondary else int(gt.shape[1])
                if int(gt.layout["l"][0]) != pos:
                    kept.append(idx)
            index = kept

        for idx in index:
            gt = panels.grobs[idx - 1]
            s = {k: gt.layout[k][0] for k in ("t", "l", "r", "b", "z")}
            gt = gtable_add_grob(
                gt,
                indicator,
                t=int(s["t"]),
                l=int(s["l"]),
                b=int(s["b"]),
                r=int(s["r"]),
                z=s["z"],
                name="nester",
                clip="off",
            )
            panels.grobs[idx - 1] = gt

        if index:
            offset = [int(panels.grobs[idx - 1].layout["l"][0]) for idx in index]
            if not is_secondary:
                nlevels = int(panels.grobs[index[0] - 1].shape[1])
                offset = [nlevels - o for o in offset]
            z_list = panels.layout["z"]
            for idx, off in zip(index, offset):
                z_list[idx - 1] = z_list[idx - 1] + off

    return panels
