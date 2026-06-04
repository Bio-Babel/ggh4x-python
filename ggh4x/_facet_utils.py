"""Facet-layout helpers not present in ``gtable_py`` / ``ggplot2_py``.

R sources:

- ``panel_cols`` / ``panel_rows`` — ggplot2 internal (``R/facet-.R``). Locate panel
  cells in an assembled gtable by name and return their unique column / row spans.
- ``weave_tables_row`` / ``weave_tables_col`` — ggplot2 internal, copied into ggh4x
  ``R/borrowed_ggplot2.R`` (matrix-style axis-band weaving for ``facet_grid2`` /
  ``facet_wrap2``).
- ``weave_panel_rows`` / ``weave_panel_cols`` — ggh4x ``R/utils_gtable.R``
  (data-frame-style axis-band weaving for ``facet_manual`` / strips).
- ``split_heights_cm`` / ``split_widths_cm`` — ggh4x ``R/utils_grid.R``.
- ``df_grid`` (R ``df.grid``) — ggh4x ``R/borrowed_ggplot2.R`` (cross-product of two
  data frames, used by ``FacetGrid2.compute_layout``).
- ``render_axes`` — ggplot2 internal batch axis renderer (absent from ``ggplot2_py``).

This module is the shared foundation both ``FacetGrid2`` and ``FacetWrap2`` build on.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from grid_py import Unit, null_grob
from gtable_py import gtable_add_cols, gtable_add_grob, gtable_add_rows
from gtable_py._utils import height_cm, width_cm

__all__ = [
    "panel_cols",
    "panel_rows",
    "render_axes",
    "weave_tables_row",
    "weave_tables_col",
    "weave_panel_rows",
    "weave_panel_cols",
    "split_heights_cm",
    "split_widths_cm",
    "df_grid",
]


def _layout_frame(table: Any) -> pd.DataFrame:
    """Return a gtable's layout as a DataFrame regardless of its native form.

    ``gtable_py`` stores ``Gtable.layout`` as a ``dict`` of parallel lists
    (keys ``t, l, b, r, z, clip, name``); R / some callers use a DataFrame. Normalise.
    """
    lay = table.layout
    if isinstance(lay, pd.DataFrame):
        return lay
    return pd.DataFrame({k: list(v) for k, v in lay.items()})


def panel_cols(table: Any) -> pd.DataFrame:
    """Return the unique ``(l, r)`` column spans of a gtable's panel cells.

    Mirrors ggplot2's internal ``panel_cols``: filter ``table$layout`` to rows whose
    ``name`` begins with ``"panel"``, then take unique ``l``/``r`` pairs in
    first-occurrence order (R ``unique()`` semantics, not sorted).

    Parameters
    ----------
    table : Gtable
        An assembled panel gtable whose panel cells are named ``"panel-*"``.

    Returns
    -------
    pandas.DataFrame
        Columns ``l`` and ``r``; one row per distinct panel column span.
    """
    lay = _layout_frame(table)
    mask = lay["name"].astype(str).str.match(r"^panel")
    return lay.loc[mask, ["l", "r"]].drop_duplicates().reset_index(drop=True)


def panel_rows(table: Any) -> pd.DataFrame:
    """Return the unique ``(t, b)`` row spans of a gtable's panel cells.

    Mirrors ggplot2's internal ``panel_rows`` (see :func:`panel_cols`).

    Parameters
    ----------
    table : Gtable
        An assembled panel gtable whose panel cells are named ``"panel-*"``.

    Returns
    -------
    pandas.DataFrame
        Columns ``t`` and ``b``; one row per distinct panel row span.
    """
    lay = _layout_frame(table)
    mask = lay["name"].astype(str).str.match(r"^panel")
    return lay.loc[mask, ["t", "b"]].drop_duplicates().reset_index(drop=True)


# --- batch axis renderer (ggplot2 internal ``render_axes``) ------------------


def render_axes(
    x_ranges: Optional[Sequence[Any]],
    y_ranges: Optional[Sequence[Any]],
    coord: Any,
    theme: Any,
    transpose: bool = False,
) -> Dict[str, Any]:
    """Render a batch of panel axes, mirroring ggplot2's internal ``render_axes``.

    This is the ggplot2-internal helper ``ggplot2:::render_axes`` which
    ``ggplot2_py`` does not expose. It loops the coord's per-panel axis
    renderers over the supplied panel-parameter (range) lists.

    For every element of ``x_ranges`` it calls ``coord.render_axis_h(pp, theme)``
    (returning ``{"top": grob, "bottom": grob}``) and for every element of
    ``y_ranges`` it calls ``coord.render_axis_v(pp, theme)`` (returning
    ``{"left": grob, "right": grob}``).

    Parameters
    ----------
    x_ranges : sequence of dict or None
        Per-panel ``panel_params`` dicts used to render horizontal (x) axes. If
        ``None``, the horizontal axes are not rendered (see *Notes* for how this
        interacts with ``transpose``).
    y_ranges : sequence of dict or None
        Per-panel ``panel_params`` dicts used to render vertical (y) axes. If
        ``None``, the vertical axes are not rendered.
    coord : Coord
        A ``ggplot2_py`` coordinate system providing ``render_axis_h`` and
        ``render_axis_v`` methods.
    theme : Theme
        The resolved plot theme passed through to the coord's axis renderers.
    transpose : bool, default False
        Controls the shape of the result. When ``False`` the per-panel
        ``{top, bottom}`` / ``{left, right}`` dicts are returned grouped per panel.
        When ``True`` they are transposed to per-side lists, matching the shape
        the ggh4x facets consume.

    Returns
    -------
    dict
        When ``transpose`` is ``False``:

        ``{"x": [{"top": .., "bottom": ..}, ...], "y": [{"left": .., "right": ..}, ...]}``

        — and the ``"x"`` / ``"y"`` keys are only present when the corresponding
        ranges argument is not ``None``.

        When ``transpose`` is ``True``:

        ``{"x": {"top": [...], "bottom": [...]}, "y": {"left": [...], "right": [...]}}``

        — both keys are always present (with empty per-side lists when the
        corresponding ranges argument is ``None``), faithfully reproducing R's
        ``lapply(NULL, ...)`` behaviour.

    Notes
    -----
    R source (``ggplot2:::render_axes``)::

        axes <- list()
        if (!is.null(x)) axes$x <- lapply(x, coord$render_axis_h, theme)
        if (!is.null(y)) axes$y <- lapply(y, coord$render_axis_v, theme)
        if (transpose) {
          axes <- list(
            x = list(top    = lapply(axes$x, `[[`, "top"),
                     bottom = lapply(axes$x, `[[`, "bottom")),
            y = list(left   = lapply(axes$y, `[[`, "left"),
                     right  = lapply(axes$y, `[[`, "right")))
        }
        axes

    The asymmetry between the two modes is deliberate and matches R: in
    non-transposed mode an absent side is simply omitted from the dict, whereas
    in transposed mode the side keys always exist (with empty lists) because the
    transpose block unconditionally rebuilds the nested structure from a possibly
    ``NULL`` ``axes$x`` / ``axes$y``.
    """
    axes: Dict[str, Any] = {}
    if x_ranges is not None:
        axes["x"] = [coord.render_axis_h(pp, theme) for pp in x_ranges]
    if y_ranges is not None:
        axes["y"] = [coord.render_axis_v(pp, theme) for pp in y_ranges]

    if transpose:
        x_list = axes.get("x", [])
        y_list = axes.get("y", [])
        axes = {
            "x": {
                "top": [a["top"] for a in x_list],
                "bottom": [a["bottom"] for a in x_list],
            },
            "y": {
                "left": [a["left"] for a in y_list],
                "right": [a["right"] for a in y_list],
            },
        }
    return axes


# --- matrix-style weaving (ggplot2 internal, ggh4x borrowed_ggplot2.R) -------


def _matrix_col(matrix: Sequence[Sequence[Any]], j: int, nrow: int) -> List[Any]:
    """Extract column ``j`` (0-based) across all rows of a row-major matrix.

    Mirrors R's ``matrix[, j]`` for a grob matrix stored as a list-of-lists in
    row-major order (``matrix[row][col]``).
    """
    return [matrix[r][j] for r in range(nrow)]


def weave_tables_col(
    table: Any,
    table2: Optional[Sequence[Sequence[Any]]] = None,
    col_shift: int = 0,
    col_width: Optional[Unit] = None,
    name: str = "",
    z: float = 1,
    clip: str = "off",
) -> Any:
    """Weave a column of axis grobs into a panel gtable, one per panel column.

    Faithful port of ggplot2's internal ``weave_tables_col`` (copied into ggh4x's
    ``R/borrowed_ggplot2.R``). For each panel column (right-to-left) it inserts a
    new gtable column at ``panel_l + col_shift`` and, if ``table2`` is supplied,
    places that column's grobs (one per panel row) into the freshly inserted
    column.

    Parameters
    ----------
    table : Gtable
        Panel gtable whose panel cells are named ``"panel-*"``.
    table2 : sequence of sequence of grob, optional
        A grob matrix in row-major order (``table2[row][col]``) with one row per
        panel row and one column per panel column. When omitted, only empty
        columns are inserted (no grobs placed).
    col_shift : int, default 0
        Offset relative to each panel column's left index at which the new column
        is inserted (R: ``-1`` for left axes, ``0`` for right axes).
    col_width : Unit, optional
        Widths (length = number of panel columns) for the inserted columns.
    name : str, default ""
        Prefix for the inserted grob names (``"{name}-{row}-{col}"``).
    z : float, default 1
        Drawing order of the inserted grobs (ggh4x uses ``3`` so axes sit above
        panel backgrounds).
    clip : str, default "off"
        Clipping for the inserted grobs.

    Returns
    -------
    Gtable
        ``table`` augmented with the inserted axis columns / grobs.
    """
    panel_col = list(panel_cols(table)["l"])
    panel_row = list(panel_rows(table)["t"])
    nrow = len(panel_row)
    for i in reversed(range(len(panel_col))):
        col_ind = int(panel_col[i]) + col_shift
        table = gtable_add_cols(table, col_width[i], pos=col_ind)
        if table2 is not None:
            grobs = _matrix_col(table2, i, nrow)
            table = gtable_add_grob(
                table,
                grobs,
                t=[int(x) for x in panel_row],
                l=col_ind + 1,
                clip=clip,
                name=[f"{name}-{k + 1}-{i + 1}" for k in range(len(panel_row))],
                z=z,
            )
    return table


def weave_tables_row(
    table: Any,
    table2: Optional[Sequence[Sequence[Any]]] = None,
    row_shift: int = 0,
    row_height: Optional[Unit] = None,
    name: str = "",
    z: float = 1,
    clip: str = "off",
) -> Any:
    """Weave a row of axis grobs into a panel gtable, one per panel row.

    Faithful port of ggplot2's internal ``weave_tables_row`` (copied into ggh4x's
    ``R/borrowed_ggplot2.R``). For each panel row (bottom-to-top) it inserts a new
    gtable row at ``panel_t + row_shift`` and, if ``table2`` is supplied, places
    that row's grobs (one per panel column) into the freshly inserted row.

    Parameters
    ----------
    table : Gtable
        Panel gtable whose panel cells are named ``"panel-*"``.
    table2 : sequence of sequence of grob, optional
        A grob matrix in row-major order (``table2[row][col]``) with one row per
        panel row and one column per panel column. When omitted, only empty rows
        are inserted (no grobs placed).
    row_shift : int, default 0
        Offset relative to each panel row's top index at which the new row is
        inserted (R: ``-1`` for top axes, ``0`` for bottom axes).
    row_height : Unit, optional
        Heights (length = number of panel rows) for the inserted rows.
    name : str, default ""
        Prefix for the inserted grob names (``"{name}-{col}-{row}"``).
    z : float, default 1
        Drawing order of the inserted grobs.
    clip : str, default "off"
        Clipping for the inserted grobs.

    Returns
    -------
    Gtable
        ``table`` augmented with the inserted axis rows / grobs.
    """
    panel_col = list(panel_cols(table)["l"])
    panel_row = list(panel_rows(table)["t"])
    for i in reversed(range(len(panel_row))):
        row_ind = int(panel_row[i]) + row_shift
        table = gtable_add_rows(table, row_height[i], pos=row_ind)
        if table2 is not None:
            grobs = list(table2[i])
            table = gtable_add_grob(
                table,
                grobs,
                t=row_ind + 1,
                l=[int(x) for x in panel_col],
                clip=clip,
                name=[f"{name}-{k + 1}-{i + 1}" for k in range(len(panel_col))],
                z=z,
            )
    return table


# --- data-frame-style weaving (ggh4x utils_gtable.R) ------------------------


def _panel_layout_frame(table: Any) -> pd.DataFrame:
    """Return the ``"panel-*"`` rows of a gtable layout as a DataFrame."""
    lay = _layout_frame(table)
    mask = lay["name"].astype(str).str.match(r"^panel")
    return lay.loc[mask].reset_index(drop=True)


def weave_panel_rows(
    table: Any,
    table2: Optional[pd.DataFrame] = None,
    row_shift: int = 0,
    row_height: Optional[Unit] = None,
    name: str = "",
    z: float = 1,
    clip: str = "off",
    pos: Optional[str] = None,
    grob_var: str = "grobs",
) -> Any:
    """Insert rows into a panel table relative to the panels (ggh4x weave).

    Faithful port of ggh4x's ``weave_panel_rows`` (``R/utils_gtable.R``). Unlike
    :func:`weave_tables_row` (which consumes a grob *matrix*), this consumes a
    *data frame* ``table2`` with integer columns ``t``, ``b``, ``l``, ``r``
    indexing into the (sorted-unique) panel positions, plus a list-column named by
    ``grob_var`` holding the grobs to place. This shape is used by
    ``facet_manual`` and the strip subsystem.

    Parameters
    ----------
    table : Gtable
        Panel gtable whose panel cells are named ``"panel-*"``.
    table2 : pandas.DataFrame, optional
        Frame with integer columns ``t``, ``b``, ``l``, ``r`` (1-based indices
        into the panels) and a list-column ``grob_var``. When omitted, only empty
        rows are inserted.
    row_shift : int, default 0
        Offset relative to the panel position ``pos`` at which to insert rows.
    row_height : Unit, optional
        Heights for the inserted rows (length = number of unique panel rows).
    name : str, default ""
        Prefix for the inserted grob names.
    z : float, default 1
        Drawing order of the inserted grobs.
    clip : str, default "off"
        Clipping for the inserted grobs.
    pos : {"t", "b"} or None, default None
        Which panel edge to index against. ``None`` is interpreted verbatim as
        ``"t"`` with the opposite edge ``"b"`` used for the grob's bottom; when
        given, the same edge is used for both top and bottom of placed grobs.
    grob_var : str, default "grobs"
        Name of the list-column in ``table2`` holding the grobs.

    Returns
    -------
    Gtable
        ``table`` augmented with the inserted rows / grobs.
    """
    if pos is None:
        pos = "t"
        alt = "b"
    else:
        alt = pos

    rows = panel_rows(table)
    rows = sorted(set(int(v) for v in rows[pos]))

    for i in reversed(range(len(rows))):
        table = gtable_add_rows(table, row_height[i], pos=rows[i] + row_shift)

    if table2 is not None:
        if row_shift > -1:
            row_shift = 1 + row_shift
        panels = _panel_layout_frame(table)
        panels["t"] = panels["t"] + row_shift
        panels["b"] = panels["b"] + row_shift

        t_idx = [int(x) for x in table2["t"]]
        b_idx = [int(x) for x in table2["b"]]
        l_idx = [int(x) for x in table2["l"]]
        r_idx = [int(x) for x in table2["r"]]
        n = len(l_idx)

        table = gtable_add_grob(
            table,
            list(table2[grob_var]),
            t=[int(panels[pos].iloc[k - 1]) for k in t_idx],
            b=[int(panels[alt].iloc[k - 1]) for k in b_idx],
            l=[int(panels["l"].iloc[k - 1]) for k in l_idx],
            r=[int(panels["r"].iloc[k - 1]) for k in r_idx],
            clip=clip,
            z=z,
            name=[f"{name}-{k + 1}-{k + 1}" for k in range(n)],
        )
    return table


def weave_panel_cols(
    table: Any,
    table2: Optional[pd.DataFrame] = None,
    col_shift: int = 0,
    col_width: Optional[Unit] = None,
    name: str = "",
    z: float = 1,
    clip: str = "off",
    pos: Optional[str] = None,
    grob_var: str = "grobs",
) -> Any:
    """Insert columns into a panel table relative to the panels (ggh4x weave).

    Faithful port of ggh4x's ``weave_panel_cols`` (``R/utils_gtable.R``); the
    column-wise counterpart of :func:`weave_panel_rows`.

    Parameters
    ----------
    table : Gtable
        Panel gtable whose panel cells are named ``"panel-*"``.
    table2 : pandas.DataFrame, optional
        Frame with integer columns ``t``, ``b``, ``l``, ``r`` (1-based indices
        into the panels) and a list-column ``grob_var``. When omitted, only empty
        columns are inserted.
    col_shift : int, default 0
        Offset relative to the panel position ``pos`` at which to insert columns.
    col_width : Unit, optional
        Widths for the inserted columns (length = number of unique panel cols).
    name : str, default ""
        Prefix for the inserted grob names.
    z : float, default 1
        Drawing order of the inserted grobs.
    clip : str, default "off"
        Clipping for the inserted grobs.
    pos : {"l", "r"} or None, default None
        Which panel edge to index against. ``None`` is interpreted verbatim as
        ``"l"`` with the opposite edge ``"r"`` used for the grob's right edge.
    grob_var : str, default "grobs"
        Name of the list-column in ``table2`` holding the grobs.

    Returns
    -------
    Gtable
        ``table`` augmented with the inserted columns / grobs.
    """
    if pos is None:
        pos = "l"
        alt = "r"
    else:
        alt = pos

    cols = panel_cols(table)
    cols = sorted(set(int(v) for v in cols[pos]))

    for i in reversed(range(len(cols))):
        table = gtable_add_cols(table, col_width[i], pos=cols[i] + col_shift)

    if table2 is not None:
        if col_shift > -1:
            col_shift = 1 + col_shift
        panels = _panel_layout_frame(table)
        panels["l"] = panels["l"] + col_shift
        panels["r"] = panels["r"] + col_shift

        t_idx = [int(x) for x in table2["t"]]
        b_idx = [int(x) for x in table2["b"]]
        l_idx = [int(x) for x in table2["l"]]
        r_idx = [int(x) for x in table2["r"]]
        n = len(t_idx)

        table = gtable_add_grob(
            table,
            list(table2[grob_var]),
            t=[int(panels["t"].iloc[k - 1]) for k in t_idx],
            b=[int(panels["b"].iloc[k - 1]) for k in b_idx],
            l=[int(panels[pos].iloc[k - 1]) for k in l_idx],
            r=[int(panels[alt].iloc[k - 1]) for k in r_idx],
            clip=clip,
            z=z,
            name=[f"{name}-{k + 1}-{k + 1}" for k in range(n)],
        )
    return table


# --- grob-size splitting (ggh4x utils_grid.R) -------------------------------


def _split_max_cm(values: Sequence[float], split: Sequence[Any]) -> List[float]:
    """Group ``values`` by ``split`` and return the max per group.

    Mirrors R's ``vapply(split(values, split, drop = TRUE), max, numeric(1))``:
    groups are ordered by the *sorted unique* keys of ``split`` (or by factor
    level order, with unused levels dropped, when ``split`` is categorical).
    """
    values = list(values)
    if isinstance(split, pd.Categorical):
        # Factor: preserve level order, drop unused levels (drop = TRUE).
        cats = pd.Categorical(split)
        keys = [c for c in cats.categories if (cats == c).any()]
        codes = list(cats)
    elif isinstance(split, pd.Series) and isinstance(split.dtype, pd.CategoricalDtype):
        cats = split.cat
        present = pd.unique(split.dropna())
        keys = [c for c in cats.categories if c in set(present)]
        codes = list(split)
    else:
        codes = list(split)
        # sorted unique of non-NA keys (R's split orders by sort(unique)).
        keys = sorted(set(k for k in codes if not pd.isna(k)))

    out: List[float] = []
    for key in keys:
        group = [v for v, c in zip(values, codes) if c == key]
        out.append(max(group))
    return out


def split_heights_cm(grobs: Sequence[Any], split: Sequence[Any]) -> Unit:
    """Group grobs and report the max height (cm) per group.

    Faithful port of ggh4x's ``split_heights_cm`` (``R/utils_grid.R``): measure
    each grob's height in centimetres (``grobHeight`` -> ``convertHeight``), then
    return, per ``split`` group, the maximum height as a ``"cm"`` :class:`Unit`.

    Parameters
    ----------
    grobs : sequence of grob
        Grobs whose heights are measured.
    split : sequence
        Grouping vector (same length as ``grobs``). Group ordering follows R's
        ``split`` (sorted-unique keys, or factor level order with unused levels
        dropped).

    Returns
    -------
    Unit
        Centimetre heights, one per group, in group order.
    """
    vals = [height_cm(g) for g in grobs]
    out = _split_max_cm(vals, split)
    return Unit(out, "cm")


def split_widths_cm(grobs: Sequence[Any], split: Sequence[Any]) -> Unit:
    """Group grobs and report the max width (cm) per group.

    Faithful port of ggh4x's ``split_widths_cm`` (``R/utils_grid.R``); the
    width-wise counterpart of :func:`split_heights_cm`.

    Parameters
    ----------
    grobs : sequence of grob
        Grobs whose widths are measured.
    split : sequence
        Grouping vector (same length as ``grobs``); see :func:`split_heights_cm`.

    Returns
    -------
    Unit
        Centimetre widths, one per group, in group order.
    """
    vals = [width_cm(g) for g in grobs]
    out = _split_max_cm(vals, split)
    return Unit(out, "cm")


# --- cross-product of data frames (ggh4x df.grid) ---------------------------


def df_grid(a: Optional[pd.DataFrame], b: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Cross-product (Cartesian join) of two data frames.

    Faithful port of ggh4x's ``df.grid`` (``R/borrowed_ggplot2.R``). When either
    frame is ``None`` or empty the other is returned unchanged. Otherwise every
    row of ``a`` is combined with every row of ``b``, with ``a``'s row index
    varying fastest (R's ``expand.grid(i_a, i_b)`` order).

    Parameters
    ----------
    a : pandas.DataFrame or None
        Left frame.
    b : pandas.DataFrame or None
        Right frame.

    Returns
    -------
    pandas.DataFrame
        Column-bound cross-product (``a`` columns followed by ``b`` columns) with
        ``len(a) * len(b)`` rows and a fresh ``RangeIndex``.

    Notes
    -----
    R source::

        df.grid = function(a, b) {
          if (is.null(a) || nrow(a) == 0) return(b)
          if (is.null(b) || nrow(b) == 0) return(a)
          indexes <- expand.grid(i_a = seq_len(nrow(a)), i_b = seq_len(nrow(b)))
          vec_cbind(unrowname(a[indexes$i_a, ]), unrowname(b[indexes$i_b, ]))
        }
    """
    if a is None or len(a) == 0:
        return b
    if b is None or len(b) == 0:
        return a

    na, nb = len(a), len(b)
    # expand.grid(i_a, i_b): i_a cycles fastest -> tile a, repeat b.
    i_a = np.tile(np.arange(na), nb)
    i_b = np.repeat(np.arange(nb), na)

    left = a.iloc[i_a].reset_index(drop=True)
    right = b.iloc[i_b].reset_index(drop=True)
    return pd.concat([left, right], axis=1)
