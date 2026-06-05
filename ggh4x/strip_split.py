"""Split strips for ggh4x facets (port of ggh4x ``strip_split.R``).

This module ports :class:`StripSplit` and the :func:`strip_split` constructor.
Split strips let each faceting variable be placed on a different side of the
panel (``"top"`` / ``"bottom"`` / ``"left"`` / ``"right"``), overruling
``strip.position`` / ``switch``.  A single-variable strip is spanned across the
panels that share its value.

``StripSplit`` extends :class:`ggh4x.strip_nested.StripNested` (so it inherits
the RLE-merge ``assemble_strip`` / ``finish_strip``).  It overrides:

* :meth:`StripSplit.setup` -- does *not* separate cols from rows; builds one
  ``vars`` frame from ``union(rows, cols)`` (grid) or ``facets`` (wrap), de-dups
  the layout, and calls its own single-``vars`` :meth:`get_strips` signature.
* :meth:`StripSplit.get_strips` -- per-side strip construction driven by
  ``params["position"]``; builds an :func:`ggh4x._borrowed_ggplot2.id`
  composite-key hierarchy and spans single-variable strips.
* :meth:`StripSplit.incorporate_grid` -- four independent side-blocks placing
  each side's strips into the panel gtable.
* :meth:`StripSplit.incorporate_wrap` -- trivial delegation to
  :meth:`incorporate_grid`.

R source: ``ggh4x/R/strip_split.R``.

Notes
-----
* **id() composite keys.**  R uses the ggplot2-internal ``id()`` to build a
  per-variable integer key table; this port reuses
  :func:`ggh4x._borrowed_ggplot2.id` (verified to match R's lexicographic,
  level-aware codes).  The ids drive both the ``!duplicated(ids)`` selection and
  the ``split(layout$ROW/COL, ids)`` span extension.
* **Span logic.**  When a side has a single variable and every strip is a single
  panel (``all(strp.t == strp.b)`` for x, ``all(strp.l == strp.r)`` for y), the
  bottom/right edge is extended to the panel at the maximum ROW/COL within each
  id group (``vapply(split(...), max)`` then ``match`` back to a ``PANEL``).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ggplot2_py.ggproto import ggproto
from gtable_py import gtable_add_cols, gtable_add_grob, gtable_add_rows

from ggh4x._borrowed_ggplot2 import empty, id
from ggh4x._cli import cli_abort
from ggh4x._facet_utils import split_heights_cm, split_widths_cm
from ggh4x._rlang import arg_match0
from ggh4x.strip_nested import StripNested
from ggh4x.strip_vanilla import (
    _format_labels,
    _panel_layout,
    validate_element_list,
)

__all__ = ["StripSplit", "strip_split"]


def _arg_match_multiple(
    arg: Sequence[str],
    values: Sequence[str],
    arg_name: str = "arg",
) -> List[str]:
    """Port of R ``rlang::arg_match(arg, values, multiple = TRUE)``.

    Validates every element of *arg* against the allowed *values*, aborting on
    the first invalid element.

    Parameters
    ----------
    arg : sequence of str
        The supplied vector of choices.
    values : sequence of str
        The allowed values.
    arg_name : str, default ``"arg"``
        Argument name (for the error message).

    Returns
    -------
    list of str
        *arg* as a list when every element is valid.

    Raises
    ------
    ValueError
        When any element of *arg* is not among *values*.
    """
    out = list(arg)
    for x in out:
        if x not in values:
            choices = ", ".join(repr(v) for v in values)
            cli_abort(f"`{arg_name}` must be one of {choices}, not {x!r}.")
    return out


def _match_first(values: Sequence[Any], table: Sequence[Any]) -> List[int]:
    """Port of R ``match(values, table)`` (1-based first-occurrence index).

    Parameters
    ----------
    values : sequence
        Values to look up.
    table : sequence
        The lookup table.

    Returns
    -------
    list of int
        For each value, the 1-based index of its first occurrence in *table*
        (0 when absent, mirroring how the result is only ever used as a valid
        index here).
    """
    lookup: Dict[Any, int] = {}
    for i, t in enumerate(table):
        if t not in lookup:
            lookup[t] = i + 1
    return [lookup.get(v, 0) for v in values]


def _split_max(values: Sequence[Any], keys: Sequence[Any]) -> List[Any]:
    """Port of R ``vapply(split(values, keys), max, integer(1))``.

    Groups *values* by *keys* (sorted key levels, as R ``split`` does) and
    returns the per-group maximum.

    Parameters
    ----------
    values : sequence
        Values to group and reduce.
    keys : sequence
        Grouping key per value.

    Returns
    -------
    list
        The maximum of each group, in sorted-key order.
    """
    order = sorted(set(keys))
    groups: Dict[Any, List[Any]] = {k: [] for k in order}
    for v, k in zip(values, keys):
        groups[k].append(v)
    return [max(groups[k]) for k in order]


class StripSplit(StripNested):
    """Strip that places different faceting variables on different sides.

    Subclass of :class:`ggh4x.strip_nested.StripNested`.  See the module
    docstring for the per-side / id-span algorithm.

    Attributes
    ----------
    params : dict
        Adds ``position`` (a list of side names) to the base params.
    """

    _class_name = "StripSplit"

    def setup(
        self,
        layout: pd.DataFrame,
        params: Dict[str, Any],
        theme: Any,
        type: str,
    ) -> None:
        """Build a single ``vars`` frame (cols+rows) and delegate to get_strips.

        Port of R ``StripSplit$setup`` (``strip_split.R:110-135``).  Unlike the
        base, split strips do not separate column from row variables.

        Parameters
        ----------
        layout : pandas.DataFrame
            The facet layout.
        params : dict
            Facet params (``facets`` for wrap; ``rows`` / ``cols`` for grid; plus
            ``labeller``).
        theme : Theme
            The active theme.
        type : str
            ``"wrap"`` or ``"grid"`` (kept as ``type`` for facet compatibility).
        """
        self._set(elements=self.setup_elements(theme, type))

        if type == "wrap":
            facets = params.get("facets") or {}
            facet_names = list(facets.keys()) if hasattr(facets, "keys") else list(facets)
            if len(facet_names) == 0:
                vars_frame = pd.DataFrame({"(all)": ["(all)"]})
                layout_sel = layout
            else:
                vars_frame = layout[facet_names].reset_index(drop=True)
                layout_sel = layout
        else:
            row_names = _names(params.get("rows"))
            col_names = _names(params.get("cols"))
            # R: union(names(rows), names(cols)) -- order preserved, dedup.
            var_names: List[str] = []
            for nm in row_names + col_names:
                if nm not in var_names:
                    var_names.append(nm)
            mask = _not_duplicated(layout, var_names)
            layout_sel = layout.loc[mask]
            vars_frame = layout_sel[var_names] if var_names else _empty_frame(layout_sel)

        self.get_strips(
            vars=vars_frame,
            labeller=params.get("labeller"),
            theme=theme,
            params=self.params,
            layout=layout_sel,
        )

    def get_strips(  # type: ignore[override]
        self,
        vars: Any = None,
        labeller: Any = None,
        theme: Any = None,
        params: Optional[Dict[str, Any]] = None,
        layout: Optional[pd.DataFrame] = None,
    ) -> None:
        """Construct the per-side strips and span single-variable strips.

        Port of R ``StripSplit$get_strips`` (``strip_split.R:137-213``).  Note
        the signature differs from the base ``get_strips`` (a single ``vars``
        frame and ``layout``, not separate x / y).

        Parameters
        ----------
        vars : pandas.DataFrame
            The de-duplicated variable frame (columns = faceting variables).
        labeller : callable or str
            Labeller spec.
        theme : Theme
            Active theme.
        params : dict
            Strip params (carries ``position``).
        layout : pandas.DataFrame
            The de-duplicated layout (carries ``PANEL`` / ``ROW`` / ``COL``).
        """
        if empty(vars):
            self._set(
                strips={
                    "x": {"top": None, "bottom": None},
                    "y": {"left": None, "right": None},
                }
            )
            return

        positions = list(params["position"])
        elem = self.elements
        ncol_vars = vars.shape[1]
        var_cols = list(vars.columns)

        # Recycle position to the number of facet variables (with a warning).
        if len(positions) != ncol_vars:
            warnings.warn(
                "The `position` argument in `strip_split()` is being recycled "
                "to match the length of the facetting variables, as provided in "
                "the `facets`, `rows`, or `cols` arguments in the facet function.",
                stacklevel=2,
            )
            positions = [positions[i % len(positions)] for i in range(ncol_vars)]

        # id() composite-key table controlling the strip hierarchy.  R
        # (strip_split.R:167-168): ids[[k]] <- id(vars[, 1:k]) — a CUMULATIVE
        # composite of columns 1..k, not just column k.  Using the single
        # column k mis-merges strips whenever a non-first facet variable sits
        # on its own side.
        ids = pd.DataFrame(
            {
                var_cols[i]: np.asarray(id(vars[var_cols[: i + 1]]), dtype=int)
                for i in range(ncol_vars)
            }
        )

        layout = layout.reset_index(drop=True)
        ids = ids.reset_index(drop=True)
        vars = vars.reset_index(drop=True)

        result: Dict[str, Any] = {}
        for pos in ("top", "bottom", "left", "right"):
            if pos not in positions:
                result[pos] = None
                continue

            # Variables assigned to this side.
            cn = [var_cols[i] for i in range(ncol_vars) if positions[i] == pos]
            side_id_cols = [var_cols[i] for i in range(ncol_vars) if positions[i] == pos]

            # De-duplicate by the composite id of this side's variables.
            keep = _not_duplicated(ids[side_id_cols], side_id_cols)
            lay_sel = layout.loc[keep].reset_index(drop=True)

            # Format labels for the selected variables.
            sub = lay_sel[cn].reset_index(drop=True)
            lab = _format_labels(sub, labeller)
            if pos == "right":
                lab = lab[:, ::-1]

            strp = self.assemble_strip(lab, pos, elem, params, lay_sel)

            # Span single-variable strips across panels.
            if len(cn) == 1:
                col = cn[0]
                key = list(ids[col])
                t = [int(v) for v in strp["t"]]
                b = [int(v) for v in strp["b"]]
                l = [int(v) for v in strp["l"]]
                r = [int(v) for v in strp["r"]]
                if all(tt == bb for tt, bb in zip(t, b)):
                    max_row = _split_max(list(layout["ROW"]), key)
                    panel_at = _match_first(max_row, list(layout["ROW"]))
                    panel_vals = list(layout["PANEL"])
                    strp = strp.copy()
                    strp["b"] = [int(panel_vals[p - 1]) for p in panel_at]
                if all(ll == rr for ll, rr in zip(l, r)):
                    max_col = _split_max(list(layout["COL"]), key)
                    panel_at = _match_first(max_col, list(layout["COL"]))
                    panel_vals = list(layout["PANEL"])
                    strp = strp.copy()
                    strp["r"] = [int(panel_vals[p - 1]) for p in panel_at]
            result[pos] = strp

        self._set(
            strips={
                "x": {"top": result["top"], "bottom": result["bottom"]},
                "y": {"left": result["left"], "right": result["right"]},
            }
        )

    def incorporate_grid(self, panels: Any, switch: Any) -> Any:
        """Place all four sides independently into the panel gtable.

        Port of R ``StripSplit$incorporate_grid`` (``strip_split.R:215-331``).
        The four side-blocks re-query the panel-cell layout between insertions
        because ``gtable_add_rows`` / ``gtable_add_cols`` shift indices.

        Parameters
        ----------
        panels : Gtable
            The assembled panel gtable.
        switch : Any
            Unused (split strips overrule ``switch``).

        Returns
        -------
        Gtable
            The panel gtable with all sides' strips inserted.
        """
        inside = self.elements["inside"]
        padding = self.elements["padding"]
        strips = self.strips

        # --- top ----------------------------------------------------------
        pos_cols = _panel_layout(panels)
        side = strips["x"]["top"]
        if side is not None:
            strip = list(side["grobs"])
            tbl = _tlbr(side)
            names = ["strip-t-" + str(i + 1) for i in range(len(strip))]
            stripheight = split_heights_cm(strip, tbl["t"])
            where = [pos_cols["t"][ti] - 1 for ti in tbl["t"]]
            if not inside["x"]:
                where = [w - 1 for w in where]
                for w in sorted(set(where), reverse=True):
                    panels = gtable_add_rows(panels, padding, w)
                uniq = _unique(where)
                where = [w + _match_first([w], uniq)[0] - 1 for w in where]
            for w in sorted(set(where), reverse=True):
                idx = [i for i, ww in enumerate(where) if ww == w]
                panels = gtable_add_rows(panels, _unit_at(stripheight, idx[0]), w)
                panels = gtable_add_grob(
                    panels,
                    [strip[i] for i in idx],
                    name=[names[i] for i in idx],
                    t=[where[i] + 1 for i in idx],
                    l=[pos_cols["l"][tbl["l"][i]] for i in idx],
                    r=[pos_cols["r"][tbl["r"][i]] for i in idx],
                    clip="on",
                    z=2,
                )

        # --- bottom -------------------------------------------------------
        pos_cols = _panel_layout(panels)
        side = strips["x"]["bottom"]
        if side is not None:
            strip = list(side["grobs"])
            tbl = _tlbr(side)
            names = ["strip-b-" + str(i + 1) for i in range(len(strip))]
            stripheight = split_heights_cm(strip, tbl["t"])
            where = [pos_cols["b"][bi] for bi in tbl["b"]]
            if not inside["x"]:
                where = [w + 1 for w in where]
                for w in sorted(set(where), reverse=True):
                    panels = gtable_add_rows(panels, padding, w)
                uniq = _unique(where)
                where = [w + _match_first([w], uniq)[0] for w in where]
            for w in sorted(set(where), reverse=True):
                idx = [i for i, ww in enumerate(where) if ww == w]
                panels = gtable_add_rows(panels, _unit_at(stripheight, idx[0]), w)
                panels = gtable_add_grob(
                    panels,
                    [strip[i] for i in idx],
                    name=[names[i] for i in idx],
                    t=[where[i] + 1 for i in idx],
                    l=[pos_cols["l"][tbl["l"][i]] for i in idx],
                    r=[pos_cols["r"][tbl["r"][i]] for i in idx],
                    clip="on",
                    z=2,
                )

        # --- left ---------------------------------------------------------
        pos_rows = _panel_layout(panels)
        side = strips["y"]["left"]
        if side is not None:
            strip = list(side["grobs"])
            tbl = _tlbr(side)
            names = ["strip-l-" + str(i + 1) for i in range(len(strip))]
            stripwidth = split_widths_cm(strip, tbl["l"])
            where = [pos_rows["l"][li] - 2 for li in tbl["l"]]
            if not inside["y"]:
                for w in _unique(where):
                    panels = gtable_add_cols(panels, padding, w)
                uniq = _unique(where)
                where = [w + _match_first([w], uniq)[0] - 1 for w in where]
            for w in sorted(set(where), reverse=True):
                idx = [i for i, ww in enumerate(where) if ww == w]
                panels = gtable_add_cols(panels, _unit_at(stripwidth, idx[0]), w)
                panels = gtable_add_grob(
                    panels,
                    [strip[i] for i in idx],
                    name=[names[i] for i in idx],
                    t=[pos_rows["t"][tbl["t"][i]] for i in idx],
                    b=[pos_rows["b"][tbl["b"][i]] for i in idx],
                    l=[where[i] + 1 for i in idx],
                    clip="on",
                    z=2,
                )

        # --- right --------------------------------------------------------
        pos_rows = _panel_layout(panels)
        side = strips["y"]["right"]
        if side is not None:
            strip = list(side["grobs"])
            tbl = _tlbr(side)
            names = ["strip-r-" + str(i + 1) for i in range(len(strip))]
            stripwidth = split_widths_cm(strip, tbl["r"])
            where = [pos_rows["r"][ri] for ri in tbl["r"]]
            if not inside["y"]:
                where = [w + 1 for w in where]
                for w in sorted(set(where), reverse=True):
                    panels = gtable_add_cols(panels, padding, w)
                uniq = _unique(where)
                where = [w + _match_first([w], uniq)[0] for w in where]
            for w in sorted(set(where), reverse=True):
                idx = [i for i, ww in enumerate(where) if ww == w]
                panels = gtable_add_cols(panels, _unit_at(stripwidth, idx[0]), w)
                panels = gtable_add_grob(
                    panels,
                    [strip[i] for i in idx],
                    name=[names[i] for i in idx],
                    t=[pos_rows["t"][tbl["t"][i]] for i in idx],
                    b=[pos_rows["b"][tbl["b"][i]] for i in idx],
                    l=[where[i] + 1 for i in idx],
                    clip="on",
                    z=2,
                )

        return panels

    def incorporate_wrap(
        self,
        panels: Any,
        position: str,
        clip: str = "off",
        sizes: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Reuse the grid placement algorithm for wrapped facets.

        Port of R ``StripSplit$incorporate_wrap`` (``strip_split.R:333-337``).
        The ``clip`` / ``sizes`` arguments are accepted for facet-call
        compatibility but ignored (only ``panels`` is forwarded).

        Parameters
        ----------
        panels : Gtable
            The assembled panel gtable.
        position : str
            Unused.
        clip : str, default ``"off"``
            Unused.
        sizes : dict, optional
            Unused.

        Returns
        -------
        Gtable
            The panel gtable with all sides' strips inserted.
        """
        return self.incorporate_grid(panels, False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _names(param: Any) -> List[str]:
    """Return the variable names from a facet ``rows`` / ``cols`` param.

    Parameters
    ----------
    param : Any
        A ``rows`` / ``cols`` spec (mapping, list, or ``None``).

    Returns
    -------
    list of str
    """
    if param is None:
        return []
    if hasattr(param, "keys"):
        return list(param.keys())
    if isinstance(param, (list, tuple)):
        return [str(p) for p in param]
    return []


def _empty_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a 0-column frame with the same row count as *df*."""
    return pd.DataFrame(index=df.index)


def _not_duplicated(frame: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    """Port of R ``!duplicated(frame[names])`` (boolean keep-mask).

    Parameters
    ----------
    frame : pandas.DataFrame
        The frame to de-duplicate.
    names : sequence of str
        Columns to de-duplicate on.

    Returns
    -------
    numpy.ndarray of bool
        ``True`` for the first occurrence of each unique combination.
    """
    n = frame.shape[0]
    names = list(names)
    if not names:
        mask = np.zeros(n, dtype=bool)
        if n > 0:
            mask[0] = True
        return mask
    return ~frame[names].duplicated().to_numpy()


def _tlbr(side: pd.DataFrame) -> Dict[str, List[int]]:
    """Return ``{t,l,b,r}`` panel-id lists from a built strip side.

    Parameters
    ----------
    side : pandas.DataFrame
        A built strip placement frame.

    Returns
    -------
    dict
    """
    return {
        "t": [int(v) for v in side["t"]],
        "l": [int(v) for v in side["l"]],
        "b": [int(v) for v in side["b"]],
        "r": [int(v) for v in side["r"]],
    }


def _unique(seq: Sequence[Any]) -> List[Any]:
    """Port of R ``unique()`` -- first-occurrence order preserved.

    Parameters
    ----------
    seq : sequence

    Returns
    -------
    list
    """
    seen: Dict[Any, None] = {}
    for v in seq:
        if v not in seen:
            seen[v] = None
    return list(seen.keys())


def _unit_at(u: Any, i: int) -> Any:
    """Return the single unit at 0-based position *i* (R ``u[i+1]``).

    Parameters
    ----------
    u : grid_py.Unit
        Source unit vector.
    i : int
        0-based index.

    Returns
    -------
    grid_py.Unit
    """
    n = len(u)
    if n <= 1:
        return u
    return u[i % n]


# R's ``StripSplit`` ggproto instance used as the parent of every clone.
_STRIP_SPLIT_SINGLETON: "StripSplit" = StripSplit()


def strip_split(
    position: Any = ("top", "left"),
    clip: str = "inherit",
    size: str = "constant",
    bleed: bool = False,
    text_x: Any = None,
    text_y: Any = None,
    background_x: Any = None,
    background_y: Any = None,
    by_layer_x: bool = False,
    by_layer_y: bool = False,
) -> StripSplit:
    """Create a split strip (per-variable side placement).

    Port of R ``strip_split()`` (``strip_split.R:65-99``).

    Parameters
    ----------
    position : sequence of str, default ``("top", "left")``
        Where each faceting variable's strip is placed; each of ``"top"`` /
        ``"bottom"`` / ``"left"`` / ``"right"``.  Recycled to the number of
        variables (with a warning) when the lengths differ.
    clip : str, default ``"inherit"``
        Whether labels are clipped to background boxes.
    size : str, default ``"constant"``
        Whether strip margins across layers are ``"constant"`` or ``"variable"``.
    bleed : bool, default ``False``
        Whether lower-layer strips may merge across higher-layer boundaries.
    text_x, text_y, background_x, background_y : list or element or None
        Per-strip themed elements (see :func:`ggh4x.strip_themed.strip_themed`).
    by_layer_x, by_layer_y : bool, default ``False``
        Map elements to layers (``True``) or strips (``False``).

    Returns
    -------
    StripSplit
        A ``StripSplit`` ggproto instance usable in ggh4x facets.
    """
    if isinstance(position, str):
        position = [position]
    params = {
        "clip": arg_match0(clip, ["on", "off", "inherit"], arg_name="clip"),
        "size": arg_match0(size, ["constant", "variable"], arg_name="size"),
        "bleed": bool(bleed),
        "position": _arg_match_multiple(
            list(position),
            ["top", "bottom", "left", "right"],
            arg_name="position",
        ),
    }
    given_elements = {
        "text_x": validate_element_list(text_x, "element_text"),
        "text_y": validate_element_list(text_y, "element_text"),
        "background_x": validate_element_list(background_x, "element_rect"),
        "background_y": validate_element_list(background_y, "element_rect"),
        "by_layer_x": bool(by_layer_x),
        "by_layer_y": bool(by_layer_y),
    }
    return ggproto(
        None,
        _STRIP_SPLIT_SINGLETON,
        params=params,
        given_elements=given_elements,
    )
