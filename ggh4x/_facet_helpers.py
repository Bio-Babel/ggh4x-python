"""Shared argument-normalisation helpers for ggh4x extended facets.

Both :mod:`ggh4x.facet_grid2` and :mod:`ggh4x.facet_wrap2` need to translate the
``character(1)`` / ``logical(1)`` facet arguments (``scales``, ``space``, ``axes``,
``remove_labels``, ``independent``) into the ``{"x": bool, "y": bool}`` dicts the
ggproto classes consume, and ``facet_grid2`` additionally validates the
``independent`` interactions.  These live here (rather than in either facet module)
so the two facet modules can import them without importing one another.

R sources:

- ``.match_facet_arg`` — ggh4x ``R/facet_wrap2.R:418-433``.
- ``.validate_independent`` — ggh4x ``R/facet_grid2.R:438-485``.

Also provides the ``AspectRatio`` struct that stands in for R's
``attr(aspect_ratio, "respect")`` (a Python float cannot carry attributes) and the
identity-default ``reshape_add_margins`` used by ``FacetGrid2.compute_layout``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ggh4x._cli import cli_abort, cli_warn
from ggh4x._rlang import arg_match0

__all__ = [
    "AspectRatio",
    "_match_facet_arg",
    "_validate_independent",
    "reshape_add_margins",
]


@dataclass(frozen=True)
class AspectRatio:
    """A panel aspect ratio together with its ``respect`` flag.

    R attaches ``attr(aspect_ratio, "respect")`` onto a bare numeric scalar in
    ``setup_aspect_ratio``; a Python ``float`` cannot carry attributes, so the
    ``(value, respect)`` pair is threaded as this small immutable struct.  The
    consumers (``setup_panel_table``) read :attr:`value` for ``abs(aspect)``
    null-unit heights and :attr:`respect` for the gtable ``respect=`` flag.

    Attributes
    ----------
    value : float
        The aspect ratio (R ``aspect_ratio``).
    respect : bool
        Whether the gtable should respect the aspect (R
        ``attr(aspect_ratio, "respect")``).
    """

    value: float
    respect: bool

    def __float__(self) -> float:  # convenience for ``abs(aspect)`` style use
        return float(self.value)


def _match_facet_arg(
    value: Any,
    options: Sequence[str],
    x: int = 2,
    y: int = 3,
    both: int = 4,
    neither: int = 1,
    nm: str = "value",
) -> Dict[str, bool]:
    """Normalise a facet argument to a ``{"x": bool, "y": bool}`` dict.

    Faithful port of ggh4x's ``.match_facet_arg`` (``R/facet_wrap2.R:418-433``).
    Accepts either a single non-``NA`` logical (``True`` -> the ``both`` option,
    ``False`` -> the ``neither`` option) or one of the option strings (validated
    with :func:`ggh4x._rlang.arg_match0`).  The chosen option then sets the ``x``
    / ``y`` booleans by membership in the ``x``/``both`` and ``y``/``both``
    option positions.

    Parameters
    ----------
    value : bool or str
        The user-supplied argument.  A bool selects ``both``/``neither``; a string
        is matched against *options*.
    options : sequence of str
        The four allowed option strings, ordered ``[neither, x, y, both]`` by
        default (see the ``neither``/``x``/``y``/``both`` index arguments).  The R
        defaults are e.g. ``c("fixed", "free_x", "free_y", "free")``.
    x, y, both, neither : int, default 2, 3, 4, 1
        1-based positions (R indices) into *options* for the ``x``-only,
        ``y``-only, both and neither cases.
    nm : str, default ``"value"``
        Argument name used in the ``arg_match0`` error message.

    Returns
    -------
    dict
        ``{"x": bool, "y": bool}``.

    Notes
    -----
    R source::

        .match_facet_arg <- function(value, options, x = 2, y = 3, both = 4,
                                      neither = 1, nm = deparse(substitute(value))) {
          if (is.logical(value) && length(value) == 1 && !is.na(value)) {
            if (value) value <- options[both] else value <- options[neither]
          } else {
            value <- rlang::arg_match0(value, options, arg_nm = nm)
          }
          list(x = any(value %in% options[c(x, both)]),
               y = any(value %in% options[c(y, both)]))
        }
    """
    opts = list(options)
    if isinstance(value, bool):
        value = opts[both - 1] if value else opts[neither - 1]
    else:
        value = arg_match0(value, opts, arg_name=nm)

    x_set = {opts[x - 1], opts[both - 1]}
    y_set = {opts[y - 1], opts[both - 1]}
    return {"x": value in x_set, "y": value in y_set}


def _validate_independent(
    independent: Dict[str, bool],
    free: Dict[str, bool],
    space_free: Dict[str, bool],
    rmlab: Dict[str, bool],
) -> Dict[str, Dict[str, bool]]:
    """Validate and reconcile the ``independent`` facet interactions.

    Faithful port of ggh4x's ``.validate_independent`` (``R/facet_grid2.R:438-485``).
    Enforces three rules per dimension when ``independent`` is set:

    1. ``independent`` requires ``free`` in the same dimension (else abort).
    2. ``independent`` cannot coexist with free ``space`` -> force ``space_free``
       to ``False`` (with a warning).
    3. ``independent`` axes must keep their labels -> force ``rmlab`` to ``False``
       (with a warning).

    Parameters
    ----------
    independent : dict
        ``{"x": bool, "y": bool}`` -- whether scales vary within a column/row.
    free : dict
        ``{"x": bool, "y": bool}`` -- whether scales are free.
    space_free : dict
        ``{"x": bool, "y": bool}`` -- whether panel sizes are proportional.
    rmlab : dict
        ``{"x": bool, "y": bool}`` -- whether inner axis labels are removed.

    Returns
    -------
    dict
        ``{"independent", "free", "space_free", "rmlab"}`` with the
        (possibly mutated) four dicts.

    Raises
    ------
    ValueError
        When a dimension is independent but its scales are not free.
    """
    # Copy so we never mutate the caller's dicts.
    independent = dict(independent)
    free = dict(free)
    space_free = dict(space_free)
    rmlab = dict(rmlab)

    if independent["x"]:
        if not free["x"]:
            cli_abort("`x` cannot be independent if scales are not free.")
        if space_free["x"]:
            cli_warn(
                "`x` cannot have free space if axes are independent. "
                "Overriding `space` for `x` to `FALSE`."
            )
            space_free["x"] = False
        if rmlab["x"]:
            cli_warn(
                "x-axes must be labelled if they are independent. "
                "Overriding `remove_labels` for `x` to `FALSE`."
            )
            rmlab["x"] = False

    if independent["y"]:
        if not free["y"]:
            cli_abort("`y` cannot be independent if scales are not free.")
        if space_free["y"]:
            cli_warn(
                "`y` cannot have free space if axes are independent. "
                "Overriding `space` for `y` to `FALSE`."
            )
            space_free["y"] = False
        if rmlab["y"]:
            cli_warn(
                "y-axes must be labelled if they are independent. "
                "Overriding `remove_labels` for `y` to `FALSE`."
            )
            rmlab["y"] = False

    return {
        "independent": independent,
        "free": free,
        "space_free": space_free,
        "rmlab": rmlab,
    }


def reshape_add_margins(
    df: pd.DataFrame,
    vars_: Sequence[Sequence[str]],
    margins: Any = False,
    margin_nm: str = "(all)",
) -> pd.DataFrame:
    """Add facet margins to a cross-product layout frame.

    Faithful port of ggplot2's ``reshape_add_margins`` (as borrowed by ggh4x).
    Enumerates EVERY marginal variable subset via ``reshape_margins``
    (``expand.grid`` over ``{none} ∪ {downto(margin, set)}`` per facet
    dimension), which includes the empty set (= the original data) AND the full
    set (= the grand-total ``(all)/(all)`` panel) — the latter was previously
    missing.  Each subset's columns are set to *margin_nm* and the frames are
    row-bound.  Marginalised columns gain *margin_nm* as their LAST factor level
    so margin panels sort last (R ``add_all``).

    Parameters
    ----------
    df : pandas.DataFrame
        The cross-product of the row and column faceting frames.
    vars_ : sequence of sequence of str
        ``[row_var_names, col_var_names]`` -- the variables eligible for
        marginalisation.
    margins : bool or sequence of str, default False
        ``False`` / empty -> no margins (identity).  ``True`` -> all variables
        marginalised.  A list of names -> only those variables.

    Returns
    -------
    pandas.DataFrame
        *df* with any requested marginal rows appended.
    """
    if margins is False or margins is None:
        return df
    if isinstance(margins, (list, tuple)) and len(margins) == 0:
        return df

    from itertools import product

    def _downto(a: str, b: Sequence[str]) -> List[str]:
        # R: downto(a, b) = rev(upto(a, rev(b))) = elements of b from a to end
        # (so marginalising an outer variable also marginalises nested ones).
        b = list(b)
        return b[b.index(a):] if a in b else []

    # reshape_margins: every marginal variable subset (R borrowed_ggplot2.R).
    if margins is True:
        margins_list = [v for grp in vars_ for v in grp]
    else:
        margins_list = list(margins)
    dims: List[List[List[str]]] = []
    for set_vars in vars_:
        sv = list(set_vars)
        inter = [v for v in sv if v in margins_list]  # R intersect: set order
        dims.append([_downto(m, sv) for m in inter])
    margin_sets: List[List[str]] = []
    for combo in product(*[range(len(d) + 1) for d in dims]):
        sel: List[str] = []
        for set_i, choice in enumerate(combo):
            if choice > 0:  # choice 0 == "no margin on this dimension"
                sel.extend(dims[set_i][choice - 1])
        margin_sets.append(sel)  # includes [] (original) and the full set

    affected: List[str] = []
    for s in margin_sets:
        for v in s:
            if v not in affected and v in df.columns:
                affected.append(v)
    if not affected:
        return df

    df = df.copy()

    def _add_all(col: pd.Series) -> pd.Categorical:
        # R add_all: factor with margin_nm appended as the LAST level.
        if isinstance(col.dtype, pd.CategoricalDtype):
            cats = list(col.cat.categories)
        else:
            cats = sorted(pd.unique(col.dropna()).tolist())
        if margin_nm not in cats:
            cats = cats + [margin_nm]
        return pd.Categorical(col, categories=cats)

    for v in affected:
        df[v] = _add_all(df[v])

    frames: List[pd.DataFrame] = []
    for s in margin_sets:
        block = df.copy()
        for v in s:
            if v in block.columns:
                block[v] = pd.Categorical(
                    [margin_nm] * len(block),
                    categories=list(block[v].cat.categories),
                )
        frames.append(block)
    out = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(
        drop=True
    )
    return out
