"""Convenience helpers (R source: conveniences.R, plus ``sep_discrete`` from scale_manual.R).

Ports of ggh4x's small user-facing convenience functions:

* :func:`distribute_args` -- vectorised argument distributor that calls a function once per
  "column" of arguments.
* :func:`elem_list_text` / :func:`elem_list_rect` -- thin wrappers over
  :func:`distribute_args` that build lists of ``element_text`` / ``element_rect`` theme
  elements.
* :func:`weave_factors` -- an ``interaction``-like factor builder with first-input-priority
  level ordering.
* :func:`center_limits` -- a function factory producing symmetric scale limits.
* :func:`sep_discrete` -- a function factory that maps separator-delimited discrete labels to
  numeric positions (R source: scale_manual.R).

All semantics were verified against a live R ``ggh4x`` session (level ordering, NA handling,
length-1 vs. length-n behaviour, and the run-length grouping in ``sep_discrete``).
"""

from __future__ import annotations

import inspect
from itertools import product
from typing import Any, Callable, List, Sequence

import numpy as np
import pandas as pd

from ._cli import cli_abort
from ._vctrs import vec_unrep

try:  # pragma: no cover - import guard; ggplot2_py is always present in this env
    from ggplot2_py.theme_elements import element_rect, element_text
except Exception:  # pragma: no cover
    element_text = None  # type: ignore[assignment]
    element_rect = None  # type: ignore[assignment]

__all__ = [
    "distribute_args",
    "elem_list_text",
    "elem_list_rect",
    "weave_factors",
    "center_limits",
    "sep_discrete",
]


# -- distribute_args -------------------------------------------------------------------------


def _is_na_scalar(x: Any) -> bool:
    """Return ``True`` for a scalar that represents R's ``NA`` / Python missing value.

    Parameters
    ----------
    x : Any
        Candidate scalar.

    Returns
    -------
    bool
        ``True`` if *x* is ``None`` or a float/NaT-style NaN.
    """
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    # numpy scalar NaN
    if isinstance(x, np.floating) and np.isnan(x):
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def _cell_is_atomic(cell: Any) -> bool:
    """Mirror R ``is.vector`` for a distribute_args matrix cell.

    R only NA-collapses *vector* cells (atomic vectors / bare lists). Classed objects such
    as ``margin`` and theme elements are not vectors and are left untouched.

    Parameters
    ----------
    cell : Any
        A single cell value.

    Returns
    -------
    bool
        ``True`` when *cell* should be subjected to the NA-collapse rule.
    """
    if cell is None:
        # NULL is *not* a vector in R (is.vector(NULL) == FALSE), but it is length-0 and gets
        # dropped downstream regardless, so treat it as non-atomic here.
        return False
    if isinstance(cell, (str, bytes)):
        return True
    if isinstance(cell, (int, float, bool, complex)):
        return True
    if isinstance(cell, (np.generic,)):
        return True
    if isinstance(cell, (list, tuple, np.ndarray, pd.Series, range)):
        return True
    return False


def _contains_na(cell: Any) -> bool:
    """Return ``True`` when an atomic cell contains any NA (R: ``any(is.na(x))``).

    Parameters
    ----------
    cell : Any
        An atomic cell (scalar or sequence of scalars).

    Returns
    -------
    bool
    """
    if isinstance(cell, (str, bytes)):
        return False
    if isinstance(cell, (list, tuple, np.ndarray, pd.Series, range)):
        return any(_is_na_scalar(v) for v in cell)
    return _is_na_scalar(cell)


def _as_cells(value: Any) -> List[Any]:
    """Split an argument into its per-column cells, mirroring R ``as.list(arg)``.

    A list/tuple/Series is treated as the explicit sequence of cells (each element becomes
    one cell; an element that is itself a sequence becomes a *vector* cell). Any other value
    is a length-1 argument occupying a single cell.

    Parameters
    ----------
    value : Any
        The raw argument value.

    Returns
    -------
    list
        The cells for this argument, in column order.
    """
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, pd.Series):
        return list(value)
    if isinstance(value, np.ndarray) and value.ndim == 1:
        return list(value)
    return [value]


def _signature_param_names(fun: Callable[..., Any]) -> List[str]:
    """Return the keyword-argument names of *fun*, mirroring R ``names(formals(.fun))``.

    Parameters
    ----------
    fun : callable

    Returns
    -------
    list of str
        Names of parameters that can be passed by keyword (``**kwargs`` is excluded, like R
        cannot name a ``...`` formal).
    """
    names: List[str] = []
    for name, param in inspect.signature(fun).parameters.items():
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            names.append(name)
    return names


def distribute_args(
    *,
    fun: Callable[..., Any] = None,
    cull: bool = True,
    **kwargs: Any,
) -> Any:
    """Distribute vectorised arguments across repeated calls of *fun*.

    Mirrors ggh4x's ``distribute_args`` (conveniences.R). The ``i``-th element of each named
    argument is passed to the ``i``-th call of *fun*. Length-1 arguments occupy **only the
    first** call (they are *not* recycled across calls), matching R's matrix construction.

    Parameters
    ----------
    fun : callable, optional
        Function to receive the distributed arguments. Defaults to
        ``ggplot2_py.theme_elements.element_text``.
    cull : bool, default True
        When ``True``, arguments whose names are not formals of *fun* are silently dropped.
    **kwargs : Any
        Vectorised arguments. A list/tuple/1-D array/Series provides one value per call (a
        nested sequence is passed through as a vector); any scalar is a single value used for
        the first call only. ``None`` and ``NaN`` mark positions that should be skipped, and
        an atomic vector containing any ``NaN``/``None`` collapses to "skip" at that position
        (mirroring R's "NA vectors become NULL").

    Returns
    -------
    list or object
        A list of *fun* outputs (one per column), **or** -- when no usable arguments remain
        after culling -- a single ``fun()`` result (matching R's ``return(.fun())``).

    Notes
    -----
    Because valid argument names are deduced from *fun*'s signature, functions whose public
    surface is a bare ``**kwargs`` are mishandled (extra arguments are dropped), exactly as
    the R documentation warns for ``...``.
    """
    if fun is None:
        fun = element_text

    args = dict(kwargs)

    # Cull unknown arguments by signature, mirroring names(formals(.fun)).
    if cull:
        allowed = set(_signature_param_names(fun))
        args = {k: v for k, v in args.items() if k in allowed}

    # Drop zero-length arguments (R: args[lengths(args) > 0]).
    args = {k: v for k, v in args.items() if not _is_zero_length(v)}

    if len(args) == 0:
        return fun()

    names = list(args.keys())
    cells_per_arg = {k: _as_cells(v) for k, v in args.items()}
    lens = {k: len(cells_per_arg[k]) for k in names}
    ncol = max(lens.values())

    # Build the per-column kwargs, applying the NA-collapse rule per cell.
    results: List[Any] = []
    for j in range(ncol):
        col_kwargs: dict = {}
        for name in names:
            cells = cells_per_arg[name]
            if j >= len(cells):
                continue  # length-1 (or short) args only fill leading columns
            cell = cells[j]
            if _cell_is_atomic(cell) and _contains_na(cell):
                continue  # NA-containing vector becomes NULL -> dropped
            if cell is None:
                continue  # explicit NULL -> dropped (length-0)
            col_kwargs[name] = _unwrap_singleton(cell)
        results.append(fun(**col_kwargs))
    return results


def _is_zero_length(value: Any) -> bool:
    """Return ``True`` for an argument R would treat as length-0 (``lengths(x) == 0``).

    Parameters
    ----------
    value : Any

    Returns
    -------
    bool
    """
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    if isinstance(value, (np.ndarray, pd.Series, pd.Index)):
        return len(value) == 0
    return False


def _unwrap_singleton(cell: Any) -> Any:
    """Unwrap a one-element sequence cell to its scalar, leaving longer vectors intact.

    Parameters
    ----------
    cell : Any

    Returns
    -------
    Any
    """
    if isinstance(cell, (list, tuple)) and len(cell) == 1:
        return cell[0]
    if isinstance(cell, np.ndarray) and cell.ndim == 1 and cell.size == 1:
        return cell.item()
    return cell


def elem_list_text(**kwargs: Any) -> Any:
    """Build a list of ``element_text`` theme elements from vectorised arguments.

    Convenience wrapper around :func:`distribute_args` with ``fun=element_text``.

    Parameters
    ----------
    **kwargs : Any
        Vectorised ``element_text`` arguments (see :func:`distribute_args`).

    Returns
    -------
    list or ElementText
        A list of ``ElementText`` objects (or a single one when no arguments remain).
    """
    return distribute_args(fun=element_text, **kwargs)


def elem_list_rect(**kwargs: Any) -> Any:
    """Build a list of ``element_rect`` theme elements from vectorised arguments.

    Convenience wrapper around :func:`distribute_args` with ``fun=element_rect``.

    Parameters
    ----------
    **kwargs : Any
        Vectorised ``element_rect`` arguments (see :func:`distribute_args`).

    Returns
    -------
    list or ElementRect
        A list of ``ElementRect`` objects (or a single one when no arguments remain).
    """
    return distribute_args(fun=element_rect, **kwargs)


# -- weave_factors ---------------------------------------------------------------------------


def _is_factor(x: Any) -> bool:
    """Return ``True`` if *x* is a pandas Categorical / factor-like input."""
    if isinstance(x, pd.Categorical):
        return True
    if isinstance(x, pd.Series) and isinstance(x.dtype, pd.CategoricalDtype):
        return True
    return False


def _as_categorical(x: Any) -> pd.Categorical:
    """Coerce *x* to a :class:`pandas.Categorical`."""
    if isinstance(x, pd.Categorical):
        return x
    if isinstance(x, pd.Series) and isinstance(x.dtype, pd.CategoricalDtype):
        return x.array  # type: ignore[return-value]
    return pd.Categorical(x)


def _r_as_character(x: Any) -> str:
    """Format a scalar the way R's ``as.character`` / ``paste`` would.

    Notably, integral floats are rendered without a trailing ``.0`` (R prints ``10``, not
    ``10.0``), and ``NA`` becomes the literal string ``"NA"``.

    Parameters
    ----------
    x : Any

    Returns
    -------
    str
    """
    if _is_na_scalar(x):
        return "NA"
    if isinstance(x, (bool, np.bool_)):
        return "TRUE" if x else "FALSE"
    if isinstance(x, (float, np.floating)):
        if float(x).is_integer():
            return str(int(x))
        return repr(float(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return str(x)


def weave_factors(
    *args: Any,
    drop: bool = True,
    sep: str = ".",
    replace_na: bool = True,
) -> pd.Categorical:
    """Combine factors into a single factor with first-input-priority level ordering.

    Mirrors ggh4x's ``weave_factors`` (conveniences.R). Resembles
    ``interaction(..., lex.order = TRUE)`` but treats non-factor inputs as if their levels
    were ``unique(as.character(x))`` (first-appearance order, not sorted).

    Parameters
    ----------
    *args : array-like
        Input vectors (factors / Categoricals or plain sequences). All must have the same
        length, or length 1 (length-1 inputs are recycled).
    drop : bool, default True
        Drop level combinations that do not occur in the data.
    sep : str, default "."
        Delimiter joining the per-input labels into the new level labels.
    replace_na : bool, default True
        Replace ``NA`` values with empty strings. For factor inputs this appends an empty
        ``""`` level and routes any ``NA`` to it; for non-factors, ``NA`` becomes ``""``.

    Returns
    -------
    pandas.Categorical
        The woven factor (``ordered=False``).

    Raises
    ------
    ValueError
        If the inputs do not all share a common length (or length 1).
    """
    inputs = list(args)
    nargs = len(inputs)
    if nargs < 1:
        return None  # type: ignore[return-value]

    lengths = [len(a) for a in inputs]
    max_len = max(lengths)
    if not all(length in (1, max_len) for length in lengths):
        cli_abort(
            "All inputs to `weave_factors` should have the same length, "
            "or length 1."
        )

    # Recycle length-1 inputs to the common length (R's paste() recycles).
    def _recycle(a: Any) -> Any:
        if len(a) == max_len:
            return a
        # length 1 -> repeat
        if _is_factor(a):
            cat = _as_categorical(a)
            return cat.repeat(max_len)
        return list(a) * max_len

    inputs = [_recycle(a) for a in inputs]

    # Per-input: string values + level labels (post replace_na).
    value_strings: List[List[str]] = []
    level_lists: List[List[str]] = []

    for a in inputs:
        if _is_factor(a):
            cat = _as_categorical(a)
            lvls = [str(c) for c in cat.categories]
            codes = np.asarray(cat.codes, dtype=int)
            if replace_na:
                # Append an empty "" level; route NA codes to it (R always extends levels).
                if "" in lvls:
                    na_index = lvls.index("")
                    ext_levels = list(lvls)
                else:
                    na_index = len(lvls)
                    ext_levels = list(lvls) + [""]
                code_for = np.where(codes < 0, na_index, codes)
                vals = [ext_levels[c] for c in code_for]
                level_lists.append(ext_levels)
                value_strings.append(vals)
            else:
                vals = ["NA" if c < 0 else lvls[c] for c in codes]
                level_lists.append(lvls)
                value_strings.append(vals)
        else:
            seq = list(a)
            if replace_na:
                vals = ["" if _is_na_scalar(v) else _r_as_character(v) for v in seq]
            else:
                vals = [_r_as_character(v) for v in seq]
            value_strings.append(vals)
            # levels() %||% as.character(unique(x)) -> unique first-appearance order
            uniq: List[str] = []
            seen = set()
            for v in vals:
                if v not in seen:
                    seen.add(v)
                    uniq.append(v)
            level_lists.append(uniq)

    # Row-wise joined values.
    n = max_len
    vals_joined = [sep.join(value_strings[k][i] for k in range(nargs)) for i in range(n)]

    # Unique observed values (first-appearance order).
    unique_vals: List[str] = []
    seen_v = set()
    for v in vals_joined:
        if v not in seen_v:
            seen_v.add(v)
            unique_vals.append(v)

    # Candidate levels: expand.grid over REVERSED level lists, then paste back reversed.
    # itertools.product varies the LAST iterable fastest; expand.grid varies the FIRST
    # column fastest. Reversing the list of (already reversed) lists reproduces R exactly:
    # product over the original-order lists, with the LAST input varying fastest, which after
    # the final reverse-join makes the FIRST input vary slowest (first-input priority).
    candidate_levels: List[str] = []
    for combo in product(*level_lists):
        candidate_levels.append(sep.join(combo))

    if drop:
        observed = set(unique_vals)
        candidate_levels = [lv for lv in candidate_levels if lv in observed]

    # unique() (defensive; sep collisions could create duplicates).
    final_levels: List[str] = []
    seen_l = set()
    for lv in candidate_levels:
        if lv not in seen_l:
            seen_l.add(lv)
            final_levels.append(lv)

    return pd.Categorical(vals_joined, categories=final_levels, ordered=False)


# -- center_limits ---------------------------------------------------------------------------


def center_limits(around: float = 0) -> Callable[[Any], np.ndarray]:
    """Return a function that centres scale limits symmetrically around *around*.

    Mirrors ggh4x's ``center_limits`` (conveniences.R). Useful for centring log2 fold-change
    colour limits at zero.

    Parameters
    ----------
    around : float, default 0
        The value about which to centre the returned limits.

    Returns
    -------
    callable
        A function mapping an input vector to
        ``[-1, 1] * max(abs(input - around)) + around``.
    """

    def _limits(input: Any) -> np.ndarray:
        arr = np.asarray(input, dtype=float)
        span = np.nanmax(np.abs(arr - around))
        return np.array([-1.0, 1.0]) * span + around

    return _limits


# -- sep_discrete ----------------------------------------------------------------------------


def sep_discrete(sep: str = ".", inv: bool = False) -> Callable[[Sequence[str]], np.ndarray]:
    """Return a function mapping separator-delimited discrete labels to numeric positions.

    Mirrors ggh4x's ``sep_discrete`` (scale_manual.R). Labels are split on the (literal,
    non-regex) separator; the first split component drives the base position
    (``1..n``) and each deeper component contributes a run-length group offset, so grouped
    labels are laid out hierarchically.

    Parameters
    ----------
    sep : str, default "."
        Literal separator used to split labels (no regular expressions).
    inv : bool, default False
        When ``True``, invert the layering of groups (split columns are reversed before
        scoring).

    Returns
    -------
    callable
        A function accepting a sequence of ``str`` labels and returning a numeric
        ``numpy.ndarray`` of positions.
    """

    def _positions(limits: Sequence[str]) -> np.ndarray:
        labels = list(limits)
        split = [lab.split(sep) for lab in labels]
        lengs = [len(s) for s in split]
        depth = max(lengs) if lengs else 0

        # Pad ragged splits with empty strings (R: c(lab, rep("", depth - length(lab)))).
        if not all(length == depth for length in lengs):
            split = [s + [""] * (depth - len(s)) for s in split]

        nrow = len(split)
        # Build the matrix of column vectors.
        mat = np.empty((nrow, depth), dtype=object)
        for i, row in enumerate(split):
            for j in range(depth):
                mat[i, j] = row[j]

        if inv:
            mat = mat[:, ::-1]

        # Per-column run-length group index (consecutive runs -> 0,1,2,... repeated).
        vals = np.zeros((nrow, depth), dtype=float)
        for j in range(depth):
            col = mat[:, j]
            unrep = vec_unrep(col)
            keys = list(unrep["key"])
            times = list(unrep["times"])
            group = np.repeat(np.arange(len(keys)), times)
            vals[:, j] = group

        # First column becomes the base position 1..nrow.
        vals[:, 0] = np.arange(1, nrow + 1)
        return vals.sum(axis=1)

    return _positions
