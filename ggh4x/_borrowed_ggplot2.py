"""ggplot2-internal helpers borrowed by ggh4x (R source: borrowed_ggplot2.R).

These are ggplot2 internals that ggh4x copies because they are not exported. Only the ones
ggh4x actually uses for facet layout / strip assembly are ported here; the rest are sourced
from ``ggplot2_py`` where available. The crown jewel is ``id``/``id_var`` — the radix-based
panel-id assignment that defines facet panel ordering. Ported verbatim and verified against R.
"""

from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "id_var",
    "id",
    "empty",
    "is_zero",
    "snake_class",
    "ulevels",
    "unique_combs",
]


def _is_factor(x: Any) -> bool:
    return isinstance(x, pd.Categorical) or (
        isinstance(x, pd.Series) and isinstance(x.dtype, pd.CategoricalDtype)
    )


def id_var(x: Sequence[Any], drop: bool = False) -> np.ndarray:
    """Assign integer ids to a single variable, mirroring ggplot2's ``id_var``.

    Parameters
    ----------
    x : sequence
        A vector (optionally a pandas Categorical/factor).
    drop : bool
        If ``True``, drop unused factor levels before id assignment.

    Returns
    -------
    np.ndarray
        1-based integer ids with an attached ``n`` (number of distinct values) accessible
        via ``result.n`` (set as an attribute on the returned ndarray subclass).
    """
    if len(x) == 0:
        out = _IdArray(np.array([], dtype=int))
        out.n = 0
        return out
    if _is_factor(x) and not drop:
        cat = x if isinstance(x, pd.Categorical) else pd.Categorical(x)
        levels = list(cat.categories)
        codes = cat.codes.astype(int)
        has_na = bool((codes < 0).any())
        # addNA(x, ifany=TRUE): NA becomes an extra level if present
        if has_na:
            ids = np.where(codes < 0, len(levels) + 1, codes + 1)
            n = len(levels) + 1
        else:
            ids = codes + 1
            n = len(levels)
        out = _IdArray(ids.astype(int))
        out.n = n
        return out
    # non-factor: sort unique (NA last), match
    s = pd.Series(list(x))
    uniq = pd.unique(s.dropna())
    levels = np.sort(uniq) if len(uniq) else np.array([])
    has_na = bool(s.isna().any())
    level_list = list(levels) + ([np.nan] if has_na else [])
    lookup = {v: i + 1 for i, v in enumerate(levels)}
    na_id = len(levels) + 1 if has_na else 0
    ids = np.array([na_id if pd.isna(v) else lookup[v] for v in s], dtype=int)
    out = _IdArray(ids)
    out.n = int(len(level_list))
    return out


class _IdArray(np.ndarray):
    """ndarray carrying an ``n`` attribute (R's ``attr(id, 'n')``)."""

    n: int

    def __new__(cls, input_array: np.ndarray) -> "_IdArray":
        obj = np.asarray(input_array, dtype=int).view(cls)
        obj.n = 0
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self.n = getattr(obj, "n", 0)


def id(variables: pd.DataFrame | Sequence[Any], drop: bool = False) -> _IdArray:
    """Compute a unique id per row across multiple variables, mirroring ggplot2's ``id``.

    Uses radix mixing: variables are reversed (so the first varies slowest), each is
    id-coded, and ids are combined as ``sum((id_i - 1) * cumprod(n_{<i})) + 1``. This is
    what determines facet ``PANEL`` ordering.

    Parameters
    ----------
    variables : pandas.DataFrame or sequence of vectors
        The faceting variables (columns).
    drop : bool
        Drop unused combinations.

    Returns
    -------
    _IdArray
        1-based row ids with ``.n`` = number of distinct combinations.
    """
    if isinstance(variables, pd.DataFrame):
        nrows = len(variables)
        cols = [variables[c] for c in variables.columns]
    else:
        nrows = None
        cols = list(variables)
    cols = [c for c in cols if len(c) > 0]
    if len(cols) == 0:
        n = nrows if nrows is not None else 0
        out = _IdArray(np.arange(1, n + 1))
        out.n = n
        return out
    if len(cols) == 1:
        return id_var(cols[0], drop=drop)
    ids = [id_var(c, drop=drop) for c in cols][::-1]  # rev()
    ndistinct = np.array([i.n for i in ids], dtype=float)
    n = int(np.prod(ndistinct))
    p = len(ids)
    combs = np.concatenate([[1.0], np.cumprod(ndistinct[: p - 1])])
    mat = np.column_stack([np.asarray(i, dtype=float) for i in ids])
    res = ((mat - 1.0) @ combs + 1.0).astype(int)
    if drop:
        return id_var(res, drop=True)
    out = _IdArray(res)
    out.n = n
    return out


def empty(df: Any) -> bool:
    """Test whether a data frame is "empty", mirroring ggplot2's ``empty``.

    Parameters
    ----------
    df : Any

    Returns
    -------
    bool
        ``True`` if *df* is ``None`` or has zero rows or zero columns.
    """
    if df is None:
        return True
    if isinstance(df, pd.DataFrame):
        return df.shape[0] == 0 or df.shape[1] == 0
    return False


def is_zero(x: Any) -> bool:
    """Test for a zero/empty grob, mirroring ggplot2's ``is.zero``.

    Parameters
    ----------
    x : Any

    Returns
    -------
    bool
        ``True`` if *x* is ``None`` or a zeroGrob/null grob.
    """
    if x is None:
        return True
    cls = type(x).__name__
    return cls in ("ZeroGrob", "zeroGrob", "NullGrob") or getattr(x, "_grid_class", None) in (
        "zeroGrob",
        "null",
    )


def snake_class(x: Any) -> str:
    """Convert a class name to snake_case, mirroring ggplot2's ``snake_class``.

    Parameters
    ----------
    x : Any
        An object (its first class name is used) or a class-name string.

    Returns
    -------
    str
        e.g. ``FacetGrid2`` -> ``facet_grid2``.
    """
    import re

    name = x if isinstance(x, str) else type(x).__name__
    name = re.sub(r"([A-Za-z])([A-Z])([a-z])", r"\1_\2\3", name)
    name = name.replace(".", "_")
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    return name.lower()


def ulevels(x: Sequence[Any]) -> np.ndarray:
    """Unique sorted levels (NA included for factors), mirroring ggplot2's ``ulevels``.

    Parameters
    ----------
    x : sequence

    Returns
    -------
    np.ndarray
    """
    if _is_factor(x):
        cat = x if isinstance(x, pd.Categorical) else pd.Categorical(x)
        return np.asarray(list(cat.categories))
    s = pd.Series(list(x))
    uniq = pd.unique(s.dropna())
    return np.sort(uniq)


def unique_combs(df: pd.DataFrame) -> pd.DataFrame:
    """All unique combinations of the columns' levels, mirroring ggplot2's ``unique_combs``.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Cross-product of per-column ``ulevels`` (first column varies fastest, like R).
    """
    if df.shape[1] == 0:
        return pd.DataFrame()
    level_lists = {c: ulevels(df[c]) for c in df.columns}
    cols = list(df.columns)
    from itertools import product

    rows = list(product(*[level_lists[c] for c in reversed(cols)]))
    data = {c: [] for c in cols}
    for combo in rows:
        for c, v in zip(reversed(cols), combo):
            data[c].append(v)
    return pd.DataFrame(data)[cols]
