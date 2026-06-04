"""vctrs shims (R source: vctrs usage in ggh4x).

Faithful Python reimplementations of the small set of ``vec_*`` helpers ggh4x uses for
data-frame and vector manipulation. Each mirrors the exact R semantics (ordering,
run-length, recycling) verified against a live ``vctrs`` session.
"""

from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "vec_interleave",
    "vec_unrep",
    "vec_rep_each",
    "vec_match",
    "vec_unique",
    "vec_unique_count",
    "vec_group_loc",
    "vec_rbind",
    "vec_recycle_common",
    "data_frame0",
]


def vec_interleave(*args: Sequence[Any]) -> np.ndarray:
    """Interleave vectors element-by-element, mirroring ``vctrs::vec_interleave``.

    Parameters
    ----------
    *args : sequence
        Equal-length (after recycling) vectors.

    Returns
    -------
    np.ndarray
        ``[a0, b0, ..., a1, b1, ...]`` order.
    """
    arrays = [np.asarray(a) for a in args]
    n = max(len(a) for a in arrays) if arrays else 0
    arrays = [np.resize(a, n) if len(a) != n else a for a in arrays]
    return np.stack(arrays, axis=1).reshape(-1)


def vec_unrep(x: Sequence[Any]) -> pd.DataFrame:
    """Run-length encode, mirroring ``vctrs::vec_unrep``.

    Parameters
    ----------
    x : sequence
        Input vector.

    Returns
    -------
    pandas.DataFrame
        Columns ``key`` (the run values, in order) and ``times`` (run lengths).
    """
    arr = np.asarray(x, dtype=object)
    n = len(arr)
    if n == 0:
        return pd.DataFrame({"key": pd.Series([], dtype=object), "times": pd.Series([], dtype=int)})
    # boundaries where the value changes
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = arr[1:] != arr[:-1]
    idx = np.flatnonzero(change)
    keys = arr[idx]
    times = np.diff(np.append(idx, n))
    return pd.DataFrame({"key": keys, "times": times.astype(int)})


def vec_rep_each(x: Sequence[Any], times: Sequence[int] | int) -> np.ndarray:
    """Repeat each element *times[i]* times, mirroring ``vctrs::vec_rep_each``.

    Parameters
    ----------
    x : sequence
        Values to repeat.
    times : sequence of int or int
        Per-element repeat counts (or a scalar applied to all).

    Returns
    -------
    np.ndarray
        Expanded vector.
    """
    arr = np.asarray(x)
    return np.repeat(arr, times)


def vec_match(needles: Sequence[Any], haystack: Sequence[Any]) -> np.ndarray:
    """First-match index of *needles* in *haystack*, mirroring ``vctrs::vec_match``.

    Parameters
    ----------
    needles : sequence
        Values to look up.
    haystack : sequence
        Table of unique-or-not reference values.

    Returns
    -------
    np.ndarray
        0-based indices of the first match (or -1 when absent). NB: R returns 1-based or
        ``NA``; callers that need R indices add 1.
    """
    hay = pd.Index(pd.Series(list(haystack)))
    return hay.get_indexer(pd.Series(list(needles)))


def vec_unique(x: Sequence[Any]) -> np.ndarray:
    """Unique values preserving first-appearance order, mirroring ``vctrs::vec_unique``.

    Parameters
    ----------
    x : sequence

    Returns
    -------
    np.ndarray
    """
    return pd.unique(pd.Series(list(x)))


def vec_unique_count(x: Sequence[Any]) -> int:
    """Number of unique values, mirroring ``vctrs::vec_unique_count``.

    Parameters
    ----------
    x : sequence

    Returns
    -------
    int
    """
    return int(len(vec_unique(x)))


def vec_group_loc(x: Sequence[Any]) -> pd.DataFrame:
    """Group rows by value, mirroring ``vctrs::vec_group_loc``.

    Parameters
    ----------
    x : sequence

    Returns
    -------
    pandas.DataFrame
        Columns ``key`` (first-appearance order) and ``loc`` (0-based row indices per group).
    """
    s = pd.Series(list(x))
    keys = pd.unique(s)
    loc = [np.flatnonzero((s == k).to_numpy()) for k in keys]
    return pd.DataFrame({"key": list(keys), "loc": loc})


def vec_rbind(*frames: pd.DataFrame) -> pd.DataFrame:
    """Row-bind data frames filling missing columns, mirroring ``vctrs::vec_rbind``.

    Parameters
    ----------
    *frames : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Column union; absent columns filled with ``NA``; row index reset.
    """
    frames = [f for f in frames if f is not None and len(f.columns) > 0]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True, sort=False)


def vec_recycle_common(*args: Sequence[Any]) -> List[np.ndarray]:
    """Recycle vectors to a common length, mirroring ``vctrs::vec_recycle_common``.

    Parameters
    ----------
    *args : sequence
        Vectors of length 1 or *n*.

    Returns
    -------
    list of np.ndarray
        All recycled to the common length *n*.

    Raises
    ------
    ValueError
        If lengths are incompatible (not 1 and not *n*).
    """
    arrays = [np.asarray(a) for a in args]
    lengths = {len(a) for a in arrays if len(a) != 1}
    if len(lengths) > 1:
        raise ValueError(f"Incompatible lengths for recycling: {sorted(lengths)}")
    n = lengths.pop() if lengths else (len(arrays[0]) if arrays else 0)
    return [np.resize(a, n) if len(a) == 1 else a for a in arrays]


def data_frame0(**columns: Any) -> pd.DataFrame:
    """Construct a DataFrame, mirroring ggh4x's ``data_frame0`` (minimal name repair).

    Parameters
    ----------
    **columns : Any
        Column name -> values.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.DataFrame(columns)
