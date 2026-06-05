"""Small internal utilities (R source: utils.R, utils_grid.R).

Grob/unit measurement helpers used by strip and facet assembly. These delegate to grid_py's
snake_case conversion API.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from ._cli import cli_abort

__all__ = [
    "width_cm",
    "height_cm",
    "seq_range",
    "has_null_unit",
]


def _is_grob(x: Any) -> bool:
    from grid_py import is_grob

    return bool(is_grob(x))


def _is_unit(x: Any) -> bool:
    from grid_py import is_unit

    return bool(is_unit(x))


def width_cm(x: Any) -> float | np.ndarray | List[Any]:
    """Width in cm of a grob, unit, or list thereof, mirroring ``utils.R::width_cm``.

    Parameters
    ----------
    x : grob | unit | list
        A grid grob, a unit object, or a list of either.

    Returns
    -------
    float or numpy.ndarray or list
        Width(s) in centimetres. A length-1 unit/grob collapses to a Python
        ``float``; a multi-element unit returns a ``numpy.ndarray`` (one value
        per element), matching R's vectorised ``convertWidth(..., valueOnly=TRUE)``.

    Raises
    ------
    ValueError
        If *x* is none of grob/unit/list.
    """
    from grid_py import convert_width, grob_width

    if _is_grob(x):
        return float(convert_width(grob_width(x), "cm", valueOnly=True))
    if _is_unit(x):
        vals = np.asarray(convert_width(x, "cm", valueOnly=True), dtype=float)
        if vals.size == 1:
            return float(vals.reshape(-1)[0])
        return vals
    if isinstance(x, (list, tuple)):
        return [width_cm(e) for e in x]
    cli_abort(f"Unknown input: {type(x).__name__}.")


def height_cm(x: Any) -> float | np.ndarray | List[Any]:
    """Height in cm of a grob, unit, or list thereof, mirroring ``utils.R::height_cm``.

    Parameters
    ----------
    x : grob | unit | list
        A grid grob, a unit object, or a list of either.

    Returns
    -------
    float or numpy.ndarray or list
        Height(s) in centimetres. A length-1 unit/grob collapses to a Python
        ``float``; a multi-element unit returns a ``numpy.ndarray`` (one value
        per element), matching R's vectorised ``convertHeight(..., valueOnly=TRUE)``.

    Raises
    ------
    ValueError
        If *x* is none of grob/unit/list.
    """
    from grid_py import convert_height, grob_height

    if _is_grob(x):
        return float(convert_height(grob_height(x), "cm", valueOnly=True))
    if _is_unit(x):
        vals = np.asarray(convert_height(x, "cm", valueOnly=True), dtype=float)
        if vals.size == 1:
            return float(vals.reshape(-1)[0])
        return vals
    if isinstance(x, (list, tuple)):
        return [height_cm(e) for e in x]
    cli_abort(f"Unknown input: {type(x).__name__}.")


def seq_range(dat: Any, step: float | None = None, length_out: int | None = None) -> np.ndarray:
    """Sequence over the data range, mirroring ``utils.R::seq_range`` (``seq.int(min, max, ...)``).

    Parameters
    ----------
    dat : array-like
        Values whose min/max bound the sequence (NA ignored).
    step : float, optional
        Step size (``by`` in R).
    length_out : int, optional
        Number of points (``length.out`` in R).

    Returns
    -------
    numpy.ndarray
    """
    arr = np.asarray(dat, dtype=float)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if length_out is not None:
        return np.linspace(lo, hi, length_out)
    if step is not None:
        return np.arange(lo, hi + step / 2.0, step)
    # R seq_range = seq.int(min, max, ...); with no step/length it is the unit
    # step sequence min, min+1, ..., <= max (NOT just the two endpoints).
    return np.arange(lo, hi + 0.5, 1.0)


def has_null_unit(x: Any) -> bool:
    """Test whether a unit object contains any ``"null"`` units, mirroring ``has_null_unit``.

    Parameters
    ----------
    x : unit
        A grid unit (possibly compound/vector).

    Returns
    -------
    bool
    """
    from grid_py import unit_type

    if x is None:
        return False
    types = unit_type(x)
    if isinstance(types, str):
        return types == "null"
    return "null" in list(types)
