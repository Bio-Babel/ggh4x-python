"""Apply a function to position coordinates per group.

Python port of ``stat_funxy.R`` from the R package **ggh4x**.

The "function xy" stat applies a user-supplied function to the ``x`` and
``y`` aesthetics of a layer's positions, computed separately per group.
:func:`stat_centroid` and :func:`stat_midpoint` are convenience wrappers
that compute group centroids (means) and midpoints (``(min + max) / 2``)
respectively.  :func:`stat_funxy` leaves the data unchanged by default but
can be supplied arbitrary functions and arguments.

R source
--------
``ggh4x/R/stat_funxy.R`` -- :class:`StatFunxy`, :func:`stat_funxy`,
:func:`stat_centroid`, :func:`stat_midpoint`.

Notes
-----
This statistic makes only a minimal attempt at ensuring that the results
of calling both functions are of equal length.  Results of length one are
recycled to match the longest result (mirroring
``vctrs::vec_recycle_common``).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ggplot2_py import Stat

from ._cli import cli_abort
from ._vctrs import data_frame0, vec_recycle_common

__all__ = [
    "StatFunxy",
    "stat_funxy",
    "stat_centroid",
    "stat_midpoint",
]


# ---------------------------------------------------------------------------
# Lazy ``layer`` import (mirrors ggplot2_py.stat._layer; avoids importing the
# heavy layer machinery -- and any circular dependency -- at module import).
# ---------------------------------------------------------------------------
def _layer(**kwargs: Any) -> Any:
    """Construct a :class:`ggplot2_py.Layer`, importing ``layer`` lazily.

    Parameters
    ----------
    **kwargs : Any
        Forwarded verbatim to :func:`ggplot2_py.layer.layer`.

    Returns
    -------
    ggplot2_py.Layer
    """
    from ggplot2_py.layer import layer

    return layer(**kwargs)


# ---------------------------------------------------------------------------
# Helper reductions used by the wrapper constructors.
#
# R forwards ``na.rm = TRUE`` to ``mean``/``range``.  NumPy's ``mean``/``min``/
# ``max`` have no ``na_rm`` keyword, so the wrappers translate to the
# NaN-aware reductions and accept (and honour) an ``na_rm`` keyword so the
# translated ``argx``/``argy`` dicts forward cleanly.
# ---------------------------------------------------------------------------
def _mean(x: Sequence[float], na_rm: bool = False) -> np.floating:
    """Arithmetic mean with optional NA removal (port of R ``mean``).

    Parameters
    ----------
    x : sequence of float
        Values to average.
    na_rm : bool, default False
        Whether to ignore ``NaN`` values (R's ``na.rm``).

    Returns
    -------
    numpy.floating
        The (NaN-aware) mean of *x*.
    """
    arr = np.asarray(x, dtype=float)
    return np.nanmean(arr) if na_rm else np.mean(arr)


def _midpoint(x: Sequence[float], na_rm: bool = True) -> np.floating:
    """Midpoint of the data range: ``(min + max) / 2``.

    Port of the closure defined inside R ``stat_midpoint`` --
    ``sum(range(x, na.rm = na.rm), na.rm = na.rm) / 2``.

    Parameters
    ----------
    x : sequence of float
        Values to summarise.
    na_rm : bool, default True
        Whether to ignore ``NaN`` values when computing the range.

    Returns
    -------
    numpy.floating
        Half the sum of the minimum and maximum of *x*.
    """
    arr = np.asarray(x, dtype=float)
    if na_rm:
        lo, hi = np.nanmin(arr), np.nanmax(arr)
    else:
        lo, hi = np.min(arr), np.max(arr)
    return (lo + hi) / 2.0


def _identity(x: Any) -> Any:
    """Return *x* unchanged (port of R ``force`` used as the default fun).

    Parameters
    ----------
    x : Any

    Returns
    -------
    Any
        The input, unchanged.
    """
    return x


# ---------------------------------------------------------------------------
# StatFunxy ggproto
# ---------------------------------------------------------------------------
class StatFunxy(Stat):
    """Apply ``funx``/``funy`` to the ``x``/``y`` positions per group.

    Mirrors the R ``StatFunxy`` ggproto object.  ``compute_group`` calls
    ``funx`` on the group's ``x`` values and ``funy`` on its ``y`` values,
    then reconciles the lengths of the remaining columns (cropping or
    recycling) before recombining.

    Attributes
    ----------
    required_aes : list of str
        ``['x', 'y']``.
    """

    required_aes = ["x", "y"]

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        funx: Callable[..., Any] = _identity,
        funy: Callable[..., Any] = _identity,
        argx: Optional[Dict[str, Any]] = None,
        argy: Optional[Dict[str, Any]] = None,
        crop_other: bool = True,
    ) -> pd.DataFrame:
        """Apply ``funx``/``funy`` to ``x``/``y`` and reconcile lengths.

        Parameters
        ----------
        data : pandas.DataFrame
            One group's data; must contain ``x`` and ``y`` columns.
        scales : Any
            Panel scales (unused; present for signature parity).
        funx, funy : callable
            Functions applied to the ``x`` and ``y`` positions
            respectively.  Each is called as ``fun(values, **arg)``.
        argx, argy : dict, optional
            Named keyword arguments forwarded to ``funx`` / ``funy``.
        crop_other : bool, default True
            Whether the remaining (non ``x``/``y``) columns should be
            cropped to the length of the longest of ``funx``/``funy``'s
            results.  Set ``False`` when the functions return length-one
            summaries that should be recycled against the full group.

        Returns
        -------
        pandas.DataFrame
            Columns: the original non ``x``/``y`` columns (in their
            original order) followed by ``x`` and ``y``, each recycled to
            a common length.
        """
        if argx is None:
            argx = {}
        if argy is None:
            argy = {}

        # Apply functions. R: do.call(funx, c(unname(data["x"]), argx)),
        # i.e. the column is the first positional argument with its name
        # stripped, followed by the named ``argx`` arguments.
        x = funx(np.asarray(data["x"].to_numpy(), dtype=float), **argx)
        y = funy(np.asarray(data["y"].to_numpy(), dtype=float), **argy)
        x = np.atleast_1d(np.asarray(x))
        y = np.atleast_1d(np.asarray(y))

        # Ensure the rest of the data is of the correct length.
        other_names = [c for c in data.columns if c not in ("x", "y")]
        size = int(max(len(x), len(y)))
        if crop_other:
            # R: lapply(data[other], `[`, i = size) where size = seq_len(...).
            # Positional crop to the first ``size`` elements (1-based 1:size
            # -> 0-based slice [0:size]).
            other = {c: data[c].to_numpy()[:size] for c in other_names}
        else:
            other = {c: data[c].to_numpy() for c in other_names}

        # Combine: other columns first, then x, then y (R column order).
        combined: Dict[str, Any] = dict(other)
        combined["x"] = x
        combined["y"] = y

        # Recycle every column to a common length (length-1 -> longest).
        recycled = vec_recycle_common(*combined.values())
        out = {name: arr for name, arr in zip(combined.keys(), recycled)}
        return data_frame0(**out)


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------
def stat_funxy(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "point",
    position: str = "identity",
    *,
    funx: Callable[..., Any] = _identity,
    funy: Callable[..., Any] = _identity,
    argx: Optional[Dict[str, Any]] = None,
    argy: Optional[Dict[str, Any]] = None,
    crop_other: bool = True,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Apply a function to a layer's ``x`` and ``y`` position coordinates.

    Port of R ``stat_funxy``.

    Parameters
    ----------
    mapping : Mapping, optional
        Aesthetic mapping (see :func:`ggplot2_py.aes`).
    data : DataFrame or callable, optional
        Layer data.
    geom : str, default ``"point"``
        Geometry used to render the computed positions.
    position : str, default ``"identity"``
        Position adjustment.
    funx, funy : callable, default identity
        Functions called on the layer's ``x`` and ``y`` positions
        respectively.  Default ``force`` (identity) leaves data as-is.
    argx, argy : dict, optional
        Named arguments forwarded to ``funx`` / ``funy``.  Default empty.
    crop_other : bool, default True
        Whether the other data should be fitted to the length of ``x`` and
        ``y``.  Set ``False`` when ``funx``/``funy`` compute length-one
        summaries that need to be recycled.
    show_legend : bool, optional
        Whether to include this layer in the legend.
    inherit_aes : bool, default True
        Whether to inherit the plot-level mapping.
    **kwargs
        Additional parameters forwarded to the layer.

    Returns
    -------
    ggplot2_py.Layer
        A layer backed by :class:`StatFunxy`.

    Raises
    ------
    ValueError
        If ``funx``/``funy`` are not callable, if ``argx``/``argy`` are not
        dicts, or if ``argx``/``argy`` contain unnamed (empty-string-key)
        elements.
    """
    if argx is None:
        argx = {}
    if argy is None:
        argy = {}

    # stopifnot(...) validations from R, translated to cli_abort.
    if not callable(funx):
        cli_abort("The `funx` argument must be a function.")
    if not callable(funy):
        cli_abort("The `funy` argument must be a function.")
    if not isinstance(argx, dict) or not isinstance(argy, dict):
        cli_abort("The `argx` and `argy` arguments must be lists.")
    # R: length(argx) == sum(nzchar(names(argx))) -- every element named.
    if any((not isinstance(k, str)) or k == "" for k in argx.keys()):
        cli_abort("The `argx` list must have named elements")
    if any((not isinstance(k, str)) or k == "" for k in argy.keys()):
        cli_abort("The `argy` list must have named elements")

    return _layer(
        data=data,
        mapping=mapping,
        stat=StatFunxy,
        geom=geom,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "funx": funx,
            "funy": funy,
            "argx": argx,
            "argy": argy,
            "crop_other": crop_other,
            **kwargs,
        },
    )


def stat_centroid(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "point",
    position: str = "identity",
    *,
    funx: Callable[..., Any] = _mean,
    funy: Callable[..., Any] = _mean,
    argx: Optional[Dict[str, Any]] = None,
    argy: Optional[Dict[str, Any]] = None,
    crop_other: bool = True,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Compute group centroids (means of ``x`` and ``y``).

    Convenience wrapper around :func:`stat_funxy` with ``funx = funy =
    mean`` and ``argx = argy = {'na_rm': True}`` (port of R
    ``stat_centroid``).

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom : str, default ``"point"``
    position : str, default ``"identity"``
    funx, funy : callable, default NaN-aware mean
        Functions applied to ``x`` / ``y``.
    argx, argy : dict, optional
        Named arguments forwarded to ``funx`` / ``funy``.  Default
        ``{'na_rm': True}``.
    crop_other : bool, default True
    show_legend : bool, optional
    inherit_aes : bool, default True
    **kwargs

    Returns
    -------
    ggplot2_py.Layer
    """
    if argx is None:
        argx = {"na_rm": True}
    if argy is None:
        argy = {"na_rm": True}
    return stat_funxy(
        mapping=mapping,
        data=data,
        geom=geom,
        position=position,
        funx=funx,
        funy=funy,
        argx=argx,
        argy=argy,
        crop_other=crop_other,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        **kwargs,
    )


def stat_midpoint(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "point",
    position: str = "identity",
    *,
    argx: Optional[Dict[str, Any]] = None,
    argy: Optional[Dict[str, Any]] = None,
    crop_other: bool = True,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Compute group midpoints (``(min + max) / 2`` of ``x`` and ``y``).

    Convenience wrapper around :func:`stat_funxy` whose ``funx``/``funy``
    is the closure ``sum(range(x, na.rm)) / 2`` (port of R
    ``stat_midpoint``).

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom : str, default ``"point"``
    position : str, default ``"identity"``
    argx, argy : dict, optional
        Named arguments forwarded to the midpoint function.  Default
        ``{'na_rm': True}``.
    crop_other : bool, default True
    show_legend : bool, optional
    inherit_aes : bool, default True
    **kwargs

    Returns
    -------
    ggplot2_py.Layer
    """
    if argx is None:
        argx = {"na_rm": True}
    if argy is None:
        argy = {"na_rm": True}
    return stat_funxy(
        mapping=mapping,
        data=data,
        geom=geom,
        position=position,
        funx=_midpoint,
        funy=_midpoint,
        argx=argx,
        argy=argy,
        crop_other=crop_other,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        **kwargs,
    )
