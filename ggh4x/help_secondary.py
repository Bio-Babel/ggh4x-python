"""Secondary axis helper.

Python port of the ggh4x R source file ``help_secondary.R``.

The public entry point :func:`help_secondary` constructs a
:class:`ggplot2_py.scale.AxisSecondary` whose reverse transformation
(``trans``) is fitted from data, and which carries an extra ``proj``
attribute (a forward-projection callable) used to map secondary data onto
the primary axis when building a plot.

R uses non-standard evaluation (``rlang::enquo``/``eval_tidy``/``as_label``)
so that users can write ``help_secondary(df, y1, y2)`` with bare column
symbols.  Python has no NSE, so this port accepts either array-likes
(``numpy``/``pandas``/sequences) **or** column-name strings that are
resolved against the ``data`` argument (a :class:`pandas.DataFrame`).

Five fitting ``method`` choices are implemented, each mirroring the
corresponding ``help_sec_*`` helper in the R source:

``"range"``
    Overlap the full ranges of primary and secondary data
    (``scales::rescale``).
``"max"``
    Make the maxima coincide (``scales::rescale_max``).
``"fit"``
    Use the coefficients of ``lm(primary ~ secondary)``
    (ported via :func:`numpy.polyfit`).
``"ccf"``
    Align series by the lag of maximum cross-correlation
    (a faithful re-implementation of ``stats::ccf``) then apply ``"fit"``.
``"sortfit"``
    Independently sort both inputs then apply ``"fit"``.

R source reference: ``help_secondary.R`` (functions ``help_secondary``,
``new_sec_axis``, ``help_sec_range``, ``help_sec_max``, ``help_sec_fit``,
``help_sec_ccf``, ``help_sec_sortfit``).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - pandas is part of the runtime env
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from ggplot2_py.scale import sec_axis, AxisSecondary
from ggplot2_py._compat import is_waiver

import scales

from ._cli import cli_abort
from ._rlang import arg_match0

__all__ = [
    "help_secondary",
    "_new_sec_axis",
    "_help_sec_range",
    "_help_sec_max",
    "_help_sec_fit",
    "_help_sec_ccf",
    "_help_sec_sortfit",
    "_help_sec_ccf_acf",
]

# Allowed ``method`` choices, in R's declaration order.
_METHODS: Tuple[str, ...] = ("range", "max", "fit", "ccf", "sortfit")

ArrayLike = Union[Sequence[float], np.ndarray, "pd.Series"]


# ---------------------------------------------------------------------------
# Input resolution (replaces R's NSE eval_tidy / as_label)
# ---------------------------------------------------------------------------

def _resolve(value: Any, data: Any) -> Tuple[np.ndarray, Optional[str]]:
    """Resolve a ``primary``/``secondary`` argument to a numeric array.

    Mirrors R's ``eval_tidy(enquo(x), data)`` plus ``as_label(x)`` for the
    default axis title.  Because Python has no bare-symbol capture, a string
    is treated as a column name looked up in ``data``; anything else is
    coerced to a numeric :class:`numpy.ndarray`.

    Parameters
    ----------
    value : Any
        A column-name string, a :class:`pandas.Series`, or any array-like /
        scalar sequence of numbers.
    data : Any
        A :class:`pandas.DataFrame` (or mapping) used to resolve string
        column names.  May be ``None``.

    Returns
    -------
    arr : numpy.ndarray
        The resolved values as a 1-D float array.
    name : str or None
        A label for the values (column name or ``Series.name``), used as the
        default secondary-axis title.  ``None`` when no label is available.
    """
    name: Optional[str] = None

    if isinstance(value, str):
        # Column-name string: resolve against ``data`` (R eval_tidy).
        name = value
        if data is None:
            cli_abort(
                "Cannot resolve column {.val %s}: `data` is `None`." % value
            )
        try:
            col = data[value]
        except Exception:
            cli_abort("Column {.val %s} not found in `data`." % value)
        arr = np.asarray(getattr(col, "to_numpy", lambda: col)(), dtype=float)
        return np.atleast_1d(arr), name

    if pd is not None and isinstance(value, pd.Series):
        nm = value.name
        name = None if nm is None else str(nm)
        return np.atleast_1d(np.asarray(value.to_numpy(), dtype=float)), name

    # Plain array-like / sequence / scalar.
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    return arr, name


# ---------------------------------------------------------------------------
# Public constructor
# ---------------------------------------------------------------------------

def help_secondary(
    data: Any = None,
    primary: ArrayLike = (0, 1),
    secondary: ArrayLike = (0, 1),
    method: str = "range",
    na_rm: bool = True,
    **kwargs: Any,
) -> AxisSecondary:
    """Construct a secondary axis with a fitted projection.

    Python port of R ``help_secondary`` (``help_secondary.R`` L70-106).

    The intent is to call this **before** building a plot.  The returned
    :class:`~ggplot2_py.scale.AxisSecondary` has its ``trans`` populated by a
    fitted reverse transformation and carries an extra ``proj`` attribute --
    a forward-projection callable that maps secondary data onto the primary
    axis (used as an aesthetic value, e.g. ``aes(y = sec.proj(psavert))``).

    Parameters
    ----------
    data : pandas.DataFrame or mapping or None, optional
        Data used to resolve ``primary``/``secondary`` when they are given as
        column-name strings.  Mirrors the ``data`` argument of the R
        function.  Default ``None``.
    primary, secondary : str or array_like, optional
        The primary and secondary values.  Either array-likes (numpy /
        pandas / sequences) or column-name strings resolved against ``data``.
        Replaces R's bare-symbol NSE expressions.  Default ``(0, 1)``.
    method : {'range', 'max', 'fit', 'ccf', 'sortfit'}, optional
        Fitting strategy (see module docstring).  Default ``'range'``.
    na_rm : bool, optional
        Whether to remove missing values (``True``) or propagate them
        (``False``).  Applies to ``method='range'`` and ``method='max'``.
        Default ``True``.
    **kwargs
        Forwarded to :func:`ggplot2_py.scale.sec_axis` (``name``, ``breaks``,
        ``labels``, ``guide``).  Mirrors ``@inheritDotParams
        ggplot2::sec_axis -trans``.

    Returns
    -------
    ggplot2_py.scale.AxisSecondary
        The secondary axis, with an added ``proj`` callable attribute and,
        when no ``name`` was supplied, ``name`` defaulted to the secondary
        label.

    See Also
    --------
    ggplot2_py.scale.sec_axis
    """
    method = arg_match0(method, _METHODS, arg_name="method")

    primary_vals, _ = _resolve(primary, data)
    secondary_vals, sec_name = _resolve(secondary, data)

    # ``name = as_label(secondary)`` (R L82). The label is derived during
    # resolution; fall back to the original string repr if unavailable.
    name = sec_name
    if name is None and isinstance(secondary, str):
        name = secondary

    if method == "range":
        help_ = _help_sec_range(primary_vals, secondary_vals, na_rm=na_rm)
    elif method == "max":
        help_ = _help_sec_max(primary_vals, secondary_vals, na_rm=na_rm)
    elif method == "fit":
        help_ = _help_sec_fit(primary_vals, secondary_vals)
    elif method == "ccf":
        help_ = _help_sec_ccf(primary_vals, secondary_vals)
    else:  # "sortfit"
        help_ = _help_sec_sortfit(primary_vals, secondary_vals)

    # R: ggproto(NULL, new_sec_axis(trans = help$reverse, ...), proj = help$forward)
    out = _new_sec_axis(trans=help_["reverse"], **kwargs)
    # Attach the forward projection. AxisSecondary is a plain class, so this
    # extra attribute is safe (R stores it as the ggproto `proj` member).
    out.proj = help_["forward"]

    # R: if (inherits(out$name, "waiver")) out$name <- name
    if is_waiver(out.name) and name is not None:
        out.name = name

    return out


def _new_sec_axis(trans: Optional[Callable] = None, **kwargs: Any) -> AxisSecondary:
    """Bridge ``trans``/``transform`` to :func:`ggplot2_py.scale.sec_axis`.

    Python port of R ``new_sec_axis`` (``help_secondary.R`` L109-115), which
    renames ``trans`` to ``transform`` for ggplot2 >= 3.5.0.  ``sec_axis`` in
    ggplot2_py already accepts both ``transform`` and ``trans``; passing
    ``transform`` avoids the deprecation warning emitted for ``trans``.

    Parameters
    ----------
    trans : callable, optional
        The reverse transformation function.
    **kwargs
        Additional ``sec_axis`` arguments (``name``, ``breaks``, ``labels``,
        ``guide``).

    Returns
    -------
    ggplot2_py.scale.AxisSecondary
    """
    return sec_axis(transform=trans, **kwargs)


# ---------------------------------------------------------------------------
# Range helpers
# ---------------------------------------------------------------------------

def _range(x: np.ndarray, na_rm: bool) -> Tuple[float, float]:
    """Compute ``base::range(x, na.rm=na_rm)``.

    Parameters
    ----------
    x : numpy.ndarray
        Numeric values.
    na_rm : bool
        If ``True``, ignore NaN (``nanmin``/``nanmax``); if ``False``,
        propagate NaN so the result is ``(nan, nan)`` when any value is NaN
        (matching R's ``range(x, na.rm = FALSE)``).

    Returns
    -------
    tuple of float
        ``(min, max)``.
    """
    x = np.asarray(x, dtype=float)
    if na_rm:
        return float(np.nanmin(x)), float(np.nanmax(x))
    return float(np.min(x)), float(np.max(x))


def _help_sec_range(
    from_: np.ndarray, to: np.ndarray, na_rm: bool = True
) -> Dict[str, Callable]:
    """Range-overlap projection.

    Python port of R ``help_sec_range`` (``help_secondary.R`` L119-130).

    Parameters
    ----------
    from_ : numpy.ndarray
        Primary values.
    to : numpy.ndarray
        Secondary values.
    na_rm : bool, optional
        Passed to :func:`_range`.  Default ``True``.

    Returns
    -------
    dict
        ``{'forward': callable, 'reverse': callable}`` where ``forward`` maps
        secondary -> primary and ``reverse`` maps primary -> secondary, both
        via :func:`scales.rescale`.
    """
    from_rng = _range(from_, na_rm=na_rm)
    to_rng = _range(to, na_rm=na_rm)

    def forward(x: ArrayLike) -> np.ndarray:
        return scales.rescale(_num(x), to=from_rng, from_range=to_rng)

    def reverse(x: ArrayLike) -> np.ndarray:
        return scales.rescale(_num(x), to=to_rng, from_range=from_rng)

    return {"forward": forward, "reverse": reverse}


def _help_sec_max(
    from_: np.ndarray, to: np.ndarray, na_rm: bool = True
) -> Dict[str, Callable]:
    """Maxima-coincidence projection.

    Python port of R ``help_sec_max`` (``help_secondary.R`` L132-143).

    Parameters
    ----------
    from_ : numpy.ndarray
        Primary values.
    to : numpy.ndarray
        Secondary values.
    na_rm : bool, optional
        Passed to :func:`_range`.  Default ``True``.

    Returns
    -------
    dict
        ``{'forward': callable, 'reverse': callable}`` via
        :func:`scales.rescale_max`.
    """
    from_rng = _range(from_, na_rm=na_rm)
    to_rng = _range(to, na_rm=na_rm)

    def forward(x: ArrayLike) -> np.ndarray:
        return scales.rescale_max(_num(x), to=from_rng, from_range=to_rng)

    def reverse(x: ArrayLike) -> np.ndarray:
        return scales.rescale_max(_num(x), to=to_rng, from_range=from_rng)

    return {"forward": forward, "reverse": reverse}


# ---------------------------------------------------------------------------
# Linear-fit helpers
# ---------------------------------------------------------------------------

def _help_sec_fit(from_: np.ndarray, to: np.ndarray) -> Dict[str, Callable]:
    """Linear-model projection.

    Python port of R ``help_sec_fit`` (``help_secondary.R`` L145-159).
    Uses ``coef(lm(from ~ to))``; the port computes the same coefficients
    with :func:`numpy.polyfit` on ``(to, from_)``.  Like R's ``lm`` (whose
    default ``na.action`` is ``na.omit``), pairs containing NaN in either
    series are dropped before fitting.

    Parameters
    ----------
    from_ : numpy.ndarray
        Primary values (the response in ``lm``).
    to : numpy.ndarray
        Secondary values (the predictor in ``lm``).

    Returns
    -------
    dict
        ``{'forward': callable, 'reverse': callable}``.  ``forward(x) =
        intercept + x * slope`` (secondary -> primary); ``reverse(x) =
        (x - intercept) / slope`` (primary -> secondary).

    Raises
    ------
    ValueError
        If ``from_`` and ``to`` have unequal length (via :func:`cli_abort`).
    """
    from_ = np.asarray(from_, dtype=float)
    to = np.asarray(to, dtype=float)
    if from_.shape[0] != to.shape[0]:
        cli_abort("The primary and secondary values must have the same length.")

    # R lm() drops rows where either variable is NA (na.action = na.omit).
    mask = ~(np.isnan(from_) | np.isnan(to))
    f_fit = from_[mask]
    t_fit = to[mask]

    # coef(lm(from ~ to)) = c(intercept, slope).
    # numpy.polyfit(to, from, 1) returns [slope, intercept]; reverse the order.
    poly = np.polyfit(t_fit, f_fit, 1)
    slope = float(poly[0])
    intercept = float(poly[1])

    def forward(x: ArrayLike) -> np.ndarray:
        return intercept + _num(x) * slope

    def reverse(x: ArrayLike) -> np.ndarray:
        return (_num(x) - intercept) / slope

    return {"forward": forward, "reverse": reverse}


def _help_sec_ccf_acf(from_: np.ndarray, to: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cross-correlation function, faithful to ``stats::ccf``.

    Reproduces R's ``ccf(from, to, lag.max = n - 1, plot = FALSE)`` exactly.
    R de-means both series, then for lag ``k`` correlates ``from[t + k]`` with
    ``to[t]``, dividing the cross-covariance by ``n`` and by the product of
    the two series' sample standard deviations (each computed with a ``1/n``
    divisor).  Lags run from ``-(n - 1)`` to ``(n - 1)``.

    Parameters
    ----------
    from_ : numpy.ndarray
        First series (``x``).
    to : numpy.ndarray
        Second series (``y``).  Must have the same length as ``from_``.

    Returns
    -------
    lags : numpy.ndarray
        Integer lags from ``-(n - 1)`` to ``(n - 1)``.
    acf : numpy.ndarray
        The cross-correlation at each lag, matching ``ccf$acf``.
    """
    from_ = np.asarray(from_, dtype=float)
    to = np.asarray(to, dtype=float)
    n = from_.shape[0]

    xc = from_ - np.mean(from_)
    yc = to - np.mean(to)

    # Denominator: sqrt( (sum(xc^2)/n) * (sum(yc^2)/n) ), i.e. product of the
    # 1/n-normalised sample standard deviations of each (de-meaned) series.
    denom = np.sqrt((np.sum(xc ** 2) / n) * (np.sum(yc ** 2) / n))

    # np.correlate(xc, yc, 'full')[i] == sum_t xc[t + (i - (n-1))] * yc[t],
    # which is exactly R's lag convention. Length is 2n - 1.
    cross = np.correlate(xc, yc, mode="full") / n
    acf = cross / denom
    lags = np.arange(-(n - 1), n)
    return lags, acf


def _help_sec_ccf(from_: np.ndarray, to: np.ndarray) -> Dict[str, Callable]:
    """Cross-correlation-aligned linear projection.

    Python port of R ``help_sec_ccf`` (``help_secondary.R`` L161-178).
    Finds the lag of maximum cross-correlation (:func:`_help_sec_ccf_acf`),
    truncates the two series to align them at that lag, then delegates to
    :func:`_help_sec_fit`.

    Parameters
    ----------
    from_ : numpy.ndarray
        Primary values.
    to : numpy.ndarray
        Secondary values.  Must have the same length as ``from_``.

    Returns
    -------
    dict
        ``{'forward': callable, 'reverse': callable}`` from the fit on the
        aligned data.

    Raises
    ------
    ValueError
        If ``from_`` and ``to`` have unequal length (via :func:`cli_abort`).
    """
    from_ = np.asarray(from_, dtype=float)
    to = np.asarray(to, dtype=float)
    n = from_.shape[0]
    if n != to.shape[0]:
        cli_abort("The primary and secondary values must have the same length.")

    lags, acf = _help_sec_ccf_acf(from_, to)
    # which.max returns the FIRST maximum; np.argmax matches that tie rule.
    lag = int(lags[int(np.argmax(acf))])

    # No block for 0-lag because the data is already optimally aligned.
    if np.sign(lag) == 1:
        # R: from <- tail(from, -lag); to <- head(to, -lag)
        from_ = from_[lag:]
        to = to[:-lag]
    elif np.sign(lag) == -1:
        # R: from <- head(from, lag); to <- tail(to, lag)
        # head(x, lag) with lag<0 drops the last |lag|; tail(x, lag) keeps last |lag|.
        from_ = from_[:lag]
        to = to[-(-lag):]

    return _help_sec_fit(from_=from_, to=to)


def _help_sec_sortfit(from_: np.ndarray, to: np.ndarray) -> Dict[str, Callable]:
    """Sorted linear projection.

    Python port of R ``help_sec_sortfit`` (``help_secondary.R`` L180-182).
    Independently sorts both series (dropping NaN, as R's ``sort`` does by
    default) then delegates to :func:`_help_sec_fit`.

    Parameters
    ----------
    from_ : numpy.ndarray
        Primary values.
    to : numpy.ndarray
        Secondary values.

    Returns
    -------
    dict
        ``{'forward': callable, 'reverse': callable}``.
    """
    return _help_sec_fit(from_=_sort(from_), to=_sort(to))


# ---------------------------------------------------------------------------
# Small numeric utilities
# ---------------------------------------------------------------------------

def _num(x: ArrayLike) -> np.ndarray:
    """Coerce an aesthetic/array value to a float :class:`numpy.ndarray`.

    Parameters
    ----------
    x : array_like
        Value to coerce (handles pandas Series via ``to_numpy``).

    Returns
    -------
    numpy.ndarray
        Float array (scalars become 0-D arrays, preserved by NumPy ops).
    """
    if pd is not None and isinstance(x, pd.Series):
        x = x.to_numpy()
    return np.asarray(x, dtype=float)


def _sort(x: np.ndarray) -> np.ndarray:
    """Sort ascending, dropping NaN (mirrors R ``sort`` defaults).

    Parameters
    ----------
    x : numpy.ndarray
        Values to sort.

    Returns
    -------
    numpy.ndarray
        Sorted finite values (NaN removed).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return np.sort(x)
