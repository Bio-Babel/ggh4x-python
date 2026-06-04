"""Manual hybrid discrete/continuous position scales (R source: ``scale_manual.R``).

Ports ggh4x's :func:`scale_x_manual` / :func:`scale_y_manual` and the
:class:`ScaleManualPosition` ggproto.  These behave like discrete position scales
(accepting discrete data and limits) but place each discrete level at an arbitrary
*continuous* coordinate, which needn't be equally spaced.

The load-bearing trick (per the R comment at ``scale_manual.R:169``) is that the
scale's continuous range (:attr:`range_c`) is trained with the scale expansion
**already baked in** at :meth:`ScaleManualPosition.train` time — this is what lets
discrete labels land at the requested continuous positions.

The :func:`sep_discrete` helper (a function factory that maps separator-delimited
grouped labels to numeric positions) lives in :mod:`ggh4x.conveniences` and is
re-exported here for convenience.

All semantics were verified against a live R ``ggh4x`` session
(``ScaleManualPosition$train``/``$map`` ``range_c`` and mapped values for plain,
named, ``c_limits`` and ``sep_discrete`` value vectors).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

import scales as _scales

from ggplot2_py._compat import waiver, is_waiver
from ggplot2_py.scale import (
    ScaleDiscretePosition,
    discrete_scale,
    expansion,
    mapped_discrete,
    _is_discrete,
)

from .._cli import cli_abort, cli_warn
from ..conveniences import sep_discrete

__all__ = [
    "scale_x_manual",
    "scale_y_manual",
    "ScaleManualPosition",
    "sep_discrete",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _is_discrete_input(x: Any) -> bool:
    """Return ``True`` for discrete data (R ``is.discrete``: factor/char/logical).

    Port of ``scale_manual.R:259-261``.  Delegates to ggplot2_py's
    :func:`_is_discrete` which covers pandas Categoricals/object Series, string and
    boolean ``ndarray``\\ s, and ``str``/``bool`` scalars and sequences.

    Parameters
    ----------
    x : Any
        Candidate data.

    Returns
    -------
    bool
    """
    return _is_discrete(x)


def _values_names(values: Any) -> Optional[List[str]]:
    """Return the names of a named ``values`` mapping, or ``None`` if unnamed.

    A dict-like ``values`` (the Python idiom for R's named numeric vector) carries
    names as its keys; anything else is unnamed.

    Parameters
    ----------
    values : Any

    Returns
    -------
    list of str or None
    """
    if isinstance(values, dict):
        return [str(k) for k in values.keys()]
    return None


def _values_numeric(values: Any) -> np.ndarray:
    """Return the numeric magnitudes of ``values`` (dict values or the vector)."""
    if isinstance(values, dict):
        return np.asarray(list(values.values()), dtype=float)
    return np.asarray(values, dtype=float)


# ---------------------------------------------------------------------------
# ScaleManualPosition ggproto class  (scale_manual.R:155-206)
# ---------------------------------------------------------------------------
class ScaleManualPosition(ScaleDiscretePosition):
    """Position scale placing discrete levels at arbitrary continuous coordinates.

    Subclass of :class:`ggplot2_py.scale.ScaleDiscretePosition` ported from R
    ``ScaleManualPosition`` (``scale_manual.R:155-206``).  Overrides
    :meth:`train` and :meth:`map` entirely.

    Attributes
    ----------
    c_limits : numpy.ndarray or None
        Optional length-2 continuous-limit override (``NaN`` entries fall back to
        the data-derived range), set by :func:`_scale_position_manual`.
    range_c : scales.ContinuousRange
        The continuous range, trained with the expansion already applied at
        :meth:`train` time.
    """

    c_limits: Any = None

    def train(self, x: Any) -> None:
        """Train the continuous range, baking in the scale expansion.

        Port of R ``ScaleManualPosition$train`` (``scale_manual.R:158-174``).
        For discrete *x* the discrete range is trained and the value range is the
        range of the palette over the limits; for continuous *x* the value range
        is ``range(x)``.  Any non-``NaN`` :attr:`c_limits` overrides the
        corresponding bound.  The expansion (``self.expand`` or
        ``expansion(add=0.6)``) is then applied and the expanded bounds trained
        into :attr:`range_c`.

        Parameters
        ----------
        x : array-like
            Layer data for this position aesthetic.
        """
        if _is_discrete_input(x):
            self.range.train(
                x, drop=self.drop, na_rm=not self.na_translate
            )
            pal = self.palette(self.get_limits())
            # A named palette (R named numeric vector) is a dict here; range() in R
            # operates on the magnitudes regardless of names.
            pal_arr = _values_numeric(pal)
            rng = np.array([np.nanmin(pal_arr), np.nanmax(pal_arr)], dtype=float)
        else:
            x_arr = np.asarray(x, dtype=float)
            rng = np.array([np.nanmin(x_arr), np.nanmax(x_arr)], dtype=float)

        # c_limits override (scale_manual.R:166-168): NA-aware ifelse.
        if self.c_limits is not None:
            c_lim = np.asarray(self.c_limits, dtype=float)
            rng = np.where(np.isnan(c_lim), rng, c_lim)

        # Hack for scale expansion (scale_manual.R:169-173). expansion() returns
        # [mul_lo, add_lo, mul_hi, add_hi]; expand_range(range, mul, add).
        expand = self.expand if not is_waiver(self.expand) else expansion(add=0.6)
        expand = np.asarray(expand, dtype=float)
        lower = _scales.expand_range(rng, expand[0], expand[1])[0]
        upper = _scales.expand_range(rng, expand[2], expand[3])[1]
        self.range_c.train(np.array([lower, upper], dtype=float))

    def map(self, x: Any, limits: Optional[Any] = None) -> Any:
        """Map discrete *x* to continuous positions via the manual palette.

        Port of R ``ScaleManualPosition$map`` (``scale_manual.R:176-205``).  For
        discrete *x* the palette is resolved (with the
        ``n_breaks_cache``/``palette_cache`` cache), honouring a *named* palette
        (match by name, blanking names absent from *limits*) or a positional match
        against *limits*; missing values are filled with :attr:`na_value` when
        :attr:`na_translate`.  Continuous *x* passes straight through.  The result
        is always wrapped as a :func:`mapped_discrete` sentinel.

        Parameters
        ----------
        x : array-like
            Data to map.
        limits : array-like, optional
            Scale limits.  Defaults to :meth:`get_limits`.

        Returns
        -------
        ggplot2_py.scale._MappedDiscrete
            The mapped continuous positions.
        """
        if limits is None:
            limits = self.get_limits()

        if _is_discrete_input(x):
            limits_list = list(limits) if limits is not None else []
            n = sum(
                1
                for v in limits_list
                if not (v is None or (isinstance(v, float) and np.isnan(v)))
            )

            if self.n_breaks_cache is not None and self.n_breaks_cache == n:
                pal = self.palette_cache
            else:
                if self.n_breaks_cache is not None:
                    cli_warn("Cached palette does not match requested.")
                pal = self.palette(limits)
                self.palette_cache = pal
                self.n_breaks_cache = n

            pal_names = _values_names(pal)
            limits_str = [str(v) for v in limits_list]
            x_str = [str(v) for v in np.asarray(x)]

            if pal_names is not None:
                # Named-palette branch (scale_manual.R:190-194).
                pal_vals = list(_values_numeric(pal))
                # Blank entries whose names are not in the limits.
                pal_vals = [
                    (np.nan if pal_names[i] not in limits_str else pal_vals[i])
                    for i in range(len(pal_names))
                ]
                name_to_val = {pal_names[i]: pal_vals[i] for i in range(len(pal_names))}
                pal_match = np.array(
                    [name_to_val.get(v, np.nan) for v in x_str], dtype=float
                )
            else:
                # Positional branch (scale_manual.R:196): pal[match(x, limits)].
                pal_vals = np.asarray(pal, dtype=float)
                idx = {v: i for i, v in enumerate(limits_str)}
                pal_match = np.array(
                    [
                        pal_vals[idx[v]] if v in idx and idx[v] < len(pal_vals) else np.nan
                        for v in x_str
                    ],
                    dtype=float,
                )

            if self.na_translate:
                x_is_na = np.array(
                    [
                        v is None or (isinstance(v, float) and np.isnan(v)) or v == "nan"
                        for v in x_str
                    ]
                )
                na_fill = float(self.na_value) if _is_number(self.na_value) else np.nan
                fill_mask = x_is_na | np.isnan(pal_match)
                pal_match = np.where(fill_mask, na_fill, pal_match)

            x = pal_match

        return mapped_discrete(np.asarray(x, dtype=float))


def _is_number(v: Any) -> bool:
    """Return ``True`` for a real numeric scalar (used for ``na_value`` coercion)."""
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


# ---------------------------------------------------------------------------
# internal constructor  (scale_manual.R:82-151)
# ---------------------------------------------------------------------------
def _scale_position_manual(
    aesthetics: Sequence[str],
    values: Any = None,
    *,
    limits: Any = None,
    c_limits: Any = None,
    breaks: Any = None,
    expand: Any = None,
    guide: Any = None,
    position: str = "bottom",
    **kwargs: Any,
) -> ScaleManualPosition:
    """Build a :class:`ScaleManualPosition` (R ``scale_position_manual``).

    Port of ``scale_manual.R:82-151``.

    Parameters
    ----------
    aesthetics : sequence of str
        The position aesthetics (e.g. ``["x", "xmin", "xmax", "xend"]``).
    values : callable or numeric or dict
        A palette function ``limits -> numeric``, or a numeric vector (optionally
        named via a ``dict``) of positions parallel to the unique values.
    limits : array-like or callable, optional
        Scale limits.  Defaults to intersecting the data with the ``values`` names
        when *values* is named.
    c_limits : array-like or None, optional
        ``None`` to use the value range, or a length-2 numeric (``NaN`` entries
        fall back to the value range) for custom continuous limits.
    breaks : array-like, optional
        Breaks; when *values* is an unnamed vector these also name it.
    expand : array-like or Waiver, optional
        Scale expansion.
    guide : Any, optional
        Guide spec.
    position : str, default ``"bottom"``
        Axis position.
    **kwargs : Any
        Extra arguments forwarded to :func:`ggplot2_py.discrete_scale`.

    Returns
    -------
    ScaleManualPosition

    Raises
    ------
    ValueError
        If *values* is neither a function nor numeric, or if *c_limits* is not
        ``None`` or a length-2 numeric vector.
    """
    if breaks is None:
        breaks = waiver()
    if expand is None:
        expand = waiver()
    if guide is None:
        guide = waiver()

    if not callable(values):
        # Validate is.numeric before extracting magnitudes (scale_manual.R:96-101).
        if not (isinstance(values, dict) or _is_numeric_vector(values)):
            cli_abort("The `values` argument must be `numeric`.", TypeError)

        names = _values_names(values)
        # If limits is None and values has names -> intersect-with-names limits.
        if limits is None and names is not None:
            names_set = list(names)

            def _limits_fn(x: Any, _names: List[str] = names_set) -> List[str]:
                # intersect(x, names(values)) %||% character()
                xs = [str(v) for v in x]
                return [v for v in xs if v in _names]

            limits = _limits_fn

        # Unnamed vector + breaks given -> name the values by breaks.
        if (
            names is None
            and not is_waiver(breaks)
            and breaks is not None
            and not callable(breaks)
        ):
            brks = list(breaks)
            vals = list(_values_numeric(values))
            if len(brks) <= len(vals):
                values = {str(brks[i]): vals[i] for i in range(len(brks))}
            else:
                values = {str(brks[i]): vals[i] for i in range(len(vals))}

        # Build the palette (scale_manual.R:118-126).
        def _pal(lims: Any, _values: Any = values) -> np.ndarray:
            vals = _values_numeric(_values)
            lims_list = list(lims) if lims is not None else []
            if len(lims_list) > len(vals):
                cli_abort(
                    f"Insufficient values in manual scale. {len(lims_list)} needed "
                    f"but {len(vals)} provided."
                )
            out = vals[: len(lims_list)]
            names2 = _values_names(_values)
            if names2 is not None:
                # Carry the names so map()'s named-palette branch fires.
                return {names2[i]: out[i] for i in range(len(out))}
            return out

        pal = _pal
    else:
        pal = values

    # Validate c_limits (scale_manual.R:131-136).
    if c_limits is not None:
        c_arr = np.asarray(c_limits, dtype=float)
        if c_arr.ndim != 1 or c_arr.size != 2:
            cli_abort(
                "The `c_limits` argument must either be `None` or a `numeric` "
                "vector of length 2."
            )

    sc = discrete_scale(
        list(aesthetics),
        pal,
        limits=limits,
        expand=expand,
        guide=guide,
        position=position,
        super_class=ScaleManualPosition,
        **kwargs,
    )
    sc.range_c = _scales.ContinuousRange()
    sc.c_limits = c_limits
    return sc


def _is_numeric_vector(values: Any) -> bool:
    """Return ``True`` when *values* is a numeric vector (R ``is.numeric``)."""
    if isinstance(values, (int, float, np.integer, np.floating)) and not isinstance(values, bool):
        return True
    if isinstance(values, (list, tuple, np.ndarray)):
        if len(values) == 0:
            return True
        try:
            np.asarray(values, dtype=float)
            return True
        except (TypeError, ValueError):
            return False
    return False


# ---------------------------------------------------------------------------
# external constructors  (scale_manual.R:48-78)
# ---------------------------------------------------------------------------
def scale_x_manual(
    values: Any,
    c_limits: Any = None,
    position: str = "bottom",
    **kwargs: Any,
) -> ScaleManualPosition:
    """A hybrid discrete/continuous manual position scale for ``x``.

    Port of R ``scale_x_manual`` (``scale_manual.R:48-61``).  Accepts discrete
    input like a discrete scale but maps each level to an arbitrary continuous
    coordinate.

    Parameters
    ----------
    values : callable or numeric or dict
        A numeric vector with the same length as the unique values (optionally
        named via a ``dict``), or a function accepting the limits and returning a
        parallel numeric vector (see :func:`sep_discrete`).
    c_limits : array-like or None, default ``None``
        ``None`` to use the value range as the continuous limits, or a length-2
        numeric (``NaN`` entries fall back to the value range) for custom limits.
    position : str, default ``"bottom"``
        Axis position.
    **kwargs : Any
        Extra arguments forwarded to :func:`ggplot2_py.discrete_scale`.

    Returns
    -------
    ScaleManualPosition
    """
    return _scale_position_manual(
        ["x", "xmin", "xmax", "xend"],
        values=values,
        c_limits=c_limits,
        position=position,
        **kwargs,
    )


def scale_y_manual(
    values: Any,
    c_limits: Any = None,
    position: str = "left",
    **kwargs: Any,
) -> ScaleManualPosition:
    """A hybrid discrete/continuous manual position scale for ``y``.

    Port of R ``scale_y_manual`` (``scale_manual.R:65-78``).  Behaves like
    :func:`scale_x_manual` but for the ``y`` aesthetic.

    Parameters
    ----------
    values : callable or numeric or dict
        Positions for the unique values, or a function mapping limits to positions.
    c_limits : array-like or None, default ``None``
        Continuous-limit override (see :func:`scale_x_manual`).
    position : str, default ``"left"``
        Axis position.
    **kwargs : Any
        Extra arguments forwarded to :func:`ggplot2_py.discrete_scale`.

    Returns
    -------
    ScaleManualPosition
    """
    return _scale_position_manual(
        ["y", "ymin", "ymax", "yend"],
        values=values,
        c_limits=c_limits,
        position=position,
        **kwargs,
    )
