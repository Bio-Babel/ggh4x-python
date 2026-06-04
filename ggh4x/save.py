"""Save a ggplot with automatic size measurement (R source: save.R).

Port of ggh4x's :func:`save_plot`, a wrapper over :func:`ggplot2_py.save.ggsave`
that guesses the plot's physical size from the built gtable. The guess is only
meaningful when the panels have an absolute size -- set either via
``theme(panel.widths=, panel.heights=)`` or via ``force_panelsizes()`` -- because
with the default *null* (proportional) panel sizing the measured dimension is
undefined and the current graphics-device size is used instead.

Deviation from R
----------------
R's ``save_plot`` returns the file-name string with ``width`` and ``height``
attributes (in inches) attached via ``attr<-``. Python strings cannot carry
attributes, so this port returns a :class:`SavePlotResult` -- a thin ``str``
subclass that *is* the path everywhere a string is expected, but additionally
exposes ``.width`` and ``.height`` (inches, or ``None`` when the size could not
be measured, matching R's ``NA_real_``).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ._cli import cli_abort
from ._rlang import arg_match0
from ._utils import has_null_unit, height_cm, width_cm

__all__ = [
    "save_plot",
    "has_null_unit",
    "SavePlotResult",
]


# Inches-per-unit divisors used by R's
# ``switch(units, `in` = 1, cm = 2.54, mm = 25.4, px = dpi)`` (save.R L52, L63).
# ``px`` is handled specially because its divisor is the runtime ``dpi``.
_UNIT_DIVISOR = {"in": 1.0, "cm": 2.54, "mm": 25.4}

_VALID_UNITS = ("in", "cm", "mm", "px")


class SavePlotResult(str):
    """File path returned by :func:`save_plot`, carrying the plot size.

    A :class:`str` subclass so it behaves as the output file path everywhere
    (``open(result)``, ``result == path``, ``os.fspath(result)``), while also
    exposing the inferred plot dimensions. Mirrors R attaching ``width`` and
    ``height`` attributes to the returned file-name string.

    Attributes
    ----------
    width : float or None
        Plot width in inches, or ``None`` when it could not be measured
        (R's ``NA_real_``; the device size is used by ``ggsave`` in that case).
    height : float or None
        Plot height in inches, or ``None`` when it could not be measured.
    """

    width: Optional[float]
    height: Optional[float]

    def __new__(
        cls,
        value: str,
        width: Optional[float] = None,
        height: Optional[float] = None,
    ) -> "SavePlotResult":
        obj = super().__new__(cls, value)
        obj.width = width
        obj.height = height
        return obj

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SavePlotResult({str.__repr__(self)}, "
            f"width={self.width!r}, height={self.height!r})"
        )


def _build_gtable(plot: Any) -> Any:
    """Build the gtable for *plot*, mirroring R's ``ggplotGrob(plot)``.

    Parameters
    ----------
    plot : GGPlot or patchwork-like
        A ggplot, or any object exposing ``to_gtable()`` (e.g. a
        patchwork composition).

    Returns
    -------
    Gtable
        The assembled gtable whose ``widths`` / ``heights`` are measured.
    """
    from ggplot2_py.plot import ggplot_build, ggplot_gtable, is_ggplot

    if is_ggplot(plot):
        built = ggplot_build(plot)
        return ggplot_gtable(built)
    if hasattr(plot, "to_gtable"):
        # patchwork-python compositions expose to_gtable(); ggsave branches the
        # same way (ggplot2_py.save.ggsave).
        return plot.to_gtable()
    # Already a gtable (or grob) -- use as-is, consistent with ggsave's fallback.
    return plot


def _measure_inches(track_units: Any, axis: str) -> Optional[float]:
    """Measure a gtable track length in inches, or ``None`` if size is *null*.

    Mirrors save.R L44-50 / L55-61: if the track contains any top-level *null*
    unit the dimension is undefined (R ``NA_real_`` -> Python ``None``);
    otherwise it is ``sum(width_cm(track)) / 2.54``.

    Parameters
    ----------
    track_units : Unit
        The gtable ``widths`` or ``heights`` unit vector.
    axis : str
        ``"width"`` or ``"height"``; selects ``width_cm`` vs ``height_cm``.

    Returns
    -------
    float or None
        The length in inches, or ``None`` when a null unit is present.
    """
    if has_null_unit(track_units):
        return None
    measure = width_cm if axis == "width" else height_cm
    cm = np.asarray(measure(track_units), dtype=float)
    return float(np.sum(cm)) / 2.54


def _supplied_to_inches(value: float, units: str, dpi: float) -> float:
    """Convert a user-supplied dimension from *units* to inches.

    Mirrors R ``value / switch(units, `in`=1, cm=2.54, mm=25.4, px=dpi)``
    (save.R L52, L63).

    Parameters
    ----------
    value : float
        Dimension expressed in *units*.
    units : str
        One of ``"in"``, ``"cm"``, ``"mm"``, ``"px"``.
    dpi : float
        Resolution, used as the divisor when ``units == "px"``.

    Returns
    -------
    float
        The dimension in inches.
    """
    if units == "px":
        return value / dpi
    return value / _UNIT_DIVISOR[units]


def save_plot(
    *args: Any,
    plot: Any = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
    units: str = "in",
    dpi: float = 300,
    **kwargs: Any,
) -> SavePlotResult:
    """Save a ggplot, guessing its size from absolute panel dimensions.

    Wrapper over :func:`ggplot2_py.save.ggsave` that measures the plot size from
    the built gtable when *width* / *height* are not given. The measurement is
    only well defined when the panels have a fixed size (set via
    ``theme(panel.widths=, panel.heights=)`` or ``force_panelsizes()``); with the
    default proportional (*null*) panels the dimension is left undefined and the
    graphics-device size is used.

    Parameters
    ----------
    *args
        Positional arguments forwarded to :func:`ggsave`; the first is the
        output ``filename`` (R passes ``...`` through verbatim).
    plot : GGPlot or patchwork-like, optional
        The plot to save. If ``None``, the last displayed plot is used
        (``get_last_plot()`` inside ``ggsave``); the size is then measured from
        that same plot.
    width, height : float, optional
        Plot size in *units*. If ``None`` (default) the size is measured from the
        plot; when the plot has no fixed size the value becomes ``None`` and
        ``ggsave`` falls back to the device size (R's ``NA``).
    units : {"in", "cm", "mm", "px"}, optional
        Units of supplied *width* / *height*. Default ``"in"``.
    dpi : float, optional
        Resolution in dots per inch (default ``300``). Also the px-to-inch
        divisor when ``units == "px"``.
    **kwargs
        Further keyword arguments forwarded to :func:`ggsave` (e.g. ``device``,
        ``path``, ``bg``, ``limitsize``, ``scale``).

    Returns
    -------
    SavePlotResult
        The output file path (a ``str`` subclass) with ``.width`` and
        ``.height`` attributes in inches (``None`` when not measured).

    Raises
    ------
    ValueError
        If *units* is not one of ``"in"``, ``"cm"``, ``"mm"``, ``"px"``.

    Notes
    -----
    R attaches ``width`` / ``height`` attributes to the returned file name; this
    port exposes them on the returned :class:`SavePlotResult` instead, because a
    plain Python ``str`` cannot carry attributes.
    """
    units = arg_match0(units, _VALID_UNITS, arg_name="units")

    # Resolve the plot once so the measured gtable matches the saved plot, even
    # when plot=None defers to the last-displayed plot.
    if plot is None:
        from ggplot2_py.plot import get_last_plot

        plot = get_last_plot()
        if plot is None:
            cli_abort("No plot to save. Supply `plot` or create a plot first.")

    gt = _build_gtable(plot)

    # --- width (save.R L44-53) ---
    if width is None:
        width = _measure_inches(gt.widths, "width")
    else:
        width = _supplied_to_inches(width, units, float(dpi))

    # --- height (save.R L55-64) ---
    if height is None:
        height = _measure_inches(gt.heights, "height")
    else:
        height = _supplied_to_inches(height, units, float(dpi))

    from ggplot2_py.save import ggsave

    out_file = ggsave(
        *args,
        plot=plot,
        width=width,
        height=height,
        units="in",
        dpi=dpi,
        **kwargs,
    )

    return SavePlotResult(str(out_file), width=width, height=height)
