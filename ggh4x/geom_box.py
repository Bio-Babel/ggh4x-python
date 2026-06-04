"""Flexible labelled rectangles (port of ggh4x ``geom_box.R``).

This module ports the R ggh4x ``geom_box()`` constructor, the ``GeomBox``
ggproto class and the ``resolve_box()`` helper.  ``geom_box()`` is a more
flexible variant of :func:`ggplot2_py.geom_rect` / ``geom_tile``: instead of
requiring either the ``(x/y)min``/``(x/y)max`` *or* the ``(x/y)``/
``(width/height)`` aesthetics, *any two* out of those four aesthetics suffice
to define a rectangle (per axis).

R source: ``ggh4x/R/geom_box.R``.

Notes
-----
* :func:`resolve_box` resolves the ``min``/``max`` of one axis from partial
  information with the priority cascade ``min``/``max`` verbatim ->
  ``center +/- 0.5 * dim`` -> ``opposite +/- dim``.  The final ``pmin``/``pmax``
  normalisation propagates ``NA`` (``na.rm = FALSE`` semantics) exactly like R.
* :meth:`GeomBox.setup_data` resolves both axes, emits a :func:`cli_warn`
  (never an error) listing the unresolved aesthetics with axis-specific tips,
  then writes ``xmin``/``xmax``/``ymin``/``ymax`` and drops ``x``/``width``/
  ``y``/``height``.
* :meth:`GeomBox.draw_panel` builds plain rectangles under linear coords
  (``rect_grob`` when ``radius is None``, otherwise one ``roundrect_grob`` per
  row), and expands each rectangle to a five-vertex polygon delegated to
  :class:`ggplot2_py.GeomPolygon` under non-linear coords (``radius`` ignored).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ggplot2_py import ggproto, ggproto_parent
from ggplot2_py.geom import (
    Geom,
    GeomPolygon,
    FromTheme,
    Mapping,
    PT,
    _coord_transform,
    _fill_alpha,
    _ggname,
    _mix_ink_paper,
    draw_key_polygon,
)
from grid_py import (
    Gpar,
    Unit,
    grob_tree,
    is_unit,
    null_grob,
    rect_grob,
    roundrect_grob,
)

from ggh4x._cli import cli_warn
from ggh4x._vctrs import vec_interleave

__all__ = [
    "geom_box",
    "GeomBox",
    "resolve_box",
]


# ---------------------------------------------------------------------------
# resolve_box helper
# ---------------------------------------------------------------------------
def _as_float_array(x: Any, n: int) -> Optional[np.ndarray]:
    """Coerce a column-like input to a length-``n`` float array, or ``None``.

    Mirrors R's treatment of a possibly-``NULL`` vector that participates in
    :func:`resolve_box`.  ``None`` propagates (so the ``%||%`` fallback can fire),
    otherwise the value is coerced to ``float`` with ``None``/``pd.NA`` mapped to
    ``np.nan``.

    Parameters
    ----------
    x : Any
        A scalar, sequence, ``pandas.Series`` or ``None``.
    n : int
        Target length (the recycled row count).

    Returns
    -------
    numpy.ndarray or None
        ``None`` if *x* is ``None``; otherwise a float ``ndarray`` of length *n*.
    """
    if x is None:
        return None
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    arr = np.asarray(x, dtype="float64")
    if arr.ndim == 0:
        arr = np.repeat(arr, n)
    return arr


def resolve_box(
    min: Any = None,
    max: Any = None,
    center: Any = None,
    dim: Any = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Resolve ``min``/``max`` of one axis from partial position information.

    Port of R ``resolve_box()`` (``geom_box.R:230-278``).  Given any two of
    ``min``, ``max``, ``center`` and ``dim`` (per element), the remaining
    bounds are inferred with the priority order:

    1. ``min``/``max`` verbatim,
    2. ``center +/- 0.5 * dim``,
    3. opposite bound ``+/- dim`` (where ``dim`` is itself derived from
       ``(center - min) * 2`` and then ``(max - center) * 2`` when absent).

    The returned bounds are normalised with ``pmin``/``pmax`` so ``min <= max``;
    this normalisation propagates ``NaN`` (R ``na.rm = FALSE``): if either of a
    pair is ``NaN`` both outputs are ``NaN``.

    Parameters
    ----------
    min, max, center, dim : array-like or None
        The four candidate inputs for the axis.  ``None`` means "not supplied".

    Returns
    -------
    dict or None
        ``{"min": ndarray, "max": ndarray}`` with float arrays, or ``None`` when
        all four inputs are ``None`` (``n == 0``).
    """
    lengths = [
        len(np.atleast_1d(v)) for v in (min, max, center, dim) if v is not None
    ]
    n = max_int(lengths)
    if n == 0:
        return None

    lo = _as_float_array(min, n)
    hi = _as_float_array(max, n)
    lo = np.full(n, np.nan) if lo is None else lo.astype("float64").copy()
    hi = np.full(n, np.nan) if hi is None else hi.astype("float64").copy()

    if not np.isnan(lo).any() and not np.isnan(hi).any():
        return {"min": np.minimum(lo, hi), "max": np.maximum(lo, hi)}

    ctr = _as_float_array(center, n)
    dm = _as_float_array(dim, n)
    ctr = np.full(n, np.nan) if ctr is None else ctr.astype("float64").copy()
    dm = np.full(n, np.nan) if dm is None else dm.astype("float64").copy()

    if np.isnan(lo).any():
        i = np.isnan(lo)
        lo[i] = ctr[i] - 0.5 * dm[i]
    if np.isnan(hi).any():
        i = np.isnan(hi)
        hi[i] = ctr[i] + 0.5 * dm[i]
    if not np.isnan(lo).any() and not np.isnan(hi).any():
        return {"min": np.minimum(lo, hi), "max": np.maximum(lo, hi)}

    if np.isnan(dm).any():
        i = np.isnan(dm)
        dm[i] = (ctr[i] - lo[i]) * 2
        if np.isnan(dm).any():
            i = np.isnan(dm)
            dm[i] = (hi[i] - ctr[i]) * 2

    if np.isnan(lo).any():
        i = np.isnan(lo)
        lo[i] = hi[i] - dm[i]
    if np.isnan(hi).any():
        i = np.isnan(hi)
        hi[i] = lo[i] + dm[i]

    return {"min": np.minimum(lo, hi), "max": np.maximum(lo, hi)}


def max_int(values: Sequence[int]) -> int:
    """Return ``max(values)`` treating an empty input as ``0`` (R ``max()``).

    Parameters
    ----------
    values : sequence of int
        Candidate lengths.

    Returns
    -------
    int
        The maximum, or ``0`` for an empty sequence.
    """
    return max(values) if len(values) else 0


# ---------------------------------------------------------------------------
# GeomBox ggproto class
# ---------------------------------------------------------------------------
class GeomBox(Geom):
    """A flexible rectangle geom defined by any two position aesthetics per axis.

    Subclass of :class:`ggplot2_py.Geom` ported from R ``GeomBox``
    (``geom_box.R:80-223``).
    """

    optional_aes = (
        "xmin",
        "xmax",
        "x",
        "width",
        "ymin",
        "ymax",
        "y",
        "height",
    )

    # R (geom_box.R:86-92):
    #   colour    = from_theme(colour %||% NA),
    #   fill      = from_theme(fill %||% col_mix(ink, paper, 0.35)),
    #   linewidth = from_theme(borderwidth),
    #   linetype  = from_theme(bordertype),
    #   alpha     = NA
    default_aes: Mapping = Mapping(
        colour=FromTheme("colour", fallback=lambda g: None),
        fill=FromTheme("fill", fallback=_mix_ink_paper(0.35)),
        linewidth=FromTheme("borderwidth"),
        linetype=FromTheme("bordertype"),
        alpha=None,
    )

    draw_key = draw_key_polygon

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Resolve ``xmin``/``xmax``/``ymin``/``ymax`` from partial position info.

        Port of R ``GeomBox$setup_data`` (``geom_box.R:94-139``).  Each axis is
        resolved independently with :func:`resolve_box` (``width``/``height``
        falling back to ``params``).  A :func:`cli_warn` lists any unresolved
        aesthetics with axis-specific tips.  The four corner columns are written
        and ``x``/``width``/``y``/``height`` are dropped.

        Parameters
        ----------
        data : pandas.DataFrame
            Layer data after stat computation.
        params : dict
            Layer parameters (may carry ``width``/``height``).

        Returns
        -------
        pandas.DataFrame
            Data with ``xmin``/``xmax``/``ymin``/``ymax`` set and
            ``x``/``width``/``y``/``height`` removed.
        """
        data = data.copy()

        def col(name: str) -> Any:
            return data[name] if name in data.columns else None

        width = col("width")
        if width is None:
            width = params.get("width")
        height = col("height")
        if height is None:
            height = params.get("height")

        x = resolve_box(col("xmin"), col("xmax"), col("x"), width)
        y = resolve_box(col("ymin"), col("ymax"), col("y"), height)

        # Check for missing rows. R uses anyNA on each resolved bound; an
        # entirely-absent axis (resolve_box -> None) means that bound is NA.
        missing: List[str] = []
        if x is None or np.isnan(x["min"]).any():
            missing.append("xmin")
        if x is None or np.isnan(x["max"]).any():
            missing.append("xmax")
        if y is None or np.isnan(y["min"]).any():
            missing.append("ymin")
        if y is None or np.isnan(y["max"]).any():
            missing.append("ymax")

        if missing:
            tip: List[str] = []
            if any(re.match("^x", m) for m in missing):
                tip.append(
                    "Have you specified exactly two of xmin, xmax, x, or "
                    "width for every row?"
                )
            if any(re.match("^y", m) for m in missing):
                tip.append(
                    "Have you specified exactly two of ymin, ymax, y, or "
                    "height for every row?"
                )
            msg = (
                "Could not resolve the position of every "
                + ", ".join("`%s`" % m for m in missing)
                + (" aesthetic." if len(missing) == 1 else " aesthetics.")
            )
            if tip:
                msg = msg + "\n" + "\n".join("i " + t for t in tip)
            cli_warn(msg)

        n = len(data)
        if x is None:
            data["xmin"] = np.full(n, np.nan)
            data["xmax"] = np.full(n, np.nan)
        else:
            data["xmin"] = x["min"]
            data["xmax"] = x["max"]
        if y is None:
            data["ymin"] = np.full(n, np.nan)
            data["ymax"] = np.full(n, np.nan)
        else:
            data["ymin"] = y["min"]
            data["ymax"] = y["max"]

        for drop in ("x", "width", "y", "height"):
            if drop in data.columns:
                data = data.drop(columns=drop)

        return data

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        lineend: str = "butt",
        linejoin: str = "mitre",
        radius: Any = None,
        **params: Any,
    ) -> Any:
        """Build a rect/roundrect/polygon grob for the resolved rectangles.

        Port of R ``GeomBox$draw_panel`` (``geom_box.R:141-220``).

        * Non-linear coord: each rectangle is expanded into a five-vertex
          polygon (winding ``xmin, xmax, xmax, xmin, xmin`` /
          ``ymax, ymax, ymin, ymin, ymin``) and drawn by
          :meth:`ggplot2_py.GeomPolygon.draw_panel` (``radius`` ignored).
        * Linear coord, ``radius is None``: a single :func:`rect_grob`.
        * Linear coord, ``radius`` given: one :func:`roundrect_grob` per row
          combined with :func:`grob_tree`.

        Parameters
        ----------
        data : pandas.DataFrame
            Resolved layer data (``xmin``/``xmax``/``ymin``/``ymax`` present).
        panel_params : Any
            Panel scales / ranges.
        coord : Any
            Active coordinate system.
        lineend : str, default ``"butt"``
            Line end style.
        linejoin : str, default ``"mitre"``
            Line join style.
        radius : grid unit, numeric, or None, default ``None``
            Corner radius.  ``numeric`` is interpreted as millimetres; any
            non-unit value falls back to ``0pt``.  Ignored under non-linear
            coords.

        Returns
        -------
        grid_py.Grob
            The assembled grob.
        """
        if not coord.is_linear():
            aesthetics = [
                c
                for c in data.columns
                if c not in ("x", "y", "xmin", "xmax", "ymin", "ymax")
            ]

            # Rectangle to polygon: replicate each row 5 times.
            idx = np.repeat(np.arange(len(data)), 5)
            new_data = data.iloc[idx].reset_index(drop=True).copy()
            new_data["x"] = vec_interleave(
                data["xmin"].to_numpy(),
                data["xmax"].to_numpy(),
                data["xmax"].to_numpy(),
                data["xmin"].to_numpy(),
                data["xmin"].to_numpy(),
            )
            new_data["y"] = vec_interleave(
                data["ymax"].to_numpy(),
                data["ymax"].to_numpy(),
                data["ymin"].to_numpy(),
                data["ymin"].to_numpy(),
                data["ymin"].to_numpy(),
            )
            for drop in ("xmin", "xmax", "ymin", "ymax"):
                if drop in new_data.columns:
                    new_data = new_data.drop(columns=drop)

            return ggproto_parent(GeomPolygon, self).draw_panel(
                new_data, panel_params, coord, lineend=lineend, linejoin=linejoin
            )

        coords = _coord_transform(coord, data, panel_params)
        coords = coords.copy()
        coords["fill"] = _fill_alpha(
            coords["fill"].to_numpy() if "fill" in coords.columns else "grey35",
            coords["alpha"].to_numpy() if "alpha" in coords.columns else None,
        )
        coords["linewidth"] = (
            coords["linewidth"].to_numpy() * PT
            if "linewidth" in coords.columns
            else 0.5 * PT
        )
        coords["width"] = coords["xmax"].to_numpy() - coords["xmin"].to_numpy()
        coords["height"] = coords["ymax"].to_numpy() - coords["ymin"].to_numpy()

        col = coords["colour"].to_numpy() if "colour" in coords.columns else None
        lty = coords["linetype"].to_numpy() if "linetype" in coords.columns else 1

        if radius is None:
            return _ggname(
                "geom_box",
                rect_grob(
                    coords["xmin"].to_numpy(),
                    coords["ymax"].to_numpy(),
                    width=coords["width"].to_numpy(),
                    height=coords["height"].to_numpy(),
                    default_units="native",
                    just=("left", "top"),
                    gp=Gpar(
                        col=col,
                        fill=coords["fill"].to_numpy(),
                        lwd=coords["linewidth"].to_numpy(),
                        lty=lty,
                        linejoin=linejoin,
                        lineend=lineend,
                    ),
                ),
            )

        if isinstance(radius, (int, float)) and not is_unit(radius):
            radius = Unit(radius, "mm")
        if not is_unit(radius):
            radius = Unit(0, "pt")

        fill = coords["fill"].to_numpy()
        lwd = coords["linewidth"].to_numpy()
        col_arr = np.asarray(col) if col is not None else None
        lty_arr = np.asarray(lty) if not np.isscalar(lty) else None
        xmin = coords["xmin"].to_numpy()
        ymax = coords["ymax"].to_numpy()
        w = coords["width"].to_numpy()
        h = coords["height"].to_numpy()

        grobs: List[Any] = []
        for i in range(len(coords)):
            grobs.append(
                roundrect_grob(
                    xmin[i],
                    ymax[i],
                    width=w[i],
                    height=h[i],
                    r=radius,
                    default_units="native",
                    just=("left", "top"),
                    gp=Gpar(
                        col=(col_arr[i] if col_arr is not None else None),
                        fill=fill[i],
                        lwd=lwd[i],
                        lty=(lty_arr[i] if lty_arr is not None else lty),
                        linejoin=linejoin,
                        lineend=lineend,
                    ),
                )
            )
        return _ggname("geom_box", grob_tree(*grobs))


def geom_box(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    linejoin: str = "mitre",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    radius: Any = None,
    **kwargs: Any,
) -> Any:
    """Create a flexible rectangle layer.

    Port of R ``geom_box()`` (``geom_box.R:45-72``).  A more flexible variant of
    :func:`ggplot2_py.geom_rect` / ``geom_tile`` that accepts any two of the
    ``(x/y)min``/``(x/y)max``/``(x/y)``/``(width/height)`` aesthetics per axis.

    Parameters
    ----------
    mapping : Mapping, optional
        Aesthetic mapping created by :func:`ggplot2_py.aes`.
    data : Any, optional
        Layer data.
    stat : str, default ``"identity"``
        Statistical transformation.
    position : str, default ``"identity"``
        Position adjustment.
    linejoin : str, default ``"mitre"``
        Line join style for the rectangle borders.
    na_rm : bool, default ``False``
        If ``True``, silently remove missing values.
    show_legend : bool or None, default ``None``
        Whether to show a legend for this layer.
    inherit_aes : bool, default ``True``
        Whether to inherit the plot's default aesthetics.
    radius : grid unit, numeric, or None, default ``None``
        Rounded-corner radius.  ``numeric`` is interpreted as millimetres.  Does
        not work under non-linear coordinates.
    **kwargs : Any
        Additional aesthetic parameters passed to the layer.

    Returns
    -------
    ggplot2_py.Layer
        A layer object that can be added to a plot.
    """
    from ggplot2_py.layer import layer

    return layer(
        data=data,
        mapping=mapping,
        stat=stat,
        geom=GeomBox,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "linejoin": linejoin,
            "na_rm": na_rm,
            "radius": radius,
            **kwargs,
        },
    )
