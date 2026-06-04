"""Single per-panel position scales (port of ggh4x ``R/scale_facet.R``).

:func:`scale_x_facet` / :func:`scale_y_facet` build a :class:`ScaleFacet` add-on
carrying a panel *predicate* and a position scale.  When added to a plot, the
handler either appends to an existing ``FreeScaled<...>`` facet's scale lists or
lowers to a single-entry :class:`~ggh4x.panel_scales.facetted_pos_scales.FacettedPosScales`,
reusing that machinery.

NSE deviation
-------------
R captures ``expr`` via ``enquo`` and tidy-evaluates it against the plot layout.
Python has no NSE: ``expr`` is a *predicate* -- either a callable
``layout_df -> bool-array`` or a string evaluated with
:meth:`pandas.DataFrame.eval` over the layout columns (``PANEL`` / ``ROW`` /
``COL`` / ``SCALE_*`` + facet variables).
"""

from __future__ import annotations

from typing import Any

import ggplot2_py as _gg
from ggplot2_py import ggproto
from ggplot2_py.plot import update_ggplot

from ggh4x._cli import cli_abort
from ggh4x._rlang import arg_match0

from .facetted_pos_scales import FacettedPosScales, _ScaleList

__all__ = [
    "scale_facet",
    "scale_x_facet",
    "scale_y_facet",
    "ScaleFacet",
]


# ---------------------------------------------------------------------------
# ScaleFacet container (R: structure(list(lhs=, rhs=), class = "scale_facet"))
# ---------------------------------------------------------------------------
class ScaleFacet:
    """Deferred container of one per-panel position scale + its predicate.

    Port of R's ``structure(list(lhs =, rhs =), class = "scale_facet")``.
    Consumed by :func:`_update_scale_facet` at ``+``-time.

    Attributes
    ----------
    lhs : callable or str
        The panel predicate (formula LHS equivalent).
    rhs : Scale
        The position scale to apply (formula RHS equivalent).
    """

    def __init__(self, lhs: Any, rhs: Any) -> None:
        self.lhs = lhs
        self.rhs = rhs


# ---------------------------------------------------------------------------
# scale_facet (R scale_facet.R:76-116)
# ---------------------------------------------------------------------------
def scale_facet(expr: Any, aes: str, *args: Any, type: str = "continuous", **kwargs: Any) -> ScaleFacet:
    """Build a per-panel position scale (generic over aesthetic).

    Faithful port of ggh4x's ``scale_facet`` (``R/scale_facet.R:76-116``).
    Resolves the constructor ``scale_<aes>_<type>`` from the :mod:`ggplot2_py`
    namespace (R's ``find_global``), instantiates it with the extra arguments,
    and pairs it with the panel predicate *expr*.

    Parameters
    ----------
    expr : callable or str
        Panel predicate evaluated against the plot layout (see module docstring).
    aes : {"x", "y"}
        The position aesthetic.
    *args, **kwargs
        Extra arguments forwarded to the resolved scale constructor.
    type : str, default "continuous"
        Scale type, such that ``scale_<aes>_<type>`` names a constructor.

    Returns
    -------
    ScaleFacet
        An add-on object that can be added to a plot with ``+``.

    Raises
    ------
    ValueError
        When ``type == "facet"`` (circular), the constructor cannot be found, or
        *expr* is missing.
    """
    candidate = f"scale_{aes}_{type}"
    if type == "facet":
        cli_abort(
            f"Cannot circularly define `{candidate}` as template for `{candidate}`."
        )

    fun = getattr(_gg, candidate, None)
    scale = None
    if fun is not None:
        scale = fun(*args, **kwargs)

    if scale is None or not hasattr(scale, "aesthetics"):
        cli_abort(
            f"Cannot find a `{candidate}` function. "
            "Did you misspell the `type` argument?"
        )

    if expr is None:
        cli_abort("`expr` must be a valid expression.")

    return ScaleFacet(lhs=expr, rhs=scale)


def scale_x_facet(expr: Any, *args: Any, type: str = "continuous", **kwargs: Any) -> ScaleFacet:
    """Per-panel x position scale (thin wrapper, R ``scale_x_facet``).

    Parameters
    ----------
    expr : callable or str
        Panel predicate (see :func:`scale_facet`).
    *args, **kwargs
        Forwarded to the resolved ``scale_x_<type>`` constructor.
    type : str, default "continuous"

    Returns
    -------
    ScaleFacet
    """
    return scale_facet(expr, "x", *args, type=type, **kwargs)


def scale_y_facet(expr: Any, *args: Any, type: str = "continuous", **kwargs: Any) -> ScaleFacet:
    """Per-panel y position scale (thin wrapper, R ``scale_y_facet``).

    Parameters
    ----------
    expr : callable or str
        Panel predicate (see :func:`scale_facet`).
    *args, **kwargs
        Forwarded to the resolved ``scale_y_<type>`` constructor.
    type : str, default "continuous"

    Returns
    -------
    ScaleFacet
    """
    return scale_facet(expr, "y", *args, type=type, **kwargs)


# ---------------------------------------------------------------------------
# ggplot_add.scale_facet (R scale_facet.R:133-184)
# ---------------------------------------------------------------------------
@update_ggplot.register(ScaleFacet)
def _update_scale_facet(obj: ScaleFacet, plot: Any, object_name: str = "") -> Any:
    """Add a :class:`ScaleFacet` to *plot* (R ``ggplot_add.scale_facet``).

    Appends to an existing ``FreeScaled<...>`` facet's per-panel scale lists (and
    its parallel predicate list), or lowers to a single-entry
    :class:`FacettedPosScales` otherwise.

    Parameters
    ----------
    obj : ScaleFacet
    plot : ggplot2_py.plot.GGPlot
    object_name : str, optional

    Returns
    -------
    ggplot2_py.plot.GGPlot

    Raises
    ------
    ValueError
        When the plot has no facets (``FacetNull``).
    """
    aes = obj.rhs.aesthetics[0]
    aes = arg_match0(aes, ["x", "y"], arg_name="scale$aesthetics[1]")

    facet = plot.facet
    cls_name = type(facet).__name__

    if cls_name.startswith("FacetNull") or cls_name == "FacetNull":
        nm = f"scale_{aes}_facet"
        cli_abort(
            f"`{nm}` cannot be added to a plot without facets. "
            f"Try adding facets before adding `{nm}`."
        )

    if cls_name.startswith("FreeScaled"):
        if aes == "x":
            old = facet.new_x_scales
            old_lhs = getattr(old, "lhs", None) or []
            new = _ScaleList(
                list(old) + [obj.rhs],
                lhs=list(old_lhs) + [obj.lhs],
            )
            plot.facet = ggproto(None, facet, new_x_scales=new)
        else:
            old = facet.new_y_scales
            old_lhs = getattr(old, "lhs", None) or []
            new = _ScaleList(
                list(old) + [obj.rhs],
                lhs=list(old_lhs) + [obj.lhs],
            )
            plot.facet = ggproto(None, facet, new_y_scales=new)
    else:
        if aes == "x":
            fps = FacettedPosScales(
                x=_ScaleList([obj.rhs], lhs=[obj.lhs]),
                y=_ScaleList([None], lhs=None),
            )
        else:
            fps = FacettedPosScales(
                x=_ScaleList([None], lhs=None),
                y=_ScaleList([obj.rhs], lhs=[obj.lhs]),
            )
        plot = plot + fps

    return plot
