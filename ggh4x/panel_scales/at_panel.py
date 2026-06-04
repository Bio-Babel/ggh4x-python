"""Constrain a layer to specific panels (port of ggh4x ``R/at_panel.R``).

:func:`at_panel` clones a layer's geom so that, at draw time, the layer's data is
subset to only those panels for which a *predicate* (evaluated against the plot
layout) is ``True``.  This makes panel-specific annotations possible.

NSE deviation
-------------
R captures ``expr`` via ``enquo`` and tidy-evaluates it against the plot layout.
Python has no NSE: ``expr`` is a *predicate* -- either a callable
``layout_df -> bool-array`` or a string evaluated with
:meth:`pandas.DataFrame.eval` over the layout columns (``PANEL`` / ``ROW`` /
``COL`` / ``SCALE_*`` + facet variables).
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd

from ggplot2_py import ggproto
from ggplot2_py.ggproto import ggproto_parent
from ggplot2_py.layer import Layer

from ggh4x._cli import cli_abort

__all__ = ["at_panel"]


def _eval_keep(expr: Any, panels: pd.DataFrame) -> np.ndarray:
    """Evaluate the panel predicate against the layout, returning a bool array.

    Parameters
    ----------
    expr : callable or str
        The panel predicate.
    panels : pandas.DataFrame
        The plot layout.

    Returns
    -------
    numpy.ndarray
        Boolean keep-mask, recycled to ``len(panels)``.
    """
    if callable(expr):
        res = expr(panels)
    elif isinstance(expr, str):
        res = panels.eval(expr, engine="python")
    else:
        res = expr
    arr = np.asarray(res)
    if arr.dtype != bool:
        arr = arr.astype(bool)
    n = len(panels)
    if arr.ndim == 0:
        arr = np.repeat(arr, n)
    if len(arr) != n:
        arr = np.resize(arr, n)
    return arr


def at_panel(layer: Any, expr: Any) -> Any:
    """Constrain a layer to the panels matching *expr*.

    Faithful port of ggh4x's ``at_panel`` (``R/at_panel.R:43-82``).  Clones the
    layer's geom with a ``draw_layer`` override that evaluates *expr* against the
    plot layout, subsets the layer data to the kept panels, then delegates to the
    original geom's ``draw_layer``.  A bare list of layers is handled by
    recursing over its layer elements.

    Parameters
    ----------
    layer : Layer or list of Layer
        The layer (or bare list of layers) to constrain.
    expr : callable or str
        Panel predicate evaluated against the plot layout (see module docstring).

    Returns
    -------
    Layer or list
        A layer clone whose geom only draws on matched panels (or the input list
        with each layer element so cloned).

    Raises
    ------
    ValueError
        When *expr* is missing, or *layer* is neither a layer nor a bare list of
        layers.
    """
    if expr is None:
        cli_abort("`expr` must be an expression, it cannot be missing.")

    if not isinstance(layer, Layer):
        # Accept bare lists of layers (e.g. geom_sf()).
        if isinstance(layer, list):
            return [
                at_panel(el, expr) if isinstance(el, Layer) else el
                for el in layer
            ]
        cli_abort(f"`layer` must be a layer, not {type(layer).__name__}.")

    old_geom = layer.geom

    def draw_layer(self: Any, data: Any, params: Any, layout: Any, coord: Any) -> List[Any]:
        panels = layout.layout
        keep = _eval_keep(expr, panels)
        kept_panels = panels.loc[keep, "PANEL"]

        data = data[data["PANEL"].isin(kept_panels)]
        return ggproto_parent(old_geom, self).draw_layer(data, params, layout, coord)

    new_geom = ggproto(None, old_geom, draw_layer=draw_layer)
    return ggproto(None, layer, geom=new_geom)
