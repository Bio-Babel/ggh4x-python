"""The :class:`MultiScale` container and its ``ggplot_add`` handler.

R source: ``scale_listed.R:135-203`` (the shared ``ggplot_add.MultiScale`` S3
method) plus the ``structure(..., class = "MultiScale")`` tagged container built
by :func:`scale_multi.scale_fill_multi` / :func:`scale_listed.scale_listed`.

A :class:`MultiScale` is **not** a :class:`ggplot2_py.scale.Scale` or a ggproto;
it is a small deferred-mutation container.  Its handler, registered on
:func:`ggplot2_py.plot.update_ggplot` via :func:`functools.singledispatch`,
runs at ``+``-time and:

1. adds each carried scale to ``plot.scales`` (``ScalesList.add``); and
2. rewrites every affected layer's *geom* so that the standard aesthetic
   (``fill``/``colour``) it understands is exposed under the non-standard
   aesthetic name (``fill1``/``spec``/...).

The geom rewrite is the **same machinery** as ggnewscale's ``bump_aes_layer``
geom branch — clone the geom, install ``handle_na``/``draw_key`` column-rename
shims (using the ``__func__``-capture + ``object.__setattr__`` install to dodge
GGProto's auto-bind arity bug) and rewrite the geom's aes-name slots — but in the
*opposite direction*: ``handle_na``/``draw_key`` rename ``new_aes -> replaced_aes``
on the data columns (so the inner geom sees the standard name), while
``default_aes`` / ``non_missing_aes`` / ``optional_aes`` / ``required_aes`` rename
``replaced_aes -> new_aes`` on the slot names.  Unlike ggnewscale, the layer's
*stat* is left untouched.

Importing this module has the side effect of registering the handler on
``update_ggplot``; :mod:`ggh4x.multiscale` imports it for that side effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import ggplot2_py as _gg
from ggplot2_py.plot import update_ggplot

# Reuse ggnewscale's verified bumping primitives (clone-with-renamed-geom).
from ggnewscale._bump import (
    _rename_columns,
    _safe_aes_slot,
    _safe_default_aes,
)
from ggnewscale._change_name import change_name

__all__ = [
    "MultiScale",
]


# ---------------------------------------------------------------------------
# MultiScale container
# ---------------------------------------------------------------------------
@dataclass
class MultiScale:
    """Deferred container of non-standard scales for one *replaced* aesthetic.

    Port of R's ``structure(list(scales=, aes=, replaced_aes=), class="MultiScale")``
    (``scale_multi.R:72-75`` / ``scale_listed.R:127-130``).  It is intentionally a
    distinct class (not a ``dict``/``list``) so :func:`functools.singledispatch`
    resolves it to :func:`_add_multiscale` rather than to the list/Mapping
    handlers.

    Attributes
    ----------
    scales : list of ggplot2_py.scale.Scale
        The scale objects to add to the plot.
    aes : list of str
        Non-standard aesthetic names (e.g. ``["fill1", "fill2"]``).
    replaced_aes : str
        The standard aesthetic those scales replace (``"fill"`` / ``"colour"``).
    """

    scales: List[Any] = field(default_factory=list)
    aes: List[str] = field(default_factory=list)
    replaced_aes: str = ""


# ---------------------------------------------------------------------------
# Per-layer geom rewrite
# ---------------------------------------------------------------------------
def _rewrite_layer_geom(layer: Any, new_aes: List[str], replaced_aes: str) -> Any:
    """Clone *layer*'s geom so *replaced_aes* is exposed as *new_aes*.

    Port of the per-layer rewrite in R ``ggplot_add.MultiScale``
    (``scale_listed.R:150-201``), mirroring ggnewscale's ``bump_aes_layer`` geom
    branch but renaming in the opposite direction and leaving the stat untouched.

    Parameters
    ----------
    layer : ggplot2_py.layer.Layer
        The layer whose geom is rewritten (mutated in place, matching R's
        ``lay$geom <- new_geom``).
    new_aes : list of str
        The non-standard aesthetic name(s) used by this layer's mapping.
    replaced_aes : str
        The standard aesthetic name those replace.

    Returns
    -------
    ggplot2_py.layer.Layer
        The same *layer*, with its geom replaced.
    """
    old_geom = layer.geom

    # handle_na wrapper: rename data columns new_aes -> replaced_aes, then
    # delegate to the *original* geom's handle_na (scale_listed.R:157-163).  R
    # captures ``old_geom$handle_na`` as a closure, so the delegate runs against
    # the original geom whose ``required_aes``/``non_missing_aes`` still carry the
    # standard names that the renamed data now matches.  ``old_geom.handle_na`` is
    # a bound method of *old_geom*, so calling it with ``(data, params)`` keeps
    # ``self`` pointing at the original geom (not the renamed clone).
    old_handle_na = getattr(old_geom, "handle_na", None)
    new_geom_kwargs: dict = {}
    if old_handle_na is not None:

        def _new_handle_na(self: Any, data: Any, params: Any, _fn: Any = old_handle_na) -> Any:
            renamed = _rename_columns(data, {a: replaced_aes for a in new_aes})
            return _fn(renamed, params)

        new_geom_kwargs["handle_na"] = _new_handle_na

    new_geom = _gg.ggproto(
        f"New{'_'.join(new_aes)}{type(old_geom).__name__}",
        old_geom,
        **new_geom_kwargs,
    )

    # Slot name-rewrites: replaced_aes -> new_aes (scale_listed.R:179-195).
    # When a layer maps several non-standard aesthetics, R's gsub replaces the
    # standard slot name with *all* of them; the first match wins for downstream
    # lookups, so we use new_aes[0] as ggnewscale does for the single-aes case.
    target_aes = new_aes[0]
    new_geom.default_aes = change_name(
        _safe_default_aes(new_geom), replaced_aes, target_aes
    )
    new_geom.non_missing_aes = change_name(
        _safe_aes_slot(new_geom, "non_missing_aes"), replaced_aes, target_aes
    )
    new_geom.optional_aes = change_name(
        _safe_aes_slot(new_geom, "optional_aes"), replaced_aes, target_aes
    )
    new_geom.required_aes = change_name(
        _safe_aes_slot(new_geom, "required_aes"), replaced_aes, target_aes
    )

    # draw_key wrapper: rename data columns new_aes -> replaced_aes, then delegate
    # (scale_listed.R:165-171).  Capture __func__ and install via
    # object.__setattr__ to bypass GGProto auto-bind (arity dodge).
    old_draw_key_attr = getattr(new_geom, "draw_key", None)
    if old_draw_key_attr is not None:
        old_draw_key_fn = getattr(old_draw_key_attr, "__func__", old_draw_key_attr)

        def _new_draw_key(data: Any, params: Any, size: Any = None, _fn: Any = old_draw_key_fn) -> Any:
            renamed = _rename_columns(data, {a: replaced_aes for a in new_aes})
            return _fn(renamed, params, size)

        object.__setattr__(new_geom, "draw_key", _new_draw_key)

    layer.geom = new_geom
    return layer


# ---------------------------------------------------------------------------
# ggplot_add.MultiScale  (scale_listed.R:142-203)
# ---------------------------------------------------------------------------
@update_ggplot.register(MultiScale)
def _add_multiscale(obj: MultiScale, plot: Any, object_name: str = "") -> Any:
    """Add a :class:`MultiScale` to *plot* (R ``ggplot_add.MultiScale``).

    Adds every carried scale to ``plot.scales`` then rewrites each layer that maps
    one of the container's non-standard aesthetics so its geom exposes the
    standard ``replaced_aes`` under that non-standard name.

    Parameters
    ----------
    obj : MultiScale
        The container to add.
    plot : ggplot2_py.plot.GGPlot
        The plot to mutate.
    object_name : str, optional
        Unused (kept for the ``update_ggplot`` dispatch signature).

    Returns
    -------
    ggplot2_py.plot.GGPlot
        The mutated plot.
    """
    # 1. Add scales (scale_listed.R:143-145).
    for sc in obj.scales:
        plot.scales.add(sc)

    replaced_aes = obj.replaced_aes

    # 2. Rewrite each affected layer (scale_listed.R:150-201).
    new_layers: List[Any] = []
    for lay in plot.layers:
        mapping_keys = list(lay.mapping.keys()) if lay.mapping is not None else []
        if not any(k in obj.aes for k in mapping_keys):
            new_layers.append(lay)
            continue
        new_aes = [a for a in obj.aes if a in mapping_keys]
        new_layers.append(_rewrite_layer_geom(lay, new_aes, replaced_aes))

    plot.layers = new_layers
    return plot
