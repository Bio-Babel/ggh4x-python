"""Tag strips for ggh4x facets (port of ggh4x ``strip_tag.R``).

This module ports :class:`StripTag` and the :func:`strip_tag` constructor.  Tag
strips render the strips as fitted text boxes *inside* the panels (anchored to a
panel corner via a justified viewport), rather than as full-width strips outside
the panels.

``StripTag`` extends :class:`ggh4x.strip_themed.StripThemed` (a *sibling* of
``StripNested``, not a subclass).  It overrides:

* :meth:`StripTag.setup` -- builds *per-panel* (not de-duplicated) col/row var
  frames; reuses the base ``get_strips`` x / y shape.
* :meth:`StripTag.draw_labels` -- a *self-less* fitted-box label builder: no
  margin-equalisation, no ``unit(1, "null")`` cross-axis; measures grob
  height/width in cm (per-layer max on the strip axis, exact cm on the cross
  axis).
* :meth:`StripTag.finish_strip` -- *has* ``self`` (unlike the base self-less
  ``finish_strip``): builds fitted-box gtables with npc-fractional widths inside
  a cm outer viewport justified to a panel corner.
* :meth:`StripTag.incorporate_grid` / :meth:`StripTag.incorporate_wrap` -- place
  the tag grobs *onto* existing panel cells (no new rows/cols).

R source: ``ggh4x/R/strip_tag.R``.

Notes
-----
* **Viewport npc trick.**  Each fitted box is a ``gtable_matrix`` whose
  widths/heights are ``unit(w / sum(w), "npc")`` wrapped in a viewport sized to
  ``unit(sum(w), "cm")`` and justified to ``params["just"]``.  When ``clip ==
  "on"`` the viewport is clamped to ``unit(1, "npc")`` via :func:`unit_pmin`.
* **grid layout combine.**  ``incorporate_grid`` ``rbind``s the x and y tag
  gtables (order from ``params["order"]``), recomputing the combined viewport
  height (sum) and width (max).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ggplot2_py import element_grob
from ggplot2_py import is_theme_element as _is_theme_element
from ggplot2_py._utils import height_cm, width_cm
from ggplot2_py.ggproto import ggproto
from grid_py import (
    Unit,
    Viewport,
    edit_grob,
    grob_height,
    grob_name,
    grob_tree,
    grob_width,
    unit_c,
    unit_pmin,
)
from gtable_py import gtable_add_grob, gtable_matrix, rbind_gtable

from ggh4x._rlang import arg_match0
from ggh4x.strip_themed import StripThemed
from ggh4x.strip_vanilla import (
    _LabelGrobs,
    _is_zero_grob,
    _panel_layout,
    _split_by,
    validate_element_list,
)

__all__ = ["StripTag", "strip_tag"]


def _draw_labels_tag(
    labels: Sequence[str],
    element: Dict[str, List[Any]],
    position: str,
    layer_id: Sequence[int],
    size: Any = None,
) -> _LabelGrobs:
    """Build fitted-box label grobs (self-less; tag variant of ``draw_labels``).

    Port of R ``StripTag$draw_labels`` (``strip_tag.R:119-163``).  Unlike the
    base, there is no margin-equalisation and no ``unit(1, "null")`` cross axis:
    label sizes are measured directly in cm with :func:`grid_py.grob_height` /
    :func:`grid_py.grob_width` (per-layer max on the strip axis, exact cm on the
    cross axis).

    Parameters
    ----------
    labels : sequence of str
        Flattened (column-major) label strings, one per cell.
    element : dict
        ``{"el": [...text elements...], "bg": [...background grobs...]}``.
    position : str
        Strip side.
    layer_id : sequence of int
        1-based layer id per cell.
    size : Any, optional
        Unused (kept for signature parity with the base ``draw_labels``).

    Returns
    -------
    _LabelGrobs
        A list of GTree grobs with ``.width`` / ``.height`` unit attributes.
    """
    aes = "x" if position in ("top", "bottom") else "y"
    layer_id = list(layer_id)

    grobs: List[Any] = []
    for label, elem in zip(labels, element["el"]):
        grob = element_grob(elem, label=label, margin_x=True, margin_y=True)
        try:
            grob.name = grob_name(grob, "strip.text." + aes)
        except (AttributeError, TypeError):
            pass
        grobs.append(grob)

    zeros = [_is_zero_grob(g) for g in grobs]
    if len(grobs) == 0 or all(zeros):
        return _LabelGrobs(grobs)

    nonzero_idx = [i for i, z in enumerate(zeros) if not z]
    nonzero_layer = [layer_id[i] for i in nonzero_idx]

    heights = [grob_height(grobs[i]) for i in nonzero_idx]
    widths = [grob_width(grobs[i]) for i in nonzero_idx]

    if aes == "x":
        # per-layer max height (of grobHeight units); exact cm width.
        grouped = _split_by(heights, nonzero_layer)
        height_units = [_max_unit(g) for g in grouped]
        height = unit_c(*height_units) if len(height_units) > 1 else height_units[0]
        width = Unit(_to_list(width_cm(widths)), "cm")
    else:
        # per-layer max width (of grobWidth units); exact cm height.
        grouped = _split_by(widths, nonzero_layer)
        width_units = [_max_unit(g) for g in grouped]
        width = unit_c(*width_units) if len(width_units) > 1 else width_units[0]
        height = Unit(_to_list(height_cm(heights)), "cm")

    combined: List[Any] = []
    for x, bg in zip(grobs, element["bg"]):
        bg_grob = element_grob(bg) if _is_theme_element(bg) else bg
        tree = grob_tree(bg_grob, x)
        try:
            tree.name = grob_name(tree, "strip")
        except (AttributeError, TypeError):
            pass
        combined.append(tree)

    result = _LabelGrobs(combined)
    result.width = width
    result.height = height
    return result


def _max_unit(units: Sequence[Any]) -> Any:
    """Port of R ``max_height`` / ``max_width`` applied to a list of *units*.

    For the tag path the per-layer reduction is over already-measured unit
    objects (``grobHeight`` / ``grobWidth`` results), so it reduces to the
    element-wise maximum cm value wrapped back as a single ``unit``.

    Parameters
    ----------
    units : sequence of grid_py.Unit
        The per-cell measured units for one layer.

    Returns
    -------
    grid_py.Unit
        A single ``cm`` unit holding the maximum measured size.
    """
    if len(units) == 0:
        return Unit(0.0, "cm")
    values = [float(np.atleast_1d(width_cm(u))[0]) for u in units]
    return Unit(max(values), "cm")


class StripTag(StripThemed):
    """Strip rendered as a fitted text box inside the panels.

    Subclass of :class:`ggh4x.strip_themed.StripThemed`.  See the module
    docstring for the fitted-box / viewport algorithm.

    Attributes
    ----------
    params : dict
        Holds ``clip``, ``order`` (``["x", "y"]`` or ``["y", "x"]``) and ``just``
        (a length-2 numeric justification).
    """

    _class_name = "StripTag"

    # self-less draw_labels (first param is not `self`).
    draw_labels = staticmethod(_draw_labels_tag)

    def setup(
        self,
        layout: pd.DataFrame,
        params: Dict[str, Any],
        theme: Any,
        type: str,
    ) -> None:
        """Build per-panel (non-de-duplicated) var frames and get strips.

        Port of R ``StripTag$setup`` (``strip_tag.R:92-117``).  Unlike the base,
        tags are per-panel (they overlap panels), so the layout is *not*
        de-duplicated.

        Parameters
        ----------
        layout : pandas.DataFrame
            The facet layout.
        params : dict
            Facet params (``facets`` for wrap; ``rows`` / ``cols`` for grid; plus
            ``labeller``).
        theme : Theme
            The active theme.
        type : str
            ``"wrap"`` or ``"grid"``.
        """
        self._set(elements=self.setup_elements(theme, type))

        if type == "wrap":
            facets = params.get("facets") or {}
            facet_names = list(facets.keys()) if hasattr(facets, "keys") else list(facets)
            if len(facet_names) == 0:
                labels = pd.DataFrame({"(all)": ["(all)"]})
            else:
                labels = layout[facet_names].reset_index(drop=True)
            col_vars = labels
            row_vars = labels
        else:
            col_names = _names(params.get("cols"))
            row_names = _names(params.get("rows"))
            col_vars = layout[col_names].reset_index(drop=True) if col_names else _empty_frame(layout)
            row_vars = layout[row_names].reset_index(drop=True) if row_names else _empty_frame(layout)

        self.get_strips(
            x=col_vars,
            y=row_vars,
            labeller=params.get("labeller"),
            theme=theme,
            params=self.params,
            layout_x=layout,
            layout_y=layout,
        )

    def finish_strip(  # type: ignore[override]
        self,
        strip: Sequence[Any],
        width: Any,
        height: Any,
        position: str,
        layout: pd.DataFrame,
        dim: Any,
        clip: str = "inherit",
    ) -> pd.DataFrame:
        """Build fitted-box gtables placed inside panels via a justified vp.

        Port of R ``StripTag$finish_strip`` (``strip_tag.R:165-217``).  Has
        ``self`` (unlike the base self-less ``finish_strip``) to read
        ``self.params["just"]``.

        Parameters
        ----------
        strip : sequence
            The label grobs (one per cell).
        width, height : grid_py.Unit
            Per-cell width / height unit vectors (measured cm).
        position : str
            Strip side.
        layout : pandas.DataFrame
            The (full, per-panel) layout carrying ``PANEL``.
        dim : tuple of int
            ``(nrow, ncol)`` of the label matrix (panels x layers).
        clip : str, default ``"inherit"``
            Clip setting (``"on"`` clamps the box to ``1 npc``).

        Returns
        -------
        pandas.DataFrame
            Placement frame with ``t = l = b = r = PANEL`` and a ``grobs`` column
            of fitted-box gtables.
        """
        strip = list(strip)
        empty = len(strip) == 0 or all(_is_zero_grob(g) for g in strip)

        out_grobs: List[Any] = strip
        if not empty:
            just = self.params["just"]
            n = len(strip)
            w_cm = _recycle(_to_list(width_cm(width)), n)
            h_cm = _recycle(_to_list(height_cm(height)), n)

            nrow, ncol = int(dim[0]), int(dim[1])
            # idx = matrix(seq_along(strip), nrow, ncol) -- column-major.
            idx = np.arange(n, dtype=int).reshape(nrow, ncol, order="F")

            out_grobs = []
            for ri in range(nrow):
                if position in ("top", "bottom"):
                    # apply(idx, 1, matrix, ncol=1) -> column vector (ncol x 1).
                    sub_idx = idx[ri, :].reshape(ncol, 1, order="F")
                else:
                    # apply(idx, 1, matrix, nrow=1) -> row vector (1 x ncol).
                    sub_idx = idx[ri, :].reshape(1, ncol, order="F")
                d0, d1 = sub_idx.shape
                flat = list(sub_idx.reshape(-1, order="F"))
                m = [[strip[sub_idx[i, j]] for j in range(d1)] for i in range(d0)]
                wmat = np.array([w_cm[k] for k in flat], dtype=float).reshape(d0, d1, order="F")
                hmat = np.array([h_cm[k] for k in flat], dtype=float).reshape(d0, d1, order="F")
                w = wmat.max(axis=0)  # per column (layer)
                h = hmat.max(axis=1)  # per row
                sum_w = float(w.sum())
                sum_h = float(h.sum())

                vp_width = Unit(sum_w, "cm")
                vp_height = Unit(sum_h, "cm")
                if clip == "on":
                    vp_width = unit_pmin(vp_width, Unit(1, "npc"))
                    vp_height = unit_pmin(vp_height, Unit(1, "npc"))

                vp = Viewport(
                    x=just[0],
                    y=just[1],
                    just=list(just),
                    width=vp_width,
                    height=vp_height,
                    clip=clip,
                )
                widths = _npc_unit(w, sum_w)
                heights = _npc_unit(h, sum_h)
                # Build the fitted-box gtable then attach the justified viewport.
                # (gtable_py's ``Gtable(vp=...)`` reconstruction reads
                # ``vp.justification`` which grid_py stores as ``vp.just``, so the
                # viewport is attached post-construction rather than via the
                # ``vp=`` kwarg -- functionally identical to R's
                # ``gtable_matrix(..., vp = vp)``.)
                gt = gtable_matrix("strip-cells", m, widths, heights, clip=clip)
                gt.vp = vp
                out_grobs.append(gt)

        panel = [int(p) for p in layout["PANEL"]]
        return pd.DataFrame(
            {"t": panel, "l": panel, "b": panel, "r": panel, "grobs": out_grobs}
        )

    def incorporate_wrap(
        self,
        panels: Any,
        position: str,
        clip: str = "off",
        sizes: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Place per-panel tag grobs onto wrapped panel cells (no new rows/cols).

        Port of R ``StripTag$incorporate_wrap`` (``strip_tag.R:219-230``).

        Parameters
        ----------
        panels : Gtable
            The assembled panel gtable.
        position : str
            One of ``top`` / ``bottom`` / ``left`` / ``right`` -- selects the
            strip side from ``self.strips``.
        clip : str, default ``"off"``
            Clip setting forwarded to ``gtable_add_grob``.
        sizes : dict, optional
            Unused.

        Returns
        -------
        Gtable
            The panel gtable with this side's tags added.
        """
        strip = _flatten_strips(self.strips)[position]
        if strip is None:
            return panels
        pos = _panel_layout(panels)
        tlist = [pos["t"][ti] for ti in (int(v) for v in strip["t"])]
        llist = [pos["l"][li] for li in (int(v) for v in strip["l"])]
        names = ["strip-" + str(i + 1) for i in range(len(strip))]
        panels = gtable_add_grob(
            panels,
            list(strip["grobs"]),
            name=names,
            t=tlist,
            l=llist,
            clip=clip,
        )
        return panels

    def incorporate_grid(self, panels: Any, switch: Any) -> Any:
        """Combine x + y tags and place them onto panel cells.

        Port of R ``StripTag$incorporate_grid`` (``strip_tag.R:232-271``).
        Combines the x and y tag gtables by ``rbind`` (order from
        ``params["order"]``), recomputes the combined viewport height (sum) /
        width (max), then adds the grobs at the panel cell ``t`` / ``l`` with
        ``z=2``, ``clip="on"``.

        Parameters
        ----------
        panels : Gtable
            The assembled panel gtable.
        switch : str or None
            ``"x"`` / ``"y"`` / ``"both"`` / ``None``.

        Returns
        -------
        Gtable
            The panel gtable with tags added.
        """
        flat = _flatten_strips(self.strips)
        xstrip = flat["bottom"] if switch in ("x", "both") else flat["top"]
        ystrip = flat["right"] if switch in ("y", "both") else flat["left"]

        if xstrip is None and ystrip is None:
            return panels
        elif xstrip is None:
            strip = list(ystrip["grobs"])
        elif ystrip is None:
            strip = list(xstrip["grobs"])
        else:
            reorder = list(self.params["order"]) != ["x", "y"]
            strip = []
            for x, y in zip(list(xstrip["grobs"]), list(ystrip["grobs"])):
                vp = getattr(x, "vp", None)
                if vp is not None:
                    # R mutates vp$height/$width; grid_py viewports are immutable
                    # for these props, so rebuild one with the combined size.
                    vp = _rebuild_vp(
                        vp,
                        height=_sum_units(x.heights, y.heights),
                        width=_max_units(x.widths, y.widths),
                    )
                new = rbind_gtable(y, x) if reorder else rbind_gtable(x, y)
                if vp is not None:
                    new = edit_grob(new, vp=vp)
                strip.append(new)

        pos = _panel_layout(panels)
        t_src = xstrip if xstrip is not None else ystrip
        l_src = xstrip if xstrip is not None else ystrip
        tlist = [pos["t"][ti] for ti in (int(v) for v in t_src["t"])]
        llist = [pos["l"][li] for li in (int(v) for v in l_src["l"])]
        names = ["strip-" + str(i + 1) for i in range(len(strip))]
        panels = gtable_add_grob(
            panels, strip, name=names, t=tlist, l=llist, clip="on", z=2
        )
        return panels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _names(param: Any) -> List[str]:
    """Return variable names from a facet ``rows`` / ``cols`` param."""
    if param is None:
        return []
    if hasattr(param, "keys"):
        return list(param.keys())
    if isinstance(param, (list, tuple)):
        return [str(p) for p in param]
    return []


def _empty_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a 0-column frame with the same row count as *df*."""
    return pd.DataFrame(index=df.index)


def _flatten_strips(strips: Dict[str, Any]) -> Dict[str, Any]:
    """Port of R ``unlist(unname(self$strips), recursive=FALSE)`` (side-keyed)."""
    flat: Dict[str, Any] = {}
    for outer in strips.values():
        for key, value in outer.items():
            flat[key] = value
    return flat


def _recycle(seq: Sequence[Any], length_out: int) -> List[Any]:
    """Port of R ``rep(seq, length.out=n)`` for a plain list."""
    n = len(seq)
    if length_out <= 0 or n == 0:
        return []
    return [seq[i % n] for i in range(length_out)]


def _to_list(x: Any) -> List[float]:
    """Coerce a width_cm / height_cm result to a flat list of floats."""
    arr = np.atleast_1d(np.asarray(x, dtype=float))
    return [float(v) for v in arr]


def _npc_unit(values: np.ndarray, total: float) -> Any:
    """Build a ``unit(values / total, "npc")`` vector.

    Parameters
    ----------
    values : numpy.ndarray
        The per-cell cm sizes.
    total : float
        The summed cm size (npc denominator).

    Returns
    -------
    grid_py.Unit
        The npc-fraction unit vector.
    """
    if total == 0:
        fractions = [0.0 for _ in values]
    else:
        fractions = [float(v) / total for v in values]
    return Unit(fractions, "npc")


def _rebuild_vp(vp: Any, height: Any, width: Any) -> Any:
    """Rebuild a viewport with new *height* / *width*, preserving other props.

    R mutates ``vp$height`` / ``vp$width`` in place; grid_py viewports expose
    those as read-only properties, so this constructs a fresh
    :class:`grid_py.Viewport` carrying the original anchor / justification /
    clip and the new size.

    Parameters
    ----------
    vp : grid_py.Viewport
        The source viewport (from the x strip's fitted box).
    height : grid_py.Unit
        The new (combined, summed) height.
    width : grid_py.Unit
        The new (combined, maximum) width.

    Returns
    -------
    grid_py.Viewport
        A new viewport with the combined size.
    """
    just = getattr(vp, "just", None)
    return Viewport(
        x=getattr(vp, "x", None),
        y=getattr(vp, "y", None),
        just=list(just) if just is not None else "centre",
        width=width,
        height=height,
        clip=getattr(vp, "clip", "inherit"),
    )


def _sum_units(a: Any, b: Any) -> Any:
    """Port of R ``sum(a, b)`` over two unit vectors (cm total)."""
    av = float(np.sum(np.atleast_1d(np.asarray(height_cm(a), dtype=float))))
    bv = float(np.sum(np.atleast_1d(np.asarray(height_cm(b), dtype=float))))
    return Unit(av + bv, "cm")


def _max_units(a: Any, b: Any) -> Any:
    """Port of R ``max(a, b)`` over two unit vectors (cm max)."""
    av = float(np.max(np.atleast_1d(np.asarray(width_cm(a), dtype=float))))
    bv = float(np.max(np.atleast_1d(np.asarray(width_cm(b), dtype=float))))
    return Unit(max(av, bv), "cm")


# R's ``StripTag`` ggproto instance used as the parent of every clone.
_STRIP_TAG_SINGLETON: "StripTag" = StripTag()


def strip_tag(
    clip: str = "inherit",
    order: Any = ("x", "y"),
    just: Any = (0, 1),
    text_x: Any = None,
    text_y: Any = None,
    background_x: Any = None,
    background_y: Any = None,
    by_layer_x: bool = False,
    by_layer_y: bool = False,
) -> StripTag:
    """Create a tag strip (fitted text boxes inside the panels).

    Port of R ``strip_tag()`` (``strip_tag.R:53-85``).

    Parameters
    ----------
    clip : str, default ``"inherit"``
        Whether labels are clipped to background boxes.
    order : sequence of str, default ``("x", "y")``
        Either ``("x", "y")`` or ``("y", "x")`` -- the top-to-bottom order of
        horizontal vs "vertical" labels in a grid layout.
    just : sequence of float, default ``(0, 1)``
        Horizontal and vertical justification for the text box anchor.
    text_x, text_y, background_x, background_y : list or element or None
        Per-strip themed elements.  R defaults ``text_y`` to
        ``element_text(angle = 0)``; pass an explicit value to override.
    by_layer_x, by_layer_y : bool, default ``False``
        Map elements to layers (``True``) or strips (``False``).

    Returns
    -------
    StripTag
        A ``StripTag`` ggproto instance usable in ggh4x facets.
    """
    if text_y is None:
        # R default: text_y = element_text(angle = 0).
        from ggplot2_py import element_text

        text_y = element_text(angle=0)

    params = {
        "clip": arg_match0(clip, ["on", "off", "inherit"], arg_name="clip"),
        "order": list(order),
        "just": list(just),
        # R's ``strip_tag()`` params has no ``size`` (``params$size`` is NULL and
        # the tag's own ``draw_labels`` ignores it).  The base ``assemble_strip``
        # reads ``params["size"]`` with subscript access, so a behaviour-neutral
        # ``"constant"`` is supplied here (consumed only as the always-ignored
        # ``size`` argument of ``StripTag.draw_labels``).  Documented deviation.
        "size": "constant",
    }
    given_elements = {
        "text_x": validate_element_list(text_x, "element_text"),
        "text_y": validate_element_list(text_y, "element_text"),
        "background_x": validate_element_list(background_x, "element_rect"),
        "background_y": validate_element_list(background_y, "element_rect"),
        "by_layer_x": bool(by_layer_x),
        "by_layer_y": bool(by_layer_y),
    }
    return ggproto(
        None,
        _STRIP_TAG_SINGLETON,
        params=params,
        given_elements=given_elements,
    )
