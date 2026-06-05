"""Nested strips for ggh4x facets (port of ggh4x ``strip_nested.R``).

This module ports :class:`StripNested` and the :func:`strip_nested`
constructor.  Nested strips merge adjacent strips on the same layer that share a
label, so an outer faceting variable spans the panels of its inner variables.
It is the default strip for ``facet_nested()`` / ``facet_nested_wrap()``.

``StripNested`` extends :class:`ggh4x.strip_themed.StripThemed`.  It overrides
two methods:

* :meth:`StripNested.assemble_strip` -- run-length-encoding (RLE) merge of
  adjacent equal labels per layer.  A single-layer strip (one faceting variable)
  has nothing to merge and is delegated to the base
  :meth:`Strip.assemble_strip` via :func:`ggplot2_py.ggproto.ggproto_parent`.
* :meth:`StripNested.finish_strip` -- builds *one* multi-cell gtable per
  panel-group (the merged strip) when the redefined layout carries a ``layer``
  column; otherwise (the monolayer fast path produced a plain layout) delegates
  to the self-less :func:`Strip.finish_strip`.

R source: ``ggh4x/R/strip_nested.R``.

Notes
-----
* **Column-major / ROW-COL ordering is load-bearing.**  The labels and layout
  are re-ordered by ``order(ROW, COL)`` (x strips) or ``order(COL, ROW)`` (y
  strips); the RLE separator variable (``ROW`` for x, ``COL`` for y) is pasted
  onto each label to block merges across rows/columns.  When ``bleed`` is
  ``False`` each layer's pasted key is further pasted with all *preceding*
  layers so a lower-layer strip cannot merge across a higher-layer boundary.
* **Run encoding.**  ``rle(x)$lengths`` per layer -> ``ends = cumsum(lengths)``,
  ``starts = ends - lengths + 1`` (1-based, per-layer concatenated).  The merged
  cell takes the label at its run-start row and its layer column
  (``labels[cbind(starts, index)]``).
* **Method-binding asymmetry.**  ``assemble_strip`` / ``finish_strip`` here take
  ``self`` (instance methods).  The monolayer delegate uses
  ``ggproto_parent(Strip, self).assemble_strip(...)`` (parent dispatch on a
  *self-bearing* function), while the no-``layer``-column delegate calls the
  *self-less* :func:`Strip.finish_strip` directly (a plain function).  These two
  idioms are intentionally different and must not be unified.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from ggplot2_py.ggproto import ggproto, ggproto_parent
from grid_py import unit_c, unit_rep
from gtable_py import Gtable, gtable_add_grob

from ggh4x._rlang import arg_match0
from ggh4x.strip_themed import StripThemed
from ggh4x.strip_vanilla import (
    Strip,
    _is_zero_grob,
    validate_element_list,
)

__all__ = ["StripNested", "strip_nested"]


# ---------------------------------------------------------------------------
# Run-length helper
# ---------------------------------------------------------------------------
def _rle_lengths(x: Sequence[Any]) -> List[int]:
    """Port of R ``rle(x)$lengths`` -- run lengths of consecutive equal values.

    Parameters
    ----------
    x : sequence
        Values to run-length encode.

    Returns
    -------
    list of int
        The length of each maximal run of consecutive equal elements.
    """
    lengths: List[int] = []
    prev = object()
    for v in x:
        if v == prev and lengths:
            lengths[-1] += 1
        else:
            lengths.append(1)
            prev = v
    return lengths


def _recycle(seq: Sequence[Any], length_out: int) -> List[Any]:
    """Port of R ``rep(seq, length.out=n)`` for a plain list.

    Parameters
    ----------
    seq : sequence
        Source values.
    length_out : int
        Target length.

    Returns
    -------
    list
        *seq* cyclically recycled (or truncated) to *length_out*.
    """
    n = len(seq)
    if length_out <= 0 or n == 0:
        return []
    return [seq[i % n] for i in range(length_out)]


def _reverse_unit(u: Any) -> Any:
    """Reverse a unit vector (R ``rev(unit_vector)``).

    Parameters
    ----------
    u : grid_py.Unit
        Source unit vector.

    Returns
    -------
    grid_py.Unit
        The unit vector with its elements in reverse order.
    """
    n = len(u)
    if n <= 1:
        return u
    return unit_c(*[u[i] for i in range(n - 1, -1, -1)])


class StripNested(StripThemed):
    """Strip that merges adjacent same-label strips into spanning strips.

    Subclass of :class:`ggh4x.strip_themed.StripThemed`.  See the module
    docstring for the RLE-merge algorithm.

    Attributes
    ----------
    params : dict
        Adds ``bleed`` (bool) to the base ``clip`` / ``size`` params.
    """

    _class_name = "StripNested"

    params: Dict[str, Any] = {"bleed": False}

    def assemble_strip(
        self,
        labels: np.ndarray,
        position: str,
        elements: Dict[str, Any],
        params: Dict[str, Any],
        layout: pd.DataFrame,
    ) -> pd.DataFrame:
        """RLE-merge adjacent equal labels per layer, then draw and finish.

        Port of R ``StripNested$assemble_strip`` (``strip_nested.R:110-168``).

        Parameters
        ----------
        labels : numpy.ndarray
            2-D object array of label strings (rows = panels, cols = layers).
        position : str
            Strip side (``"top"`` / ``"bottom"`` / ``"left"`` / ``"right"``).
        elements : dict
            Resolved element bundle from :meth:`setup_elements`.
        params : dict
            Strip params (``size``, ``clip``, ``bleed``).
        layout : pandas.DataFrame
            The sliced layout for this side (carries ``PANEL`` / ``ROW`` /
            ``COL``).

        Returns
        -------
        pandas.DataFrame
            Placement frame with a ``layer`` column and per-group merged
            gtables (from :meth:`finish_strip`).
        """
        nlayers = labels.shape[1]
        # Monolayer fast path: nothing to merge -> base assembly.
        if nlayers == 1:
            return ggproto_parent(Strip, self).assemble_strip(
                labels, position, elements, params, layout
            )

        aes = "x" if position in ("top", "bottom") else "y"
        bleed = self.params["bleed"]

        # Right strip reverses label columns (inside-out) -- note the base
        # build_strip already reversed; R reverses again here so it operates on
        # outermost-first columns.
        if position == "right":
            labels = labels[:, ::-1]

        if aes == "x":
            sepvar = "ROW"
            order_keys = ["ROW", "COL"]
        else:
            sepvar = "COL"
            order_keys = ["COL", "ROW"]

        # R order(a, b) -> stable lexicographic order.
        order = np.lexsort(
            tuple(np.asarray(layout[k].to_numpy()) for k in reversed(order_keys))
        )
        labels = labels[order, :]
        layout = layout.iloc[order].reset_index(drop=True)

        sep = [str(v) for v in layout[sepvar].tolist()]

        nrow = labels.shape[0]
        # Per-layer pasted key columns (R: tmp[] <- paste0(col, layout[[sepvar]])).
        tmp_cols: List[List[str]] = []
        for j in range(nlayers):
            tmp_cols.append([str(labels[i, j]) + sep[i] for i in range(nrow)])

        if not bleed:
            # Paste each layer (from the 2nd) with all preceding pasted layers
            # to prevent lower-layer bleeding across higher-layer boundaries.
            cumulative = list(tmp_cols[0])
            for j in range(1, nlayers):
                cumulative = [cumulative[i] + tmp_cols[j][i] for i in range(nrow)]
                tmp_cols[j] = list(cumulative)

        # Run lengths per layer.
        lens = [_rle_lengths(col) for col in tmp_cols]
        flat_lens: List[int] = [length for col_lens in lens for length in col_lens]

        ends: List[int] = []
        for col_lens in lens:
            acc = 0
            for length in col_lens:
                acc += length
                ends.append(acc)
        starts = [e - flat_lens[k] + 1 for k, e in enumerate(ends)]

        panel = [int(p) for p in layout["PANEL"].tolist()]
        # layer id per run, repeated by the number of runs in each layer.
        layer = [j + 1 for j, col_lens in enumerate(lens) for _ in col_lens]

        new_layout = pd.DataFrame(
            {
                "t": [panel[s - 1] for s in starts],
                "b": [panel[e - 1] for e in ends],
                "l": [panel[s - 1] for s in starts],
                "r": [panel[e - 1] for e in ends],
                "layer": layer,
            }
        )
        index = list(new_layout["layer"])
        # labels[cbind(starts, index)] -- matrix index (1-based) -> the run-start
        # row at its layer column.
        merged = [labels[starts[k] - 1, index[k] - 1] for k in range(len(starts))]

        elems = self.init_strip(elements, position, index)
        strips = self.draw_labels(merged, elems, position, index, params["size"])

        width = _recycle_unit_attr(strips.width, nlayers)
        height = _recycle_unit_attr(strips.height, nlayers)

        return self.finish_strip(
            list(strips),
            width,
            height,
            position,
            new_layout,
            (new_layout.shape[0], nlayers),
            params["clip"],
        )

    def finish_strip(
        self,
        strip: Sequence[Any],
        width: Any,
        height: Any,
        position: str,
        layout: pd.DataFrame,
        dim: Any,
        clip: str = "inherit",
    ) -> pd.DataFrame:
        """Build one multi-cell gtable per merged group (or delegate).

        Port of R ``StripNested$finish_strip`` (``strip_nested.R:170-200``).
        When the *layout* lacks a ``layer`` column (the monolayer fast path
        produced a plain placement frame) this delegates to the *self-less*
        :func:`Strip.finish_strip`.  Otherwise each merged strip is placed into a
        bare :class:`gtable_py.Gtable` at its layer index (``t=index, l=1`` for
        horizontal; ``t=1, l=index`` for vertical); for ``"bottom"`` / ``"right"``
        the index and the width/height vectors are reversed so the outermost
        layer sits furthest from the panel.

        Parameters
        ----------
        strip : sequence
            The merged label grobs (one per run).
        width, height : grid_py.Unit
            Per-layer width / height unit vectors (length ``nlayers``).
        position : str
            Strip side.
        layout : pandas.DataFrame
            The redefined layout (carries ``layer`` when merged).
        dim : tuple of int
            ``(nrow, nlayers)``.
        clip : str, default ``"inherit"``
            Clip setting.

        Returns
        -------
        pandas.DataFrame
            The *layout* with an attached object ``grobs`` column.
        """
        # No 'layer' column -> delegate to the self-less base finish_strip.
        if "layer" not in layout.columns:
            return Strip.finish_strip(
                strip, width, height, position, layout, dim, clip
            )

        strip = list(strip)
        empty_strips = len(strip) == 0 or all(_is_zero_grob(g) for g in strip)

        out_grobs: List[Any] = strip
        if not empty_strips:
            index = [int(v) for v in layout["layer"]]
            ncols = int(dim[1])
            if position in ("bottom", "right"):
                index = [ncols - i + 1 for i in index]
                width = _reverse_unit(width)
                height = _reverse_unit(height)
            out_grobs = []
            if position in ("top", "bottom"):
                # gt = gtable(widths = width[1], heights = height)
                for g, i in zip(strip, index):
                    gt = Gtable(widths=_unit_slice(width, 0), heights=height)
                    gt = gtable_add_grob(gt, g, t=i, l=1, clip=clip)
                    out_grobs.append(gt)
            else:
                for g, i in zip(strip, index):
                    gt = Gtable(widths=width, heights=_unit_slice(height, 0))
                    gt = gtable_add_grob(gt, g, t=1, l=i, clip=clip)
                    out_grobs.append(gt)

        result = layout.copy()
        result["grobs"] = pd.Series(out_grobs, index=result.index, dtype=object)
        return result


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------
def _recycle_unit_attr(u: Any, length_out: int) -> Any:
    """Recycle the ``width``/``height`` unit attribute to ``nlayers``.

    Port of R ``rep(attr(strips, 'width'), length.out = nlayers)``.

    Parameters
    ----------
    u : grid_py.Unit or None
        The unit vector attribute from ``draw_labels``.
    length_out : int
        Target length (number of layers).

    Returns
    -------
    grid_py.Unit or None
        The recycled unit vector (``None`` passes through).
    """
    if u is None:
        return None
    return unit_rep(u, length_out=length_out)


def _unit_slice(u: Any, i: int) -> Any:
    """Return the single-element unit at position *i* (R ``u[i+1]``).

    Parameters
    ----------
    u : grid_py.Unit
        Source unit vector.
    i : int
        0-based index.

    Returns
    -------
    grid_py.Unit
        The unit at *i* (or *u* itself when it is already scalar).
    """
    n = len(u)
    if n <= 1:
        return u
    return u[i % n]


# R's ``StripNested`` ggproto instance used as the parent of every clone.
_STRIP_NESTED_SINGLETON: "StripNested" = StripNested()


def strip_nested(
    clip: str = "inherit",
    size: str = "constant",
    bleed: bool = False,
    text_x: Any = None,
    text_y: Any = None,
    background_x: Any = None,
    background_y: Any = None,
    by_layer_x: bool = False,
    by_layer_y: bool = False,
) -> StripNested:
    """Create a nested (label-merging) strip.

    Port of R ``strip_nested()`` (``strip_nested.R:66-97``).

    Parameters
    ----------
    clip : str, default ``"inherit"``
        Whether labels are clipped to background boxes.
    size : str, default ``"constant"``
        Whether strip margins across layers are ``"constant"`` or ``"variable"``.
    bleed : bool, default ``False``
        Whether lower-layer strips may merge across higher-layer boundaries.
    text_x, text_y, background_x, background_y : list or element or None
        Per-strip themed elements (see :func:`ggh4x.strip_themed.strip_themed`).
    by_layer_x, by_layer_y : bool, default ``False``
        Map elements to layers (``True``) or strips (``False``).

    Returns
    -------
    StripNested
        A ``StripNested`` ggproto instance usable in ggh4x facets.
    """
    params = {
        "clip": arg_match0(clip, ["on", "off", "inherit"], arg_name="clip"),
        "size": arg_match0(size, ["constant", "variable"], arg_name="size"),
        "bleed": bool(bleed),
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
        _STRIP_NESTED_SINGLETON,
        params=params,
        given_elements=given_elements,
    )
