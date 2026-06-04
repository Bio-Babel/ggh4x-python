"""Default (vanilla) strips for ggh4x facets (port of ggh4x ``strip_vanilla.R``).

This module ports the **base** ``Strip`` ggproto class, the ``strip_vanilla()``
constructor, and the strip helpers ``resolve_strip`` / ``assert_strip`` /
``validate_element_list`` / ``inherit_element``.  The themed/nested/split/tag
subclasses live in separate modules.

The ``Strip`` hierarchy is a *self-rooted* ggproto hierarchy (its own base, not
a ggplot2 ``Facet``).  It is consumed by the ggh4x facet subsystem, which stores
a ``Strip`` instance and during panel drawing calls
``strip.setup(layout, params, theme, type)`` followed by
``strip.incorporate_grid(panels, switch)`` (grid facets) or
``strip.incorporate_wrap(panels, position, clip, sizes)`` (wrap/manual facets).

R source: ``ggh4x/R/strip_vanilla.R`` (the ``Strip`` base) and the helpers
``validate_element_list`` / ``inherit_element`` from ``ggh4x/R/strip_themed.R``.

Notes on faithful porting decisions (verified against ggplot2 4.0.2 + ggh4x
0.3.1.9000 run through ``Rscript``):

* **Self-less methods.**  In R, ``Strip$draw_labels``, ``Strip$init_strip`` and
  ``Strip$finish_strip`` are plain functions *without* a ``self`` argument.
  They are installed here as class-level functions whose first parameter is not
  named ``self``, so ``ggproto``'s auto-self-binding leaves them unbound -- a
  call like ``self.draw_labels(labels, ...)`` passes ``labels`` as the first
  positional, never ``self``.  ``assemble_strip`` / ``build_strip`` etc. *do*
  take ``self`` and are bound normally.

* **Guide system path.**  R ``draw_labels`` branches on the package-internal
  ``new_guide_system`` flag.  On ggplot2 4.0.2 this flag is ``TRUE`` (verified),
  so the **new-guide path** is the gold standard and the only one ported: label
  height/width come straight from ``grob_height`` / ``grob_width`` of the title
  grob (which already includes margins via ``_TitleGrob``), and the
  old-guide-system ``vp$parent$layout`` margin surgery is *not* executed.  See
  the module-level note in :func:`_draw_labels_impl`.

* **strip.placement precedence.**  R ``calc_element('strip.placement.x', theme)
  %||% 'inside' == 'inside'`` parses (verified via the R parse tree) as
  ``(calc_element(...) %||% 'inside') == 'inside'`` because ``%||%`` binds
  tighter than ``==``.  The faithful Python form is therefore
  ``(el if el is not None else 'inside') == 'inside'``.

* **Column-major order.**  R ``as.vector(col(labels))``, ``matrix()`` reshape
  and ``apply(mat, 1, ...)`` all rely on Fortran (column-major) ordering.  The
  port uses ``order='F'`` reshapes / explicit column-major iteration so strip
  cells land in the right panels.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ggplot2_py import (
    calc_element,
    element_grob,
    element_render,
    ggproto,
    is_ggproto,
    is_theme_element,
    max_height,
    max_width,
)
from ggplot2_py.ggproto import GGProto
from grid_py import (
    Unit,
    convert_unit,
    grob_name,
    grob_tree,
    unit_c,
    unit_rep,
)
from gtable_py import gtable_matrix

from ggh4x._cli import cli_abort
from ggh4x._rlang import arg_match0

__all__ = [
    "Strip",
    "strip_vanilla",
    "resolve_strip",
    "assert_strip",
    "validate_element_list",
    "inherit_element",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _is_zero_grob(grob: Any) -> bool:
    """Return ``True`` for R ``is.zero`` -- a ``zeroGrob`` / null grob.

    ``grid_py`` has no dedicated ``zeroGrob`` class; ``null_grob()`` returns a
    plain :class:`grid_py.Grob` named ``GRID.null.N`` (or with ``_grid_class ==
    "null"``).  This mirrors ggplot2_py's own ``_is_null_grob`` detection.

    Parameters
    ----------
    grob : Any
        Candidate grob (or ``None``).

    Returns
    -------
    bool
        ``True`` when *grob* is ``None`` or a null/zero grob.
    """
    if grob is None:
        return True
    cls = getattr(grob, "_grid_class", "")
    name = getattr(grob, "_name", getattr(grob, "name", ""))
    return cls == "null" or "null" in str(name).lower() or "zero" in str(name).lower()


def _rep_len(seq: Sequence[Any], length_out: int) -> List[Any]:
    """Port of R ``rep_len(seq, length_out)`` (cyclic recycling).

    Parameters
    ----------
    seq : sequence
        Source values (must be non-empty when ``length_out > 0``).
    length_out : int
        Desired output length.

    Returns
    -------
    list
        *seq* recycled (or truncated) to exactly *length_out* elements.
    """
    n = len(seq)
    if length_out <= 0 or n == 0:
        return []
    return [seq[i % n] for i in range(length_out)]


class _LabelGrobs(list):
    """A list of label grobs carrying R ``attr(., "width")`` / ``"height")``.

    R attaches ``width`` and ``height`` attributes onto the returned label list
    in ``draw_labels``; ``assemble_strip`` reads them back.  pandas / Python
    ``list`` carry no R-style attributes, so this thin ``list`` subclass holds
    the two unit vectors as plain attributes.

    Attributes
    ----------
    width : grid_py.Unit or None
        Per-layer width unit vector (set by :func:`_draw_labels_impl`).
    height : grid_py.Unit or None
        Per-layer height unit vector.
    """

    width: Any = None
    height: Any = None


# ---------------------------------------------------------------------------
# Self-less Strip methods (R: plain functions WITHOUT a `self` argument)
# ---------------------------------------------------------------------------
def _init_strip_impl(
    elements: Dict[str, Any],
    position: str,
    layer_index: Sequence[int],
) -> Dict[str, List[Any]]:
    """Pick + expand the text/background elements for a strip side.

    Port of R ``Strip$init_strip`` (``strip_vanilla.R:254-276``) -- a *self-less*
    method.  Selects ``elements[['text']][aes][position]`` and
    ``elements[['background']][aes]``, wraps singletons in a list, then expands
    to one element per cell either by-layer (``pmin(layer_index, len)``) or by
    cyclic ``rep_len``.

    Parameters
    ----------
    elements : dict
        The resolved element bundle from ``setup_elements`` (keys ``text``,
        ``background``, optionally ``by_layer``).
    position : str
        One of ``"top"``, ``"bottom"``, ``"left"``, ``"right"``.
    layer_index : sequence of int
        1-based column index per cell (column-major), from ``col(labels)``.

    Returns
    -------
    dict
        ``{"el": [...], "bg": [...]}`` -- one text element and one background
        element per cell.
    """
    aes = "x" if position in ("top", "bottom") else "y"

    el = elements["text"][aes][position]
    el = el if isinstance(el, list) else [el]

    bg = elements["background"][aes]
    bg = bg if isinstance(bg, list) else [bg]

    by_layer_map = elements.get("by_layer")
    if by_layer_map is None:
        by_layer = False
    else:
        by_layer = by_layer_map[aes]

    layer_index = list(layer_index)
    if by_layer:
        # R: el[pmin(layer_index, length(el))] -- 1-based indexing.
        el = [el[min(i, len(el)) - 1] for i in layer_index]
        bg = [bg[min(i, len(bg)) - 1] for i in layer_index]
    else:
        el = _rep_len(el, len(layer_index))
        bg = _rep_len(bg, len(layer_index))

    return {"el": el, "bg": bg}


def _draw_labels_impl(
    labels: Sequence[str],
    element: Dict[str, List[Any]],
    position: str,
    layer_id: Sequence[int],
    size: str,
) -> _LabelGrobs:
    """Build per-label title+background grobs and size them per layer.

    Port of R ``Strip$draw_labels`` (``strip_vanilla.R:145-223``) -- a
    *self-less* method.  **Only the new-guide-system path is ported**: on
    ggplot2 4.0.2 the package-internal ``new_guide_system`` flag is ``TRUE``
    (verified), so

    * label height/width come directly from :func:`grid_py.grob_height` /
      :func:`grid_py.grob_width` of each title grob (which, as a ``_TitleGrob``,
      already includes the element margins);
    * per-layer maxima are taken via :func:`ggplot2_py.max_height` /
      :func:`ggplot2_py.max_width` over groups defined by ``layer_id``;
    * the cross-axis dimension is set to ``unit(1, "null")``.

    The old-guide-system branch (R lines 152-211) -- which re-injects equalised
    widths/heights into ``grob$widths``, ``grob$heights`` *and* the four-deep
    ``grob[[c("vp", "parent", "layout", "widths/heights")]]`` path -- is **not**
    executed, because that path is the new-guide gold standard.  ``ggplot2_py``'s
    ``_TitleGrob`` exposes ``grob_height``/``grob_width`` that fold in the
    margins, so the new-guide path reproduces R's strip sizing without any
    ``vp.parent.layout`` surgery (which does not exist on the Python grob).

    Parameters
    ----------
    labels : sequence of str
        Flattened (column-major) label strings, one per cell.
    element : dict
        ``{"el": [...text elements...], "bg": [...background grobs/elements...]}``
        from :func:`_init_strip_impl`, one per cell.
    position : str
        Strip side (``"top"``/``"bottom"``/``"left"``/``"right"``).
    layer_id : sequence of int
        1-based layer (column) id per cell.
    size : str
        ``"constant"`` (all layers share one margin set -> ``layer_id`` collapsed
        to all ``1``) or ``"variable"``.

    Returns
    -------
    _LabelGrobs
        A list of GTree grobs (background + text per cell) with ``.width`` /
        ``.height`` unit-vector attributes attached.
    """
    layer_id = list(layer_id)
    if size == "constant":
        layer_id = [1] * len(layer_id)

    aes = "x" if position in ("top", "bottom") else "y"

    # Build the title grob per label (background composited later).
    grobs: List[Any] = []
    for label, elem in zip(labels, element["el"]):
        grob = element_grob(elem, label=label, margin_x=True, margin_y=True)
        # new-guide-system: no add_margins fix-up needed (titleGrob carries
        # margins). Re-name to mirror R grobName(grob, "strip.text.<aes>").
        try:
            grob.name = grob_name(grob, "strip.text." + aes)
        except Exception:
            pass
        grobs.append(grob)

    zeros = [_is_zero_grob(g) for g in grobs]

    out = _LabelGrobs(grobs)
    if len(grobs) == 0 or all(zeros):
        out.width = None
        out.height = None
        return out

    nonzero_idx = [i for i, z in enumerate(zeros) if not z]
    nonzero_layer = [layer_id[i] for i in nonzero_idx]

    if aes == "x":
        # Per-layer max height; width is 1 null per layer.
        heights = [grobs[i] for i in nonzero_idx]
        grouped = _split_by(heights, nonzero_layer)
        height_units = [max_height(g) for g in grouped]
        height = unit_c(*height_units) if len(height_units) > 1 else height_units[0]
        width = unit_rep(Unit(1, "null"), length_out=len(height_units))
    else:
        widths = [grobs[i] for i in nonzero_idx]
        grouped = _split_by(widths, nonzero_layer)
        width_units = [max_width(g) for g in grouped]
        width = unit_c(*width_units) if len(width_units) > 1 else width_units[0]
        height = unit_rep(Unit(1, "null"), length_out=len(width_units))

    # Combine each label grob with its background into a gTree.
    combined: List[Any] = []
    for x, bg in zip(grobs, element["bg"]):
        bg_grob = element_grob(bg) if is_theme_element(bg) else bg
        tree = grob_tree(bg_grob, x)
        try:
            tree.name = grob_name(tree, "strip")
        except Exception:
            pass
        combined.append(tree)

    result = _LabelGrobs(combined)
    result.width = width
    result.height = height
    return result


def _finish_strip_impl(
    strip: Sequence[Any],
    width: Any,
    height: Any,
    position: str,
    layout: pd.DataFrame,
    dim: Tuple[int, int],
    clip: str = "inherit",
) -> pd.DataFrame:
    """Reshape the flat grob list into one gtable per panel; build placement.

    Port of R ``Strip$finish_strip`` (``strip_vanilla.R:225-252``) -- a
    *self-less* method.  The grob list is reshaped ``matrix(strip, ncol=dim[2],
    nrow=dim[1])`` (column-major), then ``apply(strip, 1, ...)`` iterates over
    **rows** (panels) producing a 1-column (horizontal) or 1-row (vertical)
    sub-matrix per panel, each wrapped in :func:`gtable_py.gtable_matrix`.

    Parameters
    ----------
    strip : sequence
        Flattened (column-major) label grobs, length ``dim[0] * dim[1]``.
    width : grid_py.Unit
        Per-layer width unit vector.
    height : grid_py.Unit
        Per-layer height unit vector.
    position : str
        Strip side.
    layout : pandas.DataFrame
        The sliced layout carrying ``PANEL`` (and ``ROW``/``COL``/...).
    dim : tuple of int
        ``(nrow, ncol)`` of the label matrix (panels x layers).
    clip : str, default ``"inherit"``
        Clip setting forwarded to ``gtable_matrix``.

    Returns
    -------
    pandas.DataFrame
        Placement frame with integer ``t``/``l``/``b``/``r`` == ``PANEL`` and an
        object ``grobs`` column holding the per-panel gtables (or the raw grob
        list when empty).
    """
    strip = list(strip)
    empty_strips = len(strip) == 0 or all(_is_zero_grob(g) for g in strip)

    horizontal = position in ("top", "bottom")
    out_grobs: List[Any] = strip
    if not empty_strips:
        nrow, ncol = int(dim[0]), int(dim[1])
        # R matrix(strip, ncol, nrow) fills column-major.
        # R matrix(strip, ncol, nrow) fills column-major: element (i, j) is
        # flat[i + j * nrow]. Build the object matrix explicitly so numpy never
        # tries to introspect the GTree grobs as nested sequences.
        mat = [[strip[i + j * nrow] for j in range(ncol)] for i in range(nrow)]

        out_grobs = []
        for i in range(nrow):
            row = mat[i]  # the layers belonging to panel-row i
            if horizontal:
                # apply(strip, 1, matrix, ncol=1) -> one column, nrow == ncol layers.
                sub = [[row[j]] for j in range(ncol)]
                widths = _recycle_unit(width, length_out=1)
                heights = _recycle_unit(height, length_out=ncol)
            else:
                # apply(strip, 1, matrix, nrow=1) -> one row, ncol == ncol layers.
                sub = [[row[j] for j in range(ncol)]]
                widths = _recycle_unit(width, length_out=ncol)
                heights = _recycle_unit(height, length_out=1)
            out_grobs.append(
                gtable_matrix("strip", sub, widths, heights, clip=clip)
            )

    panel = [int(p) for p in layout["PANEL"]]
    return pd.DataFrame(
        {
            "t": panel,
            "l": panel,
            "b": panel,
            "r": panel,
            "grobs": out_grobs,
        }
    )


def _split_by(values: Sequence[Any], keys: Sequence[Any]) -> List[List[Any]]:
    """Port of R ``split(values, keys)`` preserving sorted-key group order.

    R ``split`` groups by the sorted unique key levels.  Since ``keys`` here are
    contiguous 1-based layer ids, the groups come out in layer order.

    Parameters
    ----------
    values : sequence
        Values to group.
    keys : sequence
        Grouping key per value.

    Returns
    -------
    list of list
        One group per sorted-unique key.
    """
    order = sorted(set(keys))
    groups: Dict[Any, List[Any]] = {k: [] for k in order}
    for v, k in zip(values, keys):
        groups[k].append(v)
    return [groups[k] for k in order]


def _recycle_unit(u: Any, length_out: int) -> Any:
    """Port of R ``rep(unit, length.out=n)`` for a unit vector.

    Parameters
    ----------
    u : grid_py.Unit
        Source unit vector.
    length_out : int
        Target length.

    Returns
    -------
    grid_py.Unit
        *u* recycled to *length_out* elements.
    """
    return unit_rep(u, length_out=length_out)


# ---------------------------------------------------------------------------
# Strip base ggproto class
# ---------------------------------------------------------------------------
class Strip(GGProto):
    """Base strip class for ggh4x facets (port of R ``Strip``).

    Subclass of :class:`ggplot2_py.ggproto.GGProto`.  Builds strip grobs from a
    facet ``layout`` + ``params`` + ``theme`` and weaves them into the assembled
    panel gtable.  Instances are produced by :func:`strip_vanilla` (and the
    subclass constructors).

    Attributes
    ----------
    clip : str
        Class-level default ``"inherit"`` (instances override via ``params``).
    elements : dict
        Resolved theme element bundle (set by :meth:`setup`).
    params : dict
        Strip parameters (``clip``, ``size`` for vanilla).
    strips : dict
        Built strip placement frames, shape ``{"x": {"top", "bottom"},
        "y": {"left", "right"}}`` (set by :meth:`get_strips`).

    Notes
    -----
    ``draw_labels`` / ``init_strip`` / ``finish_strip`` are installed as
    *self-less* class attributes (see module docstring): their first parameter is
    not ``self`` so ggproto does not inject a receiver.
    """

    _class_name = "Strip"

    clip: str = "inherit"
    elements: Dict[str, Any] = {}
    params: Dict[str, Any] = {}
    strips: Dict[str, Any] = {}

    # --- self-less methods (NOT auto-bound: first arg is not `self`) --------
    draw_labels = staticmethod(_draw_labels_impl)
    init_strip = staticmethod(_init_strip_impl)
    finish_strip = staticmethod(_finish_strip_impl)

    def setup_elements(self, theme: Any, type: str) -> Dict[str, Any]:
        """Resolve strip theme elements for one facet kind.

        Port of R ``Strip$setup_elements`` (``strip_vanilla.R:70-103``).
        Resolves backgrounds (rendered grobs via ``element_render``), per-side
        text elements (``calc_element``), inside/outside placement booleans, and
        the switch padding (converted to cm).

        Parameters
        ----------
        theme : Theme
            The active theme.
        type : str
            ``"wrap"`` selects ``strip.switch.pad.wrap`` padding; anything else
            (``"grid"``) selects ``strip.switch.pad.grid``.  (Kept as the
            parameter name ``type`` for facet-call compatibility.)

        Returns
        -------
        dict
            ``{"padding", "background", "text", "inside"}``.
        """
        background = {
            "x": element_render(theme, "strip.background.x"),
            "y": element_render(theme, "strip.background.y"),
        }
        text = {
            "x": {
                "top": calc_element("strip.text.x.top", theme),
                "bottom": calc_element("strip.text.x.bottom", theme),
            },
            "y": {
                "left": calc_element("strip.text.y.left", theme),
                "right": calc_element("strip.text.y.right", theme),
            },
        }
        inside = {
            "x": _placement_inside(calc_element("strip.placement.x", theme)),
            "y": _placement_inside(calc_element("strip.placement.y", theme)),
        }
        pad_name = (
            "strip.switch.pad.wrap" if type == "wrap" else "strip.switch.pad.grid"
        )
        padding = calc_element(pad_name, theme)
        padding = convert_unit(padding, "cm")

        return {
            "padding": padding,
            "background": background,
            "text": text,
            "inside": inside,
        }

    def setup(
        self,
        layout: pd.DataFrame,
        params: Dict[str, Any],
        theme: Any,
        type: str,
    ) -> None:
        """Entry point invoked by the facet; resolve elements + build strips.

        Port of R ``Strip$setup`` (``strip_vanilla.R:105-132``).  Stores the
        resolved element bundle on the instance, then builds the per-side label
        frames (``col_vars`` / ``row_vars``) and the sliced ``layout_x`` /
        ``layout_y`` frames, branching on wrap vs grid, and delegates to
        :meth:`get_strips`.

        Parameters
        ----------
        layout : pandas.DataFrame
            The facet layout (carries ``PANEL``/``ROW``/``COL`` and the facet
            variable columns).
        params : dict
            Facet params (``facets`` for wrap; ``cols``/``rows`` for grid; plus
            ``labeller``).
        theme : Theme
            The active theme.
        type : str
            ``"wrap"`` or ``"grid"`` (kept as ``type`` for facet compatibility).
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
            layout_x = layout
            layout_y = layout
        else:
            col_names = _param_names(params.get("cols"))
            row_names = _param_names(params.get("rows"))
            col_mask = _not_duplicated(layout, col_names)
            row_mask = _not_duplicated(layout, row_names)
            layout_x = layout.loc[col_mask]
            layout_y = layout.loc[row_mask]
            col_vars = layout_x[col_names] if col_names else _empty_frame(layout_x)
            row_vars = layout_y[row_names] if row_names else _empty_frame(layout_y)

        self.get_strips(
            x=_VarFrame(col_vars, type="cols", facet=type),
            y=_VarFrame(row_vars, type="rows", facet=type),
            labeller=params.get("labeller"),
            theme=theme,
            params=self.params,
            layout_x=layout_x,
            layout_y=layout_y,
        )

    def get_strips(
        self,
        x: Any = None,
        y: Any = None,
        labeller: Any = None,
        theme: Any = None,
        params: Optional[Dict[str, Any]] = None,
        layout_x: Optional[pd.DataFrame] = None,
        layout_y: Optional[pd.DataFrame] = None,
    ) -> None:
        """Build the x and y strips and store them on the instance.

        Port of R ``Strip$get_strips`` (``strip_vanilla.R:135-143``).  Calls
        :meth:`build_strip` for the horizontal (x) and vertical (y) sides and
        writes ``self.strips = {"x": {top, bottom}, "y": {left, right}}``.

        Parameters
        ----------
        x, y : _VarFrame or pandas.DataFrame
            The column / row label frames.
        labeller : callable or str
            Labeller spec.
        theme : Theme
            Active theme.
        params : dict
            Strip params.
        layout_x, layout_y : pandas.DataFrame
            The sliced layout frames for the x / y sides.
        """
        self._set(
            strips={
                "x": self.build_strip(x, labeller, theme, True, params, layout_x),
                "y": self.build_strip(y, labeller, theme, False, params, layout_y),
            }
        )

    def assemble_strip(
        self,
        labels: np.ndarray,
        position: str,
        elements: Dict[str, Any],
        params: Dict[str, Any],
        layout: pd.DataFrame,
    ) -> pd.DataFrame:
        """Index, init, draw and finish one strip side.

        Port of R ``Strip$assemble_strip`` (``strip_vanilla.R:279-290``).
        Computes the column-major layer index ``col(labels)``, delegates to the
        self-less ``init_strip`` / ``draw_labels`` / ``finish_strip``.

        Parameters
        ----------
        labels : numpy.ndarray
            2-D object array of label strings (rows = panels, cols = layers).
        position : str
            Strip side.
        elements : dict
            Resolved element bundle.
        params : dict
            Strip params (``size``, ``clip``).
        layout : pandas.DataFrame
            The sliced layout for this side.

        Returns
        -------
        pandas.DataFrame
            The placement frame from ``finish_strip``.
        """
        index = _col_index(labels)
        elems = self.init_strip(elements, position, index)
        strips = self.draw_labels(
            _flatten_col_major(labels), elems, position, index, params["size"]
        )
        width = strips.width
        height = strips.height
        return self.finish_strip(
            strips, width, height, position, layout,
            (labels.shape[0], labels.shape[1]), params["clip"],
        )

    def build_strip(
        self,
        data: Any,
        labeller: Any,
        theme: Any,
        horizontal: bool,
        params: Dict[str, Any],
        layout: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Format labels into a matrix and assemble strips per side.

        Port of R ``Strip$build_strip`` (``strip_vanilla.R:293-319``).  When the
        data is empty, returns a named ``None`` pair.  Otherwise applies the
        labeller per variable (R ``do.call(cbind, lapply(labels(data), cbind))``
        -> rows = panels, cols = variables) and assembles both sides; the right
        strip reverses label columns (inside-out).

        Parameters
        ----------
        data : _VarFrame or pandas.DataFrame
            The per-side label frame.
        labeller : callable or str
            Labeller spec.
        theme : Theme
            Active theme.
        horizontal : bool
            ``True`` -> top/bottom strips; ``False`` -> left/right.
        params : dict
            Strip params.
        layout : pandas.DataFrame
            The sliced layout for this side.

        Returns
        -------
        dict
            ``{"top", "bottom"}`` (horizontal) or ``{"left", "right"}``.
        """
        frame = data.frame if isinstance(data, _VarFrame) else data
        if _empty_data(frame):
            if horizontal:
                return {"top": None, "bottom": None}
            return {"left": None, "right": None}

        labels = _format_labels(frame, labeller)
        elem = self.elements

        if horizontal:
            top = self.assemble_strip(labels, "top", elem, params, layout)
            bottom = self.assemble_strip(labels, "bottom", elem, params, layout)
            return {"top": top, "bottom": bottom}
        else:
            revlab = labels[:, ::-1]
            right = self.assemble_strip(revlab, "right", elem, params, layout)
            left = self.assemble_strip(labels, "left", elem, params, layout)
            return {"left": left, "right": right}

    def incorporate_wrap(
        self,
        panels: Any,
        position: str,
        clip: str = "off",
        sizes: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Insert strips for one position into wrapped panels.

        Port of R ``Strip$incorporate_wrap`` (``strip_vanilla.R:322-375``).
        Uses ``weave_panel_rows`` / ``weave_panel_cols`` from
        :mod:`ggh4x._facet_utils` (imported lazily so module import never
        depends on those helpers being present).

        Parameters
        ----------
        panels : Gtable
            The assembled panel gtable.
        position : str
            ``"top"``/``"bottom"``/``"left"``/``"right"``.
        clip : str, default ``"off"``
            Clip setting for the strips.
        sizes : dict
            Per-position size unit vectors (used for padding placement).

        Returns
        -------
        Gtable
            The panel gtable with this position's strips woven in.
        """
        from ggh4x._facet_utils import (
            split_heights_cm,
            split_widths_cm,
            weave_panel_cols,
            weave_panel_rows,
        )

        strip_padding = self.elements["padding"]
        size_vec = sizes[position]
        padding = _padding_from_sizes(size_vec, strip_padding)

        strip = _flatten_strips(self.strips)[position]
        inside = self.elements["inside"]
        side = position[0]
        strip_name = "strip-" + side
        offset = {
            "t": -2 + int(inside["x"]),
            "b": 1 - int(inside["x"]),
            "l": -2 + int(inside["y"]),
            "r": 1 - int(inside["y"]),
        }[side]

        if side in ("t", "b"):
            strip_height = split_heights_cm(list(strip["grobs"]), list(strip["t"]))
            panels = weave_panel_rows(
                panels, strip, offset, strip_height, strip_name, 2, clip, side
            )
            if not inside["x"]:
                panels = weave_panel_rows(
                    panels, row_shift=offset, row_height=padding
                )
        else:
            strip_width = split_widths_cm(list(strip["grobs"]), list(strip["l"]))
            panels = weave_panel_cols(
                panels, strip, offset, strip_width, strip_name, 2, clip, side
            )
            if not inside["y"]:
                panels = weave_panel_cols(
                    panels, col_shift=offset, col_width=padding
                )
        return panels

    def incorporate_grid(self, panels: Any, switch: Any) -> Any:
        """Insert x then y strips into the assembled grid panel gtable.

        Port of R ``Strip$incorporate_grid`` (``strip_vanilla.R:378-449``).
        Reads panel cell coordinates from the gtable layout (rows named
        ``^panel-``), handles inside/outside placement (adds a padding row/col
        when outside), inserts the strip row/col and adds the strip grobs with
        ``z=2``, ``clip="on"``.  Panel positions are re-derived after the x block
        because ``gtable_add_rows`` shifts indices.

        Parameters
        ----------
        panels : Gtable
            The assembled panel gtable (axes already attached).
        switch : str or None
            ``"x"`` / ``"y"`` / ``"both"`` / ``None`` -- which axes' strips
            switch sides.

        Returns
        -------
        Gtable
            The panel gtable with strips inserted.
        """
        from gtable_py import gtable_add_cols, gtable_add_grob, gtable_add_rows

        switch_x = switch in ("both", "x")
        switch_y = switch in ("both", "y")
        inside = self.elements["inside"]
        padding = self.elements["padding"]
        strips = self.strips

        pos_cols = _panel_layout(panels)

        if switch_x:
            side = strips["x"]["bottom"]
            prefix = "strip-b-"
        else:
            side = strips["x"]["top"]
            prefix = "strip-t-"
        strip = _grobs_of(side)
        table = _tlbr_of(side)

        if strip is not None:
            stripnames = [prefix + str(i + 1) for i in range(len(strip))]
            stripheight = max_height(strip)
            if inside["x"]:
                where = -2 if switch_x else 1
            else:
                where = 0 - int(switch_x)
                panels = gtable_add_rows(panels, padding, where)
            panels = gtable_add_rows(panels, stripheight, where)
            panels = gtable_add_grob(
                panels, strip, name=stripnames,
                t=where + (0 if switch_x else 1),
                l=[pos_cols["l"][li] for li in table["l"]],
                r=[pos_cols["r"][ri] for ri in table["r"]],
                clip="on", z=2,
            )

        pos_rows = _panel_layout(panels)

        if switch_y:
            side = strips["y"]["left"]
            prefix = "strip-l-"
        else:
            side = strips["y"]["right"]
            prefix = "strip-r-"
        strip = _grobs_of(side)
        table = _tlbr_of(side)

        if strip is not None:
            stripnames = [prefix + str(i + 1) for i in range(len(strip))]
            stripwidth = max_width(strip)
            if inside["y"]:
                where = 1 if switch_y else -2
            else:
                where = -1 + int(switch_y)
                panels = gtable_add_cols(panels, padding, where)
            panels = gtable_add_cols(panels, stripwidth, where)
            panels = gtable_add_grob(
                panels, strip, name=stripnames,
                t=[pos_rows["t"][ti] for ti in table["t"]],
                b=[pos_rows["b"][bi] for bi in table["b"]],
                l=where + int(switch_y),
                clip="on", z=2,
            )

        return panels


# Re-bind the self-less methods on the class so attribute access returns the
# raw functions (staticmethod is unwrapped on access, leaving no `self`).
Strip.draw_labels = staticmethod(_draw_labels_impl)
Strip.init_strip = staticmethod(_init_strip_impl)
Strip.finish_strip = staticmethod(_finish_strip_impl)

# R's ``Strip`` is a ggproto *instance* (env), used as the parent in every
# constructor.  The Python ``Strip`` is a class; this module-level singleton is
# the instance the constructors clone (instance-as-parent path).  Subclass
# constructors should likewise clone their own singletons.
_STRIP_SINGLETON: "Strip" = Strip()


# ---------------------------------------------------------------------------
# Internal data-frame side-channel + layout helpers
# ---------------------------------------------------------------------------
class _VarFrame:
    """Carry R ``attr(., "type")`` / ``attr(., "facet")`` alongside a frame.

    R attaches ``type = "cols"/"rows"`` and ``facet = type`` onto the var frame
    via ``structure()`` / ``attr<-``; pandas has no per-object attribute slot, so
    this thin wrapper carries them out of band.

    Parameters
    ----------
    frame : pandas.DataFrame
        The label var frame.
    type : str
        ``"cols"`` or ``"rows"``.
    facet : str
        The facet kind (``"grid"`` / ``"wrap"``).
    """

    __slots__ = ("frame", "type", "facet")

    def __init__(self, frame: pd.DataFrame, type: str, facet: str) -> None:
        self.frame = frame
        self.type = type
        self.facet = facet


def _param_names(param: Any) -> List[str]:
    """Return the variable names from a facet ``cols``/``rows`` param.

    R uses ``names(params$cols)``.  The param may be a dict / mapping, a list of
    names, or ``None``.

    Parameters
    ----------
    param : Any
        A ``cols``/``rows`` spec.

    Returns
    -------
    list of str
    """
    if param is None:
        return []
    if hasattr(param, "keys"):
        return list(param.keys())
    if isinstance(param, (list, tuple)):
        return [str(p) for p in param]
    return []


def _empty_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a 0-column frame with the same row count as *df*.

    Mirrors R ``layout[character(0)]`` -- a frame with ``nrow(layout)`` rows and
    no columns, whose ``duplicated()`` is all-but-first ``TRUE``.
    """
    return pd.DataFrame(index=df.index)


def _not_duplicated(layout: pd.DataFrame, names: List[str]) -> np.ndarray:
    """Port of R ``!duplicated(layout[names])`` (boolean keep-mask).

    When *names* is empty, R ``layout[character(0)]`` is a 0-column frame and
    ``duplicated()`` treats every row as identical -- only the first row is kept.
    pandas ``DataFrame.duplicated()`` on a 0-column frame returns a length-0
    array, so that case is handled explicitly.

    Parameters
    ----------
    layout : pandas.DataFrame
        The facet layout.
    names : list of str
        Column names to de-duplicate on.

    Returns
    -------
    numpy.ndarray of bool
        ``True`` for the first occurrence of each unique combination.
    """
    n = layout.shape[0]
    if not names:
        mask = np.zeros(n, dtype=bool)
        if n > 0:
            mask[0] = True
        return mask
    return ~layout[names].duplicated().to_numpy()


def _placement_inside(el: Any) -> bool:
    """Faithful port of R ``calc_element('strip.placement.x', th) %||% 'inside' == 'inside'``.

    Verified against the R parse tree: ``%||%`` binds tighter than ``==``, so the
    expression is ``(el %||% 'inside') == 'inside'``.

    Parameters
    ----------
    el : str or None
        The resolved ``strip.placement.*`` value.

    Returns
    -------
    bool
        ``True`` for ``"inside"`` (or unset / ``None``); ``False`` otherwise.
    """
    return (el if el is not None else "inside") == "inside"


def _empty_data(frame: Any) -> bool:
    """Port of R ``empty(data)`` for the var frame.

    R ``empty()`` is ``TRUE`` for ``NULL`` / zero-row / zero-column frames.

    Parameters
    ----------
    frame : Any
        The label var frame (or ``None``).

    Returns
    -------
    bool
    """
    if frame is None:
        return True
    if isinstance(frame, pd.DataFrame):
        return frame.shape[0] == 0 or frame.shape[1] == 0
    return len(frame) == 0


def _format_labels(frame: pd.DataFrame, labeller: Any) -> np.ndarray:
    """Apply the labeller per variable and column-stack into a string matrix.

    Port of R ``do.call(cbind, lapply(labels(data), cbind))`` where
    ``labels(data)`` returns a per-variable list of character vectors.  The
    ggplot2_py labellers collapse multiple variables into one flat list, so the
    labeller is applied **per column** to reproduce R's per-variable matrix
    (rows = panels, cols = variables/layers).

    Parameters
    ----------
    frame : pandas.DataFrame
        The label var frame.
    labeller : callable or str
        Labeller spec (resolved via ``as_labeller``).

    Returns
    -------
    numpy.ndarray
        2-D object array, rows = panels, cols = variables.
    """
    from ggplot2_py.labeller import as_labeller, label_value

    if labeller is None:
        labeller_fn = label_value
    elif callable(labeller):
        labeller_fn = labeller
    else:
        labeller_fn = as_labeller(labeller)

    cols: List[List[str]] = []
    for name in frame.columns:
        values = [str(v) for v in frame[name].tolist()]
        out = labeller_fn({str(name): values})
        # Labeller may return a dict (per-variable) or a flat list.
        if isinstance(out, dict):
            out = list(out.values())[0]
        cols.append([str(v) for v in out])

    nrow = frame.shape[0]
    ncol = len(cols)
    mat = np.empty((nrow, ncol), dtype=object)
    for j, col in enumerate(cols):
        for i in range(nrow):
            mat[i, j] = col[i]
    return mat


def _col_index(labels: np.ndarray) -> List[int]:
    """Port of R ``as.vector(col(labels))`` -- column number per cell, column-major.

    For a ``nrow x ncol`` matrix this is ``rep(1:ncol, each=nrow)``.

    Parameters
    ----------
    labels : numpy.ndarray
        2-D label matrix.

    Returns
    -------
    list of int
        1-based column index per flattened (column-major) cell.
    """
    nrow, ncol = labels.shape
    return [j + 1 for j in range(ncol) for _ in range(nrow)]


def _flatten_col_major(labels: np.ndarray) -> List[Any]:
    """Flatten a 2-D matrix column-major (R ``as.vector`` order).

    Parameters
    ----------
    labels : numpy.ndarray
        2-D matrix.

    Returns
    -------
    list
        Column-major flattened values.
    """
    return list(labels.reshape(-1, order="F"))


def _panel_layout(panels: Any) -> Dict[str, List[int]]:
    """Return panel-cell ``t``/``b``/``l``/``r`` lists from a gtable layout.

    Mirrors R ``panels$layout[grepl('^panel-', panels$layout$name), ]`` -- but
    keeps **all** matching rows (not de-duplicated) and 1-based indexable by the
    strip table's ``t``/``l``/``b``/``r`` panel ids.  ``gtable_py`` stores the
    layout as a dict of parallel lists.

    Parameters
    ----------
    panels : Gtable
        The assembled panel gtable.

    Returns
    -------
    dict
        ``{"t": [...], "b": [...], "l": [...], "r": [...]}`` with a leading
        ``None`` so the lists are 1-based addressable (``[panel_id]``).
    """
    import re

    lay = panels.layout
    if isinstance(lay, pd.DataFrame):
        names = list(lay["name"])
        t = list(lay["t"]); b = list(lay["b"]); l = list(lay["l"]); r = list(lay["r"])
    else:
        names = list(lay["name"])
        t = list(lay["t"]); b = list(lay["b"]); l = list(lay["l"]); r = list(lay["r"])

    rx = re.compile(r"^panel-")
    sel_t: List[Optional[int]] = [None]
    sel_b: List[Optional[int]] = [None]
    sel_l: List[Optional[int]] = [None]
    sel_r: List[Optional[int]] = [None]
    for i, nm in enumerate(names):
        if rx.match(str(nm)):
            sel_t.append(int(t[i]))
            sel_b.append(int(b[i]))
            sel_l.append(int(l[i]))
            sel_r.append(int(r[i]))
    return {"t": sel_t, "b": sel_b, "l": sel_l, "r": sel_r}


def _grobs_of(side: Any) -> Optional[List[Any]]:
    """Return the ``grobs`` list of a built strip side (or ``None``).

    Parameters
    ----------
    side : pandas.DataFrame or None
        A built strip placement frame.

    Returns
    -------
    list or None
    """
    if side is None:
        return None
    grobs = list(side["grobs"])
    return grobs


def _tlbr_of(side: Any) -> Optional[Dict[str, List[int]]]:
    """Return ``{t,l,b,r}`` panel-id lists, or ``None`` for a ``None`` side.

    The strip placement frame's ``t``/``l``/``b``/``r`` are panel ids used to
    index into the panel-layout lists.  Mirrors R ``NULL[c("t","b","l","r")]``
    -> ``NULL`` when the strip side is empty.

    Parameters
    ----------
    side : pandas.DataFrame or None
        A built strip placement frame.

    Returns
    -------
    dict or None
    """
    if side is None:
        return None
    return {
        "t": [int(v) for v in side["t"]],
        "l": [int(v) for v in side["l"]],
        "b": [int(v) for v in side["b"]],
        "r": [int(v) for v in side["r"]],
    }


def _flatten_strips(strips: Dict[str, Any]) -> Dict[str, Any]:
    """Port of R ``unlist(unname(self$strips), recursive=FALSE)``.

    Flattens ``{"x": {"top", "bottom"}, "y": {"left", "right"}}`` into a single
    dict keyed by the inner side names (``top``/``bottom``/``left``/``right``).

    Parameters
    ----------
    strips : dict
        The nested strip dict.

    Returns
    -------
    dict
        Keyed by side name.
    """
    flat: Dict[str, Any] = {}
    for outer in strips.values():
        for key, value in outer.items():
            flat[key] = value
    return flat


def _padding_from_sizes(size_vec: Any, strip_padding: Any) -> Any:
    """Port of R ``padding[as.numeric(padding) != 0] <- strip_padding``.

    Copies the per-position size unit vector and overwrites its non-zero entries
    with the scalar strip padding (so padding lands only where an axis sits).

    Parameters
    ----------
    size_vec : grid_py.Unit
        The ``sizes[[position]]`` unit vector.
    strip_padding : grid_py.Unit
        Scalar padding (in cm).

    Returns
    -------
    grid_py.Unit
        The reconstructed padding unit vector.
    """
    values = convert_unit(size_vec, "cm", valueOnly=True)
    values = np.atleast_1d(np.asarray(values, dtype="float64"))
    pad_cm = float(np.atleast_1d(convert_unit(strip_padding, "cm", valueOnly=True))[0])
    units: List[Any] = []
    for v in values:
        if v != 0:
            units.append(Unit(pad_cm, "cm"))
        else:
            units.append(Unit(float(v), "cm"))
    if len(units) == 0:
        return Unit([], "cm")
    if len(units) == 1:
        return units[0]
    return unit_c(*units)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------
def strip_vanilla(clip: str = "inherit", size: str = "constant") -> Strip:
    """Create a default (vanilla ggplot2 style) strip.

    Port of R ``strip_vanilla()`` (``strip_vanilla.R:41-51``).

    Parameters
    ----------
    clip : str, default ``"inherit"``
        Whether text labels are clipped to the background boxes; one of
        ``"inherit"``, ``"on"``, ``"off"``.
    size : str, default ``"constant"``
        Whether strip margins across layers remain ``"constant"`` or are
        ``"variable"``.

    Returns
    -------
    Strip
        A ``Strip`` ggproto instance usable in ggh4x facets.
    """
    params = {
        "clip": arg_match0(clip, ["on", "off", "inherit"], arg_name="clip"),
        "size": arg_match0(size, ["constant", "variable"], arg_name="size"),
    }
    # R: ggproto(NULL, Strip, params=...). R's ``Strip`` is itself a ggproto
    # *instance*, so this is the instance-as-parent path -> returns an instance
    # (a clone of the ``Strip`` singleton with ``params`` overridden).
    return ggproto(None, _STRIP_SINGLETON, params=params)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def resolve_strip(strip: Any, arg: str = "strip", env: Any = None) -> Strip:
    """Resolve a strip specification into a ``Strip`` instance.

    Port of R ``resolve_strip()`` (``strip_vanilla.R:454-470``).  A string
    ``"vanilla"`` maps to the ``strip_vanilla`` constructor (R ``find_global``);
    a callable is invoked; a ``Strip`` instance passes through.  Anything else
    raises.

    Parameters
    ----------
    strip : str or callable or Strip
        The strip spec.  Strings name a ``strip_<name>`` constructor in this
        module's namespace.
    arg : str, default ``"strip"``
        Argument name for the error message.
    env : Any, optional
        Unused (kept for R signature parity / ``find_global`` env).

    Returns
    -------
    Strip
        A resolved ``Strip`` instance.

    Raises
    ------
    ValueError
        When *strip* cannot be resolved to a ``Strip``.
    """
    if isinstance(strip, str):
        fn = globals().get("strip_" + strip)
        if fn is None:
            # Allow subclass constructors registered elsewhere (lazy).
            try:
                import ggh4x as _pkg  # noqa: F401

                fn = getattr(_pkg, "strip_" + strip, None)
            except Exception:
                fn = None
        strip = fn

    if callable(strip):
        strip = strip()

    if is_ggproto(strip) and isinstance(strip, Strip):
        return strip

    cli_abort(
        f"The `{arg}` argument must be a valid strip specification."
    )


# fallback for {deeptime}
assert_strip = resolve_strip


def validate_element_list(
    elem: Any,
    prototype: str = "element_text",
) -> Optional[List[Any]]:
    """Validate a user-supplied list of theme elements.

    Port of R ``validate_element_list()`` (``strip_themed.R:188-207``).  ``None``
    passes through; a non-list is wrapped in a list; every item must be a blank
    element, an element of the *prototype* class, or ``None`` -- else it aborts.

    Parameters
    ----------
    elem : Any
        ``None``, a single element, or a list of elements.
    prototype : str, default ``"element_text"``
        Expected element class name (``"element_text"`` / ``"element_rect"``);
        the leading ``"element_"`` is stripped for the type check.

    Returns
    -------
    list or None
        The validated list (or ``None``).

    Raises
    ------
    ValueError
        When any item is not blank / prototype-typed / ``None``.
    """
    if elem is None:
        return None
    if not isinstance(elem, list):
        elem = [elem]

    proto_type = prototype[len("element_"):] if prototype.startswith("element_") else prototype

    invalid = []
    for x in elem:
        ok = (
            is_theme_element(x, "blank")
            or is_theme_element(x, proto_type)
            or x is None
        )
        invalid.append(not ok)

    if any(invalid):
        cli_abort(
            f"The argument should be a list of `{prototype}` objects."
        )
    return elem


def inherit_element(child: Any, parent: Any) -> Any:
    """Resolve a child element's ``None`` properties from a parent element.

    Port of R ``inherit_element()`` (``strip_themed.R:211-248``), a
    ``combine_elements``-equivalent with ggh4x's exact early-return order.  On
    ggplot2 4.0.x the theme elements are S7 objects, so the property-copy branch
    (fill ``None`` props from *parent*) is taken; the ``rel`` size-multiplication
    branch lives in the **non-S7** path and -- matching the gold standard run --
    is *not* executed (verified: ``rel(2)`` against a parent size of 10 stays at
    2, not 20).  This port reproduces the S7 behaviour: ``None`` props are filled
    from *parent* and ``Rel`` sizes are left untouched.

    Parameters
    ----------
    child : Any
        The child element (or value).
    parent : Any
        The parent element to inherit from.

    Returns
    -------
    Any
        The resolved element.
    """
    # 1. parent NULL or child blank -> child verbatim.
    if parent is None or is_theme_element(child, "blank"):
        return child
    # 2. child NULL -> parent.
    if child is None:
        return parent
    # 3. neither is a theme element -> child.
    if not is_theme_element(child) and not is_theme_element(parent):
        return child
    # 4. parent blank -> obey child.inherit.blank.
    if is_theme_element(parent, "blank"):
        if getattr(child, "inherit_blank", False):
            return parent
        return child

    # 5. Fill child's None props from parent (S7-props-copy branch).
    import copy as _copy

    result = _copy.copy(child)
    for attr in list(parent.__dict__.keys()):
        if attr in result.__dict__ and getattr(result, attr) is None:
            setattr(result, attr, getattr(parent, attr))

    return result
