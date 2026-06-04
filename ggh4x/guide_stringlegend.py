"""String legend guide (R source: ``guide_stringlegend.R``).

Ports ggh4x's :func:`guide_stringlegend` constructor and the
:class:`GuideStringlegend` ggproto, which renders colour/fill (and optionally
``family``/``fontface``) mappings as **coloured text strings** rather than as the
geom key swatches drawn by :func:`ggplot2_py.guide_legend`.

This is live, non-deprecated ggh4x code.  :class:`GuideStringlegend` extends
:class:`ggplot2_py.guide.GuideLegend`, inheriting the full legend draw
orchestration (:meth:`ggplot2_py.guide.Guide.draw`) and overriding exactly the
five points where a string legend differs from a key legend:

* :meth:`GuideStringlegend.get_layer_key` -- identity passthrough (no geom keys);
* :meth:`GuideStringlegend.setup_params` -- parent params then zero key cell sizes;
* :meth:`GuideStringlegend.setup_elements` -- pull ``legend.text`` margin onto the
  resolved text element and zero the key width/height (the load-bearing
  "text only, no swatch" mechanism);
* :meth:`GuideStringlegend.build_labels` -- a coloured text grob per key row;
* :meth:`GuideStringlegend.build_decor` -- a single empty grob (no swatches).

Parent dispatch follows the fixed strategy: ``GuideLegend.setup_params`` is a
``@staticmethod(params)`` while ``GuideLegend.setup_elements`` is an instance
method; each is invoked through :func:`ggplot2_py.ggproto_parent` with its correct
positional arity.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ggplot2_py import ggproto_parent
from ggplot2_py._compat import waiver
from ggplot2_py.guide import GuideLegend, new_guide
from ggplot2_py.theme_elements import (
    ElementText,
    calc_element,
    element_grob,
)
from grid_py import Unit, null_grob

__all__ = [
    "guide_stringlegend",
    "GuideStringlegend",
]


def _column(key: Any, name: str) -> Any:
    """Return key column *name* (a sequence) or ``None`` when absent.

    Mirrors R's ``key$<name>`` which yields ``NULL`` for a missing column.  *key*
    is a :class:`pandas.DataFrame`; ``df.get(name)`` returns ``None`` when the
    column is not present.

    Parameters
    ----------
    key : pandas.DataFrame
    name : str

    Returns
    -------
    pandas.Series or None
    """
    if key is None:
        return None
    getter = getattr(key, "get", None)
    if callable(getter):
        return getter(name)
    return None


def _at(seq: Any, i: int) -> Any:
    """Return the ``i``-th element of *seq* (``Series`` or list), or ``None``."""
    if seq is None:
        return None
    iloc = getattr(seq, "iloc", None)
    if iloc is not None:
        return iloc[i]
    return seq[i]


class GuideStringlegend(GuideLegend):
    """Legend that renders colour/fill mappings as coloured text, not key swatches.

    Subclass of :class:`ggplot2_py.guide.GuideLegend` ported from R
    ``GuideStringlegend`` (``guide_stringlegend.R:52-98``).

    Notes
    -----
    The ``available_aes`` (``["colour", "fill", "family", "fontface"]``) and
    ``name`` (``"stringlegend"``) are injected by :func:`guide_stringlegend` at
    instance-build time via :func:`ggplot2_py.guide.new_guide`.
    """

    _class_name = "GuideStringlegend"

    def get_layer_key(
        self,
        params: Dict[str, Any],
        layers: Optional[List[Any]] = None,
        data: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Return *params* unchanged (R ``get_layer_key``: identity passthrough).

        Port of ``guide_stringlegend.R:55-57``.  A string legend has no geom keys,
        so it bypasses the base machinery that resolves per-key ``draw_key`` decor.
        The arity ``(self, params, layers, data=None)`` is kept so the inherited
        ``process_layers`` still dispatches correctly.

        Parameters
        ----------
        params : dict
            The guide parameters.
        layers, data : optional
            Accepted and ignored (R uses ``...``).

        Returns
        -------
        dict
            *params* unchanged.
        """
        return params

    @staticmethod
    def setup_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Compute legend params, then zero the key cell sizes.

        Port of ``guide_stringlegend.R:59-63``.  Delegates to the parent
        ``GuideLegend.setup_params`` (a ``@staticmethod`` computing
        ``nrow``/``ncol``/``n_breaks`` and validating ``direction``), then sets
        ``params['sizes'] = {'widths': 0, 'heights': 0}`` so the key cells take no
        space.  The load-bearing zeroing is also enforced via the key unit
        elements in :meth:`setup_elements`.

        Parameters
        ----------
        params : dict

        Returns
        -------
        dict
        """
        params = GuideLegend.setup_params(params)
        params = dict(params)
        params["sizes"] = {"widths": 0, "heights": 0}
        return params

    def setup_elements(
        self,
        params: Dict[str, Any],
        elements: Optional[Dict[str, Any]] = None,
        theme: Any = None,
    ) -> Dict[str, Any]:
        """Resolve elements, injecting the text margin and zeroing the key size.

        Port of ``guide_stringlegend.R:65-73``.  Merges ``params['theme']`` into
        *theme* (then clears it to avoid a double-add), delegates to the parent
        ``GuideLegend.setup_elements``, pulls the ``legend.text`` margin onto the
        resolved text element (so :meth:`build_labels`' titleGrob margins control
        inter-key spacing), sets ``spacing_y`` from ``legend.key.spacing.y`` and
        zeroes the key width/height (so only the text shows).

        Parameters
        ----------
        params : dict
        elements : dict, optional
            Defaults to a copy of ``self.elements``.
        theme : Theme, optional

        Returns
        -------
        dict
        """
        if elements is None:
            elements = dict(self.elements)
        if theme is not None:
            # Theme.__add__ tolerates a None operand (params['theme'] may be None).
            theme = theme + params.get("theme")
        params = dict(params)
        params["theme"] = None

        elements = ggproto_parent(GuideLegend, self).setup_elements(
            params, elements, theme
        )

        elements["spacing_y"] = calc_element("legend.key.spacing.y", theme)

        # Pull the legend.text margin onto the resolved text element.  Build a
        # fresh ElementText (copying every field) rather than mutating .margin in
        # place, since element objects may be shared/cached.
        text_el = elements.get("text")
        text_margin = getattr(calc_element("legend.text", theme), "margin", None)
        if text_el is not None:
            elements["text"] = ElementText(
                family=getattr(text_el, "family", None),
                face=getattr(text_el, "face", None),
                colour=getattr(text_el, "colour", None),
                size=getattr(text_el, "size", None),
                hjust=getattr(text_el, "hjust", None),
                vjust=getattr(text_el, "vjust", None),
                angle=getattr(text_el, "angle", None),
                lineheight=getattr(text_el, "lineheight", None),
                margin=text_margin,
            )

        elements["key_height"] = Unit(0, "cm")
        elements["key_width"] = Unit(0, "cm")
        return elements

    @staticmethod
    def build_labels(
        key: Any, elements: Dict[str, Any], params: Dict[str, Any]
    ) -> List[Any]:
        """Build one coloured text grob per key row.

        Port of ``guide_stringlegend.R:75-95`` -- the core override.  When there
        are no labels, returns one :func:`grid_py.null_grob` per key row.
        Otherwise the per-row colour is ``key$colour`` falling back (whole-column)
        to ``key$fill``, and each label is drawn via :func:`element_grob` on the
        resolved text element with ``margin_x``/``margin_y`` so the
        ``legend.text`` margin (set in :meth:`setup_elements`) drives layout.
        ``family``/``fontface`` columns are passed per-row when present, else
        ``None`` so the text element's defaults apply.

        Parameters
        ----------
        key : pandas.DataFrame
            The guide key (``.label``, ``colour``/``fill``, optionally
            ``family``/``fontface``).
        elements : dict
            Resolved guide elements (``text`` carries the injected margin).
        params : dict

        Returns
        -------
        list of grob
            One grob per key row, coloured by the colour/fill aesthetic.
        """
        n_key = len(key) if key is not None else 0
        labels = _column(key, ".label")
        n_labels = 0 if labels is None else len(labels)
        if n_labels < 1:
            return [null_grob() for _ in range(n_key)]

        # colour <- key$colour %||% key$fill  (whole-column coalesce).
        colour = _column(key, "colour")
        if colour is None:
            colour = _column(key, "fill")
        family = _column(key, "family")
        fontface = _column(key, "fontface")
        text_el = elements.get("text")

        out: List[Any] = []
        for i in range(n_labels):
            out.append(
                element_grob(
                    text_el,
                    label=str(_at(labels, i)),
                    colour=_at(colour, i),
                    family=_at(family, i),
                    face=_at(fontface, i),
                    margin_x=True,
                    margin_y=True,
                )
            )
        return out

    @staticmethod
    def build_decor(
        decor: Any = None,
        grobs: Any = None,
        elements: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Return empty grobs (R ``build_decor``: no key swatches).

        Port of ``guide_stringlegend.R:97`` -- ``function(...) zeroGrob()``.  The
        colour is shown in the label text instead of in a swatch, so the decor is
        suppressed.

        R returns a *single* ``zeroGrob`` because its downstream legend assembly
        treats a single empty grob as "no decor".  The ggplot2_py procedural
        assembly (``_guide_legend.measure_legend_grobs``) instead iterates ``decor``
        by index (``decor[i]._width``), so a scalar grob raises ``len(decor)``;
        the slice-test confirmed this.  We therefore return one
        :func:`grid_py.null_grob` per key row to stay shape-compatible.  Each grob
        carries no ``_width``/``_height``, so combined with the zeroed
        ``key_width``/``key_height`` units (see :meth:`setup_elements`) the key
        cells collapse to zero -- the same visual result as R.

        Returns
        -------
        list of grid_py.Grob
            One :func:`grid_py.null_grob` per key row.
        """
        n = 0
        if params is not None:
            key = params.get("key")
            n = params.get("n_breaks", len(key) if key is not None else 0)
        return [null_grob() for _ in range(int(n))]


def guide_stringlegend(
    title: Any = waiver(),
    theme: Any = None,
    position: Optional[str] = None,
    direction: Optional[str] = None,
    nrow: Optional[int] = None,
    ncol: Optional[int] = None,
    reverse: bool = False,
    order: int = 0,
) -> GuideStringlegend:
    """Construct a string legend guide showing colour/fill mappings as text.

    Port of R ``guide_stringlegend`` (``guide_stringlegend.R:22-44``).  Builds a
    :class:`GuideStringlegend` via :func:`ggplot2_py.guide.new_guide` with
    ``available_aes = ["colour", "fill", "family", "fontface"]`` and
    ``name = "stringlegend"``.  It can be supplied to
    :func:`ggplot2_py.guides` or as a scale's ``guide`` argument.

    Parameters
    ----------
    title : str or Waiver, optional
        Legend title.  Defaults to the scale's name.
    theme : Theme, optional
        Guide-local theme overrides.
    position : str, optional
        Legend position.
    direction : str, optional
        ``"horizontal"`` or ``"vertical"``.
    nrow, ncol : int, optional
        Legend grid dimensions.
    reverse : bool, default ``False``
        Reverse the order of the legend keys.
    order : int, default ``0``
        Ordering relative to other guides.

    Returns
    -------
    GuideStringlegend
    """
    return new_guide(
        title=title,
        theme=theme,
        direction=direction,
        nrow=nrow,
        ncol=ncol,
        reverse=reverse,
        order=order,
        position=position,
        available_aes=["colour", "fill", "family", "fontface"],
        name="stringlegend",
        super=GuideStringlegend,
    )
