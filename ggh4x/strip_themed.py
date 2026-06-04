"""Themed strips for ggh4x facets (port of ggh4x ``strip_themed.R``).

This module ports :class:`StripThemed` -- the R ggproto object whose *class
string* is ``"StripElemental"`` (the object is named ``StripThemed``) -- and the
:func:`strip_themed` constructor.

``StripThemed`` extends the base :class:`ggh4x.strip_vanilla.Strip`.  It allows
per-strip theming: the user supplies *lists* of
:class:`~ggplot2_py.theme_elements.ElementText` /
:class:`~ggplot2_py.theme_elements.ElementRect` objects (``text_x`` / ``text_y``
/ ``background_x`` / ``background_y``) that are inherited onto the theme defaults
(via :func:`ggh4x.strip_vanilla.inherit_element`) and mapped either across strips
(``rep_len`` recycling) or across the layers of a strip (``by_layer_*``).

R source: ``ggh4x/R/strip_themed.R`` (the ``StripThemed`` /
``"StripElemental"`` ggproto and the ``strip_themed()`` constructor).  The
helpers ``validate_element_list`` / ``inherit_element`` live in
:mod:`ggh4x.strip_vanilla`.

Notes
-----
* **Only override is ``setup_elements``.**  Everything else (``setup`` /
  ``get_strips`` / ``assemble_strip`` / ``build_strip`` / the self-less
  ``init_strip`` / ``draw_labels`` / ``finish_strip`` / ``incorporate_*``) is
  inherited unchanged from :class:`Strip`.  The base
  :func:`ggh4x.strip_vanilla._init_strip_impl` already reads
  ``elements["by_layer"][aes]`` (defaulting to ``False`` when absent), so the
  ``by_layer`` dict produced here flows straight through.
* **``by_layer`` dict.**  ``setup_elements`` adds ``by_layer = {"x": ..., "y":
  ...}`` to the returned element bundle (the base ``Strip.setup_elements`` does
  not).
* **Per-side independent inherit.**  R re-runs ``inherit_element`` for
  ``text$x$top`` *and* ``text$x$bottom`` against the *same* user list, so the
  top and bottom calc-element defaults are inherited independently; this port
  does the same.
"""

from __future__ import annotations

from typing import Any, Dict

from ggplot2_py import calc_element, element_grob
from ggplot2_py.ggproto import ggproto

from ggh4x._rlang import arg_match0
from ggh4x.strip_vanilla import (
    Strip,
    _placement_inside,
    inherit_element,
    validate_element_list,
)
from grid_py import convert_unit

__all__ = ["StripThemed", "strip_themed"]


class StripThemed(Strip):
    """Strip with individually themed boxes and texts (R ``"StripElemental"``).

    Subclass of :class:`ggh4x.strip_vanilla.Strip`.  Overrides only
    :meth:`setup_elements`; the rest of the strip machinery is inherited.

    Attributes
    ----------
    given_elements : dict
        The user-supplied element lists and ``by_layer`` flags, set by
        :func:`strip_themed` (keys ``text_x`` / ``text_y`` / ``background_x`` /
        ``background_y`` / ``by_layer_x`` / ``by_layer_y``).

    Notes
    -----
    The R class string is ``"StripElemental"`` even though the object is named
    ``StripThemed``.  ``_class_name`` is set to ``"StripElemental"`` for repr
    parity (the :func:`strip_themed` clone reuses the Python class name, but the
    base singleton reports ``StripElemental``).
    """

    _class_name = "StripElemental"

    given_elements: Dict[str, Any] = {}

    def setup_elements(self, theme: Any, type: str) -> Dict[str, Any]:
        """Resolve per-strip themed elements, with a ``by_layer`` dict.

        Port of R ``StripThemed$setup_elements`` (``strip_themed.R:119-182``).
        Like :meth:`Strip.setup_elements` but resolves the backgrounds and
        per-side texts from the user-supplied element lists in
        ``self.given_elements``, each inherited onto the ``calc_element`` theme
        default via :func:`inherit_element`; backgrounds are then turned into
        grobs via :func:`element_grob`.  Adds a ``by_layer`` dict.

        Parameters
        ----------
        theme : Theme
            The active theme.
        type : str
            ``"wrap"`` selects ``strip.switch.pad.wrap`` padding, anything else
            (``"grid"``) ``strip.switch.pad.grid``.  Kept as the parameter name
            ``type`` for facet-call compatibility.

        Returns
        -------
        dict
            ``{"padding", "background", "text", "inside", "by_layer"}``.  The
            ``background`` values are *grobs* (single grob when the user gave no
            list, else a list of grobs); ``text`` values are elements (single
            element from ``calc_element`` or a list of inherited elements).
        """
        given = self.given_elements

        # --- backgrounds: calc_element default, inherit user list, to grob ----
        bg_x_default = calc_element("strip.background.x", theme)
        bg_y_default = calc_element("strip.background.y", theme)
        background: Dict[str, Any] = {}
        if given.get("background_x") is not None:
            inherited = [inherit_element(el, bg_x_default) for el in given["background_x"]]
            background["x"] = [element_grob(el) for el in inherited]
        else:
            background["x"] = element_grob(bg_x_default)
        if given.get("background_y") is not None:
            inherited = [inherit_element(el, bg_y_default) for el in given["background_y"]]
            background["y"] = [element_grob(el) for el in inherited]
        else:
            background["y"] = element_grob(bg_y_default)

        # --- texts: per-side calc_element default, optionally inherit list ----
        text: Dict[str, Dict[str, Any]] = {
            "x": {
                "top": calc_element("strip.text.x.top", theme),
                "bottom": calc_element("strip.text.x.bottom", theme),
            },
            "y": {
                "left": calc_element("strip.text.y.left", theme),
                "right": calc_element("strip.text.y.right", theme),
            },
        }
        if given.get("text_x") is not None:
            text["x"]["top"] = [
                inherit_element(el, text["x"]["top"]) for el in given["text_x"]
            ]
            text["x"]["bottom"] = [
                inherit_element(el, text["x"]["bottom"]) for el in given["text_x"]
            ]
        if given.get("text_y") is not None:
            text["y"]["left"] = [
                inherit_element(el, text["y"]["left"]) for el in given["text_y"]
            ]
            text["y"]["right"] = [
                inherit_element(el, text["y"]["right"]) for el in given["text_y"]
            ]

        inside = {
            "x": _placement_inside(calc_element("strip.placement.x", theme)),
            "y": _placement_inside(calc_element("strip.placement.y", theme)),
        }
        pad_name = (
            "strip.switch.pad.wrap" if type == "wrap" else "strip.switch.pad.grid"
        )
        padding = convert_unit(calc_element(pad_name, theme), "cm")

        by_layer = {
            "x": bool(given.get("by_layer_x", False)),
            "y": bool(given.get("by_layer_y", False)),
        }

        return {
            "padding": padding,
            "background": background,
            "text": text,
            "inside": inside,
            "by_layer": by_layer,
        }


# R's ``StripThemed`` is a ggproto *instance* used as the parent of every
# ``strip_themed()`` clone; this module-level singleton plays that role.
_STRIP_THEMED_SINGLETON: "StripThemed" = StripThemed()


def strip_themed(
    clip: str = "inherit",
    size: str = "constant",
    text_x: Any = None,
    text_y: Any = None,
    background_x: Any = None,
    background_y: Any = None,
    by_layer_x: bool = False,
    by_layer_y: bool = False,
) -> StripThemed:
    """Create a strip with individually themed boxes and texts.

    Port of R ``strip_themed()`` (``strip_themed.R:76-106``).

    Parameters
    ----------
    clip : str, default ``"inherit"``
        Whether labels are clipped to background boxes (``"inherit"`` / ``"on"``
        / ``"off"``).
    size : str, default ``"constant"``
        Whether strip margins across layers are ``"constant"`` or ``"variable"``.
    text_x, text_y : list of ElementText or ElementText or None
        Per-strip (or per-layer, see *by_layer_*) text elements.  ``None`` means
        the global theme applies; ``element_blank()`` omits the text.
    background_x, background_y : list of ElementRect or ElementRect or None
        Per-strip (or per-layer) background rectangles.
    by_layer_x, by_layer_y : bool, default ``False``
        When ``True`` map elements to the different *layers* of a strip; when
        ``False`` map to individual strips with ``rep_len`` recycling.

    Returns
    -------
    StripThemed
        A ``StripThemed`` ggproto instance usable in ggh4x facets.
    """
    params = {
        "clip": arg_match0(clip, ["on", "off", "inherit"], arg_name="clip"),
        "size": arg_match0(size, ["constant", "variable"], arg_name="size"),
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
        _STRIP_THEMED_SINGLETON,
        params=params,
        given_elements=given_elements,
    )
