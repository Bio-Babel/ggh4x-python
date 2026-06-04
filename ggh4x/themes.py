"""ggh4x theme-element registration (R source: themes.R).

ggh4x extends ggplot2's theme system with the ``ggh4x.facet.nestline`` element used by
``facet_nested`` / ``facet_nested_wrap``. (The ``ggh4x.axis.*`` nesting elements belong to
the deprecated axis guides and are out of scope.) R registers these in ``.onLoad``; the
Python port registers them at import time of the package.
"""

from __future__ import annotations

from ggplot2_py.theme_elements import el_def, element_blank, register_theme_elements

__all__ = ["register_ggh4x_theme_elements"]

_REGISTERED = False


def register_ggh4x_theme_elements() -> None:
    """Register ggh4x's theme elements globally (idempotent), mirroring ``ggh4x_theme_elements``.

    Registers ``ggh4x.facet.nestline`` as an ``ElementLine`` inheriting from ``line``, with a
    blank default — matching ``themes.R``.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    register_theme_elements(
        element_tree={"ggh4x.facet.nestline": el_def("ElementLine", "line")},
        **{"ggh4x.facet.nestline": element_blank()},
    )
    _REGISTERED = True
