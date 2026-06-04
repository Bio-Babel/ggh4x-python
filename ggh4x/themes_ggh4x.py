"""ggh4x theme-element registration (port of ggh4x ``themes.R``).

ggh4x extends ggplot2's theme system with a handful of extra theme elements.
The one registered here is ``ggh4x.facet.nestline`` -- an
:class:`~ggplot2_py.theme_elements.ElementLine` inheriting from the ``line``
element, used as the parent for the ``nest_line`` argument of
``facet_nested()`` / ``facet_nested_wrap()``.  It defaults to
``element_blank()`` (nest lines are off unless a theme/facet turns them on).

R source: ``ggh4x/R/themes.R`` (``ggh4x_theme_elements``).  In R the
registration happens in ``.onLoad``; this Python port mirrors that by invoking
:func:`ggh4x_theme_elements` at *module import time* (bottom of this file).

Notes
-----
* The element is consumed by the **facet** subsystem (nest lines), not by the
  strips themselves, but registration lives here to match the R file layout.
* Registration is idempotent: :func:`register_theme_elements` merges into the
  global element tree, so re-importing the module is harmless.
"""

from __future__ import annotations

from ggplot2_py.theme_elements import (
    ElementLine,
    el_def,
    element_blank,
    register_theme_elements,
)

__all__ = ["ggh4x_theme_elements"]


def ggh4x_theme_elements() -> None:
    """Register ggh4x's extended theme elements globally.

    Port of R ``ggh4x_theme_elements()`` (``themes.R:30-37``)::

        register_theme_elements(
          ggh4x.facet.nestline = element_blank(),
          element_tree = list(
            ggh4x.facet.nestline = el_def("element_line", "line")
          )
        )

    Registers ``ggh4x.facet.nestline`` in the element tree as an
    :class:`~ggplot2_py.theme_elements.ElementLine` that inherits from the
    ``line`` element, with an ``element_blank()`` default.

    Returns
    -------
    None
        Mutates the global ggplot2 element tree / default theme in place.

    Notes
    -----
    Idempotent -- calling it repeatedly merges the same entry and re-sets the
    same blank default.
    """
    register_theme_elements(
        element_tree={"ggh4x.facet.nestline": el_def(ElementLine, "line")},
        **{"ggh4x.facet.nestline": element_blank()},
    )


# R registers in .onLoad; mirror that by registering at import time.
ggh4x_theme_elements()
