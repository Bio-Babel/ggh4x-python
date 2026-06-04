"""Multiple / listed / manual scales for ggh4x.

This package ports three R ggh4x files that implement two independent
capabilities sharing the :class:`MultiScale` deferred-mutation container plus a
standalone manual position scale:

* :mod:`scale_multi` (``scale_multi.R``) -- :func:`scale_colour_multi` /
  :func:`scale_fill_multi`: map several non-standard colour/fill aesthetics each
  to its own gradient :func:`ggplot2_py.continuous_scale`.
* :mod:`scale_listed` (``scale_listed.R``) -- :func:`scale_listed`: distribute a
  user-supplied list of discrete scales bound to non-standard aesthetics, grouped
  by the standard aesthetic each replaces.  Also home of the shared
  :class:`MultiScale` container and its ``ggplot_add`` handler.
* :mod:`scale_manual` (``scale_manual.R``) -- :func:`scale_x_manual` /
  :func:`scale_y_manual`: a hybrid discrete/continuous position scale
  (:class:`ScaleManualPosition`) that places discrete levels at arbitrary
  continuous coordinates.

Importing this package registers the ``MultiScale`` handler on
:func:`ggplot2_py.plot.update_ggplot` (via the :mod:`_multiscale_add` import),
exactly as ``ggnewscale.__init__`` imports ``_ggplot_add`` for its side effect.
"""

from __future__ import annotations

# Importing _multiscale_add registers @update_ggplot.register(MultiScale).
from ._multiscale_add import MultiScale
from .scale_listed import scale_listed
from .scale_manual import (
    ScaleManualPosition,
    scale_x_manual,
    scale_y_manual,
    sep_discrete,
)
from .scale_multi import (
    scale_color_multi,
    scale_colour_multi,
    scale_fill_multi,
)

__all__ = [
    "scale_fill_multi",
    "scale_colour_multi",
    "scale_color_multi",
    "scale_listed",
    "scale_x_manual",
    "scale_y_manual",
    "sep_discrete",
    "ScaleManualPosition",
    "MultiScale",
]
