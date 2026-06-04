"""Per-panel scale and panel-size customisation for ggh4x facets.

This package ports four ggh4x R files, all implemented as **add-on objects**
(not new facet/geom bases): each is created by a constructor and consumed by an
``ggplot_add`` handler that rewrites the plot's live facet (or a layer's geom)
at ``+``-time, producing a runtime ggproto clone.

* :mod:`force_panelsize` (``force_panelsize.R``) -- :func:`force_panelsizes`:
  force the panel row heights / column widths of any facet.
* :mod:`facetted_pos_scales` (``facetted_pos_scales.R``) --
  :func:`facetted_pos_scales`: per-panel position scales.
* :mod:`scale_facet` (``scale_facet.R``) -- :func:`scale_x_facet` /
  :func:`scale_y_facet`: a single per-panel position scale targeting panels by
  predicate.
* :mod:`at_panel` (``at_panel.R``) -- :func:`at_panel`: constrain a layer to a
  subset of panels.

Importing this package registers the ``ForcedSize`` / ``FacettedPosScales`` /
``ScaleFacet`` handlers on :func:`ggplot2_py.plot.update_ggplot` (via the
module imports), exactly as ``ggh4x.multiscale`` registers ``MultiScale``.
"""

from __future__ import annotations

from .at_panel import at_panel
from .facetted_pos_scales import (
    FacettedPosScales,
    check_facetted_scale,
    facetted_pos_scales,
    finish_data_individual,
    init_scale,
    init_scales_individual,
    should_transform,
    train_scales_individual,
    validate_facetted_scale,
)
from .force_panelsize import ForcedSize, force_panelsizes, is_null_unit
from .scale_facet import ScaleFacet, scale_facet, scale_x_facet, scale_y_facet

__all__ = [
    "force_panelsizes",
    "ForcedSize",
    "is_null_unit",
    "facetted_pos_scales",
    "FacettedPosScales",
    "check_facetted_scale",
    "validate_facetted_scale",
    "init_scale",
    "init_scales_individual",
    "train_scales_individual",
    "finish_data_individual",
    "should_transform",
    "scale_facet",
    "scale_x_facet",
    "scale_y_facet",
    "ScaleFacet",
    "at_panel",
]
