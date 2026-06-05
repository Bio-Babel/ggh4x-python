"""ggh4x-python — Python port of the R ggh4x package.

A ggplot2-extension toolkit ported onto ``ggplot2_py``: extended facets
(``facet_grid2`` / ``facet_wrap2`` / ``facet_nested`` / ``facet_manual``) with
pluggable strips, per-panel scales and forced panel sizes, plus extra stats,
geoms, multi-scales, positions/coords, and the ``guide_stringlegend`` legend.

Importing this package wires every subsystem's registration side-effects
(theme elements, the ``+``-add handlers for ``MultiScale`` / ``ForcedSize`` /
``FacettedPosScales`` / ``ScaleFacet``, and the ``element_grob`` dispatch for
``ElementPartRect``).
"""

from __future__ import annotations

__version__ = "0.3.1.9000"
__r_commit__ = "63c91b7"

# --- Stats -------------------------------------------------------------------
from .stat_theodensity import StatTheoDensity, stat_theodensity
from .stat_difference import StatDifference, stat_difference
from .stat_rle import StatRle, stat_rle
from .stat_rollingkernel import StatRollingkernel, stat_rollingkernel
from .stat_funxy import StatFunxy, stat_centroid, stat_funxy, stat_midpoint

# --- Geoms -------------------------------------------------------------------
from .geom_outline_point import GeomOutlinePoint, geom_outline_point
from .geom_box import GeomBox, geom_box
from .geom_pointpath import GeomPointPath, GeomPointpath, geom_pointpath
from .geom_text_aimed import GeomTextAimed, geom_text_aimed
from .geom_polygonraster import GeomPolygonRaster, geom_polygonraster
from .geom_rectrug import (
    GeomRectMargin,
    GeomTileMargin,
    geom_rectmargin,
    geom_tilemargin,
)

# --- Helpers / utils / elements ----------------------------------------------
from .conveniences import (
    center_limits,
    distribute_args,
    elem_list_rect,
    elem_list_text,
    sep_discrete,
    weave_factors,
)
from .save import save_plot
from .help_secondary import help_secondary
from .element_part_rect import ElementPartRect, element_part_rect
from .themes_ggh4x import ggh4x_theme_elements

# --- Strips ------------------------------------------------------------------
from .strip_vanilla import Strip, resolve_strip, strip_vanilla
from .strip_themed import StripThemed, strip_themed
from .strip_nested import StripNested, strip_nested
from .strip_split import StripSplit, strip_split
from .strip_tag import StripTag, strip_tag

# --- Facets ------------------------------------------------------------------
from .facet_grid2 import FacetGrid2, facet_grid2
from .facet_wrap2 import FacetWrap2, facet_wrap2
from .facet_nested import FacetNested, facet_nested
from .facet_nested_wrap import FacetNestedWrap, facet_nested_wrap
from .facet_manual import FacetManual, facet_manual

# --- Panel sizing / per-panel scales -----------------------------------------
from .panel_scales import (
    at_panel,
    facetted_pos_scales,
    force_panelsizes,
    scale_x_facet,
    scale_y_facet,
)

# --- Coord / Position --------------------------------------------------------
from .coord_axes_inside import CoordAxesInside, coord_axes_inside
from .position_lineartrans import PositionLinearTrans, position_lineartrans
from .position_disjoint_ranges import (
    PositionDisjointRanges,
    position_disjoint_ranges,
)

# --- Multi / listed / manual scales ------------------------------------------
from .multiscale import (
    scale_color_multi,
    scale_colour_multi,
    scale_fill_multi,
    scale_listed,
    scale_x_manual,
    scale_y_manual,
)

# --- Live guide --------------------------------------------------------------
from .guide_stringlegend import GuideStringlegend, guide_stringlegend

__all__ = [
    "__version__",
    # stats
    "stat_theodensity", "StatTheoDensity",
    "stat_difference", "StatDifference",
    "stat_rle", "StatRle",
    "stat_rollingkernel", "StatRollingkernel",
    "stat_funxy", "StatFunxy", "stat_centroid", "stat_midpoint",
    # geoms
    "geom_outline_point", "GeomOutlinePoint",
    "geom_box", "GeomBox",
    "geom_pointpath", "GeomPointPath", "GeomPointpath",
    "geom_text_aimed", "GeomTextAimed",
    "geom_polygonraster", "GeomPolygonRaster",
    "geom_rectmargin", "geom_tilemargin", "GeomRectMargin", "GeomTileMargin",
    # helpers / elements
    "distribute_args", "elem_list_text", "elem_list_rect", "weave_factors",
    "center_limits", "sep_discrete", "save_plot", "help_secondary",
    "element_part_rect", "ElementPartRect", "ggh4x_theme_elements",
    # strips
    "Strip", "strip_vanilla", "resolve_strip",
    "StripThemed", "strip_themed",
    "StripNested", "strip_nested",
    "StripSplit", "strip_split",
    "StripTag", "strip_tag",
    # facets
    "facet_grid2", "FacetGrid2",
    "facet_wrap2", "FacetWrap2",
    "facet_nested", "FacetNested",
    "facet_nested_wrap", "FacetNestedWrap",
    "facet_manual", "FacetManual",
    # panel sizing / per-panel scales
    "force_panelsizes", "facetted_pos_scales",
    "scale_x_facet", "scale_y_facet", "at_panel",
    # coord / position
    "coord_axes_inside", "CoordAxesInside",
    "position_lineartrans", "PositionLinearTrans",
    "position_disjoint_ranges", "PositionDisjointRanges",
    # multi / listed / manual scales
    "scale_colour_multi", "scale_color_multi", "scale_fill_multi",
    "scale_listed", "scale_x_manual", "scale_y_manual",
    # guide
    "guide_stringlegend", "GuideStringlegend",
]
