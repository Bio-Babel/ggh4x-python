---
name: use-ggh4x
description: Reach for ggh4x when a ggplot2_py plot needs hierarchical/nested facet strips, independent per-panel scales, manual panel layouts, forced panel sizes, multiple colour scales on one plot, or one of its extra stats/geoms/positions/guide.
---

# ggh4x

`ggh4x` is a Python port of the R `ggh4x` package (teunbrand/ggh4x): a
ggplot2-extension toolkit built as a faithful extension of `ggplot2_py` — it
subclasses ggplot2's ggproto facets, stats, geoms, scales, coords, and
positions, not a generic matplotlib/plotnine wrapper. Importing `ggh4x` wires
several registration side effects (a new theme element for facet nest lines,
`+`-add handlers for `MultiScale`/`ForcedSize`/`FacettedPosScales`/
`ScaleFacet`, and an `element_grob` dispatch patch for `ElementPartRect`).

## When to use it

- Facet panels need hierarchical/merged strip labels (`facet_nested`,
  `facet_nested_wrap`) or a fully custom manual layout (`facet_manual`).
- A facet needs independent per-row/column scales, inner-axis drawing, or
  forced/relative panel sizes (`facet_grid2`, `facet_wrap2`,
  `force_panelsizes`, `facetted_pos_scales`, `scale_x_facet`/`scale_y_facet`).
- One plot needs more than one colour/fill scale (`scale_colour_multi`,
  `scale_fill_multi`, `scale_listed`).
- A stat/geom/position/guide beyond stock ggplot2 is needed: theoretical
  density fits (`stat_theodensity`), rolling-kernel trends
  (`stat_rollingkernel`), signed-difference ribbons (`stat_difference`),
  run-length encoding (`stat_rle`), per-group x/y summaries (`stat_funxy`,
  `stat_centroid`, `stat_midpoint`), aimed text (`geom_text_aimed`),
  shareable point outlines (`geom_outline_point`), transformable raster
  polygons (`geom_polygonraster`), margin rugs (`geom_rectmargin`,
  `geom_tilemargin`), a linear-transform position (`position_lineartrans`),
  a disjoint-ranges position (`position_disjoint_ranges`), axes drawn inside
  the panel (`coord_axes_inside`), or a compact text legend
  (`guide_stringlegend`).

## When not to use it

- The plotting backend is not `ggplot2_py` — ggh4x's extension points do not
  apply to matplotlib/plotnine code.
- The task only needs vanilla `facet_wrap`/`facet_grid` behavior with no
  strip, per-panel-scale, or panel-size customization.

## Entry points

`facet_grid2`/`facet_wrap2`/`facet_nested`/`facet_nested_wrap`/
`facet_manual` (facets), `strip_vanilla`/`strip_themed`/`strip_nested`/
`strip_split`/`strip_tag` (strips, passed as a facet's `strip=`),
`force_panelsizes`/`facetted_pos_scales`/`scale_x_facet`/`scale_y_facet`
(panel sizing and per-panel scales), and the `stat_*`/`geom_*` extras above.

## Minimal usage sketch

```python
import pandas as pd
from ggplot2_py import ggplot, aes, geom_point, vars
from ggh4x import facet_nested, force_panelsizes
from ggh4x._datasets import mpg

mpg = mpg.copy()
mpg["cyl_f"] = mpg["cyl"].astype("category")

(
    ggplot(mpg, aes("displ", "hwy", colour="cyl_f"))
    + geom_point()
    + facet_nested(cols=vars("drv", "cyl"))
    + force_panelsizes(cols=[2, 1, 1])
)
```

For more: `biobabel.describe_package(import_name="ggh4x")`.
