# ggh4x-python

A Python port of the R **[ggh4x](https://teunbrand.github.io/ggh4x/)** package, built as a
faithful extension of [`ggplot2_py`](https://github.com/Bio-Babel) — *not* a matplotlib or
plotnine wrapper. It subclasses ggplot2's ggproto facets, stats, geoms, scales, coords and
positions so the extension points line up one-to-one with the R original.

Ported from ggh4x **0.3.1.9000** against ggplot2_py **4.0.2** (gtable_py 0.3.6, scales 1.4.0).

## What it adds

| Area | Functions |
|---|---|
| **Extended facets** | `facet_grid2`, `facet_wrap2` — inner axes (`axes=`), label removal (`remove_labels=`), independent scales |
| **Nested facets** | `facet_nested`, `facet_nested_wrap` — outer strips spanning inner strips, with nesting lines |
| **Manual facets** | `facet_manual` — `layout()`-style designs where a panel can span multiple cells |
| **Strips** | `strip_vanilla`, `strip_themed`, `strip_nested`, `strip_split`, `strip_tag` — per-strip theming, merging, in-panel tags |
| **Panel control** | `force_panelsizes`, `facetted_pos_scales`, `scale_x_facet`, `scale_y_facet`, `at_panel` |
| **Stats** | `stat_theodensity`, `stat_difference`, `stat_rle`, `stat_rollingkernel`, `stat_funxy`, `stat_centroid`, `stat_midpoint` |
| **Geoms** | `geom_pointpath`, `geom_text_aimed`, `geom_polygonraster`, `geom_outline_point`, `geom_box`, `geom_rectmargin`, `geom_tilemargin` |
| **Scales** | `scale_colour_multi`, `scale_fill_multi`, `scale_listed`, `scale_x_manual`, `scale_y_manual` |
| **Coords / positions** | `coord_axes_inside`, `position_lineartrans`, `position_disjoint_ranges` |
| **Guides** | `guide_stringlegend` |

## Installation

```bash
pip install -e .
```

Requires `ggplot2_py`, `gtable_py`, `grid_py`, `scales`, plus `numpy`, `pandas`, `scipy`.

## Quick start

```python
import pandas as pd
from ggplot2_py import ggplot, aes, geom_point, vars
from ggh4x import facet_nested, force_panelsizes
from ggh4x._datasets import mpg

mpg = mpg.copy(); mpg["cyl_f"] = mpg["cyl"].astype("category")

(ggplot(mpg, aes("displ", "hwy", colour="cyl_f")) + geom_point()
 + facet_nested(cols=vars("drv", "cyl"))
 + force_panelsizes(cols=[2, 1, 1]))
```

## Tutorials

Worked notebooks reproducing the original ggh4x vignettes:

- **[Facets](tutorials/Facets.ipynb)** — extended/nested/manual facets, strips, per-panel scales, panel sizes.
- **[Statistics](tutorials/Statistics.ipynb)** — theoretical densities, rolling kernels, differences, summary functions, RLE.
- **[Miscellaneous](tutorials/Miscellaneous.ipynb)** — multi-scales, extra geoms, aimed text, coord/position helpers.

## Validation

Every in-scope export is validated against the R original by **both-side internal-computation
comparison** (the R source is the gold standard). Facets are checked by exact **gtable
layout-table parity**; numeric subsystems by Pearson/exact agreement against R `layer_data`.
See the [API Reference](api.md) for per-function documentation.
