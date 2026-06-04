# ggh4x-python

Python port of the R **ggh4x** package (version 0.3.1.9000+63c91b7), built on
`ggplot2_py` 4.0.2. Ports the extended facets, strips, per-panel scales, stats,
geoms, multi-scales, coords/positions and `guide_stringlegend` — every in-scope
export validated against the R original (see `docs/` and the port reports).

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from ggplot2_py import ggplot, aes, geom_point, vars
from ggh4x import facet_nested, force_panelsizes
from ggh4x._datasets import mpg

mpg = mpg.copy(); mpg["cyl_f"] = mpg["cyl"].astype("category")
(ggplot(mpg, aes("displ", "hwy", colour="cyl_f")) + geom_point()
 + facet_nested(cols=vars("drv", "cyl")) + force_panelsizes(cols=[2, 1, 1]))
```

See the **Tutorials** (`tutorials/*.ipynb`) for the full Facets / Statistics /
Miscellaneous walkthroughs.

## Documentation

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Not ported — `deprecated.R` (deferred)

The 12 exports in ggh4x's `deprecated.R` are deprecation shims in R 0.3.1 (they
`lifecycle::deprecate_warn()` then fall back to a plain guide/scale). Their real
logic has moved upstream, so they are intentionally **not** reimplemented; use the
successor APIs instead:

| ggh4x (deprecated) | Use instead |
|---|---|
| `guide_axis_truncated`, `guide_axis_manual`, `guide_axis_minor`, `guide_axis_colour`/`_color` | ggplot2 4.0 native `guide_axis()` args (`cap=`, `minor.ticks=`, `theme=`) |
| `guide_axis_logticks` | ggplot2 `guide_axis_logticks()` |
| `guide_axis_nested`, `guide_axis_scalebar`, `guide_dendro` | the R successor package **legendry** (`guide_axis_nested`, `primitive_bracket`, `*_dendro`) — no Python equivalent yet |
| `scale_x_dendrogram`, `scale_y_dendrogram` | **legendry** dendrogram scales |
| `ggsubset` | pass a filtered `data=` / a lambda to a layer's `data` argument |

`guide_stringlegend` is **not** deprecated and **is** ported.

