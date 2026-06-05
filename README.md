# ggh4x-python

Python version of R **ggh4x** package (version 0.3.1.9000+63c91b7)

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

