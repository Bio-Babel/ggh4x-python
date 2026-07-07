"""Smoke test for ggh4x._biobabel.

Builds a tiny synthetic DataFrame and exercises one real ggh4x path: a
nested facet with a custom strip, plus a forced relative panel size added
after the facet. No network access, no GUI, no large datasets.
"""

from __future__ import annotations

import pandas as pd


def main() -> None:
    from ggplot2_py import ggplot, aes, geom_point, vars
    from ggh4x import facet_nested, strip_nested, force_panelsizes

    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [10, 20, 15, 25],
            "drv": ["f", "f", "4", "4"],
            "cyl": [4, 6, 4, 6],
        }
    )

    plot = (
        ggplot(df, aes("x", "y"))
        + geom_point()
        + facet_nested(cols=vars("drv", "cyl"), strip=strip_nested(bleed=False))
        + force_panelsizes(cols=[1, 1], respect=True)
    )

    assert plot is not None
    print("built a ggh4x facet_nested + force_panelsizes plot successfully")


if __name__ == "__main__":
    main()
