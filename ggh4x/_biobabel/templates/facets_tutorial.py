"""End-to-end template ported from tutorials/Facets.ipynb.

Builds a small synthetic dataset and exercises extended/nested facets,
per-strip theming, per-panel scales, and forced panel sizes. Uses only a
tiny in-memory DataFrame — no network access, no GUI, no large datasets.
"""

from __future__ import annotations

import pandas as pd
from ggplot2_py import ggplot, aes, geom_point, vars, theme, element_blank, element_line

from ggh4x import (
    facet_wrap2,
    facet_grid2,
    facet_nested,
    strip_themed,
    elem_list_rect,
    elem_list_text,
    scale_x_facet,
    force_panelsizes,
)


def build_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "displ": [1.6, 2.0, 2.4, 3.0, 3.5, 4.0, 1.8, 2.2],
            "hwy": [33, 29, 27, 22, 20, 18, 31, 26],
            "class": ["compact", "compact", "midsize", "suv", "suv", "suv", "compact", "midsize"],
            "drv": ["f", "f", "f", "4", "4", "4", "f", "f"],
            "year": [1999, 2008, 1999, 2008, 1999, 2008, 1999, 2008],
        }
    )


def main() -> None:
    mpg = build_data()
    p = ggplot(mpg, aes("displ", "hwy")) + geom_point()

    # Extended wrap facet with inner axes drawn but x labels removed.
    p + facet_wrap2(vars("class"), axes="all", remove_labels="x")

    # Extended grid facet with independent per-column x scales.
    p + facet_grid2(vars("year"), vars("drv"), scales="free_x", independent="x")

    # Nested facet with per-strip theming.
    p + facet_nested(
        cols=vars("drv", "class"),
        strip=strip_themed(
            background_x=elem_list_rect(fill=["limegreen", "dodgerblue"]),
            text_x=elem_list_text(colour=["dodgerblue", "limegreen"]),
            by_layer_x=True,
        ),
    ) + theme(strip_background=element_blank())

    # Per-panel scale override, requires a free axis.
    p + facet_grid2(cols=vars("drv"), scales="free_x") + scale_x_facet(
        lambda d: d["COL"] == 1, breaks=[2, 3, 4]
    )

    # Forced relative panel widths, added after the facet.
    p + facet_grid2(cols=vars("drv")) + force_panelsizes(cols=[1, 0.5], respect=True)


if __name__ == "__main__":
    main()
