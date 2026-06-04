"""Regression test: axes='all' + remove_labels must not crash.

Caught by Step-9 Facets validation: ``purge_guide_labels`` reset the axis
guide's reported size by assigning ``vp.width``/``vp.height`` directly, but
``grid_py.Viewport`` is immutable. The combination only fires when interior
axes are duplicated (``axes="all"``) *and* their labels are stripped
(``remove_labels=``), so plain unit tests of either feature alone missed it.
Fixed by rebuilding the viewport via ``grid_py.edit_viewport``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ggplot2_py import aes, geom_point, ggplot, ggplotGrob

from ggh4x import facet_grid2, facet_wrap2

_DF = pd.DataFrame(
    {"x": [1, 2, 3, 4, 5, 6], "y": [2, 3, 1, 4, 6, 5],
     "a": [1, 1, 2, 2, 1, 2], "b": [1, 2, 1, 2, 2, 1]}
)


@pytest.mark.parametrize("remove", ["x", "y", "all"])
def test_facet_grid2_axes_all_with_remove_labels_builds(remove):
    p = (ggplot(_DF, aes("x", "y")) + geom_point()
         + facet_grid2(rows="a", cols="b", axes="all", remove_labels=remove))
    g = ggplotGrob(p)  # must not raise
    assert sum(1 for n in g.layout["name"] if str(n).startswith("panel")) == 4


@pytest.mark.parametrize("remove", ["x", "y", "all"])
def test_facet_wrap2_axes_all_with_remove_labels_builds(remove):
    p = (ggplot(_DF, aes("x", "y")) + geom_point()
         + facet_wrap2(facets="a", axes="all", remove_labels=remove))
    g = ggplotGrob(p)  # must not raise
    assert sum(1 for n in g.layout["name"] if str(n).startswith("panel")) == 2
