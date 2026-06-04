"""Polygon parameterisation for rasters (port of ggh4x ``geom_polygonraster.R``).

``geom_polygonraster()`` takes equally-sized raster pixels and re-parametrises
each as a four-vertex polygon.  This is less efficient than a true raster but
lets the pixels be transformed by non-linear ``coord``-functions (e.g.
:func:`ggplot2_py.coord_polar`) or ``position``-functions such as
:func:`ggh4x.position_lineartrans`.

R source: ``ggh4x/R/geom_polygonraster.R``.

Notes
-----
* :meth:`GeomPolygonRaster.setup_data` completely overrides
  :class:`ggplot2_py.GeomRaster`'s setup (which uses a diff-of-unique pixel
  size); it uses :func:`ggplot2_py._utils.resolution` (``zero=True``) and a
  *specific* four-vertex corner winding (``geom_polygonraster.R:92-99``).
* :meth:`GeomPolygonRaster.draw_panel` overwrites ``group`` with the pixel
  ``id`` before ``coord_munch`` so each pixel is its own polygon, and forces an
  invisible border (``col=0, lwd=0, lty=0``).
* The default position is :func:`ggh4x.position_lineartrans` (lazily imported
  to avoid an import cycle).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ggplot2_py.geom import (
    GeomRaster,
    Gpar,
    Mapping,
    _coord_transform,
    _ggname,
    polygon_grob,
    null_grob,
    scales_alpha,
)
from ggplot2_py.coord import coord_munch
from ggplot2_py._utils import resolution

__all__ = [
    "geom_polygonraster",
    "GeomPolygonRaster",
]


class GeomPolygonRaster(GeomRaster):
    """Raster geom that draws each pixel as a four-vertex polygon.

    Subclass of :class:`ggplot2_py.GeomRaster` ported from R
    ``GeomPolygonRaster`` (``geom_polygonraster.R:82-130``).
    """

    def setup_data(self, data: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Reparametrise each pixel into four corner vertices.

        Port of R ``GeomPolygonRaster$setup_data``
        (``geom_polygonraster.R:84-107``).  Each pixel row is replicated four
        times and assigned the four corner coordinates with the exact R corner
        winding, then sorted by pixel ``id``.

        Parameters
        ----------
        data : pandas.DataFrame
            Layer data with ``x``/``y`` pixel centres.
        params : dict
            Layer parameters (``hjust``/``vjust``).

        Returns
        -------
        pandas.DataFrame
            Data with four vertex rows per pixel and an ``id`` column.
        """
        data = data.copy().reset_index(drop=True)
        x = data["x"].to_numpy(dtype="float64")
        y = data["y"].to_numpy(dtype="float64")
        w = resolution(x)
        h = resolution(y)
        hjust = params.get("hjust", 0.5)
        if hjust is None:
            hjust = 0.5
        vjust = params.get("vjust", 0.5)
        if vjust is None:
            vjust = 0.5

        n = len(data)
        data["id"] = np.arange(1, n + 1)

        # R corner matrix (ncol = 2), built from these concatenations:
        #   xs = [x-w*(1-hjust) (×2), x+w*hjust (×2)]
        #   ys = [y-h*(1-vjust),      y+h*vjust (×2), y-h*(1-vjust)]
        x_lo = x - w * (1 - hjust)
        x_hi = x + w * hjust
        y_lo = y - h * (1 - vjust)
        y_hi = y + h * vjust
        xs = np.concatenate([x_lo, x_lo, x_hi, x_hi])
        ys = np.concatenate([y_lo, y_hi, y_hi, y_lo])

        # rbind(data, data, data, data): copy block k holds vertex k.
        rep = pd.concat([data] * 4, ignore_index=True)
        rep["x"] = xs
        rep["y"] = ys
        # order(id) groups the four vertices of each pixel together.
        rep = rep.sort_values("id", kind="stable").reset_index(drop=True)
        return rep

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        hjust: float = 0.5,
        vjust: float = 0.5,
        **params: Any,
    ) -> Any:
        """Draw one polygon per pixel.

        Port of R ``GeomPolygonRaster$draw_panel``
        (``geom_polygonraster.R:108-129``).

        Parameters
        ----------
        data : pandas.DataFrame
            Layer data (four vertex rows per pixel).
        panel_params : Any
            Panel scales / ranges.
        coord : Any
            Active coordinate system.
        hjust, vjust : float, default ``0.5``
            Pixel anchor justifications (accepted for parameter parity).
        **params : Any
            Ignored extra parameters.

        Returns
        -------
        grid_py.Grob
            A single :func:`grid_py.polygon_grob` (or
            :func:`grid_py.null_grob` when there is a single row).
        """
        if len(data) == 1:
            return null_grob()

        data = data.copy()
        data["group"] = data["id"]
        coords = coord_munch(coord, data, panel_params)
        coords = coords.reset_index(drop=True)

        # first <- coords[!duplicated(data$id), ]
        id_arr = coords["id"].to_numpy()
        _, first_idx = np.unique(id_arr, return_index=True)
        first = coords.iloc[np.sort(first_idx)]

        fill = (
            scales_alpha(
                first["fill"].to_numpy() if "fill" in first.columns else "grey35",
                first["alpha"].to_numpy() if "alpha" in first.columns else None,
            )
        )

        return _ggname(
            "geom_polygon",
            polygon_grob(
                x=coords["x"].to_numpy(dtype="float64"),
                y=coords["y"].to_numpy(dtype="float64"),
                default_units="native",
                id=coords["id"].to_numpy(),
                gp=Gpar(col=0, fill=fill, lwd=0, lty=0),
            ),
        )


def geom_polygonraster(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: Any = None,
    hjust: float = 0.5,
    vjust: float = 0.5,
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a polygon-raster layer.

    Port of R ``geom_polygonraster()`` (``geom_polygonraster.R:51-74``).  Each
    raster pixel becomes a four-vertex polygon, enabling non-linear coord and
    position transformations.

    Parameters
    ----------
    mapping : Mapping, optional
        Aesthetic mapping created by :func:`ggplot2_py.aes`.
    data : Any, optional
        Layer data.
    stat : str, default ``"identity"``
        Statistical transformation.
    position : Position, optional
        Position adjustment.  Defaults to :func:`ggh4x.position_lineartrans`
        (R's default).
    hjust, vjust : float, default ``0.5``
        Pixel anchor justifications.  Must be a length-1 numeric.
    na_rm : bool, default ``False``
        If ``True``, silently remove missing values.
    show_legend : bool or None, default ``None``
        Whether to show a legend for this layer.
    inherit_aes : bool, default ``True``
        Whether to inherit the plot's default aesthetics.
    **kwargs : Any
        Additional aesthetic parameters passed to the layer.

    Returns
    -------
    ggplot2_py.Layer
        A layer object that can be added to a plot.

    Raises
    ------
    TypeError
        If ``hjust``/``vjust`` are not length-1 numerics.
    """
    from ggplot2_py.layer import layer

    if not isinstance(hjust, (int, float)):
        raise TypeError("`hjust` must be a length-1 numeric.")
    if not isinstance(vjust, (int, float)):
        raise TypeError("`vjust` must be a length-1 numeric.")

    if position is None:
        # Lazy import to avoid a cross-subsystem import cycle.
        from ggh4x.position_lineartrans import position_lineartrans

        position = position_lineartrans()

    return layer(
        data=data,
        mapping=mapping,
        stat=stat,
        geom=GeomPolygonRaster,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "hjust": hjust,
            "vjust": vjust,
            "na_rm": na_rm,
            **kwargs,
        },
    )
