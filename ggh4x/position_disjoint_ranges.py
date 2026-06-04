"""Segregate overlapping ranges (port of ggh4x ``position_disjoint_ranges.R``).

This module ports the R ggh4x ``position_disjoint_ranges()`` constructor and the
``PositionDisjointRanges`` ggproto class.  One-dimensional ranged data in the
x-direction (``xmin``/``xmax``) is segregated in the y-direction so that no two
ranges overlap in two-dimensional space.  Overlapping ranges are assigned to
different y-direction *bins* (lower bins filled first), preserving x-information.

R source: ``ggh4x/R/position_disjoint_ranges.R``.

Notes
-----
* The disjoint-bin assignment is a sweep-line algorithm inspired by
  ``IRanges::disjointBins()`` but generalised to any numeric (not just integer)
  ranges.  This port reimplements it in pure Python (no Bioconductor dependency).
* **Boundary semantics (load-bearing):** R uses a *strict* ``<`` test
  (``track_bins < dat$xmin``).  A range is placed into an existing bin only when
  that bin's running ``xmax`` is *strictly less* than the new ``xmin``; touching
  endpoints (``xmax == xmin`` after extension) are treated as overlapping and
  forced into a new bin.  Free bins are filled lowest-index-first; bins are
  1-based.
* The returned ``data`` preserves the original row order; only the internal
  ``ranges`` table is sorted.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ggplot2_py.position import Position

from ggh4x._cli import cli_warn

__all__ = [
    "position_disjoint_ranges",
    "PositionDisjointRanges",
]


class PositionDisjointRanges(Position):
    """Segregate overlapping x-ranges into disjoint y-direction bins.

    Subclass of :class:`ggplot2_py.position.Position` ported from R
    ``PositionDisjointRanges`` (``position_disjoint_ranges.R:58-129``).

    Attributes
    ----------
    extend : float or None
        How far (in total) a range is extended when computing overlaps.  A
        positive value leaves space between ranges sharing a bin.
    stepsize : float or None
        Vertical space added between bins.  Positive grows bins bottom-to-top,
        negative grows them top-to-bottom.
    required_aes : tuple of str
        ``("xmin", "xmax", "ymin", "ymax")``.
    """

    extend: Optional[float] = None
    stepsize: Optional[float] = None
    required_aes = ("xmin", "xmax", "ymin", "ymax")

    def __init__(self, **kwargs: Any) -> None:
        """Store constructor members directly on the instance.

        Parameters
        ----------
        **kwargs : Any
            Members (``extend``, ``stepsize``) assigned verbatim, mirroring the
            ``ggproto(NULL, PositionDisjointRanges, ...)`` clone.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def setup_params(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate the x-direction ranges and collect parameters.

        Port of R ``PositionDisjointRanges$setup_params``
        (``position_disjoint_ranges.R:64-73``).  Emits a :func:`cli_warn` when
        ``xmin``/``xmax`` are absent, then returns ``extend``/``stepsize``.

        Parameters
        ----------
        data : pandas.DataFrame
            Layer data.

        Returns
        -------
        dict
            ``{"extend": ..., "stepsize": ...}``.
        """
        if "xmin" not in data.columns or "xmax" not in data.columns:
            cli_warn(
                "Undefined ranges in the x-direction.\n"
                "i Please supply xmin and xmax."
            )
        return {"extend": self.extend, "stepsize": self.stepsize}

    def compute_panel(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        scales: Any = None,
    ) -> pd.DataFrame:
        """Assign disjoint y-bins to overlapping x-ranges within a panel.

        Port of R ``PositionDisjointRanges$compute_panel``
        (``position_disjoint_ranges.R:74-128``).

        Parameters
        ----------
        data : pandas.DataFrame
            Panel data with ``xmin``/``xmax``/``group`` (and optionally
            ``ymin``/``ymax``).
        params : dict
            Carries ``extend`` and ``stepsize``.
        scales : Any, optional
            Panel scales (unused; signature parity with the base class).

        Returns
        -------
        pandas.DataFrame
            *data* with ``ymin``/``ymax`` shifted by ``stepsize * (bin - 1)`` per
            assigned bin, in the original row order.
        """
        group = data["group"].to_numpy()

        # --- Simplify groups to ranges (R L76-92) -----------------------------
        if len(np.unique(group)) > 1:
            # One range per group: (min(xmin), max(xmax)) collapsed per group.
            # ``sort=False`` keeps groups in order of first appearance, matching
            # R ``by()`` which iterates over ``sort(unique(group))`` -> but the
            # subsequent sort by xmin and ``match`` make the iteration order
            # irrelevant; we only require the per-group aggregate to be correct.
            agg = (
                data.groupby("group", sort=False)
                .agg(xmin=("xmin", "min"), xmax=("xmax", "max"))
                .reset_index()
            )
            ranges = agg[["xmin", "xmax", "group"]].copy()
        elif np.all(group == -1):
            # One range per row, synthetic 1-based group ids (R ``row(data)[,1]``).
            ranges = data[["xmin", "xmax"]].copy()
            ranges["group"] = np.arange(1, len(data) + 1)
            group = ranges["group"].to_numpy()
        else:
            # Single non-(-1) group -> nothing to disjoin; return unchanged.
            return data

        # --- Extend & sort ranges (R L95-98) ----------------------------------
        extend = params["extend"]
        stepsize = params["stepsize"]
        ranges["xmin"] = ranges["xmin"].to_numpy(dtype="float64") - 0.5 * extend
        ranges["xmax"] = ranges["xmax"].to_numpy(dtype="float64") + 0.5 * extend
        order = np.argsort(ranges["xmin"].to_numpy(), kind="stable")
        ranges = ranges.iloc[order].reset_index(drop=True)

        # --- Sweep-line disjoint bins (R L102-118) ----------------------------
        # Strict ``<`` boundary: a range reuses a bin only if its running xmax is
        # strictly less than the new xmin; touching ranges overlap -> new bin.
        xmin = ranges["xmin"].to_numpy(dtype="float64")
        xmax = ranges["xmax"].to_numpy(dtype="float64")
        track_bins = [xmax[0]]
        bins = [1]
        for i in range(1, len(ranges)):
            free = [k for k, end in enumerate(track_bins) if end < xmin[i]]
            if free:
                ans = free[0]
                track_bins[ans] = xmax[i]
                bins.append(ans + 1)  # 1-based
            else:
                track_bins.append(xmax[i])
                bins.append(len(track_bins))
        ranges["bin"] = bins

        # --- Map back & shift (R L120-127) ------------------------------------
        # R ``match(group, ranges$group)`` -> positional index into the sorted
        # ``ranges`` (first match, 1-based); ``ranges$bin[map]`` then indexes by
        # position.  pandas Index.get_indexer gives the 0-based positional index.
        map_idx = pd.Index(ranges["group"].to_numpy()).get_indexer(group)
        bin_vals = ranges["bin"].to_numpy()[map_idx]

        if "ymin" in data.columns and "ymax" in data.columns:
            data = data.copy()
            shift = stepsize * (bin_vals - 1)
            data["ymax"] = data["ymax"].to_numpy(dtype="float64") + shift
            data["ymin"] = data["ymin"].to_numpy(dtype="float64") + shift

        return data


def position_disjoint_ranges(
    extend: float = 1,
    stepsize: float = 1,
) -> PositionDisjointRanges:
    """Create a disjoint-ranges position adjustment.

    Port of R ``position_disjoint_ranges()``
    (``position_disjoint_ranges.R:48-50``).  Segregates one-dimensional ranged
    data (``xmin``/``xmax``) in the y-direction so that no two ranges overlap.
    This positioning is most useful when y-coordinates carry no relevant
    information; it pairs well with ``geom_rect`` / ``geom_tile``.

    Parameters
    ----------
    extend : float, optional
        Total amount by which a range is extended when computing overlaps.  A
        positive value leaves space between ranges in the same bin.  Default
        ``1``.
    stepsize : float, optional
        Vertical space added between bins.  Positive grows bins bottom-to-top,
        negative grows them top-to-bottom.  Default ``1``.

    Returns
    -------
    PositionDisjointRanges
        A position object that can be passed to a layer's ``position`` argument.

    Examples
    --------
    >>> position_disjoint_ranges(extend=0.1)  # doctest: +ELLIPSIS
    <ggh4x.position_disjoint_ranges.PositionDisjointRanges object at ...>
    """
    return PositionDisjointRanges(
        extend=extend,
        stepsize=stepsize,
    )
