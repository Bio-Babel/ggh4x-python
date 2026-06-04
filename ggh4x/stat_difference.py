"""Difference ribbon stat (R source: ``ggh4x/R/stat_difference.R``).

Port of ggh4x's :func:`stat_difference` and the :class:`StatDifference`
ggproto object onto the Bio-Babel ``ggplot2_py`` stack.

``stat_difference()`` builds a ribbon whose ``fill`` aesthetic encodes the sign
of the difference ``ymax - ymin`` (or ``xmax - xmin`` when the orientation is
flipped). The stat re-orders the ``group`` aesthetic so that the positive and
negative segments of the difference receive distinct fills, and it interpolates
the exact crossover positions so that the ribbon does not look "stumpy" where
the two series cross.

R reference behaviour was captured live via ``StatDifference$compute_group`` /
``StatDifference$compute_panel`` / ``StatDifference$setup_params`` and the
parity tests in ``tests/test_stat_difference.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ggplot2_py import ggproto_parent
from ggplot2_py.aes import AfterStat, aes
from ggplot2_py.stat import Stat, _flip_data, _has_flipped_aes

from ._vctrs import data_frame0, vec_rep_each, vec_unrep

__all__ = ["StatDifference", "stat_difference"]


# ---------------------------------------------------------------------------
# ggproto
# ---------------------------------------------------------------------------
class StatDifference(Stat):
    """Stat computing a signed difference ribbon between two series.

    Mirrors the R ``StatDifference`` ggproto object. For each group the stat
    sorts the data by the main (continuous) axis, computes the sign of
    ``ymax - ymin``, run-length encodes that sign, and interpolates exact
    crossover positions where the sign changes. A per-run ``id`` marker is
    emitted so that :meth:`compute_panel` can renumber the ``group`` aesthetic
    (one group per monotone-sign run), giving the positive and negative
    segments distinct fills.

    Attributes
    ----------
    required_aes : list of str
        ``['x|y', 'ymin|xmin', 'ymax|xmax']`` -- the main axis can be either
        ``x`` or ``y`` (orientation aware), with a matching ``min``/``max`` pair.
    default_aes : Mapping
        ``aes(fill=after_stat(sign))``.
    extra_params : list of str
        ``['na_rm', 'orientation', 'levels']``.
    """

    required_aes = ["x|y", "ymin|xmin", "ymax|xmax"]
    default_aes = aes(fill=AfterStat("sign"))
    extra_params = ["na_rm", "orientation", "levels"]

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve the flip orientation of the stat.

        Mirrors R::

            params$flipped_aes <- has_flipped_aes(
                data, params, main_is_orthogonal = FALSE, main_is_continuous = TRUE
            )

        Parameters
        ----------
        data : pandas.DataFrame
            Layer data.
        params : dict
            Layer parameters.

        Returns
        -------
        dict
            *params* with ``flipped_aes`` set.
        """
        params["flipped_aes"] = _has_flipped_aes(
            data,
            params,
            main_is_orthogonal=False,
            main_is_continuous=True,
        )
        return params

    def compute_panel(
        self,
        data: pd.DataFrame,
        scales: Any,
        flipped_aes: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute the panel, renumbering groups by sign-run.

        Flips the data for the requested orientation, delegates to the base
        :meth:`Stat.compute_panel` for the split-by-group dispatch, then sets
        ``group = cumsum(id)`` so that each monotone-sign run becomes its own
        group, drops the ``id`` marker, records ``flipped_aes`` and flips back.

        Mirrors R::

            data <- flip_data(data, flipped_aes)
            data <- ggproto_parent(Stat, self)$compute_panel(data, scales, ...)
            data$group <- cumsum(data$id)
            data$id <- NULL
            data$flipped_aes <- flipped_aes
            flip_data(data, flipped_aes)

        Parameters
        ----------
        data : pandas.DataFrame
            Panel data (one PANEL).
        scales : Any
            Panel scales (``dict``-like or ``None``); unused here, forwarded.
        flipped_aes : bool, optional
            Whether the orientation is flipped (``x``/``y`` swapped).
        **kwargs
            Forwarded to :meth:`compute_group` (e.g. ``levels``, ``na_rm``).
            ``orientation`` (injected by ``ggplot2_py``'s ``compute_layer``
            via ``extra_params``) is consumed here and not forwarded, matching
            R where ``compute_layer`` only forwards ``self$parameters()``.

        Returns
        -------
        pandas.DataFrame
            Panel data with renumbered ``group`` and a ``flipped_aes`` column.
        """
        # ``orientation`` rides along via ggplot2_py's parameters(extra=True);
        # R's compute_group has no such formal, so strip it before delegating.
        kwargs.pop("orientation", None)

        data = _flip_data(data, flipped_aes)
        data = ggproto_parent(Stat, self).compute_panel(data, scales, **kwargs)
        if data is None or data.empty:
            return data if data is not None else pd.DataFrame()
        data = data.copy()
        data["group"] = np.cumsum(np.asarray(data["id"], dtype=float)).astype(int)
        data = data.drop(columns=["id"])
        data["flipped_aes"] = flipped_aes
        return _flip_data(data, flipped_aes)

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        levels: Sequence[str] = ("+", "-"),
        na_rm: bool = False,
        flipped_aes: bool = False,
    ) -> pd.DataFrame:
        """Compute the signed difference ribbon for a single group.

        Sorts by ``x``, run-length encodes ``sign(ymax - ymin)``, interpolates
        the crossover ``x`` wherever the sign changes, and assembles a frame
        with interpolated ``ymin``/``ymax`` (via :func:`numpy.interp`), a
        per-run ``id`` marker and a ``sign`` factor labelled with *levels*.

        Mirrors R ``StatDifference$compute_group`` exactly, including:

        * crossover formula
          ``-y[d]*(x[d+1]-x[d])/(y[d+1]-y[d]) + x[d]`` over run-ends ``d``
          (excluding the last run),
        * trimming the first and last interpolated ``sign`` entries,
        * dropping rows where ``sign == 0`` *after* assembly, and
        * factor coercion ``factor(sign, levels=c("1","-1"), labels=levels)``.

        Parameters
        ----------
        data : pandas.DataFrame
            One group's data; must contain ``x``, ``ymin``, ``ymax``.
        scales : Any
            Panel scales; unused.
        levels : sequence of str, optional
            Two labels for the ``fill`` factor: ``levels[0]`` when
            ``ymax > ymin`` and ``levels[1]`` when ``ymax < ymin``.
        na_rm : bool, optional
            Unused (missing handling occurs upstream); kept for parity.
        flipped_aes : bool, optional
            Unused here (the flip is applied in :meth:`compute_panel`).

        Returns
        -------
        pandas.DataFrame
            Columns ``x``, ``ymin``, ``ymax``, ``id``, ``sign``.
        """
        # Sort by x (stable, mirroring R's order()).
        order = np.argsort(data["x"].to_numpy(), kind="stable")
        data = data.iloc[order].reset_index(drop=True)

        x = data["x"].to_numpy(dtype=float)
        ymin = data["ymin"].to_numpy(dtype=float)
        ymax = data["ymax"].to_numpy(dtype=float)
        n = len(data)

        y = ymax - ymin
        sign = np.sign(y)

        # Run-length encode the sign: run values (key) and run lengths (times).
        sign_rle = vec_unrep(sign)
        rle_key = np.asarray(sign_rle["key"].to_numpy(), dtype=float)
        rle_times = sign_rle["times"].to_numpy()

        # Crossing points at run boundaries (all run-ends except the last).
        ends = np.cumsum(rle_times)
        dups = ends[:-1]  # 1-based run-end indices in R; convert below.
        if dups.size:
            # R is 1-based: x[dups] / x[dups + 1]; here dups-1 / dups (0-based).
            d0 = dups.astype(int) - 1
            d1 = dups.astype(int)
            cross = (
                -y[d0] * (x[d1] - x[d0]) / (y[d1] - y[d0]) + x[d0]
            )
        else:
            cross = np.empty(0, dtype=float)

        # Interpolate ymin/ymax at the doubled crossover positions.
        x_cross = vec_rep_each(cross, 2)
        if x_cross.size:
            ymin_cross = np.interp(x_cross, x, ymin)
            ymax_cross = np.interp(x_cross, x, ymax)
        else:
            ymin_cross = np.empty(0, dtype=float)
            ymax_cross = np.empty(0, dtype=float)

        # Match metadata: doubled run keys, trimmed at both ends.
        sign_meta = vec_rep_each(rle_key, 2)
        if sign_meta.size >= 2:
            sign_meta = sign_meta[1:-1]
        else:
            sign_meta = np.empty(0, dtype=float)

        # Per-crossover id marker (0,1 pairs) and order key.
        n_cross = cross.size
        id_cross = np.tile(np.array([0, 1]), n_cross) if n_cross else np.empty(0, dtype=int)
        ord_cross = (np.cumsum(id_cross) + 1) if n_cross else np.empty(0, dtype=int)

        # Order key for the *original* data rows: each run index repeated by
        # its run length (1-based to match ord_cross domain).
        data_ord = vec_rep_each(np.arange(1, len(rle_times) + 1), rle_times)

        # id for original rows: 1 for the first row, 0 elsewhere.
        id_data = np.zeros(n, dtype=int)
        if n:
            id_data[0] = 1

        new = data_frame0(
            x=np.concatenate([x, x_cross]),
            ymin=np.concatenate([ymin, ymin_cross]),
            ymax=np.concatenate([ymax, ymax_cross]),
            ord=np.concatenate([data_ord.astype(float), ord_cross.astype(float)]),
            id=np.concatenate([id_data, id_cross]).astype(int),
            sign=np.concatenate([sign, sign_meta]),
        )

        # Order by (ord, x) -- stable, mirroring R's order(new$ord, new$x).
        sort_idx = np.lexsort((new["x"].to_numpy(), new["ord"].to_numpy()))
        new = new.iloc[sort_idx].reset_index(drop=True)

        # Drop zero-difference rows.
        new = new[new["sign"].to_numpy() != 0].reset_index(drop=True)

        # Factor: numeric sign -> char level "1"/"-1" -> labels levels[0:2].
        new["sign"] = _sign_factor(new["sign"].to_numpy(), levels)
        new = new.drop(columns=["ord"])
        return new


def _sign_factor(sign: np.ndarray, levels: Sequence[str]) -> pd.Categorical:
    """Coerce numeric signs into a labelled, ordered categorical.

    Mirrors R ``factor(sign, levels = c("1", "-1"), labels = levels[1:2])``:
    ``+1 -> levels[0]``, ``-1 -> levels[1]``, anything else -> ``NaN``. The
    resulting categories are ``[levels[0], levels[1]]`` in that order.

    Parameters
    ----------
    sign : numpy.ndarray
        Numeric signs (typically ``+1`` / ``-1`` after zeros are dropped).
    levels : sequence of str
        Two-element label sequence.

    Returns
    -------
    pandas.Categorical
        Categorical with categories ``[levels[0], levels[1]]``.
    """
    cats = [levels[0], levels[1]]
    out = np.full(len(sign), None, dtype=object)
    out[sign == 1] = levels[0]
    out[sign == -1] = levels[1]
    return pd.Categorical(out, categories=cats, ordered=False)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------
def stat_difference(
    mapping: Optional[Any] = None,
    data: Any = None,
    geom: str = "ribbon",
    position: str = "identity",
    *,
    levels: Tuple[str, str] = ("+", "-"),
    na_rm: bool = False,
    orientation: Any = None,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Difference ribbon.

    Makes a ribbon that is filled depending on whether ``ymax`` is higher than
    ``ymin``. Useful for displaying differences between two series. The stat may
    reorder the ``group`` aesthetic to accommodate two different fills for the
    signs of the difference, and interpolates the series at crossovers so the
    ribbon does not look stumpy.

    Parameters
    ----------
    mapping : Mapping, optional
        Aesthetic mapping created with ``aes()``.
    data : DataFrame or callable, optional
        Layer data.
    geom : str, optional
        Geom used to render the stat output. Defaults to ``"ribbon"``.
    position : str, optional
        Position adjustment. Defaults to ``"identity"``.
    levels : tuple of str, optional
        A ``character(2)`` giving factor levels for the ``fill`` aesthetic for
        the cases where (1) ``ymax > ymin`` and (2) ``ymax < ymin``. Defaults to
        ``("+", "-")``.
    na_rm : bool, optional
        If ``False``, missing values are removed with a warning. Defaults to
        ``False``.
    orientation : {None, "x", "y"}, optional
        The axis the stat should run along. The default (``None``) infers the
        orientation from the aesthetics.
    show_legend : bool, optional
        Whether this layer is included in the legend.
    inherit_aes : bool, optional
        Whether to inherit the plot-level aesthetic mapping.
    **kwargs
        Other arguments passed to the layer (e.g. ``alpha``).

    Returns
    -------
    Layer
        A layer object that can be added to a plot.

    Notes
    -----
    When there is a run of more than two zero-difference values, the inner
    values are ignored (matching R).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from ggh4x.stat_difference import stat_difference
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     "x": np.arange(1, 101),
    ...     "y": np.cumsum(rng.standard_normal(100)),
    ...     "z": np.cumsum(rng.standard_normal(100)),
    ... })
    >>> layer = stat_difference(aes(x="x", ymin="y", ymax="z"), data=df, alpha=0.3)
    """
    from ggplot2_py.layer import layer as _layer

    return _layer(
        data=data,
        mapping=mapping,
        stat=StatDifference,
        geom=geom,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "na_rm": na_rm,
            "orientation": orientation,
            "levels": levels,
            **kwargs,
        },
    )
