"""Run length encoding stat (R source: ggh4x/R/stat_rle.R).

Port of ggh4x's :func:`stat_rle` / ``StatRle``. Run length encoding takes a
vector of values (the ``label`` aesthetic) and, after ordering the data on the
``x`` aesthetic, computes the lengths of consecutive repeated values, turning
each run into a rectangle spanning the run's ``x`` extent.

In contrast to :func:`base::rle`, ``NA`` values are considered equivalent
(consecutive ``NA`` form a single run), mirroring ``vctrs::vec_unrep`` as used
by the R implementation.

The computed columns are ``start``, ``end``, ``start_id``, ``end_id``,
``run_id``, ``runlength`` and ``runvalue``; the geom defaults map these onto a
``geom_rect`` via :func:`after_stat`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ggplot2_py import ggproto_parent  # noqa: F401  (kept for parity / future use)
from ggplot2_py.aes import AfterStat
from ggplot2_py.layer import layer
from ggplot2_py.stat import Stat

from ._rlang import arg_match0

__all__ = ["stat_rle", "StatRle"]


# -- helpers ----------------------------------------------------------------


def _vec_unrep(x: "np.ndarray | pd.Series | List[Any]") -> pd.DataFrame:
    """Run-length encode consecutive equal values, ``NA``-as-equal.

    Faithful reimplementation of ``vctrs::vec_unrep`` for the semantics ggh4x
    relies on: consecutive equal values (including consecutive ``NA`` / ``None``)
    collapse into a single run. This differs from the more general
    :func:`ggh4x._vctrs.vec_unrep`, which compares with ``!=`` and therefore
    splits runs of ``NaN`` (because ``NaN != NaN``).

    Parameters
    ----------
    x : numpy.ndarray, pandas.Series, or list
        Input vector. May contain ``NA``/``NaN``/``None``. A pandas
        ``Categorical``/factor dtype is preserved in the returned ``key``.

    Returns
    -------
    pandas.DataFrame
        Two columns: ``key`` (the run value in first-appearance order, dtype
        preserved from the input) and ``times`` (``int`` run length).

    Raises
    ------
    ValueError
        If ``x`` is ``None`` (mirrors R's ``vec_unrep(NULL)`` scalar-type error).
    """
    if x is None:
        raise ValueError("`x` must be a vector, not `None`.")

    s = x if isinstance(x, pd.Series) else pd.Series(list(x))
    n = len(s)
    if n == 0:
        return pd.DataFrame(
            {
                "key": pd.Series([], dtype=s.dtype),
                "times": pd.Series([], dtype=int),
            }
        )

    # Boundaries: a new run starts at position 0 and wherever the current value
    # differs from the previous one, treating NA as equal to a preceding NA.
    cur = s.iloc[1:].reset_index(drop=True)
    prev = s.iloc[:-1].reset_index(drop=True)
    cur_na = cur.isna().to_numpy()
    prev_na = prev.isna().to_numpy()
    # equal if both NA, or (neither NA and values match)
    eq = (cur_na & prev_na) | ((~cur_na) & (~prev_na) & (cur.to_numpy() == prev.to_numpy()))

    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = ~eq
    idx = np.flatnonzero(change)
    times = np.diff(np.append(idx, n)).astype(int)
    key = s.iloc[idx].reset_index(drop=True)
    return pd.DataFrame({"key": key, "times": times})


# -- ggproto ----------------------------------------------------------------


class StatRle(Stat):
    """Run length encoding stat turning runs of ``label`` into rectangles.

    Notes
    -----
    Port of ``StatRle`` (``ggh4x/R/stat_rle.R`` L113-162). The data is ordered
    on ``x`` and the ``label`` aesthetic is run-length encoded (``NA``-as-equal).
    Four ``align`` modes (``none``/``centre``/``start``/``end``) control how the
    ``start``/``end`` ``x`` positions of each run are computed.
    """

    required_aes: List[str] = ["x", "label"]
    default_aes: Dict[str, Any] = {
        "xmin": AfterStat("start"),
        "xmax": AfterStat("end"),
        "ymin": AfterStat(lambda d: np.full(len(d), -np.inf)),
        "ymax": AfterStat(lambda d: np.full(len(d), np.inf)),
        "fill": AfterStat("runvalue"),
    }
    dropped_aes: List[str] = ["x", "label"]
    extra_params: List[str] = ["na_rm", "orientation", "align"]

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set ``flipped_aes`` from the requested orientation.

        Parameters
        ----------
        data : pandas.DataFrame
            Layer data (unused; present for signature parity).
        params : dict
            Stat parameters. Read ``orientation``.

        Returns
        -------
        dict
            ``params`` with ``flipped_aes`` set to ``orientation == 'y'``.
        """
        params["flipped_aes"] = params.get("orientation") == "y"
        return params

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any = None,
        flipped_aes: bool = False,
        align: str = "none",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Run-length encode ``label`` and compute per-run ``x`` extents.

        Parameters
        ----------
        data : pandas.DataFrame
            Group data, must contain ``x`` and ``label`` columns.
        scales : Any, optional
            Panel scales (unused; present for signature parity).
        flipped_aes : bool, optional
            Orientation flag (unused by the computation; kept for parity).
        align : {'none', 'centre', 'start', 'end'}, optional
            How to derive ``start``/``end`` positions:

            * ``none`` -- exact ``x`` at run boundaries.
            * ``centre`` -- midpoint between a run boundary and its neighbour.
            * ``start`` -- align run starts to the previous run's end.
            * ``end`` -- align run ends to the next run's start.
        **kwargs : Any
            Ignored extra parameters.

        Returns
        -------
        pandas.DataFrame
            Columns ``start``, ``end``, ``start_id``, ``end_id``, ``run_id``,
            ``runlength`` and ``runvalue``.
        """
        # Order on x (stable, mirroring R's order()).
        order = np.argsort(data["x"].to_numpy(), kind="stable")
        data = data.iloc[order].reset_index(drop=True)
        n = len(data)

        run = _vec_unrep(data["label"])
        times = run["times"].to_numpy()

        # 1-based boundary indices (R semantics).
        end_id = np.cumsum(times)
        start_id = end_id - times + 1

        xvals = data["x"].to_numpy()

        def _x(idx_1based: np.ndarray) -> np.ndarray:
            # idx_1based is a 1-based index array (already clamped to [1, n]).
            return xvals[idx_1based.astype(int) - 1]

        if align == "centre":
            start = (
                _x(np.maximum(start_id, 1)) + _x(np.maximum(start_id - 1, 1))
            ) / 2.0
            end = (_x(np.minimum(end_id, n)) + _x(np.minimum(end_id + 1, n))) / 2.0
        elif align == "end":
            start = _x(np.maximum(start_id - 1, 1))
            end = _x(end_id)
        elif align == "start":
            start = _x(start_id)
            end = _x(np.minimum(end_id + 1, n))
        else:  # "none"
            start = _x(start_id)
            end = _x(end_id)

        run_id = np.arange(1, len(run) + 1)

        return pd.DataFrame(
            {
                "start": start,
                "end": end,
                "start_id": start_id.astype(int),
                "end_id": end_id.astype(int),
                "run_id": run_id.astype(int),
                "runlength": times.astype(int),
                "runvalue": run["key"].reset_index(drop=True),
            }
        )


# -- constructor ------------------------------------------------------------


def stat_rle(
    mapping: Optional[Any] = None,
    data: Optional[Any] = None,
    geom: str = "rect",
    position: str = "identity",
    *,
    align: str = "none",
    na_rm: bool = False,
    orientation: str = "x",
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Run length encoding layer.

    Run length encoding takes a vector of values (the ``label`` aesthetic) and
    calculates the lengths of consecutive repeated values, after ordering the
    data on ``x``. Each run is rendered as a rectangle by default.

    Parameters
    ----------
    mapping : Mapping, optional
        Aesthetic mapping. Requires ``x`` and ``label``.
    data : DataFrame or callable, optional
        Layer data.
    geom : str, optional
        Geom to use. Defaults to ``"rect"``.
    position : str, optional
        Position adjustment. Defaults to ``"identity"``.
    align : {'none', 'centre', 'center', 'start', 'end'}, optional
        Effects the computed ``start`` and ``end`` variables. ``"center"`` is
        normalised to ``"centre"``. Defaults to ``"none"``.
    na_rm : bool, optional
        If ``False`` (default), missing values are removed with a warning.
    orientation : {'x', 'y'}, optional
        Orientation of the stat. Defaults to ``"x"``.
    show_legend : bool, optional
        Whether to show a legend for this layer.
    inherit_aes : bool, optional
        Whether to inherit aesthetics from the plot. Defaults to ``True``.
    **kwargs : Any
        Additional parameters passed to the layer.

    Returns
    -------
    Layer
        A ggplot2_py layer.
    """
    align = arg_match0(align, ["none", "centre", "center", "start", "end"], "align")
    if align == "center":
        align = "centre"

    params: Dict[str, Any] = {
        "na_rm": na_rm,
        "orientation": orientation,
        "align": align,
    }
    params.update(kwargs)

    return layer(
        data=data,
        mapping=mapping,
        stat=StatRle,
        geom=geom,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params=params,
    )
