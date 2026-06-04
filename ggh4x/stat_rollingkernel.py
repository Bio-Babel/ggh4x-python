"""Rolling-kernel smoother stat.

Python port of ``stat_roll.R`` from the R package **ggh4x**
(``StatRollingkernel`` ggproto object, the ``stat_rollingkernel`` constructor,
and the three kernel helpers ``.kernel_norm`` / ``.kernel_unif`` /
``.kernel_cauchy``).

A rolling kernel moves along one of the axes and assigns weights to datapoints
depending on the distance to the kernel's location. It then computes a weighted
average of the y-values, creating a trendline. Unlike a (weighted) rolling
average, the spacing between datapoints need not be constant.

The computation follows the R source verbatim:

#. ``flip_data`` the input when ``orientation == "y"``.
#. Resolve the bandwidth: a :class:`~ggplot2_py.Rel` object becomes
   ``bw.value * diff(range(x))``; a string names one of the ``stats::bw.*``
   rules (delegated to :func:`ggplot2_py.stat._precompute_bw`).
#. Drop non-finite ``x``/``y`` rows.
#. Build an evaluation sequence of length ``n`` spanning
   ``mid +/- (1 + expand) * 0.5 * diff(range(x))``.
#. Form the outer difference matrix ``outer(x, seq, "-")``, apply the kernel,
   column-normalize by ``colSums`` (this yields ``NaN`` columns where the kernel
   has zero total weight, exactly as in R), multiply by ``y`` and sum down the
   columns to obtain the smoothed value.
#. ``flip_data`` the result back.

The kernels mirror ``stats::dnorm`` / ``stats::dunif`` / ``stats::dcauchy``
through ``scipy.stats`` PDFs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from ggplot2_py import Rel
from ggplot2_py.layer import layer as _layer
from ggplot2_py.stat import Stat, _flip_data, _precompute_bw

from ._cli import cli_abort

__all__ = [
    "StatRollingkernel",
    "stat_rollingkernel",
    "_kernel_norm",
    "_kernel_unif",
    "_kernel_cauchy",
]


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def _kernel_norm(x: np.ndarray, bw: float) -> np.ndarray:
    """Gaussian kernel.

    Port of ``.kernel_norm <- function(x, bw) dnorm(x, sd = bw)``.

    Parameters
    ----------
    x : numpy.ndarray
        Distances from the kernel location.
    bw : float
        Bandwidth, used as the standard deviation.

    Returns
    -------
    numpy.ndarray
        Relative weights, i.e. ``dnorm(x, sd = bw)``.
    """
    from scipy import stats as _st

    return _st.norm.pdf(x, scale=bw)


def _kernel_unif(x: np.ndarray, bw: float) -> np.ndarray:
    """Uniform kernel.

    Port of ``.kernel_unif <- function(x, bw) dunif(x, min = -0.5 * bw,
    max = 0.5 * bw)``. Equivalent to a simple unweighted moving average.

    Parameters
    ----------
    x : numpy.ndarray
        Distances from the kernel location.
    bw : float
        Bandwidth; the uniform support is ``[-0.5 * bw, 0.5 * bw]``.

    Returns
    -------
    numpy.ndarray
        Relative weights, i.e. ``dunif(x, -0.5 * bw, 0.5 * bw)``.
    """
    from scipy import stats as _st

    # scipy's ``uniform`` is parameterised by (loc, scale) == (min, max - min).
    # Its support is the closed interval [loc, loc + scale], matching R's
    # ``dunif`` which returns 1/(max-min) at both endpoints.
    return _st.uniform.pdf(x, loc=-0.5 * bw, scale=bw)


def _kernel_cauchy(x: np.ndarray, bw: float) -> np.ndarray:
    """Cauchy kernel.

    Port of ``.kernel_cauchy <- function(x, bw) dcauchy(x, scale = bw)``. The
    Cauchy distribution has fatter tails than the normal distribution.

    Parameters
    ----------
    x : numpy.ndarray
        Distances from the kernel location.
    bw : float
        Bandwidth, used as the scale parameter (location fixed at 0).

    Returns
    -------
    numpy.ndarray
        Relative weights, i.e. ``dcauchy(x, scale = bw)``.
    """
    from scipy import stats as _st

    return _st.cauchy.pdf(x, scale=bw)


# Mapping from R ``switch`` kernel names to the callable kernels.
_KERNEL_LOOKUP: Dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "gaussian": _kernel_norm,
    "norm": _kernel_norm,
    "unif": _kernel_unif,
    "mean": _kernel_unif,
    "cauchy": _kernel_cauchy,
}


# ---------------------------------------------------------------------------
# ggproto
# ---------------------------------------------------------------------------

class StatRollingkernel(Stat):
    """Rolling-kernel smoother (port of ``StatRollingkernel``).

    Computed variables
    ------------------
    x : float
        A sequence of ordered x positions.
    y : float
        The weighted value of the rolling kernel.
    weight : float
        The sum of weight strengths at a position.
    scaled : float
        ``weight / sum(weight)`` by group.
    """

    required_aes = ["x", "y"]
    extra_params = ["na_rm", "orientation"]

    def setup_params(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve orientation and the kernel callable.

        Mirrors the R ``setup_params`` (which carries an unused third
        ``scales`` argument). Sets ``flipped_aes`` from ``orientation`` and
        resolves a string ``kernel`` into a callable.

        Parameters
        ----------
        data : pandas.DataFrame
            Layer data (unused, kept for signature parity).
        params : dict
            Layer parameters.

        Returns
        -------
        dict
            Updated parameters with ``flipped_aes`` set and ``kernel``
            resolved to a callable.

        Raises
        ------
        Exception
            Via :func:`ggh4x._cli.cli_abort` if the kernel name is unknown.
        """
        params["flipped_aes"] = params.get("orientation") == "y"
        kernel = params.get("kernel")
        if isinstance(kernel, str):
            resolved = _KERNEL_LOOKUP.get(kernel)
            if resolved is None:
                cli_abort(f"Unknown kernel specification: {kernel}.")
            params["kernel"] = resolved
        return params

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        n: int = 256,
        bw: Any = 0.02,
        expand: float = 0,
        kernel: Callable[[np.ndarray, float], np.ndarray] = _kernel_norm,
        flipped_aes: bool = False,
    ) -> pd.DataFrame:
        """Compute the rolling-kernel trendline for one group.

        Parameters
        ----------
        data : pandas.DataFrame
            Group data with at least ``x`` and ``y`` columns.
        scales : Any
            Panel scales (unused).
        n : int, default 256
            Number of evaluation points to return.
        bw : float, str, or ggplot2_py.Rel, default 0.02
            Bandwidth. A :class:`~ggplot2_py.Rel` becomes
            ``bw.value * diff(range(x))``; a string names a ``bw.*`` rule.
        expand : float, default 0
            Fraction by which to expand the evaluation range beyond the data.
        kernel : callable, default :func:`_kernel_norm`
            Kernel ``f(distances, bw) -> weights``.
        flipped_aes : bool, default False
            Whether x/y are swapped (orientation ``"y"``).

        Returns
        -------
        pandas.DataFrame
            Columns ``x``, ``y``, ``weight`` and ``scaled`` (flipped back when
            ``flipped_aes`` is true).
        """
        data = _flip_data(data, flipped_aes)

        x_full = np.asarray(data["x"], dtype=float)

        # -- Resolve bandwidth -------------------------------------------------
        if isinstance(bw, Rel):
            bw = bw.value * (np.nanmax(x_full) - np.nanmin(x_full))
        elif isinstance(bw, str):
            # ``_precompute_bw`` maps nrd0/nrd/sj/sj-ste/sj-dpi/ucv/bcv.
            lname = bw.lower()
            if lname not in ("nrd0", "nrd", "ucv", "bcv", "sj", "sj-ste", "sj-dpi"):
                cli_abort(f"Unknown bandwidth rule: {bw}.")
            bw = _precompute_bw(x_full, lname)
        bw = float(bw)

        # -- Drop non-finite rows ---------------------------------------------
        x = np.asarray(data["x"], dtype=float)
        y = np.asarray(data["y"], dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        x = x[keep]
        y = y[keep]

        # -- Evaluation sequence ----------------------------------------------
        lo = float(np.min(x))
        hi = float(np.max(x))
        mid = (lo + hi) / 2.0
        half = (1.0 + expand) * (0.5 * (hi - lo))
        seq_range = np.linspace(mid - half, mid + half, int(n))

        # -- Kernel-weighted average ------------------------------------------
        # krnl[i, j] = x[i] - seq_range[j]   (R: outer(data$x, seq_range, "-"))
        krnl = x[:, None] - seq_range[None, :]
        krnl = kernel(krnl, bw)
        krnl = np.asarray(krnl, dtype=float)

        # weight = colSums(krnl); column-normalise (0/0 -> NaN, as in R).
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = krnl.sum(axis=0)
            krnl = krnl / weight[None, :]
        krnl = krnl * y[:, None]
        y_out = krnl.sum(axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = weight / weight.sum()

        out = pd.DataFrame(
            {
                "x": seq_range,
                "y": y_out,
                "weight": weight,
                "scaled": scaled,
            }
        )
        return _flip_data(out, flipped_aes)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

def stat_rollingkernel(
    mapping: Optional[Any] = None,
    data: Any = None,
    geom: str = "line",
    position: str = "identity",
    *,
    bw: Any = "nrd",
    kernel: Any = "gaussian",
    n: int = 256,
    expand: float = 0.1,
    na_rm: bool = False,
    orientation: str = "x",
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Rolling-kernel trendline layer.

    A rolling kernel moves along one of the axes, assigns distance-based
    weights to datapoints and computes a weighted average of the y-values,
    producing a trendline.

    Parameters
    ----------
    mapping : aes, optional
        Aesthetic mapping.
    data : DataFrame or callable, optional
        Layer data.
    geom : str, default ``"line"``
        Geom used to draw the trendline.
    position : str, default ``"identity"``
        Position adjustment.
    bw : float, str, or ggplot2_py.Rel, default ``"nrd"``
        Bandwidth. One of: a numeric kernel width in data units; a
        :class:`~ggplot2_py.Rel` for a width relative to the group data range;
        or a string naming one of the ``stats::bw.nrd`` family of rules.
    kernel : str or callable, default ``"gaussian"``
        Either a callable ``f(distances, bw) -> weights`` or one of
        ``"gaussian"``/``"norm"``, ``"unif"``/``"mean"`` or ``"cauchy"``.
    n : int, default 256
        Number of points to return per group.
    expand : float, default 0.1
        How much to expand the evaluation range beyond the extreme datapoints.
    na_rm : bool, default False
        Whether to silently remove missing values.
    orientation : str, default ``"x"``
        Axis along which the rolling occurs, either ``"x"`` or ``"y"``.
    show_legend : bool, optional
        Whether to show a legend.
    inherit_aes : bool, default True
        Whether to inherit aesthetics from the plot.
    **kwargs
        Additional parameters passed to the layer.

    Returns
    -------
    Layer
        A ggplot2 layer ggproto object.
    """
    return _layer(
        data=data,
        mapping=mapping,
        stat=StatRollingkernel,
        geom=geom,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "bw": bw,
            "kernel": kernel,
            "n": n,
            "expand": expand,
            "na_rm": na_rm,
            "orientation": orientation,
            **kwargs,
        },
    )
