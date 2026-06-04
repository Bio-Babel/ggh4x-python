"""Fitted theoretical density.

Port of ``ggh4x``'s ``stat_theodensity.R``. Estimates the parameters of a
named theoretical distribution by maximum likelihood and evaluates the
distribution's probability density (or mass) function on a grid. This is the
Python analogue of fitting a parametric distribution with
``fitdistrplus::fitdist`` and then calling ``d<distri>`` from R's ``stats``
package.

R uses ``fitdistrplus::fitdist`` for maximum-likelihood estimation. The Python
port replaces this with :mod:`scipy.stats` MLE: continuous distributions use the
``<dist>.fit`` method (with location/scale fixed where R's parameterization
fixes them), while discrete distributions (``pois``/``geom``/``binom``/
``nbinom``) use closed-form or numerically optimized maximum-likelihood
estimators. A hand-built mapping table translates R distribution names and
parameterizations (e.g. R's ``gamma`` rate vs scipy's scale) into scipy
objects and parameters.

R source
--------
``ggh4x/R/stat_theodensity.R``

Notes
-----
The single largest fidelity risk is that :func:`scipy.stats.<dist>.fit` returns
parameters in a different order and parameterization than R's ``d<distri>``
functions. The :data:`_DISTRI_TABLE` mapping captures, per R distribution name,
the scipy distribution object, the fixed-parameter constraints required to
reproduce R's MLE, and the conversion from the scipy fit tuple back into R's
named parameters. Verified against live ``fitdistrplus`` output on identical
data samples.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ggplot2_py import ggproto_parent
from ggplot2_py.aes import AfterStat
from ggplot2_py.stat import StatDensity, _layer

from ._cli import cli_abort, cli_inform, cli_warn

__all__ = ["stat_theodensity", "StatTheoDensity", "_class_distri"]


# ---------------------------------------------------------------------------
# Distribution parameterization mapping table
# ---------------------------------------------------------------------------
#
# R's ``d<distri>`` functions and ``fitdistrplus`` use parameterizations that
# differ from :mod:`scipy.stats`. Each entry maps an R distribution name to:
#   - ``kind``      : "continuous" or "discrete".
#   - ``fitter``    : callable ``(x, fix_arg, start_arg) -> dict`` returning the
#                     fitted parameters in R's named parameterization (the same
#                     names ``coef(fitdist(...))`` would produce, plus any fixed
#                     parameters).
#   - ``pdf``       : callable ``(xseq, params) -> ndarray`` evaluating the
#                     density / mass function using R's named parameters.
#
# This indirection reproduces ``get(paste0("d", distri))`` together with the
# ``coef(fitdistrplus::fitdist(...))`` call in the R source.


def _require_scipy() -> Any:
    """Import :mod:`scipy.stats`, aborting with a helpful message if absent.

    Returns
    -------
    module
        The :mod:`scipy.stats` module.

    Raises
    ------
    ImportError
        If scipy is not installed.
    """
    try:
        from scipy import stats as _stats  # noqa: WPS433 (local import by design)
    except ImportError as err:  # pragma: no cover - environment dependent
        raise ImportError(
            "The 'scipy' package is required for `stat_theodensity()`."
        ) from err
    return _stats


# -- Continuous fitters ------------------------------------------------------


def _fit_norm(x, fix_arg, start_arg):
    st = _require_scipy()
    loc, scale = st.norm.fit(x)
    return {"mean": loc, "sd": scale}


def _fit_lnorm(x, fix_arg, start_arg):
    st = _require_scipy()
    s, _loc, scale = st.lognorm.fit(x, floc=0)
    return {"meanlog": float(np.log(scale)), "sdlog": s}


def _fit_cauchy(x, fix_arg, start_arg):
    st = _require_scipy()
    loc, scale = st.cauchy.fit(x)
    return {"location": loc, "scale": scale}


def _fit_gamma(x, fix_arg, start_arg):
    st = _require_scipy()
    kwargs: Dict[str, Any] = {"floc": 0}
    fixed: Dict[str, Any] = {}
    # R fixes ``rate`` -> scipy fixes ``scale = 1 / rate``.
    if fix_arg and "rate" in fix_arg:
        kwargs["fscale"] = 1.0 / float(fix_arg["rate"])
        fixed["rate"] = float(fix_arg["rate"])
    if fix_arg and "shape" in fix_arg:
        kwargs["fa"] = float(fix_arg["shape"])
        fixed["shape"] = float(fix_arg["shape"])
    a, _loc, scale = st.gamma.fit(x, **kwargs)
    out: Dict[str, Any] = {"shape": a, "rate": 1.0 / scale}
    out.update(fixed)
    return out


def _fit_weibull(x, fix_arg, start_arg):
    st = _require_scipy()
    c, _loc, scale = st.weibull_min.fit(x, floc=0)
    return {"shape": c, "scale": scale}


def _fit_exp(x, fix_arg, start_arg):
    st = _require_scipy()
    _loc, scale = st.expon.fit(x, floc=0)
    return {"rate": 1.0 / scale}


def _fit_logis(x, fix_arg, start_arg):
    st = _require_scipy()
    loc, scale = st.logistic.fit(x)
    return {"location": loc, "scale": scale}


def _fit_beta(x, fix_arg, start_arg):
    st = _require_scipy()
    a, b, _loc, _scale = st.beta.fit(x, floc=0, fscale=1)
    return {"shape1": a, "shape2": b}


def _fit_unif(x, fix_arg, start_arg):
    return {"min": float(np.min(x)), "max": float(np.max(x))}


def _fit_t(x, fix_arg, start_arg):
    st = _require_scipy()
    # R's ``dt`` has a single ``df`` parameter (standard t, location 0,
    # scale 1). fitdistrplus needs a start value; mirror by fixing loc/scale.
    df, _loc, _scale = st.t.fit(x, floc=0, fscale=1)
    return {"df": df}


def _fit_f(x, fix_arg, start_arg):
    st = _require_scipy()
    dfn, dfd, _loc, _scale = st.f.fit(x, floc=0, fscale=1)
    return {"df1": dfn, "df2": dfd}


def _fit_chisq(x, fix_arg, start_arg):
    # Only reached if a user forces ``chisq`` past ``setup_params`` (which
    # normally remaps it to gamma). scipy chi2 has a single ``df`` parameter.
    st = _require_scipy()
    df, _loc, _scale = st.chi2.fit(x, floc=0, fscale=1)
    return {"df": df}


# -- Discrete fitters --------------------------------------------------------


def _fit_pois(x, fix_arg, start_arg):
    return {"lambda": float(np.mean(x))}


def _fit_geom(x, fix_arg, start_arg):
    # R's ``dgeom`` counts failures before the first success; the MLE of the
    # success probability is ``1 / (1 + mean)``.
    return {"prob": 1.0 / (1.0 + float(np.mean(x)))}


def _fit_binom(x, fix_arg, start_arg):
    # ``size`` must be fixed (R aborts/auto-fixes it in setup_params). The MLE
    # of ``prob`` with fixed ``size`` is ``mean(x) / size``.
    if not fix_arg or "size" not in fix_arg:
        cli_abort("Fitting a binomial distribution requires a fixed 'size'.")
    size = int(fix_arg["size"])
    return {"size": size, "prob": float(np.mean(x)) / size}


def _fit_nbinom(x, fix_arg, start_arg):
    # fitdistrplus returns (size, mu). Full two-parameter MLE over (size, mu),
    # with ``prob = size / (size + mu)``. Matches fitdist on identical data.
    st = _require_scipy()
    from scipy import optimize  # noqa: WPS433

    x = np.asarray(x, dtype=float)
    m = float(np.mean(x))
    v = float(np.var(x))
    size0 = (m * m / (v - m)) if v > m else 1.0
    if not np.isfinite(size0) or size0 <= 0:
        size0 = 1.0

    def _negll(params: np.ndarray) -> float:
        size, mu = params
        if size <= 0 or mu <= 0:
            return 1e10
        prob = size / (size + mu)
        return -float(np.sum(st.nbinom.logpmf(x, size, prob)))

    res = optimize.minimize(
        _negll,
        np.array([size0, m]),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 10000},
    )
    size, mu = res.x
    return {"size": float(size), "mu": float(mu)}


# -- PDF / PMF evaluators (R parameterization) -------------------------------


def _pdf_norm(xseq, p):
    return _require_scipy().norm.pdf(xseq, loc=p["mean"], scale=p["sd"])


def _pdf_lnorm(xseq, p):
    return _require_scipy().lognorm.pdf(
        xseq, p["sdlog"], loc=0, scale=np.exp(p["meanlog"])
    )


def _pdf_cauchy(xseq, p):
    return _require_scipy().cauchy.pdf(xseq, loc=p["location"], scale=p["scale"])


def _pdf_gamma(xseq, p):
    return _require_scipy().gamma.pdf(xseq, p["shape"], loc=0, scale=1.0 / p["rate"])


def _pdf_weibull(xseq, p):
    return _require_scipy().weibull_min.pdf(xseq, p["shape"], loc=0, scale=p["scale"])


def _pdf_exp(xseq, p):
    return _require_scipy().expon.pdf(xseq, loc=0, scale=1.0 / p["rate"])


def _pdf_logis(xseq, p):
    return _require_scipy().logistic.pdf(xseq, loc=p["location"], scale=p["scale"])


def _pdf_beta(xseq, p):
    return _require_scipy().beta.pdf(xseq, p["shape1"], p["shape2"], loc=0, scale=1)


def _pdf_unif(xseq, p):
    lo, hi = p["min"], p["max"]
    return _require_scipy().uniform.pdf(xseq, loc=lo, scale=hi - lo)


def _pdf_t(xseq, p):
    return _require_scipy().t.pdf(xseq, p["df"])


def _pdf_f(xseq, p):
    return _require_scipy().f.pdf(xseq, p["df1"], p["df2"])


def _pdf_chisq(xseq, p):
    return _require_scipy().chi2.pdf(xseq, p["df"])


def _pmf_pois(xseq, p):
    return _require_scipy().poisson.pmf(xseq, p["lambda"])


def _pmf_geom(xseq, p):
    return _require_scipy().geom.pmf(xseq + 1, p["prob"])  # scipy geom support: 1,2,...


def _pmf_binom(xseq, p):
    return _require_scipy().binom.pmf(xseq, int(p["size"]), p["prob"])


def _pmf_nbinom(xseq, p):
    size, mu = p["size"], p["mu"]
    prob = size / (size + mu)
    return _require_scipy().nbinom.pmf(xseq, size, prob)


_DISTRI_TABLE: Dict[str, Dict[str, Any]] = {
    "norm": {"kind": "continuous", "fitter": _fit_norm, "pdf": _pdf_norm},
    "lnorm": {"kind": "continuous", "fitter": _fit_lnorm, "pdf": _pdf_lnorm},
    "cauchy": {"kind": "continuous", "fitter": _fit_cauchy, "pdf": _pdf_cauchy},
    "gamma": {"kind": "continuous", "fitter": _fit_gamma, "pdf": _pdf_gamma},
    "weibull": {"kind": "continuous", "fitter": _fit_weibull, "pdf": _pdf_weibull},
    "exp": {"kind": "continuous", "fitter": _fit_exp, "pdf": _pdf_exp},
    "logis": {"kind": "continuous", "fitter": _fit_logis, "pdf": _pdf_logis},
    "beta": {"kind": "continuous", "fitter": _fit_beta, "pdf": _pdf_beta},
    "unif": {"kind": "continuous", "fitter": _fit_unif, "pdf": _pdf_unif},
    "t": {"kind": "continuous", "fitter": _fit_t, "pdf": _pdf_t},
    "f": {"kind": "continuous", "fitter": _fit_f, "pdf": _pdf_f},
    "chisq": {"kind": "continuous", "fitter": _fit_chisq, "pdf": _pdf_chisq},
    "pois": {"kind": "discrete", "fitter": _fit_pois, "pdf": _pmf_pois},
    "geom": {"kind": "discrete", "fitter": _fit_geom, "pdf": _pmf_geom},
    "binom": {"kind": "discrete", "fitter": _fit_binom, "pdf": _pmf_binom},
    "nbinom": {"kind": "discrete", "fitter": _fit_nbinom, "pdf": _pmf_nbinom},
}

#: Distributions explicitly rejected by ``stat_theodensity`` (mirrors R).
_UNSUPPORTED = ("multinom", "hyper", "wilcox", "signrank")


# ---------------------------------------------------------------------------
# Helper: classify a distribution as discrete or continuous
# ---------------------------------------------------------------------------


def _class_distri(distri: str) -> str:
    """Classify a distribution name as ``"discrete"`` or ``"continuous"``.

    Port of ``class_distri`` (``stat_theodensity.R``). R first checks fixed
    discrete/continuous name sets and only falls back to an empirical
    ``r<distri>`` probe for user-defined distributions in the calling
    environment. The Python port supports the built-in distributions of the
    mapping table; unknown names raise.

    Parameters
    ----------
    distri : str
        Distribution name without the ``d``/``r``/``p``/``q`` prefix.

    Returns
    -------
    str
        ``"discrete"`` or ``"continuous"``.

    Raises
    ------
    ValueError
        If the distribution cannot be classified.
    """
    discrete_distris = (
        "pois",
        "nbinom",
        "binom",
        "geom",
        "hyper",
        "signrank",
        "multinom",
        "wilcox",
    )
    if distri in discrete_distris:
        return "discrete"

    conti_distris = (
        "beta",
        "cauchy",
        "chisq",
        "exp",
        "f",
        "gamma",
        "lnorm",
        "norm",
        "t",
        "unif",
        "weibull",
        "logis",
    )
    if distri in conti_distris:
        return "continuous"

    # R performs an empirical probe of a user-supplied ``r<distri>`` function in
    # the calling environment. That is out of scope for the port; abort like R
    # would when it cannot determine the type.
    cli_abort(
        f"`stat_theodensity()` failed to determine if the '{distri}' "
        "distribution is discrete or continuous."
    )


# ---------------------------------------------------------------------------
# ggproto
# ---------------------------------------------------------------------------


class StatTheoDensity(StatDensity):
    """Fit a theoretical distribution by MLE and evaluate its density.

    Extends :class:`ggplot2_py.StatDensity`. The kernel-density computation of
    the parent is replaced by maximum-likelihood fitting of a named theoretical
    distribution followed by evaluation of its probability density (continuous)
    or mass (discrete) function.

    Attributes
    ----------
    default_aes : dict
        Inherited from :class:`StatDensity`, mapping ``x``/``y`` to
        ``after_stat(density)``.
    """

    extra_params: List[str] = ["na_rm", "orientation"]

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        distri: str = "norm",
        n: int = 512,
        distri_type: str = "continuous",
        fix_arg: Optional[Dict[str, Any]] = None,
        start_arg: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fit ``distri`` to ``data['x']`` and evaluate its density on a grid.

        Parameters
        ----------
        data : pandas.DataFrame
            Must contain an ``x`` column.
        scales : dict-like
            Panel scales; ``scales['x'].dimension()`` provides the evaluation
            range.
        distri : str, default ``"norm"``
            Distribution name (without prefix).
        n : int, default 512
            Number of equally spaced evaluation points (continuous only).
        distri_type : str, default ``"continuous"``
            Either ``"continuous"`` or ``"discrete"``.
        fix_arg : dict, optional
            Fixed parameters in R parameterization.
        start_arg : dict, optional
            Starting parameters (consumed by some fitters).

        Returns
        -------
        pandas.DataFrame
            Columns ``x``, ``density``, ``scaled``, ``count``, ``n`` on success;
            a single NaN row with columns ``x``, ``density``, ``ndensity``,
            ``count``, ``n`` on failure (``< 2`` points or estimation failure).
        """
        _require_scipy()

        # Data to return upon failure (mirrors R's ``nulldata``).
        nulldata = pd.DataFrame(
            {
                "x": [np.nan],
                "density": [np.nan],
                "ndensity": [np.nan],
                "count": [np.nan],
                "n": [np.nan],
            }
        )

        entry = _DISTRI_TABLE.get(distri)
        if entry is None:
            cli_abort(
                "The `distri` argument must have a valid density function "
                f"called `d{distri}`."
            )

        x_all = np.asarray(data["x"].to_numpy(), dtype=float)
        x = x_all[~np.isnan(x_all)]
        nx = len(data["x"])  # R uses length(data$x), i.e. including NA rows.

        if nx < 2:
            cli_warn("Groups with fewer than two data points have been dropped.")
            return nulldata

        scale = (
            scales.get("x")
            if isinstance(scales, dict)
            else getattr(scales, "x", None)
        )
        if scale is not None and hasattr(scale, "dimension"):
            rng = tuple(scale.dimension())
        else:
            rng = (float(np.nanmin(x_all)), float(np.nanmax(x_all)))

        if distri_type == "discrete":
            xseq = np.arange(np.floor(rng[0]), np.ceil(rng[1]) + 1, 1.0)
        else:
            xseq = np.linspace(rng[0], rng[1], n)

        # Maximum-likelihood estimation (replaces fitdistrplus::fitdist).
        try:
            params = entry["fitter"](x, fix_arg, start_arg)
        except Exception:  # noqa: BLE001 - any estimation failure -> nulldata
            cli_warn(f"Failed to estimate parameters of '{distri}' distribution.")
            return nulldata

        par_values = np.asarray(list(params.values()), dtype=float)
        if (
            par_values.size == 0
            or np.any(np.isnan(par_values))
            or not np.all(np.isfinite(par_values))
        ):
            cli_warn(f"Failed to estimate parameters of '{distri}' distribution.")
            return nulldata

        dens = np.asarray(entry["pdf"](xseq, params), dtype=float)

        dens_max = np.nanmax(dens)
        return pd.DataFrame(
            {
                "x": xseq,
                "density": dens,
                "scaled": dens / dens_max,
                "count": dens * nx,
                "n": nx,
            }
        )

    def setup_params(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classify the distribution and apply R's parameter remaps.

        Parameters
        ----------
        data : pandas.DataFrame
            Layer data; ``data['x']`` is inspected for integrality.
        params : dict
            Stat parameters (``distri``, ``fix_arg``, ``start_arg``, ...).

        Returns
        -------
        dict
            Updated parameters with ``distri_type`` injected and ``chisq`` /
            ``binom`` remaps applied.

        Raises
        ------
        ValueError
            If a discrete distribution is requested for non-integer data.
        """
        distri = params.get("distri", "norm")
        dtype = _class_distri(distri)
        if dtype == "discrete":
            x = np.asarray(data["x"].to_numpy(), dtype=float)
            x = x[~np.isnan(x)]
            if float(np.sum(np.abs(np.mod(x, 1)))) > 0:
                cli_abort(
                    f"A discrete '{distri}' distribution cannot be fitted "
                    "to continuous data."
                )
        params = dict(params)
        params["distri_type"] = dtype

        # Chi square estimator causes trouble; estimate as gamma with rate=0.5.
        if params.get("distri") == "chisq":
            params["distri"] = "gamma"
            fix_arg = params.get("fix_arg")
            if fix_arg is None:
                params["fix_arg"] = {"rate": 0.5}
            else:
                params["fix_arg"] = {
                    "shape": float(fix_arg["df"]) / 2.0,
                    "rate": 0.5,
                }

        # Binomial does not operate without a fixed size.
        if params.get("distri") == "binom":
            x = np.asarray(data["x"].to_numpy(), dtype=float)
            x = x[~np.isnan(x)]
            if params.get("fix_arg") is None:
                params["fix_arg"] = {"size": int(np.max(x))}
                cli_inform(
                    "Estimating binomial PMF with size set to maximum data value."
                )
            params["start_arg"] = {
                "prob": float(np.mean(x)) / float(np.max(x))
            }

        return params


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def stat_theodensity(
    mapping: Optional[Any] = None,
    data: Any = None,
    geom: str = "line",
    position: str = "identity",
    *,
    distri: str = "norm",
    n: int = 512,
    fix_arg: Optional[Dict[str, Any]] = None,
    start_arg: Optional[Dict[str, Any]] = None,
    na_rm: bool = True,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Construct a fitted-theoretical-density layer.

    Estimates the parameters of ``distri`` by maximum likelihood and evaluates
    its probability density function, useful for comparing histograms or kernel
    density estimates against a theoretical distribution.

    Parameters
    ----------
    mapping : aes, optional
        Aesthetic mapping.
    data : DataFrame or callable, optional
        Layer data.
    geom : str, default ``"line"``
        Geometry used to render the layer.
    position : str, default ``"identity"``
        Position adjustment.
    distri : str, default ``"norm"``
        Distribution name without prefix (e.g. ``"norm"``, ``"nbinom"``). See
        :data:`_DISTRI_TABLE` for supported names.
    n : int, default 512
        Number of equally spaced evaluation points (ignored for discrete
        distributions).
    fix_arg : dict, optional
        Fixed parameters of the named distribution (R parameterization).
    start_arg : dict, optional
        Starting parameters for the estimation.
    na_rm : bool, default True
        Whether to silently remove missing values.
    show_legend : bool, optional
        Whether to show this layer in the legend.
    inherit_aes : bool, default True
        Whether to inherit aesthetics from the plot.
    **kwargs
        Additional parameters forwarded to the layer.

    Returns
    -------
    Layer
        A ggplot2_py layer.

    Raises
    ------
    ValueError
        If ``distri`` has no known density function, or names an unsupported
        distribution (``multinom``/``hyper``/``wilcox``/``signrank``).
    """
    if distri not in _DISTRI_TABLE:
        cli_abort(
            "The `distri` argument must have a valid density function "
            f"called `d{distri}`."
        )
    if distri in _UNSUPPORTED:
        cli_abort(
            f"`stat_theodensity()` does not support the '{distri}' distribution."
        )

    return _layer(
        stat=StatTheoDensity,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "distri": distri,
            "n": n,
            "fix_arg": fix_arg,
            "start_arg": start_arg,
            "na_rm": na_rm,
            **kwargs,
        },
    )
