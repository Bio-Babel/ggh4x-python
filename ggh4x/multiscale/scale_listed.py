"""Distribute a list of non-standard-aesthetic scales (R source: ``scale_listed.R``).

Ports ggh4x's :func:`scale_listed`, which takes a user-supplied list of ready-made
discrete scales (each bound to a single *non-standard* aesthetic) together with a
parallel ``replaces`` vector naming the *standard* aesthetic each one substitutes,
validates them, materialises their guides (setting ``available_aes`` so the legend
system matches the non-standard aesthetic) and groups them by ``replaces`` into one
:class:`~ggh4x.multiscale._multiscale_add.MultiScale` per distinct standard
aesthetic.

The grouping mirrors R's ``split(scalelist, replaces)``: groups are emitted in the
sorted order of the distinct ``replaces`` names (R orders ``split`` by factor
levels, i.e. alphabetically for a character vector).  ``scale_listed`` returns the
*list* of :class:`MultiScale` objects; the per-layer geom rewrite happens later in
:func:`ggh4x.multiscale._multiscale_add._add_multiscale`.

All semantics were verified against a live R ``ggh4x`` session (validity checks,
the ``any``-union vs assign ``available_aes`` logic, and the alphabetical
``split`` ordering).
"""

from __future__ import annotations

from typing import Any, List

import ggplot2_py as _gg
from ggplot2_py import standardise_aes_names

from .._cli import cli_abort
from ._multiscale_add import MultiScale

__all__ = [
    "scale_listed",
    "ALL_AESTHETICS",
]


# Verbatim port of R ``.all_aesthetics`` (scale_listed.R:205-211).
ALL_AESTHETICS = frozenset(
    {
        "adj", "alpha", "angle", "bg", "cex", "col", "color", "colour", "fg",
        "fill", "group", "hjust", "label", "linetype", "lower", "lty", "lwd",
        "max", "middle", "min", "pch", "radius", "sample", "shape", "size",
        "srt", "upper", "vjust", "weight", "width", "x", "xend", "xmax", "xmin",
        "xintercept", "y", "yend", "ymax", "ymin", "yintercept", "z",
    }
)


def _is_guide_proto(obj: Any) -> bool:
    """Return ``True`` when *obj* is a :class:`ggplot2_py.guide.Guide` instance."""
    Guide = getattr(_gg, "Guide", None)
    if Guide is None:
        return False
    try:
        return isinstance(obj, Guide)
    except TypeError:
        return False


def scale_listed(scalelist: List[Any], replaces: Any = None) -> List[MultiScale]:
    """Distribute a list of non-standard-aesthetic scales across the plot.

    Port of R ``scale_listed`` (``scale_listed.R:56-133``).  Validates the scale
    list, materialises each scale's guide with the correct ``available_aes``, and
    groups the scales by the standard aesthetic each replaces into one
    :class:`MultiScale` per distinct ``replaces`` value.

    This should only be added to a plot **after** every layer the non-standard
    aesthetics affect has been added, since :class:`MultiScale` rewrites those
    layers' geoms at ``+``-time.

    Parameters
    ----------
    scalelist : list of ggplot2_py.scale.Scale
        Scales each created with a single non-standard ``aesthetics`` argument.
    replaces : sequence of str
        Parallel to *scalelist*; the standard aesthetic (typically ``"colour"``
        or ``"fill"``) each scale replaces.

    Returns
    -------
    list of MultiScale
        One :class:`MultiScale` per distinct ``replaces`` value, in the sorted
        order of the distinct ``replaces`` names.

    Raises
    ------
    ValueError
        If *replaces* is not parallel to *scalelist*, contains an invalid
        aesthetic, if a list element is not a :class:`ggplot2_py.scale.Scale`, or
        if any scale does not have exactly one aesthetic.
    """
    if replaces is None:
        replaces = []
    replaces = list(replaces)

    # Check replaces validity (scale_listed.R:58-69).
    if len(scalelist) != len(replaces):
        cli_abort(
            "The `replaces` argument must be parallel to and of the same length "
            "as the `scalelist` argument."
        )
    replaces = standardise_aes_names(replaces)
    if not all(r in ALL_AESTHETICS for r in replaces):
        cli_abort(
            "The aesthetics in the `replaces` argument must be valid aesthetics."
        )

    # Check scalelist validity (scale_listed.R:71-82).
    Scale = getattr(_gg, "Scale", None)
    if not all(Scale is not None and isinstance(s, Scale) for s in scalelist):
        cli_abort(
            "The `scalelist` argument must have valid `Scale` objects as "
            "list-elements."
        )

    # Check scale aesthetics (scale_listed.R:85-98).
    aes_per_scale = [list(s.aesthetics or []) for s in scalelist]
    if any(len(a) > 1 for a in aes_per_scale):
        cli_abort("`scale_listed()` can only accept 1 aesthetic per scale.")
    aesthetics: List[str] = [a for sub in aes_per_scale for a in sub]
    if len(aesthetics) != len(replaces):
        cli_abort(
            "Every scale in the `scalelist` argument must have set valid "
            "aesthetics."
        )

    # Interpret guides (scale_listed.R:101-121).
    for scale in scalelist:
        guide = scale.guide
        if guide == "none" or guide is False:
            continue
        if not _is_guide_proto(guide):
            # match.fun(paste0("guide_", guide))()
            guide_fn = getattr(_gg, f"guide_{guide}", None)
            if not callable(guide_fn):
                cli_abort(
                    f"`scale_listed()` cannot find a guide constructor for "
                    f"`{guide}`."
                )
            guide = guide_fn()
        if _is_guide_proto(guide):
            # ggproto(NULL, old): clone so the original guide is untouched.
            guide = _gg.ggproto(None, guide)
        avail = list(getattr(guide, "available_aes", []) or [])
        if "any" not in avail:
            guide.available_aes = list(scale.aesthetics)
        else:
            guide.available_aes = avail + list(scale.aesthetics)
        scale.guide = guide

    # Split by replaced aes (scale_listed.R:123-132).  R's ``split`` keys are the
    # distinct ``replaces`` names ordered as factor levels (alphabetical for a
    # character vector).
    distinct = sorted(set(replaces))
    out: List[MultiScale] = []
    for aes_name in distinct:
        idx = [i for i, r in enumerate(replaces) if r == aes_name]
        out.append(
            MultiScale(
                scales=[scalelist[i] for i in idx],
                aes=[aesthetics[i] for i in idx],
                replaced_aes=aes_name,
            )
        )
    return out
