"""Multiple gradient colour/fill scales (R source: ``scale_multi.R``).

Ports ggh4x's :func:`scale_colour_multi` / :func:`scale_fill_multi`, which map
several *non-standard* colour/fill aesthetics (``fill1``, ``fill2``, ...) each to
its own :func:`ggplot2_py.continuous_scale` gradient.  The constructors build a
:class:`~ggh4x.multiscale._multiscale_add.MultiScale` container that defers the
plot rewrite to ``+``-time (see :mod:`ggh4x.multiscale._multiscale_add`).

The R *listed-argument* convention is reproduced exactly by :func:`_pickvalue`:
an argument that is a Python ``list`` is interpreted *per aesthetic* (its ``i``-th
element belongs to the ``i``-th aesthetic, wrapping to the first element when the
index exceeds the list length); any other value (scalar, colour vector, tuple,
``ndarray``) is broadcast unchanged to every aesthetic.

Notes
-----
* ggplot2_py always runs the *new guide system*, so the ``trans`` extra argument
  is unconditionally renamed to ``transform`` and the R ``scale_name`` argument is
  never forwarded (it does not exist on :func:`ggplot2_py.continuous_scale`).
* All semantics were verified against a live R ``ggh4x`` session
  (``distribute_scale_multi``, ``pickvalue``, the materialised ``GuideColourbar``
  ``available_aes`` and the default ``white``/``black`` gradient palette).
"""

from __future__ import annotations

from typing import Any, Dict, List

import scales as _scales

import ggplot2_py as _gg
from ggplot2_py import standardise_aes_names
from ggplot2_py.scale import continuous_scale

from .._cli import cli_abort
from ._multiscale_add import MultiScale

__all__ = [
    "scale_fill_multi",
    "scale_colour_multi",
    "scale_color_multi",
    "MultiScale",
]

# Sentinel marking "argument not supplied" so the R ``missing()`` cascade for
# ``colours`` / ``colors`` can be reproduced faithfully.
_MISSING = object()


# ---------------------------------------------------------------------------
# pickvalue  (scale_multi.R:174-183)
# ---------------------------------------------------------------------------
def _is_per_aesthetic_list(x: Any) -> bool:
    """Return ``True`` when *x* is a Python list used as a *per-aesthetic* container.

    R distinguishes ``list(c("white","red"), c("black","blue"))`` (a *list* of
    vectors -- per aesthetic) from ``c("white","black")`` (a *vector* -- broadcast)
    via ``class(x)[[1]] == "list"``.  Python collapses both onto ``list``, so the
    faithful disambiguation is: a list is per-aesthetic iff at least one of its
    elements is itself a non-string sequence (a nested vector), mirroring the
    documented ``colours = [["white", "red"], ...]`` idiom.  A flat list of scalars
    (e.g. ``["white", "black"]``) is a bare colour vector and is broadcast.

    Parameters
    ----------
    x : Any

    Returns
    -------
    bool
    """
    if type(x) is not list:
        return False
    return any(
        isinstance(el, (list, tuple)) or hasattr(el, "__len__") and not isinstance(el, (str, bytes))
        for el in x
    )


def _pickvalue(x: Any, i: int) -> Any:
    """Select the per-aesthetic value of *x* for aesthetic index *i*.

    Port of R ``pickvalue`` (``scale_multi.R:174-183``).  When *x* is a
    *per-aesthetic* list (see :func:`_is_per_aesthetic_list`) the ``i``-th element
    is returned, wrapping back to the first element when ``i`` exceeds the list
    length.  Any other value (a scalar, a bare colour vector / flat list, a
    ``tuple`` or an ``ndarray``) is *broadcast* and returned unchanged.

    Parameters
    ----------
    x : Any
        The raw argument value.
    i : int
        Zero-based aesthetic index.

    Returns
    -------
    Any
        ``x[i]`` (with wraparound to ``x[0]``) when *x* is a per-aesthetic list;
        otherwise *x* unchanged.
    """
    if not _is_per_aesthetic_list(x):
        return x
    # R: i <- if (i > length(x)) 1 else i  (1-based). Here i is 0-based, so the
    # out-of-range index wraps to the first element.
    if i >= len(x):
        i = 0
    return x[i]


# ---------------------------------------------------------------------------
# distribute_scale_multi  (scale_multi.R:115-171)
# ---------------------------------------------------------------------------
def _distribute_scale_multi(
    *,
    aesthetics: List[str],
    colours: Any,
    values: Any,
    na_value: Any,
    guide: Any,
    extra: Dict[str, Any],
) -> List[Any]:
    """Build one :func:`continuous_scale` per aesthetic, distributing arguments.

    Port of R ``distribute_scale_multi`` (``scale_multi.R:115-171``).  For every
    aesthetic ``aesthetics[i]`` the listed arguments are picked with
    :func:`_pickvalue`, the guide is materialised (so its ``available_aes`` is set
    to the single non-standard aesthetic), and a gradient
    :func:`continuous_scale` is constructed.

    Parameters
    ----------
    aesthetics : list of str
        Non-standard aesthetic names (e.g. ``["fill1", "fill2"]``).
    colours : Any
        Gradient colours (per-aesthetic ``list`` or broadcast vector).
    values : Any
        Gradient ``values`` positions (per-aesthetic ``list`` or broadcast).
    na_value : Any
        Colour for missing values (per-aesthetic ``list`` or broadcast).
    guide : Any
        Guide spec(s): ``"colourbar"``/``"colorbar"``/``"legend"`` strings or
        :class:`ggplot2_py.guide.Guide` instances (per-aesthetic or broadcast).
    extra : dict
        Extra keyword arguments forwarded to :func:`continuous_scale`, each value
        possibly a per-aesthetic ``list``.

    Returns
    -------
    list of ggplot2_py.scale.ScaleContinuous
        One continuous gradient scale per aesthetic.
    """
    n = len(aesthetics)

    # Extra args: per-aesthetic dict, with the new-guide-system trans->transform
    # rename always applied (scale_multi.R:119-127).
    extra_args: List[Dict[str, Any]] = []
    for i in range(n):
        picked: Dict[str, Any] = {k: _pickvalue(v, i) for k, v in extra.items()}
        if "trans" in picked:
            picked["transform"] = picked.pop("trans")
        extra_args.append(picked)

    # Interpret guides (scale_multi.R:130-152): materialise a string/Guide into a
    # Guide instance whose ``available_aes`` is the single non-standard aesthetic.
    guides: List[Any] = []
    for i in range(n):
        this_guide = _pickvalue(guide, i)
        if isinstance(this_guide, str):
            # standardise_aes_names('colourbar') == standardise_aes_names('colorbar')
            # i.e. colour->color normalisation; only the bar/legend distinction
            # matters here.
            if this_guide in ("colourbar", "colorbar"):
                this_guide = _gg.guide_colourbar()
            elif this_guide == "legend":
                this_guide = _gg.guide_legend()
        if _is_guide_proto(this_guide):
            # ggproto(NULL, old, available_aes = aes): clone the instance so the
            # class default is shadowed without mutating the shared original.
            cloned = _gg.ggproto(None, this_guide)
            cloned._set(available_aes=[aesthetics[i]])
            this_guide = cloned
        else:
            cli_abort(
                "`ggh4x`'s author hasn't programmed this path yet. "
                "Choose a legend or colourbar guide."
            )
        guides.append(this_guide)

    # Interpret scales (scale_multi.R:155-169).
    out: List[Any] = []
    for i in range(n):
        aes_i = aesthetics[i]
        palette = _scales.pal_gradient_n(
            colours=_pickvalue(colours, i),
            values=_pickvalue(values, i),
        )
        kwargs: Dict[str, Any] = dict(extra_args[i])
        sc = continuous_scale(
            aesthetics=aes_i,
            palette=palette,
            na_value=_pickvalue(na_value, i),
            guide=guides[i],
            **kwargs,
        )
        out.append(sc)
    return out


def _is_guide_proto(obj: Any) -> bool:
    """Return ``True`` when *obj* is a :class:`ggplot2_py.guide.Guide` instance."""
    Guide = getattr(_gg, "Guide", None)
    if Guide is None:
        return False
    try:
        return isinstance(obj, Guide)
    except TypeError:
        return False


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------
def _resolve_colours(colours: Any, colors: Any) -> Any:
    """Reproduce R's ``missing(colours)``/``missing(colors)`` default cascade.

    Port of ``scale_multi.R:53-62``: prefer ``colours``; fall back to the
    American ``colors``; default to ``["white", "black"]`` when neither is given.

    Parameters
    ----------
    colours : Any
        British-spelling argument, or :data:`_MISSING`.
    colors : Any
        American-spelling argument, or :data:`_MISSING`.

    Returns
    -------
    Any
        The resolved gradient colours.
    """
    if colours is _MISSING:
        if colors is _MISSING:
            return ["white", "black"]
        return colors
    return colours


def scale_fill_multi(
    *,
    colours: Any = _MISSING,
    values: Any = None,
    na_value: str = "transparent",
    guide: Any = "colourbar",
    aesthetics: Any = "fill",
    colors: Any = _MISSING,
    **kwargs: Any,
) -> MultiScale:
    """Map multiple non-standard fill aesthetics to multiple gradient scales.

    Port of R ``scale_fill_multi`` (``scale_multi.R:48-76``).  Distributes listed
    arguments across one :func:`ggplot2_py.continuous_scale` gradient per
    aesthetic and wraps the result in a :class:`MultiScale` container.

    This should only be added to a plot **after** every layer it affects has been
    added, since :class:`MultiScale` rewrites those layers' geoms at ``+``-time.

    Parameters
    ----------
    colours : Any, optional
        Gradient colours.  A ``list`` is per-aesthetic (e.g.
        ``[["white", "red"], ["black", "blue"]]``); a bare vector is broadcast.
        Defaults to ``["white", "black"]``.
    values : Any, optional
        Gradient ``values`` positions (per-aesthetic ``list`` or broadcast).
    na_value : str, default ``"transparent"``
        Colour for missing values (per-aesthetic ``list`` or broadcast).
    guide : Any, default ``"colourbar"``
        Guide spec(s): ``"colourbar"``/``"colorbar"``/``"legend"`` or
        :class:`ggplot2_py.guide.Guide` instances (per-aesthetic or broadcast).
    aesthetics : str or list of str, default ``"fill"``
        Non-standard aesthetic name(s) to map.
    colors : Any, optional
        American-spelling alias for *colours*.
    **kwargs : Any
        Extra arguments forwarded to :func:`continuous_scale` (each may be a
        per-aesthetic ``list``).

    Returns
    -------
    MultiScale
        A deferred-mutation container of class ``MultiScale``.
    """
    colours = _resolve_colours(colours, colors)
    aes_list = [aesthetics] if isinstance(aesthetics, str) else list(aesthetics)
    scales = _distribute_scale_multi(
        aesthetics=aes_list,
        colours=colours,
        values=values,
        na_value=na_value,
        guide=guide,
        extra=kwargs,
    )
    return MultiScale(
        scales=scales,
        aes=aes_list,
        replaced_aes=standardise_aes_names(["fill"])[0],
    )


def scale_colour_multi(
    *,
    colours: Any = _MISSING,
    values: Any = None,
    na_value: str = "transparent",
    guide: Any = "colourbar",
    aesthetics: Any = "colour",
    colors: Any = _MISSING,
    **kwargs: Any,
) -> MultiScale:
    """Map multiple non-standard colour aesthetics to multiple gradient scales.

    Port of R ``scale_colour_multi`` (``scale_multi.R:80-110``).  Behaves exactly
    like :func:`scale_fill_multi` but defaults ``aesthetics`` to ``"colour"`` and
    sets ``replaced_aes`` to ``"colour"``.

    Parameters
    ----------
    colours : Any, optional
        Gradient colours (per-aesthetic ``list`` or broadcast vector).  Defaults
        to ``["white", "black"]``.
    values : Any, optional
        Gradient ``values`` positions (per-aesthetic ``list`` or broadcast).
    na_value : str, default ``"transparent"``
        Colour for missing values (per-aesthetic ``list`` or broadcast).
    guide : Any, default ``"colourbar"``
        Guide spec(s) (per-aesthetic or broadcast).
    aesthetics : str or list of str, default ``"colour"``
        Non-standard aesthetic name(s) to map.
    colors : Any, optional
        American-spelling alias for *colours*.
    **kwargs : Any
        Extra arguments forwarded to :func:`continuous_scale`.

    Returns
    -------
    MultiScale
        A deferred-mutation container of class ``MultiScale``.
    """
    colours = _resolve_colours(colours, colors)
    aes_list = [aesthetics] if isinstance(aesthetics, str) else list(aesthetics)
    scales = _distribute_scale_multi(
        aesthetics=aes_list,
        colours=colours,
        values=values,
        na_value=na_value,
        guide=guide,
        extra=kwargs,
    )
    return MultiScale(
        scales=scales,
        aes=aes_list,
        replaced_aes=standardise_aes_names(["colour"])[0],
    )


def scale_color_multi(
    *,
    colours: Any = _MISSING,
    values: Any = None,
    na_value: str = "transparent",
    guide: Any = "colourbar",
    aesthetics: Any = "colour",
    colors: Any = _MISSING,
    **kwargs: Any,
) -> MultiScale:
    """American-spelling alias for :func:`scale_colour_multi`.

    See :func:`scale_colour_multi` for the full parameter description.

    Returns
    -------
    MultiScale
    """
    return scale_colour_multi(
        colours=colours,
        values=values,
        na_value=na_value,
        guide=guide,
        aesthetics=aesthetics,
        colors=colors,
        **kwargs,
    )
