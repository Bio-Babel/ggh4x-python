"""Per-panel position scales (port of ggh4x ``R/facetted_pos_scales.R``).

:func:`facetted_pos_scales` returns a :class:`FacettedPosScales` add-on object.
When added to a plot, its handler clones the plot's live facet into a dynamic
``FreeScaled<FacetClass>`` subclass whose ``init_scales`` / ``train_scales`` /
``finish_data`` are replaced with per-panel variants: each ``SCALE_X`` /
``SCALE_Y`` id gets its own cloned scale, user scales (with ``oob`` forced to
:func:`scales.oob_keep`) substituted at matched panels, and layer data
transformed per panel before training.

NSE deviation
-------------
R accepts a list of two-sided formulas whose LHS is tidy-evaluated against the
plot layout.  Python has no NSE: instead an element may be a *position scale*, a
``None``, or a ``(predicate, scale)`` pair where ``predicate`` is either a
callable ``layout_df -> bool-array`` or a string evaluated with
:meth:`pandas.DataFrame.eval` over the layout columns.  The predicate list is
stored parallel to the scale list (R smuggles it via ``attr(., "lhs")``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scales import oob_keep

from ggplot2_py import ggproto
from ggplot2_py.facet import Facet
from ggplot2_py.ggproto import ggproto_parent
from ggplot2_py.plot import update_ggplot

from ggh4x._cli import cli_abort, cli_warn

__all__ = [
    "facetted_pos_scales",
    "FacettedPosScales",
    "check_facetted_scale",
    "validate_facetted_scale",
    "init_scale",
    "init_scales_individual",
    "train_scales_individual",
    "finish_data_individual",
    "should_transform",
]

# Facet class-name prefixes recognised as "known" (allowlist replacing R's
# body-identity comparison of init/train/finish; see panel_scales.md risk 8).
_KNOWN_FACET_PREFIXES = (
    "FacetGrid",
    "FacetWrap",
    "FacetNull",
    "FacetManual",
    "FreeScaled",
    "Forced",
)


# ---------------------------------------------------------------------------
# Predicate evaluation (NSE replacement)
# ---------------------------------------------------------------------------
def _eval_predicate(pred: Any, layout: pd.DataFrame) -> np.ndarray:
    """Evaluate a panel predicate against the layout, returning a bool array.

    The predicate may be a callable ``layout -> array-like`` or a string
    expression evaluated with :meth:`pandas.DataFrame.eval`.  This stands in for
    R's tidy-evaluation of a formula LHS against the layout (``eval_tidy``).

    Parameters
    ----------
    pred : callable or str
        The panel predicate.
    layout : pandas.DataFrame
        The plot layout (columns ``PANEL`` / ``ROW`` / ``COL`` / ``SCALE_*`` +
        facet variables).

    Returns
    -------
    numpy.ndarray
        A boolean array, recycled to ``len(layout)``.
    """
    if callable(pred):
        res = pred(layout)
    elif isinstance(pred, str):
        res = layout.eval(pred, engine="python")
    else:
        res = pred
    arr = np.asarray(res)
    if arr.dtype != bool:
        arr = arr.astype(bool)
    n = len(layout)
    if arr.ndim == 0:
        arr = np.repeat(arr, n)
    if len(arr) != n:
        # rep_len recycling
        arr = np.resize(arr, n)
    return arr


# ---------------------------------------------------------------------------
# ScaleList: a list carrying a parallel ``lhs`` predicate list (R attr "lhs")
# ---------------------------------------------------------------------------
class _ScaleList(list):
    """A ``list`` of scales carrying an optional parallel ``lhs`` predicate list.

    Stands in for R's ``structure(rhs, lhs = lhs, class = "list")``: a plain
    list of scales (or ``None`` s) where ``self.lhs`` -- when not ``None`` --
    holds one predicate per element (the formula LHS equivalent).
    """

    lhs: Optional[List[Any]] = None

    def __init__(self, iterable: Sequence[Any] = (), lhs: Optional[List[Any]] = None) -> None:
        super().__init__(iterable)
        self.lhs = lhs


# ---------------------------------------------------------------------------
# check_facetted_scale (R facetted_pos_scales.R:115-145)
# ---------------------------------------------------------------------------
def _is_scale(x: Any) -> bool:
    """Return whether *x* looks like a ggplot2 Scale (has ``aesthetics``)."""
    return x is not None and hasattr(x, "aesthetics") and hasattr(x, "clone")


def _is_formula_pair(x: Any) -> bool:
    """Return whether *x* is a ``(predicate, scale)`` pair (formula equivalent)."""
    return (
        isinstance(x, (tuple, list))
        and len(x) == 2
        and (callable(x[0]) or isinstance(x[0], str))
        and _is_scale(x[1])
    )


def check_facetted_scale(x: Optional[Sequence[Any]], aes: str = "x", allow_null: bool = True) -> bool:
    """Validate that *x* is a list of position scales / ``None`` s / formula pairs.

    Faithful port of ggh4x's ``check_facetted_scale``
    (``R/facetted_pos_scales.R:115-145``).  Each element must be a position
    :class:`~ggplot2_py.scale.Scale` carrying the *aes* aesthetic, a ``None``
    (when ``allow_null``), or -- if *all* elements are formula pairs -- a
    ``(predicate, scale)`` pair.

    Parameters
    ----------
    x : sequence or None
        The candidate scale list.
    aes : {"x", "y"}, default "x"
        Required aesthetic.
    allow_null : bool, default True
        Whether ``None`` elements are permitted.

    Returns
    -------
    bool
        ``True`` when *x* is a valid facetted-scale list.
    """
    if x is None:
        return True

    is_scale = [_is_scale(e) for e in x]
    is_null = [e is None for e in x]
    is_form = [_is_formula_pair(e) for e in x]

    if x and all(is_form):
        return True

    # Scales must carry the right aesthetic.
    appropriate = [
        (aes in list(e.aesthetics)) for e, s in zip(x, is_scale) if s
    ]
    # is_scale[is_scale] <- is_scale[is_scale] & appropriate_aes
    ai = 0
    for i, s in enumerate(is_scale):
        if s:
            is_scale[i] = s and appropriate[ai]
            ai += 1

    if allow_null:
        if all(s or n for s, n in zip(is_scale, is_null)):
            return True
    else:
        if all(is_scale):
            return True
    return False


# ---------------------------------------------------------------------------
# validate_facetted_scale (R facetted_pos_scales.R:149-177)
# ---------------------------------------------------------------------------
def validate_facetted_scale(x: Sequence[Any], aes: str = "x") -> _ScaleList:
    """Split formula pairs into a scale list + parallel predicate (``lhs``) list.

    Faithful port of ggh4x's ``validate_facetted_scale``
    (``R/facetted_pos_scales.R:149-177``).  When *x*'s first element is not a
    formula pair the list is returned as-is.  Otherwise each ``(predicate,
    scale)`` pair is split: the predicate ``lhs`` is kept for later layout
    evaluation, the scale ``rhs`` is validated for the *aes* aesthetic.

    Parameters
    ----------
    x : sequence
        The candidate (possibly formula-pair) list.
    aes : {"x", "y"}, default "x"
        Required aesthetic.

    Returns
    -------
    _ScaleList
        A scale list, with ``.lhs`` set to the parallel predicate list when *x*
        was formula-based (else ``.lhs is None``).

    Raises
    ------
    ValueError
        When a formula pair's right-hand side is not an appropriate scale.
    """
    if not x or not _is_formula_pair(x[0]):
        return _ScaleList(x, lhs=None)

    lhs = [f[0] for f in x]
    rhs = [f[1] for f in x]

    if not check_facetted_scale(rhs, aes=aes, allow_null=False):
        cli_abort(
            "The right-hand side of formula does not result in an appropriate scale."
        )
    return _ScaleList(rhs, lhs=lhs)


# ---------------------------------------------------------------------------
# facetted_pos_scales constructor (R facetted_pos_scales.R:79-112)
# ---------------------------------------------------------------------------
class FacettedPosScales:
    """Deferred container of per-panel x / y position scales.

    Port of R's ``structure(list(x =, y =), class = "facetted_pos_scales")``.
    Consumed by :func:`_update_facetted_pos_scales` at ``+``-time.

    Attributes
    ----------
    x, y : _ScaleList
        Per-panel x / y scale lists (each possibly carrying a ``.lhs`` predicate
        list).
    """

    def __init__(self, x: _ScaleList, y: _ScaleList) -> None:
        self.x = x
        self.y = y


def facetted_pos_scales(
    x: Optional[Union[Sequence[Any], Any]] = None,
    y: Optional[Union[Sequence[Any], Any]] = None,
) -> FacettedPosScales:
    """Set individual position scales in facets.

    Faithful port of ggh4x's ``facetted_pos_scales``
    (``R/facetted_pos_scales.R:79-112``).  ``x`` / ``y`` are lists whose elements
    are position scales, ``None`` s (use the default scale at that position), or
    ``(predicate, scale)`` pairs targeting panels by predicate.  The facet must
    use free scales in the relevant direction.

    Parameters
    ----------
    x, y : list or None, default None
        Per-panel x / y position scales (or a single element, auto-wrapped).

    Returns
    -------
    FacettedPosScales
        An add-on object that can be added to a plot with ``+``.

    Raises
    ------
    ValueError
        When ``x`` or ``y`` is not a valid facetted-scale list.
    """
    if not isinstance(x, list):
        x = [x]
    if not isinstance(y, list):
        y = [y]

    x_test = check_facetted_scale(x, "x")
    y_test = check_facetted_scale(y, "y")
    if not (x_test and y_test):
        if not x_test and not y_test:
            arg, typ = "The `x` and `y` arguments ", "appropriate"
        elif not x_test:
            arg, typ = "The `x` argument ", "x"
        else:
            arg, typ = "The `y` argument ", "y"
        cli_abort(
            arg
            + "should be `None`, or a list of formulas and/or position scales "
            + f"with the {typ} aesthetic."
        )

    x = validate_facetted_scale(x, "x")
    y = validate_facetted_scale(y, "y")
    return FacettedPosScales(x=x, y=y)


# ---------------------------------------------------------------------------
# ggproto methods: init_scale / init_scales_individual (R:255-319)
# ---------------------------------------------------------------------------
def init_scale(
    old: Any,
    new: Optional[Sequence[Any]],
    layout: pd.DataFrame,
    aes: str = "x",
) -> Optional[List[Any]]:
    """Build the per-panel scale list for one aesthetic.

    Faithful port of ggh4x's ``init_scale`` (``R/facetted_pos_scales.R:255-305``).
    Clones the default *old* scale once per ``SCALE_<AES>`` id, then substitutes
    user scales (with ``oob`` forced to :func:`scales.oob_keep`).  Without
    predicates, substitution is by list position; with predicates, panels are
    matched by evaluating each predicate against *layout* and substitution
    proceeds in reverse order so earlier-added scales win.

    Parameters
    ----------
    old : Scale or None
        The default prototype scale (``None`` -> returns ``None``).
    new : sequence or _ScaleList or None
        The user scale list (possibly carrying a ``.lhs`` predicate list).
    layout : pandas.DataFrame
        The plot layout.
    aes : {"x", "y"}, default "x"
        The aesthetic.

    Returns
    -------
    list or None
        One scale per ``SCALE_<AES>`` id, or ``None`` when *old* is ``None``.
    """
    if old is None:
        return None

    scalename = "SCALE_" + aes.upper()
    n_ids = int(layout[scalename].max())
    out: List[Any] = [old.clone() for _ in range(n_ids)]

    lhs = getattr(new, "lhs", None)
    if lhs is None:
        # Regular: substitute at positions with a non-empty user scale.
        for i, sc in enumerate(new or []):
            if sc is None:
                continue
            clone = sc.clone()
            clone.oob = oob_keep
            if i < len(out):
                out[i] = clone
    else:
        n = len(layout)
        # Evaluate each predicate -> column of a logical matrix.
        cols = [_eval_predicate(p, layout) for p in lhs]
        for i in reversed(range(len(cols))):
            mask = cols[i]
            matched_rows = np.where(mask)[0]
            # unique SCALE ids among matched layout rows
            scale_ids = pd.unique(layout.iloc[matched_rows][scalename])
            for sid in scale_ids:
                clone = new[i].clone()
                clone.oob = oob_keep
                out[int(sid) - 1] = clone
    return out


def init_scales_individual(
    self: Any,
    layout: pd.DataFrame,
    x_scale: Any = None,
    y_scale: Any = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, list]:
    """Per-panel ``init_scales`` (R ``init_scales_individual``).

    Faithful port of ggh4x's ``init_scales_individual``
    (``R/facetted_pos_scales.R:308-319``).  Because ggplot2_py's layout calls
    ``init_scales`` twice (x-only, then y-only), each aesthetic is guarded on
    ``is not None`` so only the populated key is returned.

    Parameters
    ----------
    self : Facet
        The ``FreeScaled<...>`` facet instance (carries ``new_x_scales`` /
        ``new_y_scales``).
    layout : pandas.DataFrame
    x_scale, y_scale : Scale or None
        Prototype position scales (one provided per call).
    params : dict, optional

    Returns
    -------
    dict
        ``{"x": [...]}`` or ``{"y": [...]}`` -- only the populated aesthetic.
    """
    scales: Dict[str, list] = {}
    if x_scale is not None:
        res = init_scale(x_scale, self.new_x_scales, layout, aes="x")
        if res is not None:
            scales["x"] = res
    if y_scale is not None:
        res = init_scale(y_scale, self.new_y_scales, layout, aes="y")
        if res is not None:
            scales["y"] = res
    return scales


def train_scales_individual(
    self: Any,
    x_scales: list,
    y_scales: list,
    layout: pd.DataFrame,
    data: List[pd.DataFrame],
    params: Optional[Dict[str, Any]] = None,
) -> None:
    """Per-panel ``train_scales`` (R ``train_scales_individual``).

    Faithful port of ggh4x's ``train_scales_individual``
    (``R/facetted_pos_scales.R:322-332``).  Transforms each layer's data through
    :func:`finish_data_individual` *first* (so per-panel transforms precede
    training), then delegates to the parent :class:`~ggplot2_py.facet.Facet`'s
    ``train_scales``.

    Parameters
    ----------
    self : Facet
    x_scales, y_scales : list
    layout : pandas.DataFrame
    data : list of DataFrame
    params : dict, optional
    """
    data = [
        self.finish_data(ld, layout, x_scales, y_scales, params)
        for ld in data
    ]
    ggproto_parent(Facet, self).train_scales(
        x_scales, y_scales, layout, data, params
    )


def finish_data_individual(
    self: Any,
    data: pd.DataFrame,
    layout: pd.DataFrame,
    x_scales: list,
    y_scales: list,
    params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Per-panel ``finish_data`` (R ``finish_data_individual``).

    Faithful port of ggh4x's ``finish_data_individual``
    (``R/facetted_pos_scales.R:335-368``).  Splits *data* by ``PANEL`` (keeping
    the exact input row positions), matches each chunk to its ``SCALE_X`` /
    ``SCALE_Y`` ids, transforms the appropriate columns through the panel's
    scales, and recombines preserving the original row order.

    Parameters
    ----------
    self : Facet
        Carries ``new_x_scales`` / ``new_y_scales``.
    data : pandas.DataFrame
    layout : pandas.DataFrame
    x_scales, y_scales : list
    params : dict, optional

    Returns
    -------
    pandas.DataFrame
        *data* with per-panel-transformed position columns, original order.
    """
    if data is None or len(data) == 0 or "PANEL" not in data.columns:
        return data

    regular_x = _scalelist_len(self.new_x_scales) == 0
    regular_y = _scalelist_len(self.new_y_scales) == 0

    # Split by PANEL preserving positional indices.
    groups = data.groupby("PANEL", observed=True).indices  # PANEL -> int positions

    panel_codes = layout["PANEL"]
    # Numeric codes of layout PANEL for matching (R: match(as.numeric(...), layout$PANEL)).
    if isinstance(panel_codes.dtype, pd.CategoricalDtype):
        layout_panel_num = panel_codes.cat.codes.to_numpy() + 1
    else:
        layout_panel_num = pd.to_numeric(panel_codes, errors="coerce").to_numpy()

    out = data.copy()

    for panel_val, idx in groups.items():
        if len(idx) == 0:
            continue
        # numeric code of this panel
        if isinstance(data["PANEL"].dtype, pd.CategoricalDtype):
            cats = list(data["PANEL"].cat.categories)
            panel_num = cats.index(panel_val) + 1 if panel_val in cats else None
            try:
                panel_num = int(panel_val)
            except (TypeError, ValueError):
                panel_num = cats.index(panel_val) + 1 if panel_val in cats else None
        else:
            panel_num = int(panel_val)

        matches = np.where(layout_panel_num == panel_num)[0]
        if len(matches) == 0:
            continue
        panel_id = matches[0]
        xidx = int(layout.iloc[panel_id]["SCALE_X"]) - 1
        yidx = int(layout.iloc[panel_id]["SCALE_Y"]) - 1

        chunk = data.iloc[idx]
        y_vars = should_transform(
            y_scales[yidx] if 0 <= yidx < len(y_scales) else None,
            list(chunk.columns),
        )
        x_vars = should_transform(
            x_scales[xidx] if 0 <= xidx < len(x_scales) else None,
            list(chunk.columns),
        )
        if regular_x:
            x_vars = []
        if regular_y:
            y_vars = []

        for j in y_vars:
            out.iloc[idx, out.columns.get_loc(j)] = y_scales[yidx].transform(
                chunk[j].to_numpy()
            )
        for j in x_vars:
            out.iloc[idx, out.columns.get_loc(j)] = x_scales[xidx].transform(
                chunk[j].to_numpy()
            )
    return out


def _scalelist_len(scales: Optional[Sequence[Any]]) -> int:
    """Return ``sum(lengths(scales))``: count of non-``None`` scale elements."""
    if scales is None:
        return 0
    return sum(1 for s in scales if s is not None)


def should_transform(scale: Any, columns: Sequence[str]) -> List[str]:
    """Return the columns to transform for a panel's scale.

    Faithful port of ggh4x's ``should_transform``
    (``R/facetted_pos_scales.R:370-378``): no columns for a ``None`` scale, a
    discrete scale, or a date/time/hms transformation; otherwise the
    intersection of the scale's aesthetics with *columns*.

    Parameters
    ----------
    scale : Scale or None
    columns : sequence of str

    Returns
    -------
    list of str
    """
    if scale is None or scale.is_discrete():
        return []
    trans = _get_transformation(scale)
    name = getattr(trans, "name", None)
    if name in ("date", "time", "hms"):
        return []
    return [c for c in scale.aesthetics if c in columns]


def _get_transformation(scale: Any) -> Any:
    """Return a scale's transformation object (ggh4x ``get_transformation``)."""
    if hasattr(scale, "get_transformation"):
        return scale.get_transformation()
    return getattr(scale, "trans", None)


# ---------------------------------------------------------------------------
# ggplot_add.facetted_pos_scales (R facetted_pos_scales.R:186-250)
# ---------------------------------------------------------------------------
@update_ggplot.register(FacettedPosScales)
def _update_facetted_pos_scales(obj: FacettedPosScales, plot: Any, object_name: str = "") -> Any:
    """Add a :class:`FacettedPosScales` to *plot* (R ``ggplot_add.facetted_pos_scales``).

    Clones the plot's facet into a ``FreeScaled<FacetClass>`` whose
    ``init_scales`` / ``train_scales`` / ``finish_data`` are the per-panel
    variants; re-additions onto an already-``FreeScaled`` facet just update the
    new-scale lists.

    Parameters
    ----------
    obj : FacettedPosScales
    plot : ggplot2_py.plot.GGPlot
    object_name : str, optional

    Returns
    -------
    ggplot2_py.plot.GGPlot
    """
    empty_x = [e is None for e in obj.x]
    empty_y = [e is None for e in obj.y]
    if all(empty_x) and all(empty_y):
        return plot

    facet = plot.facet
    if type(facet).__name__.startswith("FreeScaled"):
        # Already initialised; just update scale lists.
        if not all(empty_x):
            facet.new_x_scales = obj.x
        if not all(empty_y):
            facet.new_y_scales = obj.y
        return plot

    # Validity warning (allowlist replaces R body-identity check).
    if not type(facet).__name__.startswith(_KNOWN_FACET_PREFIXES):
        cli_warn(
            f"Unknown facet: {type(facet).__name__}. "
            "Overriding facetted scales may be unstable."
        )

    free = facet.params.get("free") if facet.params else None
    if free is not None:
        if free.get("x") is not None and sum(not e for e in empty_x) > 0 and not free["x"]:
            cli_warn(
                "Attempting to add facetted x scales, while x scales are not free. "
                'Try adding `scales = "free_x"` to the facet.'
            )
        if free.get("y") is not None and sum(not e for e in empty_y) > 0 and not free["y"]:
            cli_warn(
                "Attempting to add facetted y scales, while y scales are not free. "
                'Try adding `scales = "free_y"` to the facet.'
            )

    new_facet = ggproto(
        f"FreeScaled{type(facet).__name__}",
        facet,
        new_x_scales=obj.x,
        new_y_scales=obj.y,
        init_scales=init_scales_individual,
        train_scales=train_scales_individual,
        finish_data=finish_data_individual,
    )
    plot.facet = new_facet
    return plot
