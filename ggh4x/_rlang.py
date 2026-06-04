"""rlang shims (R source: rlang usage in ggh4x).

ggh4x imports a handful of rlang helpers. The non-NSE ones (``arg_match0``, ``%||%``,
``inject``/``exec`` splicing) map cleanly to Python; the NSE ones (``enquo``/``eval_tidy``)
are rewritten to standard evaluation at the call sites (documented deviations), so they are
deliberately absent here.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, TypeVar

from ._cli import cli_abort

__all__ = ["arg_match0", "value_or", "exec_call"]

T = TypeVar("T")


def value_or(x: T | None, default: T) -> T:
    """Null-coalesce, mirroring rlang's ``%||%``.

    Parameters
    ----------
    x : T | None
        Candidate value.
    default : T
        Fallback used when *x* is ``None``.

    Returns
    -------
    T
        *x* if it is not ``None``, else *default*.

    Notes
    -----
    This is a whole-object coalesce (not elementwise), matching R's ``%||%``.
    """
    return x if x is not None else default


def arg_match0(
    arg: str,
    values: Sequence[str],
    arg_name: str = "arg",
) -> str:
    """Validate a string argument against allowed choices, mirroring ``rlang::arg_match0``.

    Parameters
    ----------
    arg : str
        The supplied value.
    values : sequence of str
        The allowed values.
    arg_name : str
        Name of the argument (for the error message).

    Returns
    -------
    str
        *arg* unchanged when it is one of *values*.

    Raises
    ------
    ValueError
        If *arg* is not among *values* (message lists the valid choices, like R).
    """
    if arg in values:
        return arg
    choices = ", ".join(repr(v) for v in values)
    cli_abort(
        f"`{arg_name}` must be one of {choices}, not {arg!r}.",
    )


def exec_call(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Call *fn* with spliced args, mirroring ``rlang::exec`` / ``inject``.

    Parameters
    ----------
    fn : callable
        Function to invoke.
    *args : Any
        Positional arguments (splice a list with ``*list``).
    **kwargs : Any
        Keyword arguments (splice a dict with ``**dict``).

    Returns
    -------
    T
        The result of ``fn(*args, **kwargs)``.
    """
    return fn(*args, **kwargs)
