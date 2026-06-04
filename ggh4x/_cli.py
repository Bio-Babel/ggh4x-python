"""cli message shims (R source: cli package usage in ggh4x).

ggh4x calls ``cli::cli_abort`` / ``cli::cli_warn`` / ``cli::cli_inform`` for user-facing
messages. The Python port maps these to standard exceptions / ``warnings`` so failures stay
loud (per the error-handling discipline) while stripping cli's ``{.arg}`` glue markup.
"""

from __future__ import annotations

import re
import warnings
from typing import NoReturn, Type

__all__ = ["cli_abort", "cli_warn", "cli_inform", "strip_cli_markup"]

# cli inline-markup spans like {.arg foo}, {.code x}, {.field y}, {.val 3}, {.cls C}.
_CLI_SPAN = re.compile(r"\{\.[a-zA-Z_]+\s+([^{}]*)\}")
# Leftover interpolation braces {x} -> x (we cannot evaluate R glue, just unwrap).
_CLI_BRACE = re.compile(r"\{([^{}]*)\}")


def strip_cli_markup(message: str) -> str:
    """Remove cli inline-markup so a plain message remains.

    Parameters
    ----------
    message : str
        A message possibly containing cli markup such as ``{.arg x}``.

    Returns
    -------
    str
        The message with markup spans replaced by their content.
    """
    prev = None
    out = message
    while prev != out:
        prev = out
        out = _CLI_SPAN.sub(r"\1", out)
    out = _CLI_BRACE.sub(r"\1", out)
    return out


def cli_abort(message: str, error_class: Type[Exception] = ValueError) -> NoReturn:
    """Raise an exception, mirroring ``cli::cli_abort``.

    Parameters
    ----------
    message : str
        Error message (cli markup is stripped).
    error_class : type[Exception]
        Exception type to raise (default ``ValueError``; pass ``TypeError`` where the R
        error is about an input type).

    Raises
    ------
    Exception
        Always raises *error_class*.
    """
    raise error_class(strip_cli_markup(message))


def cli_warn(message: str, category: Type[Warning] = UserWarning) -> None:
    """Emit a warning, mirroring ``cli::cli_warn``.

    Parameters
    ----------
    message : str
        Warning message (cli markup is stripped).
    category : type[Warning]
        Warning category (default ``UserWarning``).
    """
    warnings.warn(strip_cli_markup(message), category, stacklevel=2)


def cli_inform(message: str) -> None:
    """Print an informational message, mirroring ``cli::cli_inform``.

    Parameters
    ----------
    message : str
        Message to print (cli markup is stripped).
    """
    print(strip_cli_markup(message))
