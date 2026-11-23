"""Typed wrapper around numba.njit.

numba ships no type stubs, so a bare ``@numba.njit`` erases the decorated function's signature for
mypy. ``njit`` returns the function with its annotations intact while still compiling it.
Leaf kernels are cached on disk (``cache=True``) so worker processes do not recompile them.
Kernels that take other jitted functions as arguments (the estimators, which receive an
environment's step/reset) use ``njit_generic``, which compiles in memory only: numba's cache index
pickles the argument function types, and re-saving that index from parallel worker processes fails
("underlying object has vanished") once several environments share one kernel.
"""

from collections.abc import Callable
from typing import Any, cast

import numba


def njit[F: Callable[..., Any]](function: F) -> F:
    """Compile ``function`` in nopython mode with an on-disk cache, keeping its static type."""
    return cast(F, numba.njit(cache=True)(function))


def njit_generic[F: Callable[..., Any]](function: F) -> F:
    """Compile ``function`` in nopython mode without a disk cache (for kernels taking jitted functions)."""
    return cast(F, numba.njit(cache=False)(function))
