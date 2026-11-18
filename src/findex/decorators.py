"""Декоратори для профілювання часу виконання операцій індексації та пошуку."""

from __future__ import annotations

from collections.abc import Callable
from functools import update_wrapper
import logging
import time
from typing import Generic, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger("findex.timing")


class TimedCallable(Generic[P, R]):
    """Типізована обгортка функції, що заміряє та логує час виконання без # type: ignore."""

    def __init__(self, func: Callable[P, R]) -> None:
        self._func: Callable[P, R] = func
        self.last_elapsed_ms: float = 0.0
        update_wrapper(self, func)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        t_start = time.perf_counter()
        try:
            return self._func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            self.last_elapsed_ms = elapsed_ms
            logger.info("[timed] %s виконано за %.3f мс", self.__qualname__, elapsed_ms)


def timed(func: Callable[P, R]) -> TimedCallable[P, R]:
    """Декоратор вимірювання часу виконання з повним збереженням сигнатури типів."""
    return TimedCallable(func)
