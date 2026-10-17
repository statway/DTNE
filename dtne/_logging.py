"""Small, notebook-friendly progress helpers controlled by verbosity."""

from contextlib import contextmanager
from time import perf_counter


def log(verbose, message, *, level=1):
    """Emit one immediately flushed line without configuring global logging."""
    if verbose >= level:
        print(f"[DTNE] {message}", flush=True)


@contextmanager
def stage(verbose, name, **details):
    """Report stage boundaries and elapsed time, preserving any exception."""
    if verbose < 1:
        yield
        return
    suffix = ", ".join(f"{key}={value}" for key, value in details.items())
    started = perf_counter()
    log(verbose, f"{name} started" + (f": {suffix}" if suffix else ""))
    try:
        yield
    except Exception as exc:
        log(verbose, f"{name} failed: elapsed={perf_counter() - started:.2f}s, error={type(exc).__name__}")
        raise
    else:
        log(verbose, f"{name} completed: elapsed={perf_counter() - started:.2f}s")


class Progress:
    """Throttle detailed batch progress by time and always report completion."""

    def __init__(self, verbose, name, total, *, interval=5.0):
        self.verbose = verbose
        self.name = name
        self.total = total
        self.interval = interval
        self.last_report = perf_counter() if verbose >= 2 else None

    def update(self, processed):
        if self.verbose < 2:
            return
        now = perf_counter()
        if processed >= self.total or now - self.last_report >= self.interval:
            log(self.verbose, f"{self.name}: processed={processed}/{self.total}", level=2)
            self.last_report = now
