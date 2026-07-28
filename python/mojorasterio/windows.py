"""The useful, I/O-independent subset of :mod:`rasterio.windows`."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor


@dataclass(frozen=True)
class Window:
    col_off: float
    row_off: float
    width: float
    height: float

    def __post_init__(self):
        if self.width < 0 or self.height < 0:
            raise ValueError("Number of columns or rows must be non-negative")

    @classmethod
    def from_slices(
        cls,
        rows,
        cols,
        height: int | None = None,
        width: int | None = None,
        boundless: bool = False,
    ):
        row_start, row_stop = _slice_bounds(rows, height, boundless)
        col_start, col_stop = _slice_bounds(cols, width, boundless)
        return cls(col_start, row_start, col_stop - col_start, row_stop - row_start)

    def flatten(self):
        return self.col_off, self.row_off, self.width, self.height

    def toslices(self):
        row_start = max(0, floor(self.row_off))
        row_stop = max(0, ceil(self.row_off + self.height))
        col_start = max(0, floor(self.col_off))
        col_stop = max(0, ceil(self.col_off + self.width))
        return slice(row_start, row_stop), slice(col_start, col_stop)

    def round_lengths(self, **kwargs):
        return Window(self.col_off, self.row_off, round(self.width), round(self.height))

    def round_offsets(self, **kwargs):
        return Window(
            floor(self.col_off), floor(self.row_off), self.width, self.height
        )

    def crop(self, height, width):
        return crop(self, height, width)

    def intersection(self, other):
        return intersection(self, other)


def _slice_bounds(value, size, boundless):
    if isinstance(value, slice):
        start, stop = value.start, value.stop
    elif isinstance(value, tuple) and len(value) == 2:
        start, stop = value
    else:
        raise WindowError("rows and cols must be slices or 2-item tuples")
    start = 0 if start is None else start
    if stop is None:
        if size is None:
            raise WindowError("height and width are required for open-ended slices")
        stop = size
    if start < 0 and not boundless:
        if size is None:
            raise WindowError("height and width are required for negative indexes")
        start += size
    if stop < 0 and not boundless:
        if size is None:
            raise WindowError("height and width are required for negative indexes")
        stop += size
    return start, max(start, stop)


class WindowError(ValueError):
    pass


def evaluate(window, height, width, boundless=False):
    if isinstance(window, Window):
        return window
    if (
        isinstance(window, tuple)
        and len(window) == 2
        and all(isinstance(item, (tuple, slice)) for item in window)
    ):
        return Window.from_slices(
            window[0],
            window[1],
            height=height,
            width=width,
            boundless=boundless,
        )
    raise WindowError("window must be a Window or ((row_start, row_stop), (col_start, col_stop))")


def shape(window, height=-1, width=-1):
    win = evaluate(window, height, width) if not isinstance(window, Window) else window
    return win.height, win.width


def crop(window, height, width):
    win = evaluate(window, height, width)
    row_start = min(max(win.row_off, 0), height)
    col_start = min(max(win.col_off, 0), width)
    row_stop = min(max(win.row_off + win.height, 0), height)
    col_stop = min(max(win.col_off + win.width, 0), width)
    return Window(
        col_start, row_start, max(col_stop - col_start, 0), max(row_stop - row_start, 0)
    )


def intersection(*windows):
    if not windows:
        raise WindowError("at least one window is required")
    wins = [w if isinstance(w, Window) else Window(*w) for w in windows]
    col_start = max(w.col_off for w in wins)
    row_start = max(w.row_off for w in wins)
    col_stop = min(w.col_off + w.width for w in wins)
    row_stop = min(w.row_off + w.height for w in wins)
    if col_stop <= col_start or row_stop <= row_start:
        raise WindowError("Intersection is empty")
    return Window(col_start, row_start, col_stop - col_start, row_stop - row_start)


def union(*windows):
    if not windows:
        raise WindowError("at least one window is required")
    wins = [w if isinstance(w, Window) else Window(*w) for w in windows]
    col_start = min(w.col_off for w in wins)
    row_start = min(w.row_off for w in wins)
    col_stop = max(w.col_off + w.width for w in wins)
    row_stop = max(w.row_off + w.height for w in wins)
    return Window(col_start, row_start, col_stop - col_start, row_stop - row_start)


def bounds(window, transform):
    win = window if isinstance(window, Window) else Window(*window)
    left, bottom = transform * (win.col_off, win.row_off + win.height)
    right, top = transform * (win.col_off + win.width, win.row_off)
    return left, bottom, right, top
