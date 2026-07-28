"""Compute-focused Rasterio subset powered by Mojo."""

from __future__ import annotations

from . import band_math, windows
from .enums import Resampling
from .io import DatasetReader, DatasetWriter
from .windows import Window

__version__ = "0.1.0"


def open(data, mode="r", *, nodata=None, **kwargs):
    """Open an in-memory array.

    Filesystem paths and GDAL creation options are intentionally not covered.
    """
    if isinstance(data, (str, bytes)):
        raise NotImplementedError(
            "file-backed I/O is not covered; pass a NumPy-compatible array"
        )
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unsupported open options: {unknown}")
    if mode in ("r", "rb"):
        return DatasetReader(data, nodata=nodata)
    if mode in ("w", "w+"):
        return DatasetWriter(data, nodata=nodata)
    raise ValueError(f"unsupported mode: {mode!r}")


__all__ = [
    "DatasetReader",
    "DatasetWriter",
    "Resampling",
    "Window",
    "band_math",
    "open",
    "windows",
]
