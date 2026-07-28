"""In-memory dataset reads with Rasterio's covered signature."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._lib import addr, lib
from .enums import Resampling
from .windows import Window, evaluate

_METHODS = {
    Resampling.nearest: 0,
    Resampling.bilinear: 1,
    Resampling.average: 2,
}


class DatasetReader:
    """A read-only, in-memory raster with a Rasterio-shaped API.

    Arrays are band-major: ``(bands, rows, columns)``. A two-dimensional array
    is treated as a one-band dataset.
    """

    def __init__(self, data, nodata=None):
        array = np.asarray(data)
        if array.ndim == 2:
            array = array[np.newaxis, ...]
        if array.ndim != 3:
            raise ValueError("data must have shape (rows, cols) or (bands, rows, cols)")
        if any(dimension == 0 for dimension in array.shape):
            raise ValueError("raster dimensions must be positive")
        if array.dtype.kind not in "iuf":
            raise TypeError("data dtype must be a real integer or floating-point type")
        self._data = np.ascontiguousarray(array)
        if nodata is not None:
            try:
                numeric_nodata = float(nodata)
                with np.errstate(over="ignore", invalid="ignore"):
                    converted_nodata = np.array(numeric_nodata, dtype=self._data.dtype).item()
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError("nodata is not representable in the data dtype") from exc
            if np.isfinite(numeric_nodata) and not np.isfinite(converted_nodata):
                raise ValueError("nodata is not representable in the data dtype")
            if np.isfinite(numeric_nodata) and converted_nodata != numeric_nodata:
                raise ValueError("nodata is not exactly representable in the data dtype")
            nodata = numeric_nodata
        self.nodata = nodata
        self.count, self.height, self.width = self._data.shape
        self.shape = self.height, self.width
        self.dtypes = tuple(str(self._data.dtype) for _ in range(self.count))
        self.closed = False

    def __enter__(self):
        if self.closed:
            raise ValueError("I/O operation on closed dataset")
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.closed = True

    def read(
        self,
        indexes=None,
        out=None,
        window=None,
        masked=False,
        out_shape=None,
        boundless=False,
        resampling=Resampling.nearest,
        fill_value=None,
        out_dtype=None,
    ):
        if self.closed:
            raise ValueError("I/O operation on closed dataset")
        if out is not None and out_shape is not None:
            raise ValueError("out and out_shape are exclusive")
        method = _coerce_resampling(resampling)
        selected, return_2d = self._bands(indexes)
        source, win = self._window(selected, window, boundless, fill_value)
        bands, source_height, source_width = source.shape
        col, row, window_width, window_height = map(int, win.flatten())

        if out is not None:
            destination = np.asarray(out)
            expected_ndim = 2 if return_2d else 3
            if destination.ndim != expected_ndim:
                raise ValueError("out has an inconsistent number of dimensions")
            if not destination.flags.writeable:
                raise ValueError("out must be writable")
            if destination.dtype.kind not in "iuf":
                raise TypeError("out dtype must be a real integer or floating-point type")
            target_height, target_width = destination.shape[-2:]
            if target_height <= 0 or target_width <= 0:
                raise ValueError("output dimensions must be positive")
            final_dtype = destination.dtype
        else:
            target_height, target_width = _output_size(
                out_shape, bands, window_height, window_width, return_2d
            )
            final_dtype = np.dtype(out_dtype or self._data.dtype)
            if final_dtype.kind not in "iuf":
                raise TypeError("out_dtype must be a real integer or floating-point type")
            destination = None

        full_window = (
            row == 0
            and col == 0
            and window_height == source_height
            and window_width == source_width
        )
        same_size = (source_height, source_width) == (target_height, target_width)
        native_dtype = np.dtype("float32") if source.dtype == np.float32 else np.dtype("float64")
        work_dtype = source.dtype if full_window and same_size else native_dtype
        source_native = np.ascontiguousarray(source, dtype=work_dtype)
        direct_destination = (
            destination is not None
            and destination.dtype == work_dtype
            and destination.flags.c_contiguous
            and not np.shares_memory(destination, source_native)
        )
        if direct_destination:
            result = destination.reshape((bands, target_height, target_width))
        else:
            result = np.empty((bands, target_height, target_width), dtype=work_dtype)
        if full_window and same_size:
            np.copyto(result, source_native)
        else:
            fn = getattr(lib(), f"mr_resample_f{native_dtype.itemsize * 8}")
            nodata = self.nodata
            has_nodata = nodata is not None
            fn(
                addr(source_native),
                addr(result),
                bands,
                source_height,
                source_width,
                row,
                col,
                window_height,
                window_width,
                target_height,
                target_width,
                _METHODS[method],
                int(has_nodata),
                float(nodata if has_nodata else 0.0),
            )

        result_view = result[0] if return_2d else result
        if destination is not None:
            if direct_destination:
                result_view = destination
            else:
                np.copyto(destination, result_view, casting="unsafe")
                result_view = destination
        elif result_view.dtype != final_dtype:
            result_view = result_view.astype(final_dtype)

        if masked:
            nodata = self.nodata
            if nodata is None:
                mask = np.zeros(result_view.shape, dtype=bool)
            elif np.isnan(nodata):
                mask = np.isnan(result_view)
            else:
                mask = result_view == nodata
            result_view = np.ma.array(result_view, mask=mask, copy=False)
            if fill_value is not None:
                result_view.fill_value = fill_value
        return result_view

    def read_masks(
        self,
        indexes=None,
        out=None,
        window=None,
        boundless=False,
        resampling=Resampling.nearest,
    ):
        values = self.read(
            indexes=indexes,
            window=window,
            boundless=boundless,
            resampling=resampling,
            masked=True,
        )
        masks = np.where(np.ma.getmaskarray(values), 0, 255).astype("uint8")
        if out is not None:
            np.copyto(out, masks)
            return out
        return masks

    def _bands(self, indexes):
        if indexes is None:
            return self._data, False
        if isinstance(indexes, (int, np.integer)):
            index = int(indexes)
            _validate_index(index, self.count)
            return self._data[index - 1 : index], True
        if not isinstance(indexes, Sequence) or isinstance(indexes, (str, bytes)):
            raise TypeError("indexes must be an int or a sequence of ints")
        indexes = [int(index) for index in indexes]
        if not indexes:
            raise ValueError("indexes must not be empty")
        for index in indexes:
            _validate_index(index, self.count)
        zero_based = [index - 1 for index in indexes]
        if zero_based == list(range(zero_based[0], zero_based[0] + len(zero_based))):
            return self._data[zero_based[0] : zero_based[-1] + 1], False
        return self._data[zero_based], False

    def _window(self, selected, window, boundless, fill_value):
        if window is None:
            return selected, Window(0, 0, self.width, self.height)
        win = evaluate(window, self.height, self.width, boundless=boundless)
        values = (win.col_off, win.row_off, win.width, win.height)
        if any(float(value) != int(value) for value in values):
            raise NotImplementedError("fractional windows are not covered")
        col, row, width, height = map(int, values)
        if width <= 0 or height <= 0:
            raise ValueError("window dimensions must be positive")
        if not boundless:
            if row < 0 or col < 0 or row + height > self.height or col + width > self.width:
                raise ValueError("window falls outside the dataset; pass boundless=True")
            return selected, win

        fill = fill_value
        if fill is None:
            fill = self.nodata if self.nodata is not None else 0
        result = np.full((selected.shape[0], height, width), fill, dtype=selected.dtype)
        source_row0, source_col0 = max(row, 0), max(col, 0)
        source_row1 = min(row + height, self.height)
        source_col1 = min(col + width, self.width)
        if source_row1 > source_row0 and source_col1 > source_col0:
            target_row, target_col = source_row0 - row, source_col0 - col
            result[
                :,
                target_row : target_row + source_row1 - source_row0,
                target_col : target_col + source_col1 - source_col0,
            ] = selected[:, source_row0:source_row1, source_col0:source_col1]
        return result, Window(0, 0, width, height)


class DatasetWriter(DatasetReader):
    def write(self, data, indexes=None, window=None):
        if self.closed:
            raise ValueError("I/O operation on closed dataset")
        source = np.asarray(data)
        if indexes is None:
            indexes = list(range(1, self.count + 1))
        elif isinstance(indexes, int):
            indexes = [indexes]
            if source.ndim == 2:
                source = source[np.newaxis, ...]
        if window is None:
            win = Window(0, 0, self.width, self.height)
        else:
            win = evaluate(window, self.height, self.width)
        if any(float(v) != int(v) for v in win.flatten()):
            raise NotImplementedError("fractional windows are not covered")
        col, row, width, height = map(int, win.flatten())
        if width <= 0 or height <= 0:
            raise ValueError("window dimensions must be positive")
        if row < 0 or col < 0 or row + height > self.height or col + width > self.width:
            raise ValueError("window falls outside the dataset")
        if source.shape != (len(indexes), height, width):
            raise ValueError("source shape does not match indexes and window")
        for source_band, index in enumerate(indexes):
            _validate_index(index, self.count)
            self._data[index - 1, row : row + height, col : col + width] = source[source_band]


def _validate_index(index, count):
    if index < 1 or index > count:
        raise IndexError(f"band index {index} out of range (not in (1, {count}))")


def _coerce_resampling(value):
    try:
        method = Resampling(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid resampling method: {value!r}") from exc
    if method not in _METHODS:
        supported = ", ".join(item.name for item in _METHODS)
        raise NotImplementedError(f"{method.name} resampling is not covered; use {supported}")
    return method


def _output_size(out_shape, bands, height, width, return_2d):
    if out_shape is None:
        return height, width
    shape = tuple(int(value) for value in out_shape)
    if len(shape) == 2:
        if not return_2d:
            raise ValueError("out_shape must include a band dimension")
        target_height, target_width = shape
    elif len(shape) == 3:
        if shape[0] != bands:
            raise ValueError("out_shape band count does not match indexes")
        target_height, target_width = shape[-2:]
    else:
        raise ValueError("out_shape must have 2 or 3 dimensions")
    if target_height <= 0 or target_width <= 0:
        raise ValueError("output dimensions must be positive")
    return target_height, target_width
