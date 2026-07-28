"""Fused element-wise operations for raster bands."""

from __future__ import annotations

import numpy as np

from ._lib import addr, lib

_OPS = {
    "add": 0,
    "subtract": 1,
    "multiply": 2,
    "divide": 3,
    "minimum": 4,
    "maximum": 5,
}


def _dtype_for(*arrays):
    dtype = np.result_type(*arrays)
    if dtype.kind not in "iuf":
        raise TypeError("inputs must have real integer or floating-point dtypes")
    return np.dtype("float32") if dtype == np.float32 else np.dtype("float64")


def _destination(shape, dtype, out):
    if out is None:
        return np.empty(shape, dtype=dtype), None
    destination = np.asarray(out)
    if destination.shape != shape:
        raise ValueError("out has the wrong shape")
    if not destination.flags.writeable:
        raise ValueError("out must be writable")
    if destination.dtype.kind not in "iuf":
        raise TypeError("out must have a real integer or floating-point dtype")
    work = destination if destination.dtype == dtype and destination.flags.c_contiguous else np.empty(shape, dtype=dtype)
    return work, destination


def _work_dtype(dtype, *arrays):
    if dtype is None:
        return _dtype_for(*arrays)
    requested = np.dtype(dtype)
    if requested not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("dtype must be float32 or float64")
    return requested


def _binary(a, b, operation, out=None, dtype=None):
    first, second = np.broadcast_arrays(a, b)
    if first.size == 0:
        raise ValueError("inputs must not be empty")
    work_dtype = _work_dtype(dtype, first, second)
    first = np.ascontiguousarray(first, dtype=work_dtype)
    second = np.ascontiguousarray(second, dtype=work_dtype)
    result, destination = _destination(first.shape, work_dtype, out)
    fn = getattr(lib(), f"mr_binary_f{work_dtype.itemsize * 8}")
    fn(addr(first), addr(second), addr(result), first.size, _OPS[operation])
    if destination is not None and result is not destination:
        np.copyto(destination, result, casting="unsafe")
        return destination
    return result


def add(a, b, out=None, dtype=None):
    return _binary(a, b, "add", out, dtype)


def subtract(a, b, out=None, dtype=None):
    return _binary(a, b, "subtract", out, dtype)


def multiply(a, b, out=None, dtype=None):
    return _binary(a, b, "multiply", out, dtype)


def divide(a, b, out=None, dtype=None):
    return _binary(a, b, "divide", out, dtype)


def minimum(a, b, out=None, dtype=None):
    return _binary(a, b, "minimum", out, dtype)


def maximum(a, b, out=None, dtype=None):
    return _binary(a, b, "maximum", out, dtype)


def normalized_difference(a, b, *, out=None, dtype=None, zero_division=np.nan):
    first, second = np.broadcast_arrays(a, b)
    if first.size == 0:
        raise ValueError("inputs must not be empty")
    work_dtype = _work_dtype(dtype, first, second)
    first = np.ascontiguousarray(first, dtype=work_dtype)
    second = np.ascontiguousarray(second, dtype=work_dtype)
    result, destination = _destination(first.shape, work_dtype, out)
    fn = getattr(lib(), f"mr_nd_f{work_dtype.itemsize * 8}")
    fn(addr(first), addr(second), addr(result), first.size, float(zero_division))
    if destination is not None and result is not destination:
        np.copyto(destination, result, casting="unsafe")
        return destination
    return result


def ndvi(nir, red, **kwargs):
    return normalized_difference(nir, red, **kwargs)


def linear_combination(bands, weights, *, bias=0.0, out=None, dtype=None):
    if len(bands) == 0:
        raise ValueError("at least one band is required")
    arrays = np.broadcast_arrays(*bands)
    if arrays[0].size == 0:
        raise ValueError("bands must not be empty")
    coefficients = np.asarray(weights)
    if coefficients.ndim != 1 or coefficients.size != len(arrays):
        raise ValueError("weights must contain one value per band")
    work_dtype = _work_dtype(dtype, *arrays)
    if (
        isinstance(bands, np.ndarray)
        and bands.ndim >= 2
        and bands.shape[0] == len(arrays)
        and all(array.shape == bands.shape[1:] for array in arrays)
    ):
        stack = np.ascontiguousarray(bands, dtype=work_dtype)
    else:
        stack = np.ascontiguousarray(np.stack(arrays), dtype=work_dtype)
    coefficients = np.ascontiguousarray(coefficients, dtype=work_dtype)
    result, destination = _destination(arrays[0].shape, work_dtype, out)
    fn = getattr(lib(), f"mr_linear_f{work_dtype.itemsize * 8}")
    fn(
        addr(stack),
        addr(coefficients),
        addr(result),
        len(arrays),
        result.size,
        float(bias),
    )
    if destination is not None and result is not destination:
        np.copyto(destination, result, casting="unsafe")
        return destination
    return result
