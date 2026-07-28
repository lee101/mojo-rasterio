"""Numerical and behavioral parity with Rasterio's in-memory reads."""

import warnings

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.enums import Resampling as RioResampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.windows import Window as RioWindow

import mojorasterio as mr


def upstream_read(data, *, nodata=None, **kwargs):
    data = data[np.newaxis, ...] if data.ndim == 2 else data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        with MemoryFile() as memory:
            with memory.open(
                driver="GTiff",
                height=data.shape[1],
                width=data.shape[2],
                count=data.shape[0],
                dtype=data.dtype,
                nodata=nodata,
            ) as writer:
                writer.write(data)
            with memory.open() as reader:
                return reader.read(**kwargs)


@pytest.fixture
def raster():
    rng = np.random.default_rng(42)
    return rng.normal(size=(3, 37, 53)).astype("float64")


def test_metadata_and_context_manager(raster):
    with mr.open(raster) as dataset:
        assert dataset.count == 3
        assert dataset.shape == (37, 53)
        assert dataset.dtypes == ("float64",) * 3
    assert dataset.closed
    with pytest.raises(ValueError, match="closed"):
        dataset.read()


def test_full_read_and_band_index_shapes(raster):
    dataset = mr.open(raster)
    assert np.array_equal(dataset.read(), raster)
    assert np.array_equal(dataset.read(2), raster[1])
    assert np.array_equal(dataset.read([3, 1]), raster[[2, 0]])
    assert dataset.read(1).ndim == 2
    assert dataset.read([1]).shape == (1, 37, 53)


def test_full_integer_read_does_not_narrow_through_float64():
    data = np.array([[[2**63 + 1, 2**63 + 3]]], dtype="uint64")
    assert np.array_equal(mr.open(data).read(), data)


@pytest.mark.parametrize(
    "method",
    [mr.Resampling.nearest, mr.Resampling.bilinear, mr.Resampling.average],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(3, 19, 27), (3, 61, 79), (3, 23, 81)])
def test_resampling_parity(raster, method, dtype, shape):
    data = raster.astype(dtype)
    ours = mr.open(data).read(out_shape=shape, resampling=method)
    theirs = upstream_read(
        data,
        out_shape=shape,
        resampling=RioResampling(method.value),
    )
    tolerance = 2e-6 if dtype == "float32" else 2e-12
    assert ours.dtype == data.dtype
    assert np.allclose(ours, theirs, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    "method",
    [mr.Resampling.nearest, mr.Resampling.bilinear, mr.Resampling.average],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_exact_two_x_resampling_simd_tail(method, dtype):
    rng = np.random.default_rng(11)
    data = rng.normal(size=(2, 18, 22)).astype(dtype)
    shape = (2, 9, 11)
    ours = mr.open(data).read(out_shape=shape, resampling=method)
    theirs = upstream_read(
        data,
        out_shape=shape,
        resampling=RioResampling(method.value),
    )
    tolerance = 2e-6 if dtype == "float32" else 2e-12
    assert np.allclose(ours, theirs, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    "method",
    [mr.Resampling.bilinear, mr.Resampling.average],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_exact_two_x_parallel_threshold(method, dtype):
    rng = np.random.default_rng(12)
    data = rng.normal(size=(3, 512, 514)).astype(dtype)
    shape = (3, 256, 257)
    ours = mr.open(data).read(out_shape=shape, resampling=method)
    theirs = upstream_read(
        data,
        out_shape=shape,
        resampling=RioResampling(method.value),
    )
    tolerance = 2e-6 if dtype == "float32" else 2e-12
    assert np.allclose(ours, theirs, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    "method",
    [mr.Resampling.nearest, mr.Resampling.bilinear, mr.Resampling.average],
)
def test_windowed_resampling_parity(raster, method):
    ours = mr.open(raster).read(
        indexes=[1, 3],
        window=mr.Window(7, 5, 31, 23),
        out_shape=(2, 13, 17),
        resampling=method,
    )
    theirs = upstream_read(
        raster,
        indexes=[1, 3],
        window=RioWindow(7, 5, 31, 23),
        out_shape=(2, 13, 17),
        resampling=RioResampling(method.value),
    )
    assert np.allclose(ours, theirs, rtol=2e-12, atol=2e-12)


def test_slice_tuple_window_parity(raster):
    window = ((4, 19), (8, 35))
    ours = mr.open(raster).read(indexes=2, window=window)
    theirs = upstream_read(raster, indexes=2, window=window)
    assert np.array_equal(ours, theirs)


@pytest.mark.parametrize(
    "method",
    [mr.Resampling.nearest, mr.Resampling.bilinear, mr.Resampling.average],
)
def test_nodata_resampling_parity(method):
    data = np.arange(2 * 23 * 31, dtype=np.float64).reshape(2, 23, 31)
    data[:, 5:14, 11:20] = -9999.0
    ours = mr.open(data, nodata=-9999).read(
        out_shape=(2, 11, 17), resampling=method
    )
    theirs = upstream_read(
        data,
        nodata=-9999,
        out_shape=(2, 11, 17),
        resampling=RioResampling(method.value),
    )
    assert np.array_equal(ours == -9999, theirs == -9999)
    assert np.allclose(ours, theirs, rtol=2e-12, atol=2e-12)


def test_masked_read_and_masks():
    data = np.arange(48, dtype="float32").reshape(1, 6, 8)
    data[:, 2:4, 3:6] = -99
    dataset = mr.open(data, nodata=-99)
    masked = dataset.read(masked=True)
    assert np.ma.isMaskedArray(masked)
    assert np.array_equal(masked.mask, data == -99)
    assert np.array_equal(dataset.read_masks(), np.where(data == -99, 0, 255))


def test_out_buffer_and_out_dtype(raster):
    destination = np.empty((2, 11, 15), dtype="float32")
    returned = mr.open(raster).read(
        [1, 2],
        out=destination,
        resampling=mr.Resampling.average,
    )
    reference = upstream_read(
        raster,
        indexes=[1, 2],
        out_shape=(2, 11, 15),
        resampling=RioResampling.average,
        out_dtype="float32",
    )
    assert returned is destination
    assert np.allclose(returned, reference, atol=2e-5)
    converted = mr.open(raster).read(1, out_dtype="float32")
    assert converted.dtype == np.float32


def test_native_contiguous_out_buffer_is_used_directly():
    data = np.arange(2 * 18 * 22, dtype="float32").reshape(2, 18, 22)
    destination = np.empty((2, 9, 11), dtype="float32")
    returned = mr.open(data).read(
        out=destination,
        resampling=mr.Resampling.average,
    )
    assert returned is destination
    assert np.array_equal(returned, data.reshape(2, 9, 2, 11, 2).mean((2, 4)))


def test_boundless_read():
    data = np.arange(30, dtype="float64").reshape(1, 5, 6)
    ours = mr.open(data, nodata=-1).read(
        window=mr.Window(-2, -1, 9, 8), boundless=True
    )
    assert ours.shape == (1, 8, 9)
    assert np.array_equal(ours[:, 1:6, 2:8], data)
    assert np.all(ours[:, 0] == -1)
    assert np.all(ours[:, :, :2] == -1)


def test_writer_window_and_indexes():
    data = np.zeros((3, 8, 9), dtype="float32")
    with mr.open(data, mode="w") as dataset:
        values = np.arange(12, dtype="float32").reshape(3, 4)
        dataset.write(values, indexes=2, window=mr.Window(3, 2, 4, 3))
        assert np.array_equal(dataset.read(2, window=((2, 5), (3, 7))), values)
        assert not dataset.read(1).any()


def test_read_validation(raster):
    dataset = mr.open(raster)
    with pytest.raises(IndexError):
        dataset.read(0)
    with pytest.raises(ValueError, match="exclusive"):
        dataset.read(out=np.empty_like(raster), out_shape=raster.shape)
    with pytest.raises(NotImplementedError, match="cubic"):
        dataset.read(resampling=mr.Resampling.cubic)
    with pytest.raises(NotImplementedError, match="fractional"):
        dataset.read(window=mr.Window(0.5, 0, 10, 10))
    with pytest.raises(ValueError, match="outside"):
        dataset.read(window=mr.Window(-1, 0, 10, 10))
    readonly = np.empty_like(raster)
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        dataset.read(out=readonly)
    with pytest.raises(TypeError, match="real integer"):
        mr.open(raster.astype("complex128"))
    with pytest.raises(ValueError, match="representable"):
        mr.open(raster.astype("float32"), nodata=1e300)
    with pytest.raises(ValueError, match="positive"):
        mr.open(np.empty((0, 2, 2), dtype="float32"))
    with pytest.raises(ValueError, match="positive"):
        dataset.read(out=np.empty((3, 0, 2)))


def test_resampling_out_may_not_overwrite_aliased_input():
    data = np.arange(64, dtype="float64").reshape(1, 8, 8)
    destination = data.ravel()[:16].reshape(1, 4, 4)
    expected = upstream_read(
        data.copy(),
        out_shape=(1, 4, 4),
        resampling=RioResampling.average,
    )
    returned = mr.open(data).read(out=destination, resampling=mr.Resampling.average)
    assert returned is destination
    assert np.allclose(returned, expected)


def test_writer_rejects_out_of_bounds_window():
    dataset = mr.open(np.zeros((1, 4, 4), dtype="float32"), mode="w")
    with pytest.raises(ValueError, match="outside"):
        dataset.write(np.ones((1, 2, 2)), window=mr.Window(3, 3, 2, 2))
