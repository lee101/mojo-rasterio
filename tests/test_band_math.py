import numpy as np
import pytest

from mojorasterio import band_math as bm


@pytest.fixture(params=["float32", "float64"])
def pair(request):
    rng = np.random.default_rng(7)
    first = rng.uniform(0.2, 4.0, size=(4, 17, 23)).astype(request.param)
    second = rng.uniform(0.2, 4.0, size=(4, 17, 23)).astype(request.param)
    return first, second


@pytest.mark.parametrize(
    ("function", "reference"),
    [
        (bm.add, np.add),
        (bm.subtract, np.subtract),
        (bm.multiply, np.multiply),
        (bm.divide, np.divide),
        (bm.minimum, np.minimum),
        (bm.maximum, np.maximum),
    ],
)
def test_binary_math(function, reference, pair):
    first, second = pair
    assert np.allclose(function(first, second), reference(first, second), rtol=1e-6)


def test_binary_broadcast_and_out():
    first = np.arange(24, dtype="float64").reshape(2, 3, 4)
    second = np.arange(4, dtype="float64")
    destination = np.empty_like(first)
    returned = bm.add(first, second, out=destination)
    assert returned is destination
    assert np.array_equal(destination, first + second)


def test_normalized_difference(pair):
    first, second = pair
    ours = bm.normalized_difference(first, second)
    assert np.allclose(ours, (first - second) / (first + second), rtol=1e-6)


def test_normalized_difference_zero_division():
    first = np.array([1.0, 0.0, -2.0])
    second = np.array([-1.0, 0.0, 2.0])
    assert np.all(bm.ndvi(first, second, zero_division=-7) == -7)


def test_linear_combination(pair):
    first, second = pair
    ours = bm.linear_combination([first, second], [0.25, -1.5], bias=3.0)
    expected = first * 0.25 - second * 1.5 + 3.0
    assert np.allclose(ours, expected, rtol=1e-6, atol=1e-6)


def test_band_math_validation():
    with pytest.raises(ValueError, match="one value"):
        bm.linear_combination([np.ones(3), np.ones(3)], [1.0])
    with pytest.raises(ValueError, match="wrong shape"):
        bm.add(np.ones(3), np.ones(3), out=np.empty(4))
    with pytest.raises(ValueError, match="at least one"):
        bm.linear_combination([], [])
    with pytest.raises(ValueError, match="must not be empty"):
        bm.add(np.empty(0), np.empty(0))
    with pytest.raises(TypeError, match="float32 or float64"):
        bm.add(np.ones(3), np.ones(3), dtype="int16")
    with pytest.raises(TypeError, match="real integer"):
        bm.add(np.ones(3, dtype="complex64"), np.ones(3))
    readonly = np.empty(3)
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        bm.add(np.ones(3), np.ones(3), out=readonly)
