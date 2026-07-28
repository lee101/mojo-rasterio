import pytest

rasterio = pytest.importorskip("rasterio")
from affine import Affine
from rasterio import windows as rio

from mojorasterio import windows as ours


def values(window):
    return tuple(window.flatten())


def test_window_construction_and_slices():
    assert values(ours.Window.from_slices((2, 9), (3, 13))) == values(
        rio.Window.from_slices((2, 9), (3, 13))
    )
    assert ours.Window(3.2, 2.7, 9.1, 8.2).toslices() == rio.Window(
        3.2, 2.7, 9.1, 8.2
    ).toslices()


def test_negative_and_open_slices():
    assert values(
        ours.Window.from_slices(slice(-8, None), slice(-10, -2), height=30, width=40)
    ) == values(
        rio.Window.from_slices(slice(-8, None), slice(-10, -2), height=30, width=40)
    )


def test_rounding_parity():
    window = ours.Window(1.8, 2.2, 10.7, 11.1)
    reference = rio.Window(1.8, 2.2, 10.7, 11.1)
    assert values(window.round_offsets()) == values(reference.round_offsets())
    assert values(window.round_lengths()) == values(reference.round_lengths())


def test_shape_parity():
    window = ours.Window(1.2, 3.4, 10.7, 11.1)
    assert ours.shape(window) == rio.shape(rio.Window(*window.flatten()))


def test_crop_intersection_union_parity():
    first = ours.Window(-3, 4, 12, 15)
    second = ours.Window(2, 1, 10, 9)
    assert values(ours.crop(first, 20, 30)) == values(
        rio.crop(rio.Window(*first.flatten()), 20, 30)
    )
    assert values(ours.intersection(first, second)) == values(
        rio.intersection(
            rio.Window(*first.flatten()), rio.Window(*second.flatten())
        )
    )
    assert values(ours.union(first, second)) == values(
        rio.union(rio.Window(*first.flatten()), rio.Window(*second.flatten()))
    )


def test_bounds_parity():
    transform = Affine.translation(100, 500) * Affine.scale(30, -30)
    window = ours.Window(3, 4, 8, 9)
    assert ours.bounds(window, transform) == rio.bounds(
        rio.Window(*window.flatten()), transform
    )


def test_empty_intersection_error():
    with pytest.raises(ours.WindowError):
        ours.intersection(ours.Window(0, 0, 2, 2), ours.Window(5, 5, 2, 2))
