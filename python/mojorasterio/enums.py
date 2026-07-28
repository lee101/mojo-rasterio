"""Rasterio-compatible enumeration values."""

from enum import IntEnum


class Resampling(IntEnum):
    nearest = 0
    bilinear = 1
    cubic = 2
    cubic_spline = 3
    lanczos = 4
    average = 5
    mode = 6
    gauss = 7
    max = 8
    min = 9
    med = 10
    q1 = 11
    q3 = 12
    sum = 13
    rms = 14
