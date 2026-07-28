"""Repeatable mojo-rasterio benchmarks against Rasterio and NumPy."""

from __future__ import annotations

import os
import platform
import sys
import time
import warnings

import numpy as np
from rasterio.enums import Resampling as RioResampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojorasterio as mr  # noqa: E402
from mojorasterio import band_math as bm  # noqa: E402


def time_best(function, repetitions=5):
    function()
    best = float("inf")
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def row(name, mojo_seconds, reference_seconds, reference):
    speedup = reference_seconds / mojo_seconds
    print(
        f"| {name} | {mojo_seconds * 1000:.2f} ms | "
        f"{reference_seconds * 1000:.2f} ms | {speedup:.2f}x | {reference} |"
    )


def main():
    rng = np.random.default_rng(2026)
    raster = rng.normal(size=(3, 2048, 2048)).astype("float32")
    ours = mr.open(raster)

    warnings.simplefilter("ignore", NotGeoreferencedWarning)
    memory = MemoryFile()
    with memory.open(
        driver="GTiff",
        height=2048,
        width=2048,
        count=3,
        dtype="float32",
    ) as writer:
        writer.write(raster)
    upstream = memory.open()

    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}")
    print()
    print("| Kernel | Mojo | Reference | Speedup | Compared with |")
    print("|---|---:|---:|---:|---|")

    cases = [
        ("nearest 3x2048x2048 -> 1024x1024", mr.Resampling.nearest),
        ("bilinear 3x2048x2048 -> 1024x1024", mr.Resampling.bilinear),
        ("average 3x2048x2048 -> 1024x1024", mr.Resampling.average),
    ]
    for name, method in cases:
        shape = (3, 1024, 1024)
        mojo_call = lambda m=method: ours.read(out_shape=shape, resampling=m)
        rio_call = lambda m=method: upstream.read(
            out_shape=shape, resampling=RioResampling(m.value)
        )
        assert np.allclose(mojo_call(), rio_call(), rtol=2e-5, atol=2e-5)
        row(name, time_best(mojo_call), time_best(rio_call), "Rasterio 1.5 / GDAL")

    first = raster[0]
    second = raster[1]
    nd_mojo = lambda: bm.normalized_difference(first, second)
    nd_numpy = lambda: (first - second) / (first + second)
    assert np.allclose(nd_mojo(), nd_numpy(), equal_nan=True)
    row(
        "normalized difference 2048x2048",
        time_best(nd_mojo),
        time_best(nd_numpy),
        "NumPy",
    )

    linear_mojo = lambda: bm.linear_combination(
        raster, [0.2, -0.7, 1.3], bias=0.5
    )
    linear_numpy = lambda: raster[0] * 0.2 - raster[1] * 0.7 + raster[2] * 1.3 + 0.5
    assert np.allclose(linear_mojo(), linear_numpy(), rtol=2e-6, atol=2e-6)
    row(
        "linear combination 3x2048x2048",
        time_best(linear_mojo),
        time_best(linear_numpy),
        "NumPy expression",
    )

    upstream.close()
    memory.close()


if __name__ == "__main__":
    main()
