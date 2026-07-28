# mojo-rasterio

`mojo-rasterio` is a compute-focused port of Rasterio's in-memory raster paths
to Mojo. It provides a small Rasterio-shaped Python API for windowed band reads,
nearest/bilinear/average resampling, masks and nodata, and fused band
arithmetic. The implementation is standalone and open source; Rasterio and
GDAL are test and benchmark references, not runtime dependencies of the
library itself.

This is a useful subset, not a replacement for all of Rasterio. The covered
`DatasetReader.read()` arguments retain Rasterio's names and behavior, band
indexes are one-based, arrays are band-major, and resampling output has been
checked directly against Rasterio 1.5.

## Coverage

Covered:

- `DatasetReader.read()` with single or multiple band indexes
- integral `Window` objects and `((row_start, row_stop), (col_start, col_stop))`
  windows
- nearest, bilinear, and area-average resampling for `float32` and `float64`
- GDAL-compatible widened bilinear downsampling and separable nodata handling
- `out`, `out_shape`, `out_dtype`, masked reads, mask reads, nodata, and
  boundless reads
- in-memory `DatasetWriter.write()` for bands and integral windows
- `Window.from_slices`, `toslices`, rounding, crop, intersection, union, shape,
  and bounds
- fused add, subtract, multiply, divide, minimum, maximum, normalized
  difference/NDVI, and weighted linear combinations

Not covered:

- GeoTIFF or other file-backed I/O, GDAL drivers, remote datasets, or virtual
  filesystems
- CRS objects, coordinate transforms, georeferencing, features, warping, or
  overviews
- fractional windows
- cubic, Lanczos, mode, statistical, and other resampling algorithms
- integer-native resampling kernels; integer arrays are computed through
  `float64` and cast to the requested output dtype

Use Rasterio itself when those features are needed. A file-backed Rasterio
dataset can still provide an array to `mojo-rasterio` when the covered
compute-heavy operations are useful.

## Install

The repository pins the tested Mojo nightly in `pixi.toml`.

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` creates `dist/libmojo-rasterio.so`. Pixi also sets
`PYTHONPATH=python`, so examples and tests import the local package directly.

## Usage

```python
import numpy as np

import mojorasterio as rasterio
from mojorasterio import band_math
from mojorasterio.enums import Resampling
from mojorasterio.windows import Window

bands = np.arange(3 * 512 * 512, dtype=np.float32).reshape(3, 512, 512)

with rasterio.open(bands, nodata=-9999) as dataset:
    tile = dataset.read(
        indexes=[1, 3],
        window=Window(64, 96, 256, 192),
        out_shape=(2, 96, 128),
        resampling=Resampling.average,
    )

ndvi = band_math.ndvi(tile[1], tile[0])
print(tile.shape, ndvi.shape)
```

This prints:

```text
(2, 96, 128) (96, 128)
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
6.8.0-136-generic. Times are the best of five runs after one warm-up. A speedup
below `1.00x` means the Mojo implementation is slower.

| Kernel | Mojo | Reference | Speedup | Compared with |
|---|---:|---:|---:|---|
| nearest 3x2048x2048 -> 1024x1024 | 8.40 ms | 8.47 ms | 1.01x | Rasterio 1.5 / GDAL |
| bilinear 3x2048x2048 -> 1024x1024 | 12.72 ms | 102.85 ms | 8.09x | Rasterio 1.5 / GDAL |
| average 3x2048x2048 -> 1024x1024 | 15.07 ms | 21.17 ms | 1.41x | Rasterio 1.5 / GDAL |
| normalized difference 2048x2048 | 6.96 ms | 37.92 ms | 5.45x | NumPy |
| linear combination 3x2048x2048 | 9.88 ms | 48.36 ms | 4.89x | NumPy expression |

The fused band operations win because they allocate one output and traverse
each input once. Exact 2x downsampling uses native-width SIMD loads with scalar
tails. Bilinear and average split sufficiently large outputs across CPU workers;
smaller reads remain serial to avoid thread-launch overhead.

No GPU path is included. The benchmark covers only the CPU implementation.

## How it works

All kernels live in one Mojo compilation unit to avoid repeated fixed compiler
startup cost. The shared library exports concrete C-ABI entry points for
`float32` and `float64`. Python uses `ctypes`; NumPy buffer addresses cross the
ABI as 64-bit integers and are reconstructed as mutable `UnsafePointer` values
inside Mojo. Python owns every input, output, and temporary allocation.
Native contiguous NumPy inputs cross the FFI boundary without a copy, and a
compatible contiguous `out` buffer is written directly. Python validates
shapes, dtypes, writable outputs, non-empty buffers, and aliasing before the
call, and keeps all NumPy owners alive until the synchronous native call
returns.

Raster memory is C-contiguous and band-major: `(band, row, column)`, with
columns adjacent in memory. Reads pass the source dimensions and integral
window geometry to Mojo, so bilinear reads can use support pixels just outside
a requested window exactly as GDAL does. Nodata values are excluded and
weights are renormalized. Band math operates directly on contiguous NumPy
buffers, and the linear-combination kernel uses SIMD over adjacent pixels.

## Verification

The test suite contains 72 tests. It compares all three resamplers, windowed
reads, nodata behavior, output buffers, dtypes, masks, and window helpers with
real Rasterio, including SIMD remainders and the parallel size threshold. Band
math is checked numerically against NumPy. Run the complete workflow with:

```bash
pixi run build && pixi run test && pixi run bench
```

The project is licensed under the MIT License.
