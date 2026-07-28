"""ctypes bridge to the single Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "raster_kernels.mojo")
LIB = os.path.join(ROOT, "dist", "libmojo-rasterio.so")

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mr_resample_f32": ([I] * 13 + [F], None),
    "mr_resample_f64": ([I] * 13 + [F], None),
    "mr_binary_f32": ([I, I, I, I, I], None),
    "mr_binary_f64": ([I, I, I, I, I], None),
    "mr_nd_f32": ([I, I, I, I, F], None),
    "mr_nd_f64": ([I, I, I, I, F], None),
    "mr_linear_f32": ([I, I, I, I, I, F], None),
    "mr_linear_f64": ([I, I, I, I, I, F], None),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJORASTERIO_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [
            pixi,
            "run",
            "--manifest-path",
            os.path.join(ROOT, "pixi.toml"),
            "mojo",
        ]
    raise BuildError("mojo not found; set MOJORASTERIO_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(SRC)
    ):
        return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + ["build", "--emit", "shared-lib", SRC, "-o", LIB]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def addr(array: np.ndarray) -> int:
    return array.ctypes.data


def main() -> int:
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
