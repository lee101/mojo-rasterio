from .windows import WindowError


class RasterioError(Exception):
    pass


class RasterioIOError(RasterioError, OSError):
    pass


__all__ = ["RasterioError", "RasterioIOError", "WindowError"]
