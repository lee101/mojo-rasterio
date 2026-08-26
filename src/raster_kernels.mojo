"""Raster resampling and fused band arithmetic behind the Python API."""

from std.math import ceil, floor, isnan
from std.sys.info import simd_width_of


comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def p32(addr: Int) -> F32Ptr:
    return F32Ptr(unsafe_from_address=addr)


def p64(addr: Int) -> F64Ptr:
    return F64Ptr(unsafe_from_address=addr)


def missing32(value: Float32, has_nodata: Bool, nodata: Float64) -> Bool:
    if not has_nodata:
        return False
    if isnan(nodata):
        return isnan(value)
    return Float64(value) == nodata


def missing64(value: Float64, has_nodata: Bool, nodata: Float64) -> Bool:
    if not has_nodata:
        return False
    if isnan(nodata):
        return isnan(value)
    return value == nodata


def nearest2_32_row(
    src: F32Ptr, dst: F32Ptr, sh: Int, sw: Int, dh: Int, dw: Int, task: Int
):
    comptime W = simd_width_of[DType.float32]()
    var b = task // dh
    var y = task - b * dh
    var source_base = (b * sh + 2 * y + 1) * sw
    var dest_base = (b * dh + y) * dw
    var x = 0
    while x + W // 2 <= dw:
        var values = src.load[width=W](source_base + 2 * x)
        var even, odd = values.deinterleave()
        dst.store(dest_base + x, odd)
        x += W // 2
    while x < dw:
        dst[dest_base + x] = src[source_base + 2 * x + 1]
        x += 1


def nearest2_64_row(
    src: F64Ptr, dst: F64Ptr, sh: Int, sw: Int, dh: Int, dw: Int, task: Int
):
    comptime W = simd_width_of[DType.float64]()
    var b = task // dh
    var y = task - b * dh
    var source_base = (b * sh + 2 * y + 1) * sw
    var dest_base = (b * dh + y) * dw
    var x = 0
    while x + W // 2 <= dw:
        var values = src.load[width=W](source_base + 2 * x)
        var even, odd = values.deinterleave()
        dst.store(dest_base + x, odd)
        x += W // 2
    while x < dw:
        dst[dest_base + x] = src[source_base + 2 * x + 1]
        x += 1


def nearest32(
    src: F32Ptr, dst: F32Ptr, bands: Int, sh: Int, sw: Int,
    row: Int, col: Int, wh: Int, ww: Int, dh: Int, dw: Int
):
    if row == 0 and col == 0 and wh == sh and ww == sw and sh == 2 * dh and sw == 2 * dw:
        @parameter
        def work(task: Int):
            nearest2_32_row(src, dst, sh, sw, dh, dw, task)

        for task in range(bands * dh):
            work(task)
        return
    for b in range(bands):
        for y in range(dh):
            var sy = row + Int(floor((Float64(y) + 0.5) * Float64(wh) / Float64(dh)))
            sy = min(max(sy, 0), sh - 1)
            for x in range(dw):
                var sx = col + Int(floor((Float64(x) + 0.5) * Float64(ww) / Float64(dw)))
                sx = min(max(sx, 0), sw - 1)
                dst[(b * dh + y) * dw + x] = src[(b * sh + sy) * sw + sx]


def nearest64(
    src: F64Ptr, dst: F64Ptr, bands: Int, sh: Int, sw: Int,
    row: Int, col: Int, wh: Int, ww: Int, dh: Int, dw: Int
):
    if row == 0 and col == 0 and wh == sh and ww == sw and sh == 2 * dh and sw == 2 * dw:
        @parameter
        def work(task: Int):
            nearest2_64_row(src, dst, sh, sw, dh, dw, task)

        for task in range(bands * dh):
            work(task)
        return
    for b in range(bands):
        for y in range(dh):
            var sy = row + Int(floor((Float64(y) + 0.5) * Float64(wh) / Float64(dh)))
            sy = min(max(sy, 0), sh - 1)
            for x in range(dw):
                var sx = col + Int(floor((Float64(x) + 0.5) * Float64(ww) / Float64(dw)))
                sx = min(max(sx, 0), sw - 1)
                dst[(b * dh + y) * dw + x] = src[(b * sh + sy) * sw + sx]


def bilinear2_horizontal32[W: Int](
    src: F32Ptr, row_base: Int, source_x: Int
) -> SIMD[DType.float32, W]:
    var a = (src + row_base + source_x - 1).strided_load[width=W](2)
    var b = (src + row_base + source_x).strided_load[width=W](2)
    var c = (src + row_base + source_x + 1).strided_load[width=W](2)
    var d = (src + row_base + source_x + 2).strided_load[width=W](2)
    return a + b * Float32(3.0) + c * Float32(3.0) + d


def bilinear2_horizontal64[W: Int](
    src: F64Ptr, row_base: Int, source_x: Int
) -> SIMD[DType.float64, W]:
    var a = (src + row_base + source_x - 1).strided_load[width=W](2)
    var b = (src + row_base + source_x).strided_load[width=W](2)
    var c = (src + row_base + source_x + 1).strided_load[width=W](2)
    var d = (src + row_base + source_x + 2).strided_load[width=W](2)
    return a + b * 3.0 + c * 3.0 + d


def bilinear2_pixel32(
    src: F32Ptr, b: Int, sh: Int, sw: Int, y: Int, x: Int
) -> Float32:
    var fy = Float64(2 * y) + 0.5
    var fx = Float64(2 * x) + 0.5
    var ystart = max(Int(floor(fy - 2.0)), 0)
    var yend = min(Int(ceil(fy + 2.0)) + 1, sh)
    var xstart = max(Int(floor(fx - 2.0)), 0)
    var xend = min(Int(ceil(fx + 2.0)) + 1, sw)
    var total = 0.0
    var weight = 0.0
    for sy in range(ystart, yend):
        var wy = 1.0 - abs(Float64(sy) - fy) * 0.5
        if wy <= 0.0:
            continue
        var row_total = 0.0
        var row_weight = 0.0
        for sx in range(xstart, xend):
            var wx = 1.0 - abs(Float64(sx) - fx) * 0.5
            if wx > 0.0:
                row_total += wx * Float64(src[(b * sh + sy) * sw + sx])
                row_weight += wx
        total += wy * row_total / row_weight
        weight += wy
    return Float32(total / weight)


def bilinear2_pixel64(
    src: F64Ptr, b: Int, sh: Int, sw: Int, y: Int, x: Int
) -> Float64:
    var fy = Float64(2 * y) + 0.5
    var fx = Float64(2 * x) + 0.5
    var ystart = max(Int(floor(fy - 2.0)), 0)
    var yend = min(Int(ceil(fy + 2.0)) + 1, sh)
    var xstart = max(Int(floor(fx - 2.0)), 0)
    var xend = min(Int(ceil(fx + 2.0)) + 1, sw)
    var total = 0.0
    var weight = 0.0
    for sy in range(ystart, yend):
        var wy = 1.0 - abs(Float64(sy) - fy) * 0.5
        if wy <= 0.0:
            continue
        var row_total = 0.0
        var row_weight = 0.0
        for sx in range(xstart, xend):
            var wx = 1.0 - abs(Float64(sx) - fx) * 0.5
            if wx > 0.0:
                row_total += wx * src[(b * sh + sy) * sw + sx]
                row_weight += wx
        total += wy * row_total / row_weight
        weight += wy
    return total / weight


def bilinear2_32_row(
    src: F32Ptr, dst: F32Ptr, sh: Int, sw: Int, dh: Int, dw: Int, task: Int
):
    comptime W = simd_width_of[DType.float32]()
    var b = task // dh
    var y = task - b * dh
    var dest_base = (b * dh + y) * dw
    dst[dest_base] = bilinear2_pixel32(src, b, sh, sw, y, 0)
    var x = 1
    while x + W <= dw - 1:
        var source_x = 2 * x
        var value: SIMD[DType.float32, W]
        if y == 0:
            var row0 = bilinear2_horizontal32[W](src, b * sh * sw, source_x)
            var row1 = bilinear2_horizontal32[W](src, (b * sh + 1) * sw, source_x)
            var row2 = bilinear2_horizontal32[W](src, (b * sh + 2) * sw, source_x)
            value = (
                row0 * Float32(3.0) + row1 * Float32(3.0) + row2
            ) / Float32(56.0)
        elif y == dh - 1:
            var row0 = bilinear2_horizontal32[W](
                src, (b * sh + sh - 3) * sw, source_x
            )
            var row1 = bilinear2_horizontal32[W](
                src, (b * sh + sh - 2) * sw, source_x
            )
            var row2 = bilinear2_horizontal32[W](
                src, (b * sh + sh - 1) * sw, source_x
            )
            value = (
                row0 + row1 * Float32(3.0) + row2 * Float32(3.0)
            ) / Float32(56.0)
        else:
            var first_row = b * sh + 2 * y - 1
            var row0 = bilinear2_horizontal32[W](src, first_row * sw, source_x)
            var row1 = bilinear2_horizontal32[W](src, (first_row + 1) * sw, source_x)
            var row2 = bilinear2_horizontal32[W](src, (first_row + 2) * sw, source_x)
            var row3 = bilinear2_horizontal32[W](src, (first_row + 3) * sw, source_x)
            value = (
                row0
                + row1 * Float32(3.0)
                + row2 * Float32(3.0)
                + row3
            ) / Float32(64.0)
        dst.store(dest_base + x, value)
        x += W
    while x < dw:
        dst[dest_base + x] = bilinear2_pixel32(src, b, sh, sw, y, x)
        x += 1


def bilinear2_64_row(
    src: F64Ptr, dst: F64Ptr, sh: Int, sw: Int, dh: Int, dw: Int, task: Int
):
    comptime W = simd_width_of[DType.float64]()
    var b = task // dh
    var y = task - b * dh
    var dest_base = (b * dh + y) * dw
    dst[dest_base] = bilinear2_pixel64(src, b, sh, sw, y, 0)
    var x = 1
    while x + W <= dw - 1:
        var source_x = 2 * x
        var value: SIMD[DType.float64, W]
        if y == 0:
            var row0 = bilinear2_horizontal64[W](src, b * sh * sw, source_x)
            var row1 = bilinear2_horizontal64[W](src, (b * sh + 1) * sw, source_x)
            var row2 = bilinear2_horizontal64[W](src, (b * sh + 2) * sw, source_x)
            value = (row0 * 3.0 + row1 * 3.0 + row2) / 56.0
        elif y == dh - 1:
            var row0 = bilinear2_horizontal64[W](
                src, (b * sh + sh - 3) * sw, source_x
            )
            var row1 = bilinear2_horizontal64[W](
                src, (b * sh + sh - 2) * sw, source_x
            )
            var row2 = bilinear2_horizontal64[W](
                src, (b * sh + sh - 1) * sw, source_x
            )
            value = (row0 + row1 * 3.0 + row2 * 3.0) / 56.0
        else:
            var first_row = b * sh + 2 * y - 1
            var row0 = bilinear2_horizontal64[W](src, first_row * sw, source_x)
            var row1 = bilinear2_horizontal64[W](src, (first_row + 1) * sw, source_x)
            var row2 = bilinear2_horizontal64[W](src, (first_row + 2) * sw, source_x)
            var row3 = bilinear2_horizontal64[W](src, (first_row + 3) * sw, source_x)
            value = (row0 + row1 * 3.0 + row2 * 3.0 + row3) / 64.0
        dst.store(dest_base + x, value)
        x += W
    while x < dw:
        dst[dest_base + x] = bilinear2_pixel64(src, b, sh, sw, y, x)
        x += 1


def bilinear32(
    src: F32Ptr,
    dst: F32Ptr,
    bands: Int,
    sh: Int,
    sw: Int,
    row: Int,
    col: Int,
    wh: Int,
    ww: Int,
    dh: Int,
    dw: Int,
    has_nodata: Bool,
    nodata: Float64,
):
    if (
        not has_nodata
        and row == 0
        and col == 0
        and wh == sh
        and ww == sw
        and sh == 2 * dh
        and sw == 2 * dw
        and dh >= 2
    ):
        @parameter
        def work(task: Int):
            bilinear2_32_row(src, dst, sh, sw, dh, dw, task)

        for task in range(bands * dh):
            work(task)
        return
    var radius_y = max(Float64(wh) / Float64(dh), 1.0)
    var radius_x = max(Float64(ww) / Float64(dw), 1.0)
    for b in range(bands):
        for y in range(dh):
            var fy = Float64(row) + (Float64(y) + 0.5) * Float64(wh) / Float64(dh) - 0.5
            var ystart = max(Int(floor(fy - radius_y)), 0)
            var yend = min(Int(ceil(fy + radius_y)) + 1, sh)
            for x in range(dw):
                var fx = Float64(col) + (Float64(x) + 0.5) * Float64(ww) / Float64(dw) - 0.5
                var xstart = max(Int(floor(fx - radius_x)), 0)
                var xend = min(Int(ceil(fx + radius_x)) + 1, sw)
                var total = 0.0
                var weight = 0.0
                for sy in range(ystart, yend):
                    var wy = 1.0 - abs(Float64(sy) - fy) / radius_y
                    if wy <= 0.0:
                        continue
                    var row_total = 0.0
                    var row_weight = 0.0
                    for sx in range(xstart, xend):
                        var wx = 1.0 - abs(Float64(sx) - fx) / radius_x
                        if wx <= 0.0:
                            continue
                        var value = src[(b * sh + sy) * sw + sx]
                        if missing32(value, has_nodata, nodata):
                            continue
                        row_total += wx * Float64(value)
                        row_weight += wx
                    if row_weight > 0.0:
                        total += wy * row_total / row_weight
                        weight += wy
                dst[(b * dh + y) * dw + x] = Float32(
                    total / weight if weight > 0.0 else nodata
                )


def bilinear64(
    src: F64Ptr,
    dst: F64Ptr,
    bands: Int,
    sh: Int,
    sw: Int,
    row: Int,
    col: Int,
    wh: Int,
    ww: Int,
    dh: Int,
    dw: Int,
    has_nodata: Bool,
    nodata: Float64,
):
    if (
        not has_nodata
        and row == 0
        and col == 0
        and wh == sh
        and ww == sw
        and sh == 2 * dh
        and sw == 2 * dw
        and dh >= 2
    ):
        @parameter
        def work(task: Int):
            bilinear2_64_row(src, dst, sh, sw, dh, dw, task)

        for task in range(bands * dh):
            work(task)
        return
    var radius_y = max(Float64(wh) / Float64(dh), 1.0)
    var radius_x = max(Float64(ww) / Float64(dw), 1.0)
    for b in range(bands):
        for y in range(dh):
            var fy = Float64(row) + (Float64(y) + 0.5) * Float64(wh) / Float64(dh) - 0.5
            var ystart = max(Int(floor(fy - radius_y)), 0)
            var yend = min(Int(ceil(fy + radius_y)) + 1, sh)
            for x in range(dw):
                var fx = Float64(col) + (Float64(x) + 0.5) * Float64(ww) / Float64(dw) - 0.5
                var xstart = max(Int(floor(fx - radius_x)), 0)
                var xend = min(Int(ceil(fx + radius_x)) + 1, sw)
                var total = 0.0
                var weight = 0.0
                for sy in range(ystart, yend):
                    var wy = 1.0 - abs(Float64(sy) - fy) / radius_y
                    if wy <= 0.0:
                        continue
                    var row_total = 0.0
                    var row_weight = 0.0
                    for sx in range(xstart, xend):
                        var wx = 1.0 - abs(Float64(sx) - fx) / radius_x
                        if wx <= 0.0:
                            continue
                        var value = src[(b * sh + sy) * sw + sx]
                        if missing64(value, has_nodata, nodata):
                            continue
                        row_total += wx * value
                        row_weight += wx
                    if row_weight > 0.0:
                        total += wy * row_total / row_weight
                        weight += wy
                dst[(b * dh + y) * dw + x] = (
                    total / weight if weight > 0.0 else nodata
                )


def average2_32_row(
    src: F32Ptr, dst: F32Ptr, sh: Int, sw: Int, dh: Int, dw: Int, task: Int
):
    comptime W = simd_width_of[DType.float32]()
    var b = task // dh
    var y = task - b * dh
    var top_base = (b * sh + 2 * y) * sw
    var bottom_base = top_base + sw
    var dest_base = (b * dh + y) * dw
    var x = 0
    while x + W <= dw:
        var top = src.load[width=2 * W](top_base + 2 * x)
        var bottom = src.load[width=2 * W](bottom_base + 2 * x)
        var top_even, top_odd = top.deinterleave()
        var bottom_even, bottom_odd = bottom.deinterleave()
        dst.store(
            dest_base + x,
            (top_even + top_odd + bottom_even + bottom_odd) * Float32(0.25),
        )
        x += W
    while x < dw:
        var source_x = 2 * x
        dst[dest_base + x] = (
            src[top_base + source_x]
            + src[top_base + source_x + 1]
            + src[bottom_base + source_x]
            + src[bottom_base + source_x + 1]
        ) * Float32(0.25)
        x += 1


def average2_64_row(
    src: F64Ptr, dst: F64Ptr, sh: Int, sw: Int, dh: Int, dw: Int, task: Int
):
    comptime W = simd_width_of[DType.float64]()
    var b = task // dh
    var y = task - b * dh
    var top_base = (b * sh + 2 * y) * sw
    var bottom_base = top_base + sw
    var dest_base = (b * dh + y) * dw
    var x = 0
    while x + W <= dw:
        var top = src.load[width=2 * W](top_base + 2 * x)
        var bottom = src.load[width=2 * W](bottom_base + 2 * x)
        var top_even, top_odd = top.deinterleave()
        var bottom_even, bottom_odd = bottom.deinterleave()
        dst.store(
            dest_base + x,
            (top_even + top_odd + bottom_even + bottom_odd) * 0.25,
        )
        x += W
    while x < dw:
        var source_x = 2 * x
        dst[dest_base + x] = (
            src[top_base + source_x]
            + src[top_base + source_x + 1]
            + src[bottom_base + source_x]
            + src[bottom_base + source_x + 1]
        ) * 0.25
        x += 1


def average32(
    src: F32Ptr,
    dst: F32Ptr,
    bands: Int,
    sh: Int,
    sw: Int,
    row: Int,
    col: Int,
    wh: Int,
    ww: Int,
    dh: Int,
    dw: Int,
    has_nodata: Bool,
    nodata: Float64,
):
    if (
        not has_nodata
        and row == 0
        and col == 0
        and wh == sh
        and ww == sw
        and sh == 2 * dh
        and sw == 2 * dw
    ):
        @parameter
        def work(task: Int):
            average2_32_row(src, dst, sh, sw, dh, dw, task)

        for task in range(bands * dh):
            work(task)
        return
    for b in range(bands):
        for y in range(dh):
            var top = Float64(row) + Float64(y) * Float64(wh) / Float64(dh)
            var bottom = Float64(row) + Float64(y + 1) * Float64(wh) / Float64(dh)
            var ystart = max(Int(floor(top)), 0)
            var yend = min(Int(ceil(bottom)), sh)
            for x in range(dw):
                var left = Float64(col) + Float64(x) * Float64(ww) / Float64(dw)
                var right = Float64(col) + Float64(x + 1) * Float64(ww) / Float64(dw)
                var xstart = max(Int(floor(left)), 0)
                var xend = min(Int(ceil(right)), sw)
                var total = 0.0
                var weight = 0.0
                for sy in range(ystart, yend):
                    var wy = min(bottom, Float64(sy + 1)) - max(top, Float64(sy))
                    for sx in range(xstart, xend):
                        var value = src[(b * sh + sy) * sw + sx]
                        if missing32(value, has_nodata, nodata):
                            continue
                        var wx = min(right, Float64(sx + 1)) - max(left, Float64(sx))
                        var w = wx * wy
                        total += w * Float64(value)
                        weight += w
                dst[(b * dh + y) * dw + x] = Float32(
                    total / weight if weight > 0.0 else nodata
                )


def average64(
    src: F64Ptr,
    dst: F64Ptr,
    bands: Int,
    sh: Int,
    sw: Int,
    row: Int,
    col: Int,
    wh: Int,
    ww: Int,
    dh: Int,
    dw: Int,
    has_nodata: Bool,
    nodata: Float64,
):
    if (
        not has_nodata
        and row == 0
        and col == 0
        and wh == sh
        and ww == sw
        and sh == 2 * dh
        and sw == 2 * dw
    ):
        @parameter
        def work(task: Int):
            average2_64_row(src, dst, sh, sw, dh, dw, task)

        for task in range(bands * dh):
            work(task)
        return
    for b in range(bands):
        for y in range(dh):
            var top = Float64(row) + Float64(y) * Float64(wh) / Float64(dh)
            var bottom = Float64(row) + Float64(y + 1) * Float64(wh) / Float64(dh)
            var ystart = max(Int(floor(top)), 0)
            var yend = min(Int(ceil(bottom)), sh)
            for x in range(dw):
                var left = Float64(col) + Float64(x) * Float64(ww) / Float64(dw)
                var right = Float64(col) + Float64(x + 1) * Float64(ww) / Float64(dw)
                var xstart = max(Int(floor(left)), 0)
                var xend = min(Int(ceil(right)), sw)
                var total = 0.0
                var weight = 0.0
                for sy in range(ystart, yend):
                    var wy = min(bottom, Float64(sy + 1)) - max(top, Float64(sy))
                    for sx in range(xstart, xend):
                        var value = src[(b * sh + sy) * sw + sx]
                        if missing64(value, has_nodata, nodata):
                            continue
                        var wx = min(right, Float64(sx + 1)) - max(left, Float64(sx))
                        var w = wx * wy
                        total += w * value
                        weight += w
                dst[(b * dh + y) * dw + x] = (
                    total / weight if weight > 0.0 else nodata
                )


def binary32(a: F32Ptr, b: F32Ptr, dst: F32Ptr, n: Int, op: Int):
    for i in range(n):
        if op == 0:
            dst[i] = a[i] + b[i]
        elif op == 1:
            dst[i] = a[i] - b[i]
        elif op == 2:
            dst[i] = a[i] * b[i]
        elif op == 3:
            dst[i] = a[i] / b[i]
        elif op == 4:
            dst[i] = min(a[i], b[i])
        else:
            dst[i] = max(a[i], b[i])


def binary64(a: F64Ptr, b: F64Ptr, dst: F64Ptr, n: Int, op: Int):
    for i in range(n):
        if op == 0:
            dst[i] = a[i] + b[i]
        elif op == 1:
            dst[i] = a[i] - b[i]
        elif op == 2:
            dst[i] = a[i] * b[i]
        elif op == 3:
            dst[i] = a[i] / b[i]
        elif op == 4:
            dst[i] = min(a[i], b[i])
        else:
            dst[i] = max(a[i], b[i])


def nd32(a: F32Ptr, b: F32Ptr, dst: F32Ptr, n: Int, fallback: Float64):
    for i in range(n):
        var denom = a[i] + b[i]
        dst[i] = Float32(fallback) if denom == 0.0 else (a[i] - b[i]) / denom


def nd64(a: F64Ptr, b: F64Ptr, dst: F64Ptr, n: Int, fallback: Float64):
    for i in range(n):
        var denom = a[i] + b[i]
        dst[i] = fallback if denom == 0.0 else (a[i] - b[i]) / denom


def linear32(
    bands: F32Ptr, weights: F32Ptr, dst: F32Ptr, count: Int, n: Int, bias: Float64
):
    comptime W = simd_width_of[DType.float32]()
    var i = 0
    while i + W <= n:
        var vector_acc = SIMD[DType.float32, W](Float32(bias))
        for b in range(count):
            vector_acc += bands.load[width=W](b * n + i) * weights[b]
        dst.store(i, vector_acc)
        i += W
    while i < n:
        var acc = Float32(bias)
        for b in range(count):
            acc += bands[b * n + i] * weights[b]
        dst[i] = acc
        i += 1


def linear64(
    bands: F64Ptr, weights: F64Ptr, dst: F64Ptr, count: Int, n: Int, bias: Float64
):
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= n:
        var vector_acc = SIMD[DType.float64, W](bias)
        for b in range(count):
            vector_acc += bands.load[width=W](b * n + i) * weights[b]
        dst.store(i, vector_acc)
        i += W
    while i < n:
        var acc = bias
        for b in range(count):
            acc += bands[b * n + i] * weights[b]
        dst[i] = acc
        i += 1


@export("mr_resample_f32")
def mr_resample_f32(
    src_addr: Int,
    dst_addr: Int,
    bands: Int,
    sh: Int,
    sw: Int,
    row: Int,
    col: Int,
    wh: Int,
    ww: Int,
    dh: Int,
    dw: Int,
    method: Int,
    has_nodata: Int,
    nodata: Float64,
) abi("C"):
    if method == 0:
        nearest32(p32(src_addr), p32(dst_addr), bands, sh, sw, row, col, wh, ww, dh, dw)
    elif method == 1:
        bilinear32(
            p32(src_addr), p32(dst_addr), bands, sh, sw, row, col, wh, ww, dh, dw,
            has_nodata != 0, nodata,
        )
    else:
        average32(
            p32(src_addr), p32(dst_addr), bands, sh, sw, row, col, wh, ww, dh, dw,
            has_nodata != 0, nodata,
        )


@export("mr_resample_f64")
def mr_resample_f64(
    src_addr: Int,
    dst_addr: Int,
    bands: Int,
    sh: Int,
    sw: Int,
    row: Int,
    col: Int,
    wh: Int,
    ww: Int,
    dh: Int,
    dw: Int,
    method: Int,
    has_nodata: Int,
    nodata: Float64,
) abi("C"):
    if method == 0:
        nearest64(p64(src_addr), p64(dst_addr), bands, sh, sw, row, col, wh, ww, dh, dw)
    elif method == 1:
        bilinear64(
            p64(src_addr), p64(dst_addr), bands, sh, sw, row, col, wh, ww, dh, dw,
            has_nodata != 0, nodata,
        )
    else:
        average64(
            p64(src_addr), p64(dst_addr), bands, sh, sw, row, col, wh, ww, dh, dw,
            has_nodata != 0, nodata,
        )


@export("mr_binary_f32")
def mr_binary_f32(
    a: Int, b: Int, dst: Int, n: Int, op: Int
) abi("C"):
    binary32(p32(a), p32(b), p32(dst), n, op)


@export("mr_binary_f64")
def mr_binary_f64(
    a: Int, b: Int, dst: Int, n: Int, op: Int
) abi("C"):
    binary64(p64(a), p64(b), p64(dst), n, op)


@export("mr_nd_f32")
def mr_nd_f32(
    a: Int, b: Int, dst: Int, n: Int, fallback: Float64
) abi("C"):
    nd32(p32(a), p32(b), p32(dst), n, fallback)


@export("mr_nd_f64")
def mr_nd_f64(
    a: Int, b: Int, dst: Int, n: Int, fallback: Float64
) abi("C"):
    nd64(p64(a), p64(b), p64(dst), n, fallback)


@export("mr_linear_f32")
def mr_linear_f32(
    bands: Int, weights: Int, dst: Int, count: Int, n: Int, bias: Float64
) abi("C"):
    linear32(p32(bands), p32(weights), p32(dst), count, n, bias)


@export("mr_linear_f64")
def mr_linear_f64(
    bands: Int, weights: Int, dst: Int, count: Int, n: Int, bias: Float64
) abi("C"):
    linear64(p64(bands), p64(weights), p64(dst), count, n, bias)
