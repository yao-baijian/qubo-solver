"""Arithmetic-precision modes for the DistIM coupling matmul.

The engine's dominant cost is the coupling product ``J @ c``. A ``precision``
mode lowers the arithmetic precision of that product — i.e. the *hardware*
arithmetic the machine would run on — while the outer dynamics continue to
accumulate in float32:

    fp32 : native float32                          (baseline)
    fp16 : native float16 tensor-core matmul       (real speedup on GPU)
    bf16 : native bfloat16 tensor-core matmul
    int8 : 8-bit signed fixed point  (torch._int_mm on CUDA, emulated elsewhere)
    int4 : 4-bit signed fixed point  (emulated matmul)
    fp8  : 8-bit float, e4m3fn grid  (cast via torch.float8_e4m3fn; matmul emulated)
    fp4  : 4-bit float, e2m1 grid    (emulated matmul)

All low-precision modes share the same per-row scale scheme::

    J  -> row scale   s_i = max_j |J_ij| / G              (G = grid max)
    Jq = q(J / s_i)                            on [-G, G]
    c  -> c in [-1, 1], scaled onto the grid:  cq = q(c * G)
    J @ c  ~  (Jq @ cq) * (s_i / G)                       (per-row scale)

so the dequantised field stays in float32 and the rest of the engine is
unchanged. ``int8``/``int4`` use the native int8 tensor-core matmul
``torch._int_mm`` on CUDA when available (2 values/byte for int4 are not yet
packed — the 4-bit grid is used but the matmul is emulated). ``fp8``/``fp4``
use the exact hardware value grids and an emulated fp32 matmul, which keeps
the numerical behaviour faithful while remaining portable across devices.
"""

from __future__ import annotations

from typing import Optional

import torch

PRECISIONS = ("fp32", "fp16", "bf16", "int8", "int4", "fp8", "fp4")

_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)

# --------------------------------------------------------------------------- #
# value grids
# --------------------------------------------------------------------------- #
# e2m1 (fp4): 1 sign + 2 exp (bias 1) + 1 mantissa -> {0, .5, 1, 1.5, 2, 3, 4, 6}
_FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP4_MAX = 6.0

# e4m3fn (fp8): 1 sign + 4 exp (bias 7) + 3 mantissa, finite-only.
# subnormals m*2^-9 (m=1..7); normals (1+m/8)*2^(e-7), e=1..14; max finite 448.
_FP8_MAX = 448.0


def quantize_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round ``x`` onto the nearest e2m1 grid value (symmetric, max 6.0)."""
    levels = torch.tensor(_FP4_LEVELS, device=x.device)
    sign = torch.sign(x)
    # nearest level == bucketize on the level midpoints (O(log n) binary
    # search kernel, much faster than an argmin over the grid)
    bounds = (levels[1:] + levels[:-1]) / 2.0
    idx = torch.bucketize(x.abs(), bounds)
    return levels[idx] * sign


def quantize_fp8(x: torch.Tensor) -> torch.Tensor:
    """Round ``x`` onto the e4m3fn grid via the native float8 cast.

    Returns float32 tensors holding exact e4m3fn values. Values beyond the
    grid are clamped to [-448, 448] first (e4m3fn saturates to NaN otherwise).
    """
    if _FP8_DTYPE is None:
        raise RuntimeError("torch.float8_e4m3fn is not available in this build")
    return torch.clamp(x, -_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE).to(torch.float32)


# --------------------------------------------------------------------------- #
# quantized matmul
# --------------------------------------------------------------------------- #
class PrecisionMatmul:
    """Quantized ``J @ c`` producing a float32 coupling field.

    ``J`` is quantized once at construction; ``c`` (float32, in [-1, 1] after
    the engine's clipping) is quantized on every call onto the target grid.
    The result is dequantized back to float32 so the outer SimCIM dynamics
    (pump, noise, accumulation) remain in float32.
    """

    def __init__(self, J: torch.Tensor, precision: Optional[str] = "fp32",
                 device: Optional[str] = None):
        self.precision = (precision or "fp32").lower()
        if self.precision not in PRECISIONS:
            raise ValueError(
                f"Unknown precision '{precision}'; choose from {PRECISIONS}"
            )
        self.device = device or str(J.device)
        self.J32 = J.to(self.device).to(torch.float32)
        self._use_int_mm = False

        if self.precision == "fp32":
            self.J = self.J32
        elif self.precision == "fp16":
            self.J = self.J32.half()
        elif self.precision == "bf16":
            self.J = self.J32.bfloat16()
        elif self.precision in ("int8", "int4"):
            g = 127.0 if self.precision == "int8" else 7.0
            self._grid_max = g
            self.row_scale = (
                self.J32.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / g
            )
            self.J = torch.clamp(torch.round(self.J32 / self.row_scale),
                                 -g, g).to(torch.int8)
            dev = torch.device(self.device)
            self._use_int_mm = dev.type == "cuda" and hasattr(torch, "_int_mm")
            self._k_pad = 0
            if self._use_int_mm:
                # torch._int_mm needs the inner (K) dim to be a multiple of 8;
                # zero-pad the columns of J (and rows of c) when it isn't.
                if self.J.size(1) % 8 != 0:
                    self._k_pad = 8 - (self.J.size(1) % 8)
                    self.J = torch.nn.functional.pad(
                        self.J, (0, self._k_pad)).contiguous()
                else:
                    self.J = self.J.contiguous()
        elif self.precision == "fp8":
            self._grid_max = _FP8_MAX
            self.row_scale = (
                self.J32.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / _FP8_MAX
            )
            self.J = quantize_fp8(self.J32 / self.row_scale)
        elif self.precision == "fp4":
            self._grid_max = _FP4_MAX
            self.row_scale = (
                self.J32.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / _FP4_MAX
            )
            self.J = quantize_fp4(self.J32 / self.row_scale)
        else:
            raise ValueError(f"Unknown precision '{precision}'")

    def __call__(self, c: torch.Tensor) -> torch.Tensor:
        """Coupling field ``J @ c`` for a float32 state ``c`` (N, 1)."""
        c = c.to(self.device)
        if self.precision == "fp32":
            return torch.matmul(self.J, c)
        if self.precision in ("fp16", "bf16"):
            return torch.matmul(self.J, c.to(self.J.dtype)).to(torch.float32)
        if self.precision in ("int8", "int4"):
            g = self._grid_max
            cq = torch.clamp(torch.round(c * g), -g, g).to(torch.int8)
            if self._use_int_mm:
                # torch._int_mm needs the inner (K) and outer (N) dims to be
                # multiples of 8; zero-pad J's columns (done at init) and cq's
                # rows/cols, then take the first output column. The zero-padded
                # K rows contribute nothing, so the result is exact.
                if self._k_pad:
                    cq = torch.nn.functional.pad(cq, (0, 0, 0, self._k_pad))
                n_pad = (8 - cq.size(1) % 8) % 8
                if n_pad:
                    cq = torch.nn.functional.pad(cq, (0, n_pad))
                out = torch._int_mm(self.J, cq.contiguous())      # int32 (N,8)
                out = out[:, :1]
            else:
                out = torch.matmul(self.J.float(), cq.float())
            return out.to(torch.float32) * (self.row_scale / g)
        if self.precision == "fp8":
            g = self._grid_max
            cq = quantize_fp8(torch.clamp(c * g, -g, g))
            return torch.matmul(self.J, cq) * (self.row_scale / g)
        if self.precision == "fp4":
            g = self._grid_max
            cq = quantize_fp4(c * g)
            return torch.matmul(self.J, cq) * (self.row_scale / g)
        raise ValueError(self.precision)
