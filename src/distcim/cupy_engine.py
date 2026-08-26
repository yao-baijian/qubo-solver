"""CuPy backend for the DistIM broadcast-frame engine.

A pure-CuPy implementation of the freeze-field distributed SimCIM following
the same **broadcast frame** as the torch backend:

* ``J`` is partitioned into ``nparts x nparts`` blocks; node ``m`` owns column
  block ``m`` (variables ``x_m``).
* At a synchronization step every node ``m`` computes its off-diagonal
  contributions ``c_{i,m} = J_{i,m} x_m`` for every ``i != m`` and broadcasts
  them; every node ``i`` receives ``c_{i,j}`` from all ``j != i`` and combines
  them **once** into a single frozen remote field ``c_remote``.
* Between syncs the field is only
  ``field_m = J_{m,m} x_m (local block, every step) + c_remote_m + h_m`` —
  the per-node remote contributions are never re-added step after step.

All low-precision modes (``fp16``/``bf16``/``int8``/``int4``/``fp8``/``fp4``)
are implemented as grid-quantized emulations in fp32 arithmetic (native fp16
kernels are not available in every CuPy build), so the results match the torch
emulated paths and the numbers are directly comparable.

The package needs the CUDA toolkit DLL directory to be importable; on Windows
``os.add_dll_directory`` is used on ``$CUDA_PATH/bin`` before CuPy loads.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import numpy as np


def _ensure_cuda_dlls():
    """Make the CUDA runtime DLLs findable (Windows). No-op elsewhere."""
    cuda_path = os.environ.get("CUDA_PATH", "")
    bin_dir = os.path.join(cuda_path, "bin") if cuda_path else ""
    if not bin_dir or not os.path.isdir(bin_dir):
        # common default install location
        cand = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.isdir(cand):
            vers = sorted(os.listdir(cand))
            if vers:
                bin_dir = os.path.join(cand, vers[-1], "bin")
    if os.path.isdir(bin_dir):
        try:
            os.add_dll_directory(bin_dir)
        except (AttributeError, OSError):
            pass


_ensure_cuda_dlls()

import cupy as cp  # noqa: E402

PRECISIONS = ("fp32", "fp16", "bf16", "int8", "int4", "fp8", "fp4")

# --------------------------------------------------------------------------- #
# value grids (identical to src/distcim/precision.py)
# --------------------------------------------------------------------------- #
# e2m1 (fp4): {0, .5, 1, 1.5, 2, 3, 4, 6}
_FP4_LEVELS = cp.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], cp.float32)
_FP4_MAX = 6.0
# e4m3fn (fp8): subnormals m*2^-9 (m=1..7); normals (1+m/8)*2^(e-7), e=1..14;
# plus the top exponent row (1+m/8)*2^8 for m=0..6 -> max 448 (480 is NaN).
_FP8_LEVELS = cp.asarray(
    sorted({0.0}
           | {m * 2.0 ** -9 for m in range(1, 8)}
           | {(1.0 + m / 8.0) * 2.0 ** (e - 7) for e in range(1, 15)
              for m in range(8)}
           | {(1.0 + m / 8.0) * 2.0 ** 8 for m in range(7)}),
    cp.float32,
)
_FP8_MAX = 448.0


def _quantize_grid(x: cp.ndarray, levels: cp.ndarray) -> cp.ndarray:
    """Round onto the nearest value in ``levels`` (symmetric grid)."""
    sign = cp.sign(x).reshape(-1)
    ax = cp.abs(x).reshape(-1)
    d = cp.abs(ax[:, None] - levels[None, :])
    q = levels[cp.argmin(d, axis=1)]
    return (q * sign).reshape(x.shape)


def _round_grid(x: cp.ndarray, mantissa_bits: int, min_normal: float,
                max_val: float) -> cp.ndarray:
    """Round float32 onto a float grid (``mantissa_bits``) in fp32 arithmetic.

    Normalizes to [1, 2), rounds the mantissa, then rescales — this avoids the
    ``2**exp`` underflow that a direct step-size formulation hits for tiny
    values (which previously produced NaN in bf16).
    """
    sign = cp.sign(x)
    ax = cp.clip(cp.abs(x), min_normal, max_val)
    exp = cp.floor(cp.log2(ax))
    norm = ax * cp.exp2(-exp)                       # [1, 2)
    norm_q = cp.rint(norm * (2 ** mantissa_bits)) / (2 ** mantissa_bits)
    q = cp.exp2(exp) * norm_q
    q = cp.where(cp.abs(x) < min_normal * 0.5, 0.0, q)
    return sign * q


def _round_fp16(x: cp.ndarray) -> cp.ndarray:
    """Round float32 onto the fp16 grid (10 mantissa bits, emulated)."""
    return _round_grid(x, mantissa_bits=10, min_normal=6.1e-5, max_val=65504.0)


def _round_bf16(x: cp.ndarray) -> cp.ndarray:
    """Round float32 onto the bf16 grid (8 mantissa bits, emulated)."""
    return _round_grid(x, mantissa_bits=8, min_normal=1.2e-38, max_val=3.39e38)


def _quantize_fixed(x: cp.ndarray, bits: Optional[int], scale: float = 1.0):
    """Signed fixed-point quantization (same convention as the torch engine)."""
    if bits is None or scale is None:
        return x
    if scale == 0.0:
        return cp.zeros_like(x)
    levels = float(2 ** (bits - 1))
    return (cp.clip(x / scale, -1.0, 1.0) * levels).round() / levels * scale


def _pad_square_cp(t: cp.ndarray, size: int) -> cp.ndarray:
    """Zero-pad a 2D cupy array to ``(size, size)``."""
    r, c = t.shape
    if r == size and c == size:
        return cp.ascontiguousarray(t)
    out = cp.zeros((size, size), dtype=t.dtype)
    out[:r, :c] = t
    return out


def _pad_rows_cp(t: cp.ndarray, rows: int) -> cp.ndarray:
    """Zero-pad the first dim of a 2D cupy array to ``rows``."""
    r = t.shape[0]
    if r == rows:
        return cp.ascontiguousarray(t)
    out = cp.zeros((rows, t.shape[1]), dtype=t.dtype)
    out[:r] = t
    return out


def _quantize_state_cp(xp: cp.ndarray, precision: str, g: float) -> cp.ndarray:
    """Quantize the full state onto the precision grid (elementwise)."""
    if precision in ("int8", "int4"):
        return cp.clip(cp.rint(xp * g), -g, g)
    if precision == "fp8":
        return _quantize_grid(cp.clip(xp * g, -g, g), _FP8_LEVELS)
    if precision == "fp4":
        return _quantize_grid(xp * g, _FP4_LEVELS)
    raise ValueError(precision)


# --------------------------------------------------------------------------- #
# quantized matmul
# --------------------------------------------------------------------------- #
class CupyPrecisionMatmul:
    """Quantized ``J @ c`` in fp32 arithmetic producing an fp32 coupling field.

    ``J`` is quantized once at construction; ``c`` (fp32, in [-1, 1] after the
    engine's clipping) is quantized on every call. Integer/float grids use the
    same per-row scale scheme as the torch :class:`PrecisionMatmul`.
    """

    def __init__(self, J, precision: Optional[str] = "fp32"):
        self.precision = (precision or "fp32").lower()
        if self.precision not in PRECISIONS:
            raise ValueError(
                f"Unknown precision '{precision}'; choose from {PRECISIONS}"
            )
        self.J32 = cp.asarray(J, dtype=cp.float32)
        if self.precision == "fp32":
            self.J = self.J32
        elif self.precision == "fp16":
            self.J = _round_fp16(self.J32)
        elif self.precision == "bf16":
            self.J = _round_bf16(self.J32)
        elif self.precision in ("int8", "int4"):
            g = 127.0 if self.precision == "int8" else 7.0
            self._grid_max = g
            self.row_scale = (
                cp.max(cp.abs(self.J32), axis=1, keepdims=True)
                .clip(1e-8, None) / g
            )
            self.J = cp.clip(cp.rint(self.J32 / self.row_scale), -g, g)
        elif self.precision == "fp8":
            self._grid_max = _FP8_MAX
            self.row_scale = (
                cp.max(cp.abs(self.J32), axis=1, keepdims=True)
                .clip(1e-8, None) / _FP8_MAX
            )
            self.J = _quantize_grid(self.J32 / self.row_scale, _FP8_LEVELS)
        elif self.precision == "fp4":
            self._grid_max = _FP4_MAX
            self.row_scale = (
                cp.max(cp.abs(self.J32), axis=1, keepdims=True)
                .clip(1e-8, None) / _FP4_MAX
            )
            self.J = _quantize_grid(self.J32 / self.row_scale, _FP4_LEVELS)
        else:
            raise ValueError(f"Unknown precision '{precision}'")

    def __call__(self, c) -> cp.ndarray:
        c = cp.asarray(c, dtype=cp.float32)
        if self.precision == "fp32":
            return self.J @ c
        if self.precision == "fp16":
            return self.J @ _round_fp16(c)
        if self.precision == "bf16":
            return self.J @ _round_bf16(c)
        if self.precision in ("int8", "int4"):
            g = self._grid_max
            cq = cp.clip(cp.rint(c * g), -g, g)
            return (self.J @ cq) * (self.row_scale / g)
        if self.precision == "fp8":
            g = self._grid_max
            cq = _quantize_grid(cp.clip(c * g, -g, g), _FP8_LEVELS)
            return (self.J @ cq) * (self.row_scale / g)
        if self.precision == "fp4":
            g = self._grid_max
            cq = _quantize_grid(c * g, _FP4_LEVELS)
            return (self.J @ cq) * (self.row_scale / g)
        raise ValueError(self.precision)


# --------------------------------------------------------------------------- #
# SimCIM dynamics (SimplifiedSimCIM — one-component, linear + clipping)
# --------------------------------------------------------------------------- #
class CupySimCIMEngine:
    """One module of the broadcast frame: SimplifiedSimCIM in CuPy."""

    def __init__(self, n_local: int, coupler, xi: float = 1.0,
                 A_init: float = 1.0e-3, As: float = 70.0, dt: float = 0.1,
                 noise_scale: float = 1.0, x_bits: Optional[int] = None,
                 x_scale: float = 1.0):
        self.n_local = n_local
        self.coupler = coupler
        self.xi = float(xi)
        self.As = float(As)
        self.dt = float(dt)
        self.noise_scale = float(noise_scale)
        self.sqrt_dt = math.sqrt(self.dt)
        self.p = cp.float32(0.0)
        self.x_bits = x_bits
        self.x_scale = float(x_scale)
        # random-circle init: A * [cos(phi), sin(phi)]
        phases = cp.random.uniform(0.0, 2.0 * math.pi,
                                   size=(n_local, 1)).astype(cp.float32)
        self.c_comp = (cp.cos(phases) * A_init).astype(cp.float32)
        self.gauss_noise = cp.empty((n_local, 1), dtype=cp.float32)

    def _derivative(self):
        c = self.c_comp
        return [(-1.0 + self.p) * c + self.xi * self.coupler(c)]

    def _noise(self):
        c = self.c_comp
        return [cp.sqrt(c * c + 0.5) / self.As * self.noise_scale]

    def step(self):
        drift = self._derivative()[0]
        amp = self._noise()[0]
        self.gauss_noise = (cp.random.standard_normal(
            self.gauss_noise.shape, dtype=cp.float32) * self.sqrt_dt * amp)
        d_var = drift * self.dt + self.gauss_noise
        self.c_comp = cp.clip(self.c_comp + d_var, -1.0, 1.0)
        if self.x_bits is not None:
            self.c_comp = _quantize_fixed(self.c_comp, self.x_bits, self.x_scale)

    def set_p(self, value):
        self.p = cp.float32(value)

    @property
    def ising_state(self):
        return cp.sign(self.c_comp)


# --------------------------------------------------------------------------- #
# broadcast-frame coordinator (emulated, all nodes in one object)
# --------------------------------------------------------------------------- #
class CupyDistCoordinator:
    """Broadcast-frame emulation: node m computes c_{i,m}=J_{i,m} x_m and
    'sends' them; node i combines all received c_{i,j} into one frozen
    c_remote_i (the combined inter-module message)."""

    def __init__(self, J: cp.ndarray, h: cp.ndarray, slices, scheme: str,
                 time_intvl: int, quantize_bits: Optional[int],
                 precision: Optional[str]):
        self.slices = slices
        self.scheme = scheme
        self.time_intvl = time_intvl
        self.quantize_bits = quantize_bits
        self.precision = precision
        self.nparts = len(slices)
        self.blocks = [
            [CupyPrecisionMatmul(J[s_i:e_i, s_j:e_j], precision)
             for j, (s_j, e_j) in enumerate(slices)]
            for i, (s_i, e_i) in enumerate(slices)
        ]
        self.h_parts = [h[s:e] for (s, e) in slices]
        self.states: List[Optional[cp.ndarray]] = [None] * self.nparts
        self.c_remote: List[Optional[cp.ndarray]] = [None] * self.nparts
        self._old_remote: List[Optional[cp.ndarray]] = [None] * self.nparts
        self._fields: List[Optional[cp.ndarray]] = [None] * self.nparts
        self._t = 0

        # ---- batched single-GPU path (broadcast frame) ----------------
        # Uses the *padded uniform partition* [0,B), [B,2B), ... of the
        # zero-padded problem (Npad = n*B; rows N..Npad-1 zero), so every node
        # block aligns exactly to the B grid, the node matmuls batch into ONE
        # GEMM per step, and dist K=1 const == central CIM exactly.  The real
        # N variables stay in the first N rows of the padded layout, so
        # ``field[:N]`` is the true field.
        self.J_full = J
        self._B = max(e - s for (s, e) in slices)
        self._Npad = self.nparts * self._B
        self._pslices = [(m * self._B, min((m + 1) * self._B, J.shape[0]))
                         for m in range(self.nparts)]
        self._has_scale = hasattr(self.blocks[0][0], "row_scale")
        self._pad_h = cp.zeros((self._Npad, 1), dtype=cp.float32)
        self._pad_h[:h.shape[0]] = h
        self._cr_pad: Optional[cp.ndarray] = None
        self._old_cr_pad: Optional[cp.ndarray] = None

        if self._has_scale:
            # quantized grids: per-block quantization on the padded partition,
            # batched into one (n, n*B, B) bmm at each sync.
            pblocks = [
                [CupyPrecisionMatmul(J[s_i:e_i, s_j:e_j], precision)
                 for j, (s_j, e_j) in enumerate(self._pslices)]
                for i, (s_i, e_i) in enumerate(self._pslices)
            ]
            J4, scale4 = [], []
            for m in range(self.nparts):          # column block m
                for i in range(self.nparts):      # row block i -> block(i, m)
                    blk = pblocks[i][m]
                    J4.append(_pad_square_cp(blk.J, self._B))
                    scale4.append(_pad_rows_cp(blk.row_scale, self._B))
            self._J4 = cp.stack(J4).reshape(
                self.nparts, self.nparts, self._B, self._B)
            self._scale4 = cp.stack(scale4).reshape(
                self.nparts, self.nparts, self._B, 1)
            self._gm = float(pblocks[0][0]._grid_max)
            self._Jdiag = cp.stack(
                [self._J4[m, m] for m in range(self.nparts)])      # (n,B,B)
            self._scale_diag = cp.stack(
                [self._scale4[m, m] for m in range(self.nparts)])  # (n,B,1)
        else:
            # fp32/fp16/bf16 (elementwise rounded casts): diagonal blocks on
            # the padded partition batch into one bmm; the sync is one dense
            # GEMM of the full (padded) J.
            diag_blocks = []
            for m in range(self.nparts):
                s, e = self._pslices[m]
                if precision in (None, "fp32"):
                    diag_blocks.append(_pad_square_cp(J[s:e, s:e], self._B))
                elif precision == "fp16":
                    diag_blocks.append(_pad_square_cp(_round_fp16(J[s:e, s:e]),
                                                      self._B))
                else:  # bf16
                    diag_blocks.append(_pad_square_cp(_round_bf16(J[s:e, s:e]),
                                                      self._B))
            self._Jdiag = cp.stack(diag_blocks)
            if precision in (None, "fp32"):
                Jfull = J
            elif precision == "fp16":
                Jfull = _round_fp16(J)
            else:  # bf16
                Jfull = _round_bf16(J)
            self._Jfull = _pad_square_cp(Jfull, self._Npad)

    def prepare_step(self, is_sync: bool):
        n = self.nparts
        if self.scheme == "standard" or is_sync:
            new_remote = [cp.zeros_like(self.h_parts[i]) for i in range(n)]
            for j in range(n):
                xj = self.states[j]
                for i in range(n):
                    if i != j:
                        new_remote[i] = new_remote[i] + self.blocks[i][j](xj)
            for i in range(n):
                cr = new_remote[i]
                if self.quantize_bits:
                    qs = float(cp.max(cp.abs(cr))) or 1.0
                    cr = _quantize_fixed(cr, self.quantize_bits, scale=qs)
                self._old_remote[i] = self.c_remote[i]
                self.c_remote[i] = cr

        for m in range(n):
            intra = self.blocks[m][m](self.states[m])
            if self.scheme == "pulse":
                old = self._old_remote[m]
                if is_sync:
                    self._fields[m] = intra + self.time_intvl * (
                        old if old is not None else cp.zeros_like(intra)
                    ) + self.h_parts[m]
                else:
                    self._fields[m] = intra + self.h_parts[m]
            else:  # standard / const
                cr = self.c_remote[m]
                self._fields[m] = intra + (
                    cr if cr is not None else cp.zeros_like(intra)
                ) + self.h_parts[m]

        self._t += 1

    def field(self, m: int) -> cp.ndarray:
        return self._fields[m]

    def batched_field(self) -> cp.ndarray:
        """Concatenate all module fields into one ``(N, 1)`` tensor."""
        return cp.concatenate([self._fields[m] for m in range(self.nparts)])

    # ------------------------------------------------------------------ #
    # batched single-GPU path
    # ------------------------------------------------------------------ #
    def _quantize_remote(self, cr: cp.ndarray) -> cp.ndarray:
        """Per-node dynamic fixed-point range of the exchanged message."""
        crv = cr.reshape(self.nparts, self._B, 1)
        qs = cp.max(cp.abs(crv), axis=1, keepdims=True)
        qs = cp.maximum(qs, 1e-8)
        levels = float(2 ** (self.quantize_bits - 1))
        q = cp.clip(crv / qs, -1.0, 1.0) * levels
        return (q.round() / levels * qs).reshape(self._Npad, 1)

    def batched_prepare(self, x: cp.ndarray, is_sync: bool) -> cp.ndarray:
        """One-GPU batched broadcast-frame step; returns the field ``(N, 1)``.

        Same as the torch coordinator: the state lives in the zero-padded
        layout (Npad = n*B; real variables in the first N rows), so each step
        is ONE batched bmm for the local diagonal blocks (all nodes in
        parallel) and a sync is one dense GEMM (fp32/fp16/bf16) or one
        ``(n, n*B, B)`` bmm (quantized grids, per-block quantization kept)
        reproducing ``c_remote = J x - J_diag x``.
        """
        n = self.nparts
        B = self._B
        xp = cp.zeros((self._Npad, 1), dtype=cp.float32)
        xp[:x.shape[0]] = x
        xb = xp.reshape(n, B, 1)

        # ---- local diagonal blocks (every step): one batched bmm ----
        if self._has_scale:
            xq = _quantize_state_cp(xp, self.precision, self._gm)
            diag = cp.matmul(self._Jdiag, xq.reshape(n, B, 1))
            diag = (diag * (self._scale_diag / self._gm)).reshape(self._Npad, 1)
        else:
            diag = cp.matmul(self._Jdiag, xb).reshape(self._Npad, 1)

        # ---- sync: recompute the frozen remote field c_remote ----
        if self.scheme == "standard" or is_sync:
            old_cr = self._cr_pad
            if self._has_scale:
                # one (n, n*B, B) bmm: out[m, i] = J_{i,m} x_m (per-block q)
                out = cp.matmul(self._J4.reshape(n, n * B, B),
                                xq.reshape(n, B, 1))               # (n,nB,1)
                out = (out.reshape(n, n, B, 1)
                       * (self._scale4 / self._gm))                 # (n,n,B,1)
                cr = out.sum(axis=0) - diag.reshape(n, B, 1)        # (n,B,1)
                cr = cr.reshape(self._Npad, 1)
            else:
                full = cp.matmul(self._Jfull, xp)                   # (Npad,1)
                cr = full - diag
            if self.quantize_bits:
                cr = self._quantize_remote(cr)
            self._old_cr_pad = old_cr
            self._cr_pad = cr

        # ---- field = diag + frozen remote + h ----
        cr = self._cr_pad
        if self.scheme == "pulse":
            if is_sync:
                old = self._old_cr_pad
                field = diag + self.time_intvl * (
                    old if old is not None else cp.zeros_like(diag))
            else:
                field = diag
        else:  # standard / const
            field = diag + (cr if cr is not None else cp.zeros_like(diag))
        field = field + self._pad_h
        self._t += 1
        return field[:x.shape[0]]


# --------------------------------------------------------------------------- #
# public engine
# --------------------------------------------------------------------------- #
class CupyDistCIM:
    """CuPy distributed SimCIM solver (emulated broadcast frame)."""

    def __init__(
        self,
        J,
        h=None,
        nparts: int = 1,
        scheme: str = "standard",
        time_intvl: int = 10,
        xi: float = 1.0,
        A_init: float = 1.0e-3,
        As: float = 70.0,
        dt: float = 0.1,
        pump: str = "constant",
        pmax: float = 1.1,
        num_iters: int = 1000,
        seed: int = 0,
        noise_scale: float = 1.0,
        quantize_bits: Optional[int] = None,
        x_bits: Optional[int] = None,
        x_scale: float = 1.0,
        precision: Optional[str] = None,
        device: int = 0,
    ):
        if scheme not in ("standard", "const", "pulse"):
            raise ValueError(f"Unknown scheme '{scheme}'")
        with cp.cuda.Device(device):
            self.J = cp.asarray(J, dtype=cp.float32)
            N = self.J.shape[0]
            self.h = (cp.zeros((N, 1), dtype=cp.float32) if h is None
                      else cp.asarray(h, dtype=cp.float32))
            self.scheme = scheme
            self.time_intvl = time_intvl
            self.quantize_bits = quantize_bits
            self.num_iters = num_iters
            self.noise_scale = noise_scale
            self.slices = _partition_columns(N, nparts)
            self.nparts = nparts

            # resolve the paper's coupling gain
            # xi = 1/2 * (sum J^2 / (n-1))^{-1/2}  (Methods: scaled inverse RMS)
            if isinstance(xi, str):
                if xi == "inverse_interaction_rms":
                    rms = cp.sqrt(cp.sum(self.J * self.J) / (N - 1))
                    xi = float(0.5 / rms)
                else:
                    raise ValueError(f"Unknown xi mode '{xi}'")

            self.coordinator = CupyDistCoordinator(
                self.J, self.h, self.slices, scheme, time_intvl,
                quantize_bits, precision)

            if pump == "constant":
                self._pump = np.full(num_iters, pmax, dtype=np.float32)
            elif pump == "linear":
                span = int(num_iters * 0.5)
                self._pump = np.concatenate([
                    np.linspace(0.0, pmax, span),
                    np.full(num_iters - span, pmax),
                ]).astype(np.float32)
            else:
                raise ValueError(f"Unknown pump '{pump}'")

            cp.random.seed(seed)
            self.modules: List[CupySimCIMEngine] = []
            for m, (s, e) in enumerate(self.slices):
                eng = CupySimCIMEngine(
                    n_local=e - s,
                    coupler=lambda c, _m=m: self.coordinator.field(_m),
                    xi=xi, A_init=A_init, As=As, dt=dt,
                    noise_scale=noise_scale, x_bits=x_bits, x_scale=x_scale,
                )
                self.modules.append(eng)
                self.coordinator.states[m] = eng.c_comp

    def run(self) -> Tuple[np.ndarray, float]:
        # one-component SimplifiedSimCIM uses batched dynamics (vectorised
        # over all N); the cupy engine only implements the one-component model.
        xi = self.modules[0].xi
        As = self.modules[0].As
        dt = self.modules[0].dt
        sqrt_dt = math.sqrt(dt)
        noise_scale = self.modules[0].noise_scale
        x_bits = self.modules[0].x_bits
        x_scale = self.modules[0].x_scale

        # initial state = concatenated module phases (drawn in module order)
        x = cp.concatenate([eng.c_comp for eng in self.modules])   # (N, 1)
        for t in range(self.num_iters):
            p = float(self._pump[t])
            # batched broadcast frame: one bmm per step + dense GEMM / batched
            # bmm at each sync (all nodes on the one GPU)
            field = self.coordinator.batched_prepare(
                x, t % self.time_intvl == 0)

            # SimplifiedSimCIM dynamics (one vectorised pass over all N)
            noise = cp.random.standard_normal(
                x.shape, dtype=cp.float32) * sqrt_dt
            amp = cp.sqrt(x * x + 0.5) / As * noise_scale
            x = cp.clip(x + ((-1.0 + p) * x + xi * field) * dt
                        + noise * amp, -1.0, 1.0)
            if x_bits is not None:
                x = _quantize_fixed(x, x_bits, x_scale)

        for m, (s, e) in enumerate(self.slices):
            self.modules[m].c_comp = x[s:e]

        spins = cp.concatenate([eng.ising_state for eng in self.modules])
        energy = _ising_energy(spins, self.J, self.h)
        return spins.reshape(-1).get(), float(energy)


def _partition_columns(N: int, nparts: int) -> List[Tuple[int, int]]:
    base = N // nparts
    rem = N % nparts
    slices = []
    start = 0
    for m in range(nparts):
        length = base + (1 if m < rem else 0)
        slices.append((start, start + length))
        start += length
    return slices


def _ising_energy(sigma: cp.ndarray, J: cp.ndarray, h: cp.ndarray) -> float:
    s = sigma.reshape(-1, 1).astype(cp.float32)
    e = -0.5 * (s.T @ (J @ s)) - h.T @ s
    return float(e.item())


# --------------------------------------------------------------------------- #
# real multi-GPU broadcast frame over cupyx.distributed (NCCL)
# --------------------------------------------------------------------------- #
class CupyDistNCCLFieldCoupler:
    """Real multi-GPU broadcast frame over ``cupyx.distributed`` (NCCL).

    One instance per rank.  Node ``m`` owns column block ``m`` of ``J`` (its
    ``(N, part_len)`` columns).  At a sync it computes the off-diagonal
    contributions ``c_{i,m} = J_{i,m} x_m`` for every ``i`` and exchanges them
    with ``all_to_all``; it receives ``c_{m,j}`` from every ``j != m`` and
    combines them **once** into a single frozen remote field ``c_remote``.
    Between syncs the field is only
    ``field = J_{m,m} x_m (local block, every step) + c_remote + h_m``.

    This is the 4x RTX 4090 execution path (one process per GPU, NCCL
    all_to_all for the exchange).  ``cupyx.distributed`` bundles NCCL, so no
    CUDA C is needed.
    """

    def __init__(self, J_part, h_part, start: int, end: int,
                 scheme: str = "standard", time_intvl: int = 10,
                 quantize_bits: Optional[int] = None,
                 precision: Optional[str] = None,
                 rank: int = 0, nparts: int = 1, comm=None):
        self.J = J_part                       # (N, part_len) column block
        self.h_part = h_part
        self.scheme = scheme
        self.time_intvl = time_intvl
        self.quantize_bits = quantize_bits
        self.precision = precision
        self.rank = rank
        self.nparts = nparts
        self.comm = comm
        self.slices = _partition_columns(J_part.shape[0], nparts)
        # blocks[i] = J[S_i, S_m] (part_len x part_len) for this rank's column
        self.blocks = [CupyPrecisionMatmul(J_part[s:e], precision)
                       for (s, e) in self.slices]
        self.c_remote = cp.zeros_like(self.h_part)
        self._old_remote = cp.zeros_like(self.h_part)
        self._t = 0

    @property
    def is_synchronization_step(self):
        return self._t % self.time_intvl == 0

    def __call__(self, state: cp.ndarray) -> cp.ndarray:
        m, n = self.rank, self.nparts
        intra = self.blocks[m](state)          # J_{m,m} x_m (local, every step)

        if self.scheme == "standard" or self.is_synchronization_step:
            # broadcast frame: send[i] = c_{i,m} to node i (skip self)
            send = cp.stack([self.blocks[i](state) for i in range(n)])
            send[m] = 0.0
            recv = cp.empty_like(send)
            self.comm.all_to_all(send, recv)   # recv[j] = c_{m,j} from node j
            cr = cp.zeros_like(self.h_part)
            for j in range(n):
                if j != m:
                    cr = cr + recv[j]
            if self.quantize_bits:
                qs = float(cp.max(cp.abs(cr))) or 1.0
                cr = _quantize_fixed(cr, self.quantize_bits, scale=qs)
            self._old_remote = self.c_remote.copy()
            self.c_remote = cr
            self._t += 1
            if self.scheme == "pulse":
                return intra + self.time_intvl * self._old_remote + self.h_part
            return intra + self.c_remote + self.h_part   # standard / const

        # between syncs: local block + single combined frozen remote (const)
        self._t += 1
        if self.scheme == "const":
            return intra + self.c_remote + self.h_part
        elif self.scheme == "pulse":
            return intra + self.h_part
        raise ValueError(f"Unknown scheme '{self.scheme}'")


class CupyDistCIMNCCL:
    """Per-rank multi-GPU DistIM over ``cupyx.distributed`` (NCCL).

    One instance per rank; rank ``m`` runs on device ``m`` with only its
    column block of ``J``.  The broadcast-frame field is produced by the
    :class:`CupyDistNCCLFieldCoupler` (NCCL all_to_all at each sync) and the
    one-component SimCIM dynamics are integrated on the local variables.
    At the end the spins are all-gathered and the full Ising energy is
    reduced across ranks (rank 0 returns the same value on every rank).

    Launch e.g. via ``tools/run_distcim_multigpu.py`` (one process per GPU).
    """

    def __init__(self, J_part, h_part, rank: int, world_size: int, comm,
                 scheme: str = "const", time_intvl: int = 10,
                 xi: float = 1.0, A_init: float = 1.0e-3, As: float = 70.0,
                 dt: float = 0.1, pump: str = "constant", pmax: float = 1.1,
                 num_iters: int = 1000, seed: int = 0,
                 noise_scale: float = 1.0, quantize_bits: Optional[int] = None,
                 x_bits: Optional[int] = None, x_scale: float = 1.0,
                 precision: Optional[str] = None, device: Optional[int] = None):
        self.rank = rank
        self.world_size = world_size
        self.comm = comm
        self.device = cp.cuda.Device(device if device is not None else rank)
        with self.device:
            self.J_part = cp.asarray(J_part, dtype=cp.float32)
            N = self.J_part.shape[0]
            self.h_part = (cp.zeros((N, 1), dtype=cp.float32) if h_part is None
                           else cp.asarray(h_part, dtype=cp.float32))
            self.slices = _partition_columns(N, world_size)
            s, e = self.slices[rank]
            self.start_idx, self.end_idx = s, e
            # coupling gain (paper)
            if isinstance(xi, str):
                if xi == "inverse_interaction_rms":
                    rms = cp.sqrt(cp.sum(self.J_part * self.J_part) / (N - 1))
                    xi = float(0.5 / rms)
                else:
                    raise ValueError(f"Unknown xi mode '{xi}'")

            # deterministic init: rank m draws the same phases as module m of
            # the emulated single-GPU solver (same global RNG stream).
            cp.random.seed(seed)
            for _ in range(rank):
                cp.random.uniform(0.0, 2.0 * math.pi,
                                  size=(e - s, 1)).astype(cp.float32)

            self.coupler = CupyDistNCCLFieldCoupler(
                self.J_part, self.h_part, s, e, scheme, time_intvl,
                quantize_bits, precision, rank, world_size, comm)
            self.engine = CupySimCIMEngine(
                n_local=e - s,
                coupler=self.coupler,
                xi=xi, A_init=A_init, As=As, dt=dt,
                noise_scale=noise_scale, x_bits=x_bits, x_scale=x_scale,
            )
            if pump == "constant":
                self._pump = np.full(num_iters, pmax, dtype=np.float32)
            elif pump == "linear":
                span = int(num_iters * 0.5)
                self._pump = np.concatenate([
                    np.linspace(0.0, pmax, span),
                    np.full(num_iters - span, pmax)]).astype(np.float32)
            else:
                raise ValueError(f"Unknown pump '{pump}'")
            self.num_iters = num_iters

    def run(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return ``(local_spins (L,), full_spins (N,), energy)`` (same on all
        ranks).  ``full_spins`` is the all-gathered spin assignment."""
        with self.device:
            eng = self.engine
            for t in range(self.num_iters):
                eng.set_p(float(self._pump[t]))
                eng.step()
            local = cp.sign(eng.c_comp).reshape(-1).astype(cp.float32)  # (L,)
            L = local.shape[0]
            full = cp.empty((self.world_size * L,), dtype=cp.float32)
            self.comm.all_gather(local, full, count=L)               # (N,)
            # distributed energy: E = -1/2 s^T J s - h^T s
            # rank m contributes -1/2 s^T (J_part @ s_m) - h_m^T s_m
            s_all = full.reshape(-1, 1)
            col = self.J_part @ eng.c_comp                           # (N,1)
            partial = cp.float32(-0.5) * (s_all.T @ col) - (
                self.h_part.T @ eng.c_comp)
            energy = cp.empty((1,), dtype=cp.float32)
            self.comm.all_reduce(partial, energy, op="sum")
            return local.get(), full.get(), float(energy[0])


def solve_ising_cupy_nccl(
    J_part, h_part, rank: int, world_size: int, comm,
    scheme: str = "const", time_intvl: int = 10,
    xi: float = 1.0, A_init: float = 1.0e-3, As: float = 70.0, dt: float = 0.1,
    pump: str = "constant", pmax: float = 1.1, num_iters: int = 1000,
    seed: int = 0, noise_scale: float = 1.0, quantize_bits: Optional[int] = None,
    x_bits: Optional[int] = None, x_scale: float = 1.0,
    precision: Optional[str] = None, device: Optional[int] = None,
):
    """Per-rank multi-GPU solve over cupyx.distributed NCCL.

    Returns ``(local_spins (L,), full_spins (N,), energy)`` — the same result
    on every rank.  ``J_part`` is this rank's ``(N, part_len)`` column block.
    """
    engine = CupyDistCIMNCCL(
        J_part, h_part, rank, world_size, comm, scheme=scheme,
        time_intvl=time_intvl, xi=xi, A_init=A_init, As=As, dt=dt,
        pump=pump, pmax=pmax, num_iters=num_iters, seed=seed,
        noise_scale=noise_scale, quantize_bits=quantize_bits,
        x_bits=x_bits, x_scale=x_scale, precision=precision, device=device)
    return engine.run()


def solve_ising_cupy(
    J,
    h=None,
    nparts: int = 1,
    scheme: str = "standard",
    time_intvl: int = 10,
    xi: float = 1.0,
    A_init: float = 1.0e-3,
    As: float = 70.0,
    dt: float = 0.1,
    pump: str = "constant",
    pmax: float = 1.1,
    num_iters: int = 1000,
    seed: int = 0,
    noise_scale: float = 1.0,
    quantize_bits: Optional[int] = None,
    x_bits: Optional[int] = None,
    x_scale: float = 1.0,
    precision: Optional[str] = None,
    device: int = 0,
) -> Tuple[np.ndarray, float]:
    """CuPy emulated DistIM solve; returns ``(spins (N,), energy)``.

    ``J``/``h`` may be torch tensors, cupy arrays or numpy arrays.
    """
    engine = CupyDistCIM(
        J=J, h=h, nparts=nparts, scheme=scheme, time_intvl=time_intvl,
        xi=xi, A_init=A_init, As=As, dt=dt, pump=pump, pmax=pmax,
        num_iters=num_iters, seed=seed, noise_scale=noise_scale,
        quantize_bits=quantize_bits, x_bits=x_bits, x_scale=x_scale,
        precision=precision, device=device,
    )
    return engine.run()


class CupyDistCimSolver:
    """QUBO-level CuPy DistIM solver (``.solve(Q, num_vars) -> list[int]``)."""

    def __init__(self, num_iters: int = 1000, dt: float = 0.1, As: float = 70.0,
                 A_init: float = 1.0e-3, xi: float = 1.0, pump: str = "constant",
                 pmax: float = 1.1, nparts: int = 1, scheme: str = "const",
                 time_intvl: int = 10, quantize_bits: Optional[int] = None,
                 x_bits: Optional[int] = None, x_scale: float = 1.0,
                 precision: Optional[str] = None, seed: int = 0, device: int = 0):
        self.kw = dict(num_iters=num_iters, dt=dt, As=As, A_init=A_init, xi=xi,
                       pump=pump, pmax=pmax, nparts=nparts, scheme=scheme,
                       time_intvl=time_intvl, quantize_bits=quantize_bits,
                       x_bits=x_bits, x_scale=x_scale, precision=precision,
                       seed=seed, device=device)

    def solve(self, Q, num_vars) -> List[int]:
        Q_mat = cp.zeros((num_vars, num_vars), dtype=cp.float32)
        for i, j, val in Q:
            Q_mat[i, j] = val
            if i != j:
                Q_mat[j, i] = val
        J = -Q_mat / 2.0
        h = -(Q_mat.sum(axis=1, keepdims=True)) / 2.0
        spins, _ = solve_ising_cupy(J, h, **self.kw)
        return (spins > 0).astype(int).tolist()


__all__ = [
    "solve_ising_cupy",
    "solve_ising_cupy_nccl",
    "CupyDistCIM",
    "CupyDistCIMNCCL",
    "CupyDistCimSolver",
    "CupyDistNCCLFieldCoupler",
    "CupyPrecisionMatmul",
    "PRECISIONS",
]
