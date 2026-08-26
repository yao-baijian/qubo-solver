"""DistIM — distributed Ising dynamics with sparse synchronization.

Faithful port of the distribution scheme from the paper
*"Distributed Ising dynamics for real-time large-scale combinatorial
optimization"* (paper Sec. 1.2 & Methods 3.3) as implemented in the
*simulated-ising-machine* repo (``sim/parallel/update_wrapper.py`` and
``sim/parallel/distributed.py``).

The coupling matrix is split by columns into ``nparts`` modules:

    J = J_local (block-diagonal) + J_cross (inter-module)

Each module integrates its own dynamics exactly every step; only the
inter-module field ``J_c x`` is exchanged at sparse synchronization steps
(every ``time_intvl`` steps) using one of three schemes:

- ``standard`` : exchange every step (all-reduce) -> exactly the central machine
- ``const``    : hold the last message between syncs  (paper Eq. 15)
- ``pulse``    : zero between syncs, single impulse scaled by ``time_intvl``
                 at the next sync                          (paper Eq. 16)

Optional fixed-point quantization of the exchanged message reproduces a
low-precision communication link (``quantize_bits``), which is how the
algorithm is verified together with quantization.

Two backends:

- ``emulated`` : single process; all modules live in one object. Deterministic,
                 verifiable on CPU (used by the verification harness).
- ``torch``    : real ``torch.distributed`` all-reduce/all-gather; launch one
                 process per module (e.g. ``torchrun --nproc-per-node=N``).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

from .engines import SimCIMEngine, ising_energy, quantize_fixed
from .precision import (PrecisionMatmul, quantize_fp4, quantize_fp8)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def partition_columns(N: int, nparts: int) -> List[Tuple[int, int]]:
    """Contiguous column partition (same layout as the target repo's
    ``UniformIsingOptPartitioner`` with N divisible by nparts)."""
    base = N // nparts
    rem = N % nparts
    slices = []
    start = 0
    for m in range(nparts):
        length = base + (1 if m < rem else 0)
        slices.append((start, start + length))
        start += length
    return slices


def linear_schedule(num_iters: int, start: float, end: float, span: float):
    """Port of the target repo's ``LinearScheduler._set_schedule``."""
    linear_num = int(num_iters * span)
    linear_part = torch.linspace(start, end, linear_num)
    max_num = num_iters - linear_num
    max_part = end * torch.ones(max_num)
    return torch.hstack((linear_part, max_part))


def _pad_square(t: torch.Tensor, size: int) -> torch.Tensor:
    """Zero-pad a 2D tensor to ``(size, size)`` (no-op if already)."""
    r, c = t.shape
    if r == size and c == size:
        return t.contiguous()
    out = t.new_zeros(size, size)
    out[:r, :c] = t
    return out


def _pad_rows(t: torch.Tensor, rows: int) -> torch.Tensor:
    """Zero-pad the first dim of a 2D tensor to ``rows``."""
    r = t.shape[0]
    if r == rows:
        return t.contiguous()
    out = t.new_zeros(rows, t.shape[1])
    out[:r] = t
    return out


# --------------------------------------------------------------------------- #
# real torch.distributed field coupler (one process per module)
# --------------------------------------------------------------------------- #
class TorchDistFieldCoupler:
    """Per-module field wrapper using real ``torch.distributed``.

    Broadcast frame: ``J`` is a 4x4 (nparts x nparts) block matrix and node
    ``m`` owns column block ``m`` (variables ``x_m``). At a synchronization
    step node ``m`` computes its off-diagonal contributions
    ``c_{i,m} = J_{i,m} x_m`` for every ``i != m`` and broadcasts them
    (``dist.all_to_all``); every node ``i`` receives ``c_{i,j}`` from all
    ``j != i`` and combines them **once** into a single frozen remote field
    ``c_remote``. Between syncs the field is only
    ``J_{m,m} x_m (local block) + c_remote + h`` — the per-node contributions
    are never re-added step after step.

    Mirrors the target repo's ``StandardParallelUpdateWrapper`` /
    ``ConstApproxParallelUpdateWrapper`` / ``PulseApproxParallelUpdateWrapper``.
    """

    def __init__(
        self,
        J_part: torch.Tensor,          # (N, part_len) column block of this rank
        h_part: torch.Tensor,          # (part_len, 1) local external field
        start: int,
        end: int,
        scheme: str = "standard",
        time_intvl: int = 10,
        quantize_bits: Optional[int] = None,
        precision: Optional[str] = None,
        rank: int = 0,
        nparts: int = 1,
    ):
        self.J = J_part
        self.h_part = h_part
        self.start = start
        self.end = end
        self.scheme = scheme
        self.time_intvl = time_intvl
        self.quantize_bits = quantize_bits
        self.precision = precision
        self.rank = rank
        self.nparts = nparts
        dev = str(J_part.device)
        # row-block slices of the full matrix (this rank owns column block `rank`)
        self.slices = partition_columns(J_part.size(0), nparts)
        # blocks[i] = J[S_i, S_m] (part_len x part_len) for this rank's column m
        self.blocks = [
            PrecisionMatmul(J_part[s:e], precision, device=dev)
            for (s, e) in self.slices
        ]
        # single combined frozen remote field (inter-module message)
        self.c_remote = torch.zeros_like(self.h_part)
        self._old_remote = torch.zeros_like(self.h_part)
        self._t = 0

    @property
    def is_synchronization_step(self):
        return self._t % self.time_intvl == 0

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        m, n = self.rank, self.nparts
        intra = self.blocks[m](state)          # J_{m,m} x_m (local, every step)

        if self.scheme == "standard" or self.is_synchronization_step:
            # broadcast frame: compute the off-diagonal contributions c_{i,m}
            # for every node i and exchange them (all-to-all / all-gather).
            send = [self.blocks[i](state) for i in range(n)]   # c_{i,m}
            send[m] = torch.zeros_like(send[m])                # skip self
            recv = [torch.empty_like(self.h_part) for _ in range(n)]
            dist.all_to_all(recv, send)          # recv[j] = c_{m,j} from node j

            # combine the received contributions ONCE into a single remote field
            cr = torch.zeros_like(self.h_part)
            for j in range(n):
                if j != m:
                    cr = cr + recv[j]
            if self.quantize_bits:
                qs = cr.abs().max().item() or 1.0
                cr = quantize_fixed(cr, self.quantize_bits, scale=qs)
            self._old_remote = self.c_remote.clone()
            self.c_remote = cr
            self._t += 1

            if self.scheme == "pulse":
                # impulse: previous frozen remote scaled by K at the sync step
                return intra + self.time_intvl * self._old_remote + self.h_part
            return intra + self.c_remote + self.h_part   # standard / const

        # between syncs: local block + single combined frozen remote (const)
        self._t += 1
        if self.scheme == "const":
            return intra + self.c_remote + self.h_part
        elif self.scheme == "pulse":
            return intra + self.h_part
        raise ValueError(f"Unknown scheme '{self.scheme}'")


# --------------------------------------------------------------------------- #
# single-process emulated backend (all modules in one object)
# --------------------------------------------------------------------------- #
class _EmulatedCoordinator:
    """Broadcast-frame emulation of the DistIM freeze-field exchange.

    ``J`` is partitioned into ``nparts x nparts`` blocks; node ``m`` owns
    column block ``m`` (variables ``x_m``). At a synchronization step every
    node ``m`` computes its off-diagonal contributions ``c_{i,m} = J_{i,m} x_m``
    for every ``i != m`` and "broadcasts" them to node ``i``; every node ``i``
    receives ``c_{i,j}`` from all ``j != i`` and combines them **once** into a
    single frozen remote field ``c_remote``. Between syncs the field is only::

        field_m = J_{m,m} x_m (local block, every step) + c_remote_m + h_m

    so the per-node remote contributions are never re-added step after step.
    """

    def __init__(
        self,
        J: torch.Tensor,
        h: torch.Tensor,
        slices: List[Tuple[int, int]],
        scheme: str,
        time_intvl: int,
        quantize_bits: Optional[int],
        precision: Optional[str] = None,
    ):
        self.slices = slices
        self.scheme = scheme
        self.time_intvl = time_intvl
        self.quantize_bits = quantize_bits
        self.precision = precision
        self.nparts = len(slices)
        dev = str(J.device)
        # block matrix: blocks[i][j] = PrecisionMatmul(J[S_i, S_j])
        self.blocks = [
            [PrecisionMatmul(J[s_i:e_i, s_j:e_j], precision, device=dev)
             for j, (s_j, e_j) in enumerate(slices)]
            for i, (s_i, e_i) in enumerate(slices)
        ]
        self.h_parts = [h[s:e] for (s, e) in slices]
        # per-module current local states (updated by the driver)
        self.states: List[torch.Tensor] = [None] * self.nparts
        # single combined frozen remote field per node (inter-module message)
        self.c_remote: List[Optional[torch.Tensor]] = [None] * self.nparts
        # previous remote field (pulse scheme impulse)
        self._old_remote: List[Optional[torch.Tensor]] = [None] * self.nparts
        self._fields: List[Optional[torch.Tensor]] = [None] * self.nparts
        self._t = 0

        # ---- batched single-GPU path (broadcast frame) ----------------
        # The one-component batched path uses the *padded uniform partition*
        # [0,B), [B,2B), ... of the zero-padded problem (Npad = n*B; rows
        # N..Npad-1 are zero). This is the natural uniform partition the paper
        # assumes (N divisible by nparts), so every node block aligns exactly
        # to the B grid, the node matmuls batch into ONE GEMM per step, and
        # dist K=1 const == central CIM exactly.  The real N variables stay in
        # the first N rows of the padded layout, so ``field[:N]`` is the true
        # field.  (The remainder-carrying partition_columns — e.g. N=750,
        # nparts=4 -> 188/188/187/187 — would shift across B boundaries, so it
        # is only used by the per-module two-component path above.)
        self._B = max(e - s for (s, e) in slices)
        self._Npad = self.nparts * self._B
        self._pslices = [(m * self._B, min((m + 1) * self._B, J.shape[0]))
                         for m in range(self.nparts)]
        self._has_scale = hasattr(self.blocks[0][0], "row_scale")
        self._pad_h = torch.zeros(self._Npad, 1, device=J.device)
        self._pad_h[:h.size(0)] = h.to(J.device)
        self._cr_pad: Optional[torch.Tensor] = None
        self._old_cr_pad: Optional[torch.Tensor] = None

        if self._has_scale:
            # quantized grids: per-block quantization on the padded partition
            # (keeps each block's own row scales), batched into one
            # (n, n*B, B) bmm at each sync.
            pblocks = [
                [PrecisionMatmul(J[s_i:e_i, s_j:e_j], precision, device=dev)
                 for j, (s_j, e_j) in enumerate(self._pslices)]
                for i, (s_i, e_i) in enumerate(self._pslices)
            ]
            J4, scale4 = [], []
            for m in range(self.nparts):          # column block m
                for i in range(self.nparts):      # row block i -> block(i, m)
                    blk = pblocks[i][m]
                    Jb = blk.J.to(J.device).float()
                    # int8/int4 blocks carry extra zero columns padded to a
                    # multiple of 8 for torch._int_mm; trim them off (they
                    # contribute nothing) before the uniform-B pad.
                    Jb = Jb[:, :blk.J32.shape[1]]
                    J4.append(_pad_square(Jb, self._B))
                    scale4.append(_pad_rows(blk.row_scale.to(J.device),
                                            self._B))
            self._J4 = torch.stack(J4).view(self.nparts, self.nparts,
                                            self._B, self._B)
            self._scale4 = torch.stack(scale4).view(
                self.nparts, self.nparts, self._B, 1)
            self._gm = float(pblocks[0][0]._grid_max)
            self._Jdiag = torch.stack(
                [self._J4[m, m] for m in range(self.nparts)])      # (n,B,B)
            self._scale_diag = torch.stack(
                [self._scale4[m, m] for m in range(self.nparts)])  # (n,B,1)
        else:
            # fp32/fp16/bf16 (elementwise casts): diagonal blocks on the
            # padded partition batch into one bmm; the sync is one dense GEMM
            # of the full (padded) J.
            if self.precision in (None, "fp32"):
                dtype = torch.float32
            elif self.precision == "fp16":
                dtype = torch.float16
            else:  # bf16
                dtype = torch.bfloat16
            diag_blocks = []
            for m in range(self.nparts):
                s, e = self._pslices[m]
                diag_blocks.append(
                    _pad_square(J[s:e, s:e].to(J.device).to(dtype), self._B))
            self._Jdiag = torch.stack(diag_blocks)
            if self.precision in (None, "fp32"):
                Jfull = J
            elif self.precision == "fp16":
                Jfull = J.half()
            else:  # bf16
                Jfull = J.bfloat16()
            self._Jfull = _pad_square(Jfull.to(J.device), self._Npad)

        # bmm fast path (fp32, equal-sized blocks) for the per-module
        # (two-component) engine loop — kept as before.
        block_sizes = {e - s for (s, e) in slices}
        self._fast = precision in (None, "fp32") and len(block_sizes) == 1
        if self._fast:
            self._diag_J = torch.stack(
                [self.blocks[m][m].J32 for m in range(self.nparts)])  # (n,B,B)
            self._col_J = [
                torch.stack([self.blocks[i][j].J32 for i in range(self.nparts)])
                for j in range(self.nparts)
            ]  # col j -> (n, B, B)

    def _diag_field(self) -> torch.Tensor:
        """Local diagonal blocks applied to the current states (one bmm)."""
        if not self._fast:
            return torch.cat([self.blocks[m][m](self.states[m])
                              for m in range(self.nparts)], dim=0)
        xs = torch.stack([self.states[m] for m in range(self.nparts)])  # (n,B,1)
        return torch.bmm(self._diag_J, xs).reshape(-1, 1)

    def _column_contribs(self, j, xj) -> List[torch.Tensor]:
        """Node j's column contributions c_{i,j} = J_{i,j} x_j for all i."""
        if not self._fast:
            return [self.blocks[i][j](xj) for i in range(self.nparts)]
        xb = xj.unsqueeze(0).expand(self.nparts, -1, -1)          # (n, B, 1)
        out = torch.bmm(self._col_J[j], xb)                       # (n, B, 1)
        return [out[i] for i in range(self.nparts)]

    def prepare_step(self, is_sync: bool):
        """Compute per-module fields from the current state snapshot."""
        n = self.nparts
        if self.scheme == "standard" or is_sync:
            # broadcast frame: node j computes its off-diagonal contributions
            # c_{i,j} = J_{i,j} x_j and sends them to node i; node i combines
            # all received c_{i,j} (j != i) into a single c_remote_i.
            new_remote = [torch.zeros_like(self.h_parts[i]) for i in range(n)]
            for j in range(n):
                contribs = self._column_contribs(j, self.states[j])
                for i in range(n):
                    if i != j:
                        new_remote[i] = new_remote[i] + contribs[i]
            for i in range(n):
                cr = new_remote[i]
                if self.quantize_bits:
                    # dynamic fixed-point range of the exchanged message
                    qs = cr.abs().max().item() or 1.0
                    cr = quantize_fixed(cr, self.quantize_bits, scale=qs)
                self._old_remote[i] = self.c_remote[i]
                self.c_remote[i] = cr

        # field = local diagonal block (every step) + single frozen remote + h
        intra_all = self._diag_field()                       # (N, 1)
        for m in range(n):
            s, e = self.slices[m]
            intra = intra_all[s:e]
            if self.scheme == "pulse":
                old = self._old_remote[m]
                if is_sync:
                    self._fields[m] = intra + self.time_intvl * (
                        old if old is not None else torch.zeros_like(intra)
                    ) + self.h_parts[m]
                else:
                    self._fields[m] = intra + self.h_parts[m]
            else:  # standard / const
                cr = self.c_remote[m]
                self._fields[m] = intra + (
                    cr if cr is not None else torch.zeros_like(intra)
                ) + self.h_parts[m]

        self._t += 1

    def field(self, m: int) -> torch.Tensor:
        return self._fields[m]

    def batched_field(self) -> torch.Tensor:
        """Concatenate all module fields into one ``(N, 1)`` tensor."""
        return torch.cat([self._fields[m] for m in range(self.nparts)], dim=0)

    # ------------------------------------------------------------------ #
    # batched single-GPU path
    # ------------------------------------------------------------------ #
    def _quantize_state(self, xp: torch.Tensor) -> torch.Tensor:
        """Quantize a state tensor onto the grid (elementwise; the grid is
        shared by every block, so one pass serves all nodes)."""
        g = self._gm
        if self.precision in ("int8", "int4"):
            return torch.clamp(torch.round(xp * g), -g, g)
        if self.precision == "fp8":
            return quantize_fp8(torch.clamp(xp * g, -g, g))
        if self.precision == "fp4":
            return quantize_fp4(xp * g)
        raise ValueError(self.precision)

    def _quantize_remote(self, cr: torch.Tensor) -> torch.Tensor:
        """Per-node dynamic fixed-point range of the exchanged message."""
        crv = cr.view(self.nparts, self._B, 1)
        qs = crv.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        levels = float(2 ** (self.quantize_bits - 1))
        q = torch.clamp(crv / qs, -1.0, 1.0) * levels
        return (q.round() / levels * qs).view(self._Npad, 1)

    def batched_prepare(self, x: torch.Tensor, is_sync: bool) -> torch.Tensor:
        """One-GPU batched broadcast-frame step; returns the field ``(N, 1)``.

        The state lives in the zero-padded layout (Npad = n*B; real variables
        in the first N rows), so each step is ONE batched bmm for the local
        diagonal blocks (all nodes in parallel) and a sync is either one dense
        GEMM (fp32/fp16/bf16) or one ``(n, n*B, B)`` bmm (quantized grids,
        per-block quantization kept) reproducing ``c_remote = J x - J_diag x``.
        This is the broadcast frame on the *padded uniform partition*, and is
        what a single GPU runs instead of serially emulating nparts nodes.
        """
        n = self.nparts
        B = self._B
        xp = torch.zeros(self._Npad, 1, device=x.device)
        xp[:x.size(0)] = x
        xb = xp.view(n, B, 1)

        # ---- local diagonal blocks (every step): one batched bmm ----
        if self._has_scale:
            xq = self._quantize_state(xp)
            diag = torch.bmm(self._Jdiag, xq.view(n, B, 1))
            diag = (diag * (self._scale_diag / self._gm)).view(self._Npad, 1)
        else:
            diag = torch.bmm(self._Jdiag, xb.to(self._Jdiag.dtype))
            diag = diag.to(torch.float32).view(self._Npad, 1)

        # ---- sync: recompute the frozen remote field c_remote ----
        if self.scheme == "standard" or is_sync:
            old_cr = self._cr_pad
            if self._has_scale:
                # one (n, n*B, B) bmm: out[m, i] = J_{i,m} x_m (per-block q)
                out = torch.bmm(self._J4.reshape(n, n * B, B),
                                xq.view(n, B, 1))                  # (n,nB,1)
                out = (out.view(n, n, B, 1)
                       * (self._scale4 / self._gm))                 # (n,n,B,1)
                cr = out.sum(dim=0) - diag.view(n, B, 1)            # (n,B,1)
                cr = cr.view(self._Npad, 1)
            else:
                full = torch.matmul(self._Jfull, xp.to(self._Jfull.dtype))
                cr = full.to(torch.float32) - diag
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
                    old if old is not None else torch.zeros_like(diag))
            else:
                field = diag
        else:  # standard / const
            field = diag + (cr if cr is not None else torch.zeros_like(diag))
        field = field + self._pad_h
        self._t += 1
        return field[:x.size(0)]


# --------------------------------------------------------------------------- #
# public distributed engine
# --------------------------------------------------------------------------- #
class DistIMEngine:
    """Distributed SimCIM solver following the DistIM paradigm.

    Parameters
    ----------
    J, h : torch.Tensor
        Full Ising coupling matrix and external field (used by the emulated
        backend; ignored by the ``torch`` backend which loads per-rank parts).
    nparts : int
        Number of modules (partitions).
    scheme : str
        ``standard`` | ``const`` | ``pulse``.
    time_intvl : int
        Synchronization period ``K``.
    model : str
        ``StandardCIM`` | ``SimCIM`` | ``SimplifiedSimCIM``.
    xi : float or str
        Coupling gain (or ``'inverse_interaction_rms'``).
    A_init, As, dt : float
        SimCIM hyper-parameters (paper: A_init=1e-3, As=70, dt=0.5).
    pump : str
        ``constant`` (value ``pmax``) or ``linear`` (0 -> pmax).
    pmax : float
        Target pump value (paper: 1.1 for traffic, 1.0 for random graphs).
    num_iters : int
        Number of integration steps.
    seed : int
        RNG seed for initial phases and noise.
    noise_scale : float
        Multiplier for the noise term (verification helper; 0 => deterministic).
    device : str
    quantize_bits : int or None
        If set, the inter-module message is quantized to this many bits
        (communication link).
    x_bits, y_bits : int or None
        FPGA state quantization: the dynamical states are reduced to fixed
        point at every step — ``x`` (c-component) to ``x_bits`` (hardware: 8)
        and ``y`` (s-component, two-component models) to ``y_bits``
        (hardware: 16 or 32). Control params (pump, xi, dt, As) stay float.
        ``None`` disables state quantization (default).
    x_scale, y_scale : float
        Full-scale range of the fixed-point grid for ``x`` / ``y``.
    precision : str or None
        Arithmetic precision of the coupling matmul ``J @ c`` (see
        :mod:`src.distcim.precision`): ``fp32`` (default), ``fp16``, ``bf16``,
        ``int8``, ``int4``, ``fp8``, ``fp4``.
    backend : str
        ``emulated`` (single process) or ``torch`` (real torch.distributed).
    """

    def __init__(
        self,
        J: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
        nparts: int = 1,
        scheme: str = "standard",
        time_intvl: int = 10,
        model: str = "SimplifiedSimCIM",
        xi: float = 1.0,
        A_init: float = 1.0e-3,
        As: float = 70.0,
        dt: float = 0.1,
        pump: str = "constant",
        pmax: float = 1.1,
        num_iters: int = 1000,
        seed: int = 0,
        noise_scale: float = 1.0,
        device: str = "cpu",
        quantize_bits: Optional[int] = None,
        x_bits: Optional[int] = None,
        y_bits: Optional[int] = None,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        precision: Optional[str] = None,
        backend: str = "emulated",
        # torch backend overrides (when backend == "torch")
        rank: int = 0,
        world_size: int = 1,
        J_part: Optional[torch.Tensor] = None,
        h_part: Optional[torch.Tensor] = None,
        start_idx: int = 0,
        end_idx: int = 0,
    ):
        if scheme not in ("standard", "const", "pulse"):
            raise ValueError(f"Unknown scheme '{scheme}'")
        self.backend = backend
        self.scheme = scheme
        self.time_intvl = time_intvl
        self.model = model
        self.num_iters = num_iters
        self.device = device
        self.quantize_bits = quantize_bits
        self.x_bits = x_bits
        self.y_bits = y_bits
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.precision = precision
        self.seed = seed
        self.noise_scale = noise_scale
        self.rank = rank
        self.world_size = world_size

        if pump == "constant":
            self._pump_values = pmax * torch.ones(num_iters, device=self.device)
        elif pump == "linear":
            self._pump_values = linear_schedule(num_iters, 0.0, pmax,
                                                span=0.5).to(self.device)
        else:
            raise ValueError(f"Unknown pump '{pump}'")

        if backend == "emulated":
            self._build_emulated(J, h, nparts, xi, A_init, As, dt)
        elif backend == "torch":
            self._build_torch(J_part, h_part, start_idx, end_idx,
                              xi, A_init, As, dt)
        else:
            raise ValueError(f"Unknown backend '{backend}'")

    # ------------------------------------------------------------------ #
    def _build_emulated(self, J, h, nparts, xi, A_init, As, dt):
        self._J = J.to(self.device)
        self._h = h.to(self.device)
        N = self._J.size(0)
        self._slices = partition_columns(N, nparts)
        self._nparts = nparts

        self.coordinator = _EmulatedCoordinator(
            self._J, self._h, self._slices, self.scheme, self.time_intvl,
            self.quantize_bits, self.precision,
        )

        # Seed once; each module draws its own init phases in module order.
        torch.manual_seed(self.seed)
        self.modules: List[SimCIMEngine] = []
        for m, (s, e) in enumerate(self._slices):
            engine = SimCIMEngine(
                n_local=e - s,
                coupler=lambda c, _m=m: self.coordinator.field(_m),
                J=self._J,
                xi=xi,
                A_init=A_init,
                As=As,
                dt=dt,
                model=self.model,
                device=self.device,
                noise_scale=self.noise_scale,
                x_bits=self.x_bits,
                y_bits=self.y_bits,
                x_scale=self.x_scale,
                y_scale=self.y_scale,
            )
            self.modules.append(engine)
            self.coordinator.states[m] = engine.c_comp

    def _build_torch(self, J_part, h_part, start_idx, end_idx, xi, A_init, As, dt):
        J_part = J_part.to(self.device)
        if h_part is None:
            h_part = torch.zeros(end_idx - start_idx, 1, device=self.device)
        else:
            h_part = h_part.to(self.device)
        torch.manual_seed(self.seed)
        coupler = TorchDistFieldCoupler(
            J_part, h_part, start_idx, end_idx, self.scheme, self.time_intvl,
            self.quantize_bits, self.precision, self.rank, self.world_size,
        )
        self.modules = [
            SimCIMEngine(
                n_local=end_idx - start_idx,
                coupler=coupler,
                J=None,
                xi=xi,
                A_init=A_init,
                As=As,
                dt=dt,
                model=self.model,
                device=self.device,
                noise_scale=self.noise_scale,
                x_bits=self.x_bits,
                y_bits=self.y_bits,
                x_scale=self.x_scale,
                y_scale=self.y_scale,
            )
        ]
        self._start_idx = start_idx
        self._end_idx = end_idx
        self._nparts = 0  # unknown; determined at runtime

    # ------------------------------------------------------------------ #
    def run(self) -> Tuple[torch.Tensor, float]:
        """Run the dynamics and return ``(spins (N,), energy)``."""
        if self.backend == "torch":
            return self._run_torch()
        return self._run_emulated()

    def _run_emulated(self):
        """Run the emulated broadcast frame.

        One-component ``SimplifiedSimCIM`` (the benchmark model) uses **batched**
        dynamics: the coupling field for every module is produced by the
        broadcast-frame coordinator (block matmuls + single combined frozen
        ``c_remote``) and the per-variable SimCIM update is applied to the whole
        ``(N, 1)`` state in one vectorised pass — exactly what happens on real
        hardware where all compute nodes update their variables in parallel.
        Two-component models keep the per-module engine loop.
        """
        if self.modules[0].two_comp:
            return self._run_emulated_modules()
        return self._run_emulated_batched()

    def _run_emulated_batched(self):
        coord = self.coordinator
        pump = self._pump_values
        eng0 = self.modules[0]
        xi = eng0.xi
        As = eng0.As
        dt = eng0.dt
        sqrt_dt = math.sqrt(dt)
        noise_scale = eng0.noise_scale
        x_bits = eng0.x_bits
        x_scale = eng0.x_scale
        device = self.device

        # initial state = concatenated module phases (drawn in module order)
        x = torch.cat([eng.c_comp for eng in self.modules], dim=0)   # (N, 1)
        noise = torch.empty(x.size(0), 1, device=device)
        with torch.no_grad():
            for t in range(self.num_iters):
                p = pump[t]
                # batched broadcast frame: one bmm per step + dense GEMM /
                # batched bmm at each sync (all nodes on the one GPU)
                field = coord.batched_prepare(x, t % self.time_intvl == 0)

                # SimplifiedSimCIM dynamics (one vectorised pass over all N)
                noise.normal_(0.0, sqrt_dt)
                amp = torch.sqrt(x * x + 0.5).div_(As).mul_(noise_scale)
                x = x + ((-1 + p) * x + xi * field) * dt + noise * amp
                x = x.clamp(-1.0, 1.0)
                if x_bits is not None:
                    x = quantize_fixed(x, x_bits, x_scale)

        # write the final state back into the modules (readout / tests)
        for m, (s, e) in enumerate(self._slices):
            self.modules[m].c_comp = x[s:e]

        spins = torch.cat([m.ising_state for m in self.modules]).flatten()
        energy = ising_energy(spins, self._J, self._h)
        return spins, energy

    def _run_emulated_modules(self):
        """Two-component models: per-module engine loop (original behaviour)."""
        coord = self.coordinator
        pump = self._pump_values
        with torch.no_grad():
            for t in range(self.num_iters):
                p = pump[t]
                for engine in self.modules:
                    engine.set_p(p)
                for m, engine in enumerate(self.modules):
                    coord.states[m] = engine.c_comp
                coord.prepare_step(t % self.time_intvl == 0)
                for engine in self.modules:
                    engine.step()

        spins = torch.cat([m.ising_state for m in self.modules]).flatten()
        energy = ising_energy(spins, self._J, self._h)
        return spins, energy

    def _run_torch(self):
        import torch.distributed as dist

        engine = self.modules[0]
        pump = self._pump_values
        with torch.no_grad():
            for t in range(self.num_iters):
                engine.set_p(pump[t])
                engine.step()

        local = engine.ising_state                       # (part_len, 1)
        world = dist.get_world_size()
        gathered = [
            torch.empty(local.size(0), 1, device=local.device) for _ in range(world)
        ]
        dist.all_gather(gathered, local)
        spins = torch.cat(gathered).flatten()
        # energy is computed by the caller on the full J/h (rank 0 prints it)
        return spins, 0.0
