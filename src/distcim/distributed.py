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

from typing import List, Optional, Tuple

import torch

from .engines import SimCIMEngine, ising_energy, quantize_fixed


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


# --------------------------------------------------------------------------- #
# real torch.distributed field coupler (one process per module)
# --------------------------------------------------------------------------- #
class TorchDistFieldCoupler:
    """Per-module field wrapper using real ``torch.distributed``.

    Mirrors the target repo's ``StandardParallelUpdateWrapper`` /
    ``ConstApproxParallelUpdateWrapper`` / ``PulseApproxParallelUpdateWrapper``.
    The external field ``h_part`` is added locally (the target repo's
    QuadraticCoupler adds ``optimization.h`` after the ``J`` product).
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
    ):
        self.J = J_part
        self.h_part = h_part
        self.start = start
        self.end = end
        self.scheme = scheme
        self.time_intvl = time_intvl
        self.quantize_bits = quantize_bits
        self.local_J = self.J[start:end]          # intra-module block
        self.exchange_info = torch.zeros([self.J.size(0), 1])
        self._t = 0

    @property
    def is_synchronization_step(self):
        return self._t % self.time_intvl == 0

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        s, e = self.start, self.end
        if self.scheme == "standard":
            output = torch.matmul(self.J, state)
            dist.all_reduce(output)
            return output[s:e] + self.h_part

        if self.is_synchronization_step:
            old_exchange = self.exchange_info[s:e].clone()
            torch.matmul(self.J, state, out=self.exchange_info)
            intra_info = self.exchange_info[s:e].clone()
            self.exchange_info[s:e].zero_()
            dist.all_reduce(self.exchange_info)
            # Quantize the message that would be exchanged (dynamic range).
            msg = self.exchange_info[s:e]
            self.exchange_info[s:e] = quantize_fixed(
                msg, self.quantize_bits, scale=msg.abs().max().item() or 1.0
            )
            self._t += 1
            if self.scheme == "const":
                return intra_info + self.exchange_info[s:e] + self.h_part
            elif self.scheme == "pulse":
                return intra_info + self.time_intvl * old_exchange + self.h_part
        else:
            intra_info = torch.matmul(self.local_J, state)
            self._t += 1
            if self.scheme == "const":
                return intra_info + self.exchange_info[s:e] + self.h_part
            elif self.scheme == "pulse":
                return intra_info + self.h_part
        raise ValueError(f"Unknown scheme '{self.scheme}'")


# --------------------------------------------------------------------------- #
# single-process emulated backend (all modules in one object)
# --------------------------------------------------------------------------- #
class _EmulatedCoordinator:
    """Coordinates all modules in-process, mimicking lock-step distributed
    execution: every module's field for step ``t`` is computed from the same
    pre-update state snapshot (exactly what an all-reduce does)."""

    def __init__(
        self,
        J_parts: List[torch.Tensor],
        h_parts: List[torch.Tensor],
        slices: List[Tuple[int, int]],
        scheme: str,
        time_intvl: int,
        quantize_bits: Optional[int],
    ):
        self.J_parts = J_parts
        self.h_parts = h_parts
        self.slices = slices
        self.scheme = scheme
        self.time_intvl = time_intvl
        self.quantize_bits = quantize_bits
        self.nparts = len(J_parts)
        # per-module current local states (updated by the driver)
        self.states: List[torch.Tensor] = [None] * self.nparts
        # inter-module message (frozen between syncs for const/pulse)
        self.messages: List[Optional[torch.Tensor]] = [None] * self.nparts
        self._fields: List[Optional[torch.Tensor]] = [None] * self.nparts
        self._t = 0

    def prepare_step(self, is_sync: bool):
        """Compute per-module fields from the current state snapshot."""
        if self.scheme == "standard" or is_sync:
            # exact inter-module message exchange (all-reduce of J^c x)
            contribs = [
                torch.matmul(Jp, st) for Jp, st in zip(self.J_parts, self.states)
            ]  # each (N, 1)
            full = contribs[0]
            for c in contribs[1:]:
                full = full + c

            # Dynamic fixed-point range for the messages: peak |field| at this
            # sync (as a real fixed-point link would scale to).
            qscale = full.abs().max().item() if self.quantize_bits else None

            for m, (s, e) in enumerate(self.slices):
                intra = contribs[m][s:e]                       # J_m x_m (own block)
                inter = (full - contribs[m])[s:e]              # J^c x (other blocks)
                inter_q = quantize_fixed(inter, self.quantize_bits, scale=qscale)
                h_m = self.h_parts[m]

                if self.scheme == "standard":
                    self._fields[m] = full[s:e] + h_m
                elif self.scheme == "const":
                    self.messages[m] = inter_q
                    self._fields[m] = intra + inter_q + h_m
                elif self.scheme == "pulse":
                    old = self.messages[m]
                    self.messages[m] = inter_q
                    self._fields[m] = intra + self.time_intvl * (
                        old if old is not None else torch.zeros_like(intra)
                    ) + h_m
        else:
            for m, (s, e) in enumerate(self.slices):
                intra = torch.matmul(self.J_parts[m][s:e], self.states[m])
                if self.scheme == "const":
                    msg = self.messages[m]
                    self._fields[m] = intra + (
                        msg if msg is not None else torch.zeros_like(intra)
                    ) + self.h_parts[m]
                elif self.scheme == "pulse":
                    self._fields[m] = intra + self.h_parts[m]

        self._t += 1

    def field(self, m: int) -> torch.Tensor:
        return self._fields[m]


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
        self.seed = seed
        self.noise_scale = noise_scale

        if pump == "constant":
            self._pump_values = pmax * torch.ones(num_iters)
        elif pump == "linear":
            self._pump_values = linear_schedule(num_iters, 0.0, pmax, span=0.5)
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
        J_parts = [self._J[:, s:e] for (s, e) in self._slices]
        h_parts = [self._h[s:e] for (s, e) in self._slices]

        self.coordinator = _EmulatedCoordinator(
            J_parts, h_parts, self._slices, self.scheme, self.time_intvl,
            self.quantize_bits,
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
            self.quantize_bits,
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
        coord = self.coordinator
        for t in range(self.num_iters):
            p = self._pump_values[t].item()
            for engine in self.modules:
                engine.set_p(p)
            # refresh the shared state snapshot (post-update states)
            for m, engine in enumerate(self.modules):
                coord.states[m] = engine.c_comp
            is_sync = (t % self.time_intvl == 0)
            coord.prepare_step(is_sync)
            for engine in self.modules:
                engine.step()

        spins = torch.cat([m.ising_state for m in self.modules]).flatten()
        energy = ising_energy(spins, self._J, self._h)
        return spins, energy

    def _run_torch(self):
        import torch.distributed as dist

        engine = self.modules[0]
        for t in range(self.num_iters):
            engine.set_p(self._pump_values[t].item())
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
