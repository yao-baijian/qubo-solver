"""SimCIM engine — a faithful port of the target repo's CIM dynamics.

The math, operation order and RNG draw order mirror ``sim.models.cim`` and
``sim.integrators.sde.SDEEulerIntegrator`` from the *simulated-ising-machine*
repo, so that on the same problem/seed/device the results match bit-for-bit.

Supported engine variants (identical to the target repo's ``sim.models.cim``):

- ``StandardCIM``      : two-component (c, s), cubic nonlinearity, no clipping
- ``SimCIM``           : two-component (c, s), linear + clipping
- ``SimplifiedSimCIM`` : one-component (c), linear + clipping   <-- paper's SimCIM

Dynamics (paper Methods 3.2, Eq. 8-11)::

    dx = [(p - 1) * x + xi * (J x + h)] * dt
         + (1 / As) * sqrt(x**2 + 0.5) * N(0, sqrt(dt))
    x  = clip(x, -1, +1)

read out as ``sigma = sign(x)``; energy ``E = -1/2 sigma^T J sigma - h^T sigma``.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch


def random_circle_init(N: int, amplitude: float, device="cpu"):
    """Port of ``sim.initializers.base.RandomInitializer.random_circle_init``.

    Draws one uniform phase vector and returns ``[A*cos(phi), A*sin(phi)]``,
    each of shape ``(N, 1)``.
    """
    phases = torch.empty([N, 1], device=device).uniform_(0.0, 2.0 * math.pi)
    cos_comp = phases.cos().mul_(amplitude)
    sin_comp = phases.sin().mul_(amplitude)
    return cos_comp, sin_comp


def inverse_interaction_rms(J: torch.Tensor, scale: float = 0.5):
    """Port of ``IsingInitializer.inverse_interaction_rms``.

    Returns ``scale / sqrt(sum(J**2) / (N - 1))`` (a 0-dim tensor).
    """
    N = J.size(1)
    interaction_rms = J.square().sum().div(N - 1).sqrt_()
    return scale / interaction_rms


def quantize_fixed(x: torch.Tensor, bits: int, scale: float = 1.0) -> torch.Tensor:
    """Uniform signed fixed-point quantization to ``bits`` bits.

    ``x`` is clamped to ``[-scale, scale]`` and rounded onto a grid with
    ``2 ** (bits - 1)`` magnitude levels (1 sign bit + ``bits - 1`` magnitude
    bits). ``bits=None`` or ``scale=None`` returns ``x`` unchanged.
    """
    if bits is None or scale is None:
        return x
    if scale == 0.0:
        return torch.zeros_like(x)
    levels = float(2 ** (bits - 1))
    q = (torch.clamp(x / scale, -1.0, 1.0) * levels).round() / levels * scale
    return q


class SimCIMEngine:
    """Central/per-module SimCIM engine.

    Parameters
    ----------
    n_local : int
        Number of spins held by this module (for a central run this is the
        full problem size).
    coupler : callable(c_comp) -> (n_local, 1)
        Returns the coupling field for the local state. For a central run this
        is ``J @ c + h``; for a distributed run it is a DistIM wrapper that
        returns the (approximate) local field.
    xi : float or str
        Coupling gain. ``'inverse_interaction_rms'`` is resolved from ``J``.
    model : str
        One of ``StandardCIM`` / ``SimCIM`` / ``SimplifiedSimCIM``.
    seed : int or None
        If given, ``torch.manual_seed`` is called before the initial phase
        draw (same behaviour as a freshly seeded run of the target repo).
    x_bits, y_bits : int or None
        FPGA state quantization. Only the dynamical states are reduced to
        fixed point at every step: ``x`` (position / c-component) to
        ``x_bits`` (hardware: 8) and ``y`` (momentum / s-component, two-
        component models only) to ``y_bits`` (hardware: 16 or 32). All
        control parameters (pump ``p``, coupling gain ``xi``, time step
        ``dt``, noise scale ``As``) stay in float/fixed-point. ``None``
        disables state quantization (default — faithful float32 port).
    x_scale, y_scale : float
        Full-scale range of the fixed-point grid for ``x`` / ``y``
        (``x`` is clipped to [-1, 1], so ``x_scale=1.0`` is natural).
    """

    TWO_COMP = {"StandardCIM": True, "SimCIM": True, "SimplifiedSimCIM": False}
    CLIP = {"StandardCIM": False, "SimCIM": True, "SimplifiedSimCIM": True}
    CUBIC = {"StandardCIM": True, "SimCIM": False, "SimplifiedSimCIM": False}

    def __init__(
        self,
        n_local: int,
        coupler: Callable[[torch.Tensor], torch.Tensor],
        J: Optional[torch.Tensor] = None,
        xi: float = 1.0,
        A_init: float = 1.0e-3,
        As: float = 70.0,
        dt: float = 0.1,
        model: str = "SimplifiedSimCIM",
        device: str = "cpu",
        seed: Optional[int] = None,
        noise_scale: float = 1.0,
        x_bits: Optional[int] = None,
        y_bits: Optional[int] = None,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
    ):
        if model not in self.TWO_COMP:
            raise ValueError(
                f"Unknown model '{model}'; choose from {sorted(self.TWO_COMP)}"
            )
        if seed is not None:
            torch.manual_seed(seed)

        self.n_local = n_local
        self.coupler = coupler
        self.model = model
        self.two_comp = self.TWO_COMP[model]
        self.clip = self.CLIP[model]
        self.cubic = self.CUBIC[model]
        self.device = device
        self.noise_scale = float(noise_scale)

        # FPGA state quantization (only the states x / y are quantized).
        self.x_bits = x_bits
        self.y_bits = y_bits
        self.x_scale = float(x_scale)
        self.y_scale = float(y_scale)

        # Resolve xi (string modes come from the target repo's initializer).
        if isinstance(xi, str):
            if xi == "inverse_interaction_rms":
                xi = inverse_interaction_rms(J.to(device)).item()
            else:
                raise ValueError(f"Unknown xi mode '{xi}'")
        self.xi = xi

        self.As = float(As)
        self.dt = float(dt)
        self.p = torch.zeros((), device=device)

        # States: zero_vector_init + random_circle_init(amplitude=A_init).
        cos0, sin0 = random_circle_init(n_local, A_init, device)
        self.c_comp = cos0
        self.s_comp = sin0 if self.two_comp else None

        # In-place Gaussian noise buffer (same as SDEEulerIntegrator).
        self.sqrt_dt = math.sqrt(self.dt)
        self.gauss_noise = torch.empty([n_local, 1], device=device)

    # ------------------------------------------------------------------ #
    # dynamics (exact port of the target repo's derivative/noise methods) #
    # ------------------------------------------------------------------ #
    def _derivative(self):
        c = self.c_comp
        if self.two_comp:
            s = self.s_comp
            if self.cubic:
                quad = c**2 + s**2
                dcdt = (-1 + self.p - quad) * c + self.xi * self.coupler(c)
                dsdt = (-1 - self.p - quad) * s
            else:
                dcdt = (-1 + self.p) * c + self.xi * self.coupler(c)
                dsdt = (-1 - self.p) * s
            return [dcdt, dsdt]
        else:
            dcdt = (-1 + self.p) * c + self.xi * self.coupler(c)
            return [dcdt]

    def _noise(self):
        c = self.c_comp
        if self.two_comp:
            amp = (c**2 + self.s_comp**2).add(0.5).sqrt().div_(self.As)
            amp = amp * self.noise_scale
            return [amp, amp]
        else:
            amp = (c**2).add(0.5).sqrt().div_(self.As)
            amp = amp * self.noise_scale
            return [amp]

    def step(self):
        """Advance one Euler-Maruyama step (identical op order to target)."""
        drifts = self._derivative()
        amps = self._noise()

        d_vars = []
        for drift, amp in zip(drifts, amps):
            # In-place Gaussian draw scaled by sqrt(dt) then by noise amplitude.
            self.gauss_noise.normal_(0.0, self.sqrt_dt).mul_(amp)
            d_var = drift * self.dt + self.gauss_noise
            d_vars.append(d_var)

        self.c_comp = self.c_comp + d_vars[0]
        if self.two_comp:
            self.s_comp = self.s_comp + d_vars[1]
        if self.clip:
            self.c_comp = self.c_comp.clamp(-1.0, 1.0)

        # FPGA state quantization: only x (c_comp) and y (s_comp) are reduced
        # to fixed point every step; control params (p, xi, dt, As) stay float.
        if self.x_bits is not None:
            self.c_comp = quantize_fixed(self.c_comp, self.x_bits, self.x_scale)
        if self.two_comp and self.y_bits is not None:
            self.s_comp = quantize_fixed(self.s_comp, self.y_bits, self.y_scale)

    def set_p(self, value):
        """Set the pump rate for the current step."""
        self.p = torch.as_tensor(value, device=self.device)

    @property
    def ising_state(self):
        return self.c_comp.sign()


class CentralFieldCoupler:
    """``J @ c + h`` — port of the target repo's QuadraticCoupler."""

    def __init__(self, J: torch.Tensor, h: torch.Tensor):
        self.J = J
        self.h = h

    def __call__(self, c_comp: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.J, c_comp) + self.h


def ising_energy(sigma: torch.Tensor, J: torch.Tensor, h: torch.Tensor) -> float:
    """Port of ``IsingOptimizationBase.energy``: -1/2 sigma^T J sigma - h^T sigma."""
    s = sigma.reshape(-1, 1).to(J.dtype)
    return (-1 / 2 * (s.T @ (J @ s)) - h.T @ s).item()
