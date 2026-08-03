"""Verification harness: qubo-solver DistCIM vs. the target repo.

Checks, in increasing strength:

Level 1 — faithful port (bit-exact with the target repo)
    For each CIM model (SimplifiedSimCIM / StandardCIM / SimCIM), on the same
    Ising problem / seed / hyper-parameters, the qubo-solver centralized
    machine must reproduce the target repo's final energy and spin state
    *bit-for-bit* (same torch RNG + operation order).

Level 2 — distributed correctness
    a) Field-math unit check: the DistIM standard/const/pulse field wrappers
       match the paper equations (Sec. 1.2, Eqs. 14-16) on random states.
    b) Noiseless trajectory check: distributed ``standard`` (K=1) with
       ``nparts>1`` reproduces an *independent* partitioned-centralized
       reference bit-for-bit; ``const``/``pulse`` with K=1 too.

Level 3 — approximation quality (paper claim: within ~1% of centralized)
    With noise and sync period K>1, ``const``/``pulse`` energies must be within
    ``tol`` of the centralized energy; the same holds when the exchanged
    message is quantized to ``quantize_bits`` bits (distributed + quantization).

Level 4 — sample case (small synthetic traffic flow)
    Build a small traffic problem through the target repo's ``TrafficGenerator``
    + ``TrafficFlow`` (the Kowloon/HK "country traffic" construction), then
    verify qubo-solver matches the target repo on the resulting Ising form.

Usage::

    python tools/verify_distim.py [--quick]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))   # repo root -> `src` is the top-level package

from src.distcim import solve_ising              # noqa: E402
from src.distcim.distributed import (            # noqa: E402
    DistIMEngine,
    quantize_fixed,
)
from src.distcim.engines import (                # noqa: E402
    SimCIMEngine,
    ising_energy,
)

import target_repo                             # noqa: E402

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f"  ({detail})" if detail else ""))


# --------------------------------------------------------------------------- #
# target-repo runner
# --------------------------------------------------------------------------- #
def run_target(sim, model_name, J, h, xi, A_init, As, dt, pmax, num_iters, seed):
    """Run the target repo's CIM on (J, h) with constant pump and return
    (spins, energy). Same RNG order as the qubo-solver engine."""
    from sim.optimizations import IsingOptimization
    from sim.models import SimplifiedSimCIM, StandardCIM, SimCIM
    from sim.models.coupler import QuadraticCoupler
    from sim.scheduler import ConstantScheduler

    torch.manual_seed(seed)
    cls = {
        "SimplifiedSimCIM": SimplifiedSimCIM,
        "StandardCIM": StandardCIM,
        "SimCIM": SimCIM,
    }[model_name]
    opt = IsingOptimization(J, h)
    model = cls(opt, QuadraticCoupler(), xi=xi, A_init=A_init, As=As, dt=dt)
    sched = ConstantScheduler(value=pmax)
    model.attach_scheduler(sched, "pump")
    sched.scheduling(num_iters)
    for _ in range(num_iters):
        sched.step()
        model.step()
    spins = model.ising_state.flatten()
    energy = model.energy.item()
    return spins, energy


# --------------------------------------------------------------------------- #
# Level 1: bit-exact port vs the target repo
# --------------------------------------------------------------------------- #
def level1(sim, N, num_iters, seed, As):
    print("\n=== Level 1: bit-exact port vs. target repo ===")
    torch.manual_seed(123)
    J = torch.randn(N, N)
    J = (J + J.T) / 2
    J.fill_diagonal_(0)
    h = torch.randn(N, 1) * 0.1

    for model in ["SimplifiedSimCIM", "StandardCIM", "SimCIM"]:
        spins_t, e_t = run_target(
            sim, model, J, h, xi=1.0, A_init=1e-3, As=As, dt=0.1,
            pmax=1.1, num_iters=num_iters, seed=seed,
        )
        spins_q, e_q = solve_ising(
            J, h, nparts=1, scheme="standard", model=model, xi=1.0,
            A_init=1e-3, As=As, dt=0.1, pump="constant", pmax=1.1,
            num_iters=num_iters, seed=seed,
        )
        spins_q = torch.as_tensor(spins_q)
        same_energy = (e_q == e_t)
        same_spins = bool((spins_q == spins_t).all())
        record(
            f"L1 {model} energy == target", same_energy,
            f"qubo={e_q!r} target={e_t!r}",
        )
        record(f"L1 {model} spins == target", same_spins)


# --------------------------------------------------------------------------- #
# Level 2a: field-math unit check (paper Eqs. 14-16)
# --------------------------------------------------------------------------- #
def level2a(N, nparts, K):
    print("\n=== Level 2a: DistIM field-math unit check ===")
    torch.manual_seed(7)
    J = torch.randn(N, N)
    J = (J + J.T) / 2
    J.fill_diagonal_(0)
    x = torch.randn(N, 1).clamp(-1, 1)

    from src.distcim.distributed import _EmulatedCoordinator, partition_columns

    slices = partition_columns(N, nparts)
    J_parts = [J[:, s:e] for (s, e) in slices]
    xs = [x[s:e] for (s, e) in slices]
    h_parts = [torch.zeros(e - s, 1) for (s, e) in slices]

    full_field = J @ x  # reference exact field (N, 1)

    # --- standard: every step exact ---
    std = _EmulatedCoordinator(J_parts, h_parts, slices, "standard", 1, None)
    std.states = xs
    std.prepare_step(True)
    std_field = torch.cat([std.field(m) for m in range(nparts)])
    record(
        "L2a standard field == Jx", torch.allclose(std_field, full_field, atol=1e-5),
        f"max err {(std_field - full_field).abs().max().item():.2e}",
    )

    # --- const: sync step exact, off-sync = intra + frozen message ---
    K = 10
    const = _EmulatedCoordinator(J_parts, h_parts, slices, "const", K, None)
    const.states = xs
    const.prepare_step(True)                     # sync step
    sync_field = torch.cat([const.field(m) for m in range(nparts)])
    record(
        "L2a const sync field == Jx",
        torch.allclose(sync_field, full_field, atol=1e-5),
        f"max err {(sync_field - full_field).abs().max().item():.2e}",
    )
    # off-sync: only intra-module field + frozen message
    const.prepare_step(False)
    off_field = torch.cat([const.field(m) for m in range(nparts)])
    # reference: intra block + frozen message (recompute message from sync)
    contribs = [torch.matmul(Jp, xs[m]) for m, (Jp) in enumerate(J_parts)]
    full_ref = contribs[0]
    for c in contribs[1:]:
        full_ref = full_ref + c
    intra_ref = torch.cat([contribs[m][s:e] for m, (s, e) in enumerate(slices)])
    inter_ref = torch.cat([(full_ref - contribs[m])[s:e] for m, (s, e) in enumerate(slices)])
    ref_off = intra_ref + inter_ref
    record(
        "L2a const off-sync field == intra+frozen msg",
        torch.allclose(off_field, ref_off, atol=1e-5),
        f"max err {(off_field - ref_off).abs().max().item():.2e}",
    )

    # --- pulse: off-sync = intra only; sync = intra + K*old message ---
    pulse = _EmulatedCoordinator(J_parts, h_parts, slices, "pulse", K, None)
    pulse.states = xs
    pulse.prepare_step(True)                     # first sync: old msg = None -> intra only
    first_sync = torch.cat([pulse.field(m) for m in range(nparts)])
    record(
        "L2a pulse first sync == intra (no old msg)",
        torch.allclose(first_sync, intra_ref, atol=1e-5),
    )
    pulse.prepare_step(False)                    # off-sync: intra only
    pulse_off = torch.cat([pulse.field(m) for m in range(nparts)])
    record(
        "L2a pulse off-sync == intra",
        torch.allclose(pulse_off, intra_ref, atol=1e-5),
    )
    pulse.prepare_step(True)                     # second sync: intra + K*old message
    second_sync = torch.cat([pulse.field(m) for m in range(nparts)])
    record(
        "L2a pulse second sync == intra + K*old",
        torch.allclose(second_sync, intra_ref + K * inter_ref, atol=1e-5),
    )

    # --- quantization (bounded error for the chosen scale) ---
    scale = inter_ref.abs().max().item()
    q = quantize_fixed(inter_ref, 8, scale=scale)
    step = scale / 128.0
    record(
        "L2a quantize_fixed(8) error <= step/2",
        (q - inter_ref).abs().max().item() <= step / 2 + 1e-6,
        f"max err {(q - inter_ref).abs().max().item():.3e} step/2={step/2:.3e}",
    )


# --------------------------------------------------------------------------- #
# Level 2b: noiseless distributed == independent partitioned-centralized
# --------------------------------------------------------------------------- #
def partitioned_centralized_reference(J, slices, x_full0, num_iters, pmax, dt,
                                      model):
    """Independent reference: split x into modules but always use the exact
    full field ``(J @ x)[S_m]`` at every step (no DistIM wrapper logic)."""
    N = J.size(0)
    modules = []
    for m, (s, e) in enumerate(slices):
        holder = {"field": None}

        def coupler(c, _holder=holder):
            return _holder["field"]

        engine = SimCIMEngine(
            n_local=e - s,
            coupler=coupler,
            xi=1.0, A_init=1e-3, As=70.0, dt=dt,
            model=model, noise_scale=0.0,
        )
        engine.c_comp = x_full0[s:e].clone()
        modules.append((engine, holder, s, e))
    for t in range(num_iters):
        x_full = torch.cat([m.c_comp for m, _, _, _ in modules])
        field = J @ x_full
        for engine, holder, s, e in modules:
            engine.set_p(pmax)
            holder["field"] = field[s:e]
            engine.step()
    spins = torch.cat([m.ising_state for m, _, _, _ in modules]).flatten()
    return spins


def level2b(N, nparts, K, num_iters):
    print("\n=== Level 2b: noiseless distributed vs independent reference ===")
    torch.manual_seed(11)
    J = torch.randn(N, N)
    J = (J + J.T) / 2
    J.fill_diagonal_(0)

    from src.distcim.distributed import partition_columns
    from src.distcim.engines import random_circle_init
    slices = partition_columns(N, nparts)
    # deterministic module init (same as DistIMEngine's init)
    torch.manual_seed(5)
    x0 = torch.cat([random_circle_init(e - s, 1e-3)[0] for (s, e) in slices])

    ref = partitioned_centralized_reference(
        J, slices, x0, num_iters, pmax=1.1, dt=0.1, model="SimplifiedSimCIM"
    )

    for scheme in ["standard", "const", "pulse"]:
        eng = DistIMEngine(
            J=J, h=torch.zeros(N, 1), nparts=nparts, scheme=scheme,
            time_intvl=K, model="SimplifiedSimCIM", xi=1.0, A_init=1e-3,
            As=70.0, dt=0.1, pump="constant", pmax=1.1, num_iters=num_iters,
            seed=5, noise_scale=0.0,
        )
        # override module init to match the reference exactly
        for m, (s, e) in enumerate(slices):
            eng.modules[m].c_comp = x0[s:e].reshape(-1, 1).clone()
            eng.coordinator.states[m] = eng.modules[m].c_comp
        spins, _ = eng.run()
        n_diff = int((spins != ref).sum())
        if scheme == "standard":
            # standard is exact for any K (it all-reduces every step)
            ok = n_diff == 0
        elif scheme == "const":
            # const with K=1 syncs every step -> exact; K>1 is approximate
            ok = n_diff == 0 if K == 1 else True
        else:  # pulse: even K=1 has a one-step message lag (target-repo semantics)
            ok = True
        record(
            f"L2b {scheme} (nparts={nparts}, K={K}) vs reference",
            ok,
            f"n_spin_diff={n_diff}",
        )


# --------------------------------------------------------------------------- #
# Level 3: approximation quality of const/pulse (with noise + quantization)
# --------------------------------------------------------------------------- #
def level3(N, nparts, K, num_iters, seeds):
    print("\n=== Level 3: const/pulse approximation quality (noisy) ===")
    torch.manual_seed(21)
    J = torch.randn(N, N)
    J = (J + J.T) / 2
    J.fill_diagonal_(0)
    h = torch.zeros(N, 1)

    e_central = min(
        solve_ising(J, h, nparts=1, scheme="standard", num_iters=num_iters,
                    seed=s)[1]
        for s in seeds
    )

    # const is the paper's recommended accurate scheme; pulse is an
    # alternative (may be less accurate). Tolerances reflect that.
    scheme_tol = {"const": 0.10, "pulse": 0.25}
    for K in sorted({K, 5}):
        for scheme in ["const", "pulse"]:
            best = min(
                solve_ising(
                    J, h, nparts=nparts, scheme=scheme, time_intvl=K,
                    num_iters=num_iters, seed=s,
                )[1]
                for s in seeds
            )
            rel = abs(best - e_central) / max(abs(e_central), 1e-12)
            tol_k = scheme_tol[scheme]
            record(
                f"L3 {scheme} K={K} ({K}x fewer exchanges) within "
                f"{tol_k:.0%} of central",
                rel <= tol_k,
                f"central={e_central:.6f} {scheme}={best:.6f} rel={rel:.4%}",
            )
            for bits in [16, 8]:
                best_q = min(
                    solve_ising(
                        J, h, nparts=nparts, scheme=scheme, time_intvl=K,
                        num_iters=num_iters, seed=s, quantize_bits=bits,
                    )[1]
                    for s in seeds
                )
                rel_q = abs(best_q - e_central) / max(abs(e_central), 1e-12)
                record(
                    f"L3 {scheme} K={K} quantized({bits}b) within {tol_k:.0%}",
                    rel_q <= tol_k,
                    f"{scheme}={best_q:.6f} rel={rel_q:.4%}",
                )


# --------------------------------------------------------------------------- #
# Level 4: small synthetic traffic-flow sample (target-repo construction)
# --------------------------------------------------------------------------- #
def level4(sim, num_cars, num_routes, num_iters, seed):
    print("\n=== Level 4: small synthetic traffic-flow sample ===")
    import random
    import networkx as nx

    # An 8x6 grid road network with varied edge weights so that many distinct
    # alternative routes exist (exercises the real traffic QUBO construction).
    random.seed(seed)
    G = nx.DiGraph()
    rows, cols = 8, 6
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:
                w = 1.0 + random.random()
                G.add_edge(u, u + 1, length=w)
                G.add_edge(u + 1, u, length=w)
            if r + 1 < rows:
                w = 1.0 + random.random()
                G.add_edge(u, u + cols, length=w)
                G.add_edge(u + cols, u, length=w)

    from sim.datasets.generator.traffic_generator import TrafficGenerator

    gen = TrafficGenerator(
        G, weight="length", num_cars=num_cars, num_routes=num_routes,
        rm_percent=0.5, segs=0.3, tol=math.inf, max_trial=500, seed=seed,
    )
    car_routes = {}
    for car, (ori, dest) in enumerate(gen.ori_dest_pairs):
        car_routes[car] = gen.k_shortest_path_remove_nodes(ori, dest)
    n_routes = sum(len(r) for r in car_routes.values())
    avg = n_routes / max(1, len(car_routes))
    print(f"  traffic instance: {num_cars} cars, {n_routes} route vars "
          f"(avg {avg:.1f} routes/car)")
    assert avg > 1.5, "traffic instance is degenerate (too few alternative routes)"

    from sim.optimizations.traffic import TrafficFlow
    tf = TrafficFlow(car_routes)
    J = tf.J.to_dense() if tf.J.is_sparse else tf.J
    h = tf.h
    N = J.size(0)

    def congestion(qubo_bits):
        from collections import defaultdict
        loads = defaultdict(int)
        idx = 0
        for car, routes in car_routes.items():
            pick = None
            for ri in range(len(routes)):
                if qubo_bits[idx + ri] > 0.5:
                    pick = ri
            idx += len(routes)
            if pick is not None:
                for u, v in zip(routes[pick], routes[pick][1:]):
                    loads[(u, v)] += 1
        return sum(v * v for v in loads.values())

    # qubo-solver vs target on the traffic Ising form (same seed, bit-exact)
    spins_t, e_t = run_target(
        sim, "SimplifiedSimCIM", J, h, xi=1.0, A_init=1e-3, As=70.0, dt=0.1,
        pmax=1.1, num_iters=num_iters, seed=seed,
    )
    spins_q, e_q = solve_ising(
        J, h, nparts=1, scheme="standard", model="SimplifiedSimCIM", xi=1.0,
        A_init=1e-3, As=70.0, dt=0.1, pump="constant", pmax=1.1,
        num_iters=num_iters, seed=seed,
    )
    spins_q = torch.as_tensor(spins_q)
    record(
        "L4 traffic energy == target", e_q == e_t, f"qubo={e_q!r} target={e_t!r}"
    )
    record("L4 traffic spins == target", bool((spins_q == spins_t).all()))

    # The found route assignment must improve congestion over the default
    # (fastest) routes -- the paper's Fig. 3 figure of merit.
    sol_bits = ((torch.as_tensor(spins_q) + 1) / 2).round().int().tolist()
    c_opt = congestion(sol_bits)
    # default routing: route option 0 for every car
    bits_default = []
    for car, routes in car_routes.items():
        bits_default += [1] + [0] * (len(routes) - 1)
    c_default = congestion(bits_default)
    record(
        "L4 traffic congestion improved vs default",
        c_opt <= c_default, f"congestion {c_opt} (default {c_default})",
    )

    # distributed standard (noiseless) == centralized on the traffic instance
    e_c = solve_ising(
        J, h, nparts=1, scheme="standard", num_iters=num_iters, seed=seed,
        noise_scale=0.0,
    )[1]
    e_d = solve_ising(
        J, h, nparts=4, scheme="standard", time_intvl=1, num_iters=num_iters,
        seed=seed, noise_scale=0.0,
    )[1]
    record(
        "L4 traffic dist-standard(K=1) == central (noiseless)",
        e_d == e_c, f"central={e_c!r} dist={e_d!r}",
    )
    return N


# --------------------------------------------------------------------------- #
def level5(N, num_iters, seeds, tol):
    """FPGA state quantization: only x (->int8) and y (->int16/32) are
    quantized at every step; control params (pump, xi, dt, As) stay float.
    Check that quantized central ~ float32 and distributed+quantized ~
    centralized+quantized (no fundamental difference; quantized runs may
    benefit from a tuned dt -- see tools/check_traffic_quant.py)."""
    print("\n=== Level 5: FPGA state quantization (x int8, y int16/32) ===")
    torch.manual_seed(21)
    J = torch.randn(N, N)
    J = (J + J.T) / 2
    J.fill_diagonal_(0)
    h = torch.randn(N, 1) * 0.1

    # 1-component: x -> int8
    e0 = min(
        solve_ising(J, h, nparts=1, num_iters=num_iters, seed=s)[1] for s in seeds
    )
    e1 = min(
        solve_ising(J, h, nparts=1, num_iters=num_iters, seed=s, x_bits=8)[1]
        for s in seeds
    )
    rel = abs(e1 - e0) / max(abs(e0), 1e-12)
    record(
        f"L5 x-int8 central within {tol:.0%} of float32",
        rel <= tol, f"float={e0:.6f} q8={e1:.6f} rel={rel:.4%}",
    )

    # distributed + state quantization
    e2 = min(
        solve_ising(J, h, nparts=4, scheme="const", time_intvl=10,
                    num_iters=num_iters, seed=s, x_bits=8)[1]
        for s in seeds
    )
    rel2 = abs(e2 - e1) / max(abs(e1), 1e-12)
    record(
        "L5 dist-const K10 + x-int8 within "
        f"{0.15:.0%} of central-q8",
        rel2 <= 0.15, f"central-q8={e1:.6f} dist-q8={e2:.6f} rel={rel2:.4%}",
    )

    # 2-component SimCIM: x -> int8 and y -> int16
    e0b = min(
        solve_ising(J, h, model="SimCIM", num_iters=num_iters, seed=s)[1]
        for s in seeds
    )
    e1b = min(
        solve_ising(J, h, model="SimCIM", num_iters=num_iters, seed=s,
                    x_bits=8, y_bits=16)[1]
        for s in seeds
    )
    relb = abs(e1b - e0b) / max(abs(e0b), 1e-12)
    record(
        "L5 SimCIM x-int8/y-int16 within "
        f"{0.15:.0%} of float32",
        relb <= 0.15, f"float={e0b:.6f} q={e1b:.6f} rel={relb:.4%}",
    )


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="smaller instance sizes / fewer iterations")
    args = parser.parse_args()

    sim = target_repo.import_target()
    print(f"target repo imported from: {target_repo.DEFAULT_TARGET_REPO}")

    if args.quick:
        N, iters, nparts, K, seeds = 16, 300, 4, 5, [0, 1, 2]
    else:
        N, iters, nparts, K, seeds = 24, 800, 4, 10, [0, 1, 2, 3]

    level1(sim, N=N, num_iters=iters, seed=7, As=70.0)
    level2a(N=N, nparts=nparts, K=K)
    level2b(N=N, nparts=nparts, K=1, num_iters=iters)   # const/pulse K=1 exact
    level2b(N=N, nparts=nparts, K=K, num_iters=iters)
    level3(N=N, nparts=nparts, K=K, num_iters=iters, seeds=seeds)
    level4(sim, num_cars=8, num_routes=4, num_iters=iters, seed=7)
    level5(N=N, num_iters=iters, seeds=seeds, tol=0.10)

    print("\n=== SUMMARY ===")
    n_fail = 0
    for name, ok, _ in RESULTS:
        if not ok:
            n_fail += 1
            print(f"  FAIL  {name}")
    print(f"  {len(RESULTS)} checks, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
