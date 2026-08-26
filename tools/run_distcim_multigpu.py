"""Multi-GPU DistIM launcher (CuPy + NCCL, one process per GPU).

Prepares the DistIM broadcast frame for a **4x RTX 4090** cluster (or any
``--nproc`` GPUs on one node): each rank owns one column block of ``J`` and
runs the one-component SimCIM dynamics on its GPU; at every sync the
off-diagonal contributions are exchanged with a NCCL ``all_to_all`` and each
rank freezes a single combined ``c_remote`` (the paper's freeze field).

CuPy >= 12 bundles NCCL (``cupyx.distributed``), so **no CUDA C is needed** —
the multi-GPU path uses the same quantized matmuls
(:class:`src.distcim.cupy_engine.CupyDistNCCLFieldCoupler`) as the
single-GPU emulation.  (The torch ``distributed`` backend in
``src/distcim/distributed.py`` is the alternative when a torch build with
NCCL is installed.)

Usage::

    # single GPU smoke test (this machine)
    python tools/run_distcim_multigpu.py --nproc 1 --cars 250 --routes 3 --force-synthetic

    # 4x RTX 4090, single node
    python tools/run_distcim_multigpu.py --nproc 4 --cars 1250 --routes 3 --force-synthetic \
        --steps 10000 --dt 0.01 --precision fp32 --K 10

    # multi-node: pick one host, run with the same --host/--port on every node
    python tools/run_distcim_multigpu.py --nproc 8 --host 10.0.0.1 --port 29500 ...

Options mirror tools/benchmark_distcim_precision.py (paper hyper-parameters).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import numpy as np  # noqa: E402

import traffic_common  # noqa: E402
from src.distcim.cupy_engine import (  # noqa: E402
    _partition_columns, solve_ising_cupy_nccl)


def _worker(rank, nproc, host, port, args, q):
    import cupy as cp
    import cupyx.distributed
    if rank == 0:
        # NCCL must be compiled into cupy (Linux wheels bundle it; Windows
        # wheels do not). Fail with a clear message instead of a cryptic one.
        try:
            import cupy.cuda.nccl as _nccl
            if not _nccl.available:
                raise RuntimeError
        except Exception:
            msg = ("this cupy build has no NCCL support "
                   "(cupy.cuda.nccl.available is False). Multi-GPU needs a "
                   "cupy wheel built with NCCL — on Linux install "
                   "`pip install cupy-cuda12x`; on Windows NCCL is not "
                   "bundled. Use the single-GPU emulated backend "
                   "(tools/benchmark_distcim_precision.py) until the "
                   "cluster is available.")
            print("ERROR: " + msg)
            q.put(dict(error=msg))
            return
    cp.cuda.Device(rank).use()
    comm = cupyx.distributed.init_process_group(
        nproc, rank, host=host, port=port)

    # every rank rebuilds the same instance (deterministic), then slices its
    # column block of J — no large-matrix broadcast needed.
    car_routes, J, h, source = traffic_common.load_instance(
        args.cars, args.routes, args.seed,
        force_synthetic=args.force_synthetic)
    J_np = np.ascontiguousarray(J.detach().numpy() if hasattr(J, "detach")
                                else np.asarray(J), dtype=np.float32)
    h_np = (np.ascontiguousarray(h.detach().numpy()
                                 if hasattr(h, "detach") else np.asarray(h),
                                 dtype=np.float32)
            if h is not None else None)
    N = J_np.shape[0]
    s, e = _partition_columns(N, nproc)[rank]
    J_part = cp.asarray(J_np[:, s:e])          # (N, part_len) column block
    h_part = (cp.asarray(h_np[s:e]) if h_np is not None else None)

    kw = dict(
        scheme=args.scheme, time_intvl=args.K,
        xi=args.xi, A_init=args.A_init, As=args.As, dt=args.dt,
        pump=args.pump, pmax=args.pmax, num_iters=args.steps,
        seed=args.seed, noise_scale=args.noise_scale,
        precision=args.precision, device=rank,
    )
    t0 = time.perf_counter()
    local, full, energy = solve_ising_cupy_nccl(
        J_part, h_part, rank, nproc, comm, **kw)
    elapsed = time.perf_counter() - t0
    comm.stop()

    cong = traffic_common.congestion_of_spins(car_routes, full)
    q.put(dict(rank=rank, cong=int(cong), energy=float(energy),
               time_s=elapsed, source=source, N=N))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nproc", type=int, default=4, help="GPUs / ranks")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=29500)
    ap.add_argument("--cars", type=int, default=250)
    ap.add_argument("--routes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--K", type=int, default=10, help="sync period")
    ap.add_argument("--scheme", default="const", choices=["standard", "const",
                                                          "pulse"])
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--pump", default="linear")
    ap.add_argument("--pmax", type=float, default=1.1)
    ap.add_argument("--As", type=float, default=70.0)
    ap.add_argument("--A_init", type=float, default=1e-3)
    ap.add_argument("--xi", default="inverse_interaction_rms")
    ap.add_argument("--noise_scale", type=float, default=1.0)
    ap.add_argument("--force-synthetic", action="store_true")
    args = ap.parse_args()

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker,
                         args=(r, args.nproc, args.host, args.port, args, q))
             for r in range(args.nproc)]
    for p in procs:
        p.start()
    results = [q.get(timeout=3600) for _ in procs]
    for p in procs:
        p.join()

    if any("error" in r for r in results):
        for r in results:
            if "error" in r:
                print("ERROR:", r["error"])
        sys.exit(1)

    r0 = results[0]
    baseline = _baseline_congestion(args)
    print(f"\ninstance: {r0['N']} spin vars (source={r0['source']})")
    print(f"multi-GPU distcim: {args.nproc} ranks (NCCL), scheme={args.scheme}, "
          f"K={args.K}, precision={args.precision}, steps={args.steps}, "
          f"dt={args.dt}")
    print(f"  congestion = {r0['cong']}  (vs default-route baseline {baseline})")
    print(f"  energy     = {r0['energy']:.6f}")
    print(f"  wall time  = {r0['time_s']:.4f} s")
    for r in results:
        print(f"  rank {r['rank']}: cong={r['cong']} energy={r['energy']:.6f} "
              f"time={r['time_s']:.4f}s")


def _baseline_congestion(args):
    car_routes, _J, _h, _src = traffic_common.load_instance(
        args.cars, args.routes, args.seed,
        force_synthetic=args.force_synthetic)
    return traffic_common.congestion(
        car_routes, traffic_common.default_route_bits(car_routes))


if __name__ == "__main__":
    main()
