"""Profile the dense multi-GPU distCIM per-step time breakdown.

Spawns real NCCL ranks (same as ``benchmark_distcim_multigpu.py``) and on each
rank reports the per-step time split: **engine overhead** (drift/noise/clip,
everything outside the coupler) vs **coupler time** (local block matmul +
sync: off-diagonal blocks + stack + NCCL all_to_all + combine), plus how many
syncs happened. Also times a raw dense GEMM ``(N, B) @ (B, 1)`` of the local
block size to show the pure-compute floor.

This answers "why does adding GPUs / K not speed up at small N (dense)":
the per-step compute is tiny, so Python/kernel-launch + NCCL sync overhead
dominates until N is large enough.

Usage::

    python tools/profile_distcim.py --cars 1668 --routes 3 --nproc 1 4 \\
        --ks 1 10 --steps 1000 --force-synthetic
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
from src.distcim.cupy_engine import _partition_columns  # noqa: E402
from run_distcim_multigpu import _init_nccl_comm  # noqa: E402


def _worker(rank, nproc, host, port, args, K, q):
    import cupy as cp

    cp.cuda.Device(rank).use()
    comm = _init_nccl_comm(nproc, rank, host, port)

    car_routes, J, h, src = traffic_common.load_instance(
        args.cars, args.routes, args.seed,
        force_synthetic=args.force_synthetic)
    J_np = np.ascontiguousarray(J.detach().numpy()
                                if hasattr(J, "detach") else np.asarray(J),
                                dtype=np.float32)
    N = J_np.shape[0]
    Npad = ((N + nproc - 1) // nproc) * nproc
    if Npad > N:
        Jp = np.zeros((Npad, Npad), np.float32)
        Jp[:N, :N] = J_np
        J_np = Jp
    s, e = _partition_columns(Npad, nproc)[rank]
    B = e - s
    J_part = cp.asarray(J_np[:, s:e])

    from src.distcim.cupy_engine import CupyDistCIMNCCL
    kw = dict(scheme="const", time_intvl=K, xi="inverse_interaction_rms",
              A_init=1e-3, As=70.0, dt=args.dt, pump="linear", pmax=1.1,
              num_iters=args.steps, seed=args.seed, precision="fp32",
              device=rank)
    sol = CupyDistCIMNCCL(J_part, None, rank, nproc, comm, **kw)
    eng = sol.engine

    # warmup
    for _ in range(10):
        eng.set_p(1.0)
        eng.step()
    cp.cuda.Stream.null.synchronize()

    inner = eng.coupler
    ev_coupler = []

    def timed(state):
        e0 = cp.cuda.Event()
        e1 = cp.cuda.Event()
        e0.record()
        r = inner(state)
        e1.record()
        ev_coupler.append((e0, e1))
        return r

    eng.coupler = timed

    ev_start = cp.cuda.Event()
    ev_start.record()
    for t in range(args.steps):
        eng.set_p(float(sol._pump[t]))
        eng.step()
    ev_end = cp.cuda.Event()
    ev_end.record()
    ev_end.synchronize()
    total = cp.cuda.get_elapsed_time(ev_start, ev_end) / 1000.0          # s
    t_coupler = sum(cp.cuda.get_elapsed_time(e0, e1) for e0, e1 in ev_coupler) / 1000.0
    n_sync = sum(1 for _ in range(args.steps // K))
    comm.stop()

    # raw dense GEMM of the full column block (N, B) @ (B, 1) — compute floor
    a = cp.random.rand(Npad, B, dtype=cp.float32)
    c = cp.random.rand(B, 1, dtype=cp.float32)
    for _ in range(5):
        a @ c
    cp.cuda.Stream.null.synchronize()
    iters = 50
    g0 = cp.cuda.Event()
    g1 = cp.cuda.Event()
    g0.record()
    for _ in range(iters):
        a @ c
    g1.record()
    g1.synchronize()
    gemm_s = cp.cuda.get_elapsed_time(g0, g1) / 1000.0 / iters

    per_step = total / args.steps
    q.put(dict(rank=rank, N=Npad, B=B, total=total, t_coupler=t_coupler,
               steps=args.steps, n_sync=n_sync,
               per_step=per_step, engine_ps=per_step - t_coupler / args.steps,
               coupler_ps=t_coupler / args.steps, gemm_s=gemm_s))


def run(args, nproc, K, host, port):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker,
                         args=(r, nproc, host, port, args, K, q))
             for r in range(nproc)]
    for p in procs:
        p.start()
    res = [q.get(timeout=3600) for _ in procs]
    for p in procs:
        p.join()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cars", type=int, default=1668)
    ap.add_argument("--routes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--nproc", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 10])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=29510)
    ap.add_argument("--force-synthetic", action="store_true")
    args = ap.parse_args()

    print(f"profile: cars={args.cars} Npad per nproc/K below, steps={args.steps}")
    for nproc in args.nproc:
        for K in args.ks:
            res = run(args, nproc, K, args.host, args.port)
            r0 = min(res, key=lambda r: r["rank"])
            print(f"\n== nproc={nproc}  K={K}  N={r0['N']}  B={r0['B']}  "
                  f"steps={args.steps} ==")
            print(f"   per-step total    : {r0['per_step']*1e3:8.2f} ms")
            print(f"     - coupler/field : {r0['coupler_ps']*1e3:8.2f} ms "
                  f"({100*r0['coupler_ps']/r0['per_step']:.0f}%)  syncs={r0['n_sync']}")
            print(f"     - engine rest   : {r0['engine_ps']*1e3:8.2f} ms "
                  f"({100*r0['engine_ps']/r0['per_step']:.0f}%)")
            print(f"   raw GEMM (N,B)@(B,1) floor: {r0['gemm_s']*1e3:.3f} ms")
            for r in res:
                print(f"   rank {r['rank']}: coupler={r['coupler_ps']*1e3:.2f}ms "
                      f"total={r['per_step']*1e3:.2f}ms")


if __name__ == "__main__":
    main()
