"""Multi-GPU scaling benchmark: distcim broadcast frame on nproc = 1..N GPUs.

For the 4x RTX 4090 cluster (or any multi-GPU node with cupy built with
NCCL). Runs the real distributed broadcast frame (one process per GPU,
`CupyDistCIMNCCL` + NCCL all_to_all) for a grid of `nproc` values and steps,
and reports per-solve wall time, throughput and speedup vs nproc=1 and vs the
single-GPU emulated distcim / central CIM.

Usage (Linux, cupy-cuda12x with NCCL):
    python tools/benchmark_distcim_multigpu.py --cars 1250 --routes 3 \
        --force-synthetic --nproc 1 2 4 --steps 1000 10000 --dt 0.01 \
        --precision fp32 --K 10 --repeats 3

On this Windows machine cupy has no NCCL, so it prints a clear error and
exits — run it on the cluster.
"""
from __future__ import annotations

import argparse
import csv
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


def _worker(rank, nproc, host, port, args, steps, q):
    import cupy as cp
    import cupyx.distributed

    if rank == 0:
        try:
            import cupy.cuda.nccl as _nccl
            if not _nccl.available:
                raise RuntimeError
        except Exception:
            q.put(dict(error="cupy built without NCCL (Windows wheel); run on "
                             "the Linux 4090 cluster with cupy-cuda12x"))
            return
    cp.cuda.Device(rank).use()
    comm = cupyx.distributed.init_process_group(nproc, rank, host=host,
                                                port=port)
    car_routes, J, h, source = traffic_common.load_instance(
        args.cars, args.routes, args.seed, force_synthetic=args.force_synthetic)
    J_np = np.ascontiguousarray(J.detach().numpy()
                                if hasattr(J, "detach") else np.asarray(J),
                                dtype=np.float32)
    N = J_np.shape[0]
    s, e = _partition_columns(N, nproc)[rank]
    J_part = cp.asarray(J_np[:, s:e])

    from src.distcim.cupy_engine import solve_ising_cupy_nccl
    kw = dict(scheme="const", time_intvl=args.K, xi=args.xi, A_init=args.A_init,
              As=args.As, dt=args.dt, pump=args.pump, pmax=args.pmax,
              num_iters=steps, seed=args.seed, precision=args.precision,
              device=rank)
    # warmup + timed repeats (interleaved with the sync barrier keeps ranks fair)
    times = []
    for i in range(args.repeats + 1):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        solve_ising_cupy_nccl(J_part, None, rank, nproc, comm, **kw)
        cp.cuda.Stream.null.synchronize()
        t = time.perf_counter() - t0
        if i > 0:
            times.append(t)
    comm.stop()
    q.put(dict(rank=rank, nproc=nproc, steps=steps, time_s=float(np.median(times))))


def run_nproc(args, nproc, steps, host, port):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker,
                         args=(r, nproc, host, port, args, steps, q))
             for r in range(nproc)]
    for p in procs:
        p.start()
    res = [q.get(timeout=7200) for _ in procs]
    for p in procs:
        p.join()
    if any("error" in r for r in res):
        raise RuntimeError(res[0]["error"])
    return float(np.median([r["time_s"] for r in res]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cars", type=int, default=1250)
    ap.add_argument("--routes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--nproc", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--steps", nargs="+", type=int, default=[1000, 10000])
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--pump", default="linear")
    ap.add_argument("--pmax", type=float, default=1.1)
    ap.add_argument("--As", type=float, default=70.0)
    ap.add_argument("--A_init", type=float, default=1e-3)
    ap.add_argument("--xi", default="inverse_interaction_rms")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=29500)
    ap.add_argument("--force-synthetic", action="store_true")
    ap.add_argument("--out", default="multigpu_scaling")
    args = ap.parse_args()

    rows = []
    t1 = {}
    print(f"multi-GPU scaling: cars={args.cars} routes={args.routes} "
          f"nproc={args.nproc} steps={args.steps} K={args.K} "
          f"precision={args.precision}")
    for steps in args.steps:
        for nproc in args.nproc:
            try:
                t = run_nproc(args, nproc, steps, args.host, args.port)
            except RuntimeError as e:
                print("ERROR:", e)
                sys.exit(1)
            t1[(nproc, steps)] = t
            speedup = t1[(1, steps)] / t if (1, steps) in t1 else float("nan")
            rows.append(dict(nproc=nproc, steps=steps, time_s=t,
                             speedup_vs_1=speedup))
            print(f"  nproc={nproc} steps={steps:>6}  wall={t:.4f}s  "
                  f"speedup_vs_1={speedup:.2f}x")

    out_dir = REPO_ROOT / "benchmark_results" / "traffic_realistic"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.out}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["nproc", "steps", "time_s", "speedup_vs_1gpu"])
        for r in rows:
            w.writerow([r["nproc"], r["steps"], f"{r['time_s']:.4f}",
                        f"{r['speedup_vs_1']:.3f}"])
    print(f"\nresults written to {csv_path}")


if __name__ == "__main__":
    main()
