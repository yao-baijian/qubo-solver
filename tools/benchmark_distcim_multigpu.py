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


def _pad_uniform(J_np: np.ndarray, nproc: int) -> np.ndarray:
    """Zero-pad ``J`` to the next multiple of ``nproc``.

    The real-NCCL broadcast frame (:class:`CupyDistNCCLFieldCoupler`) stacks
    the per-block contributions ``J[S_i,S_m] x_m`` and exchanges them with an
    equal-split ``all_to_all``, so every rank must own the *same* number of
    variables.  Padding to ``Npad = ceil(N/nproc)*nproc`` (rows/cols
    ``N..Npad-1`` zero, the emulated coordinator's padded uniform partition)
    guarantees equal blocks for any N.  The padded variables are decoupled,
    so the first ``N`` spins are the true solution.
    """
    N = J_np.shape[0]
    Npad = ((N + nproc - 1) // nproc) * nproc
    if Npad == N:
        return J_np
    out = np.zeros((Npad, Npad), dtype=J_np.dtype)
    out[:N, :N] = J_np
    return out


def _worker(rank, nproc, host, port, args, steps, K, q):
    import cupy as cp
    from run_distcim_multigpu import _init_nccl_comm

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
    comm = _init_nccl_comm(nproc, rank, host, port)
    car_routes, J, h, source = traffic_common.load_instance(
        args.cars, args.routes, args.seed, force_synthetic=args.force_synthetic)
    J_np = np.ascontiguousarray(J.detach().numpy()
                                if hasattr(J, "detach") else np.asarray(J),
                                dtype=np.float32)
    J_np = _pad_uniform(J_np, nproc)          # equal blocks for all_to_all
    N = J_np.shape[0]
    s, e = _partition_columns(N, nproc)[rank]
    if args.sparse:
        # CSR column block: convert on CPU (scipy) then move to GPU, so the
        # dense (N, B) slice is never materialised on the GPU — this is what
        # makes N=100k (40 GB dense J) feasible.
        from scipy.sparse import csr_matrix as _scsr
        from cupyx.scipy.sparse import csr_matrix as _ccsr
        J_part = _ccsr(_scsr(J_np[:, s:e]))
    else:
        J_part = cp.asarray(J_np[:, s:e])

    from src.distcim.cupy_engine import solve_ising_cupy_nccl
    kw = dict(scheme="const", time_intvl=K, xi=args.xi, A_init=args.A_init,
              As=args.As, dt=args.dt, pump=args.pump, pmax=args.pmax,
              num_iters=steps, seed=args.seed, precision=args.precision,
              device=rank, sparse=args.sparse)
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
    q.put(dict(rank=rank, nproc=nproc, steps=steps, K=K,
               time_s=float(np.median(times)),
               times=[float(t) for t in times]))


def run_nproc(args, nproc, steps, K, host, port, timeout=900):
    """Run one (nproc, steps, K) config; returns median wall s or NaN on
    worker crash/OOM (instead of hanging on a 2h queue timeout)."""
    import queue as _queue
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker,
                         args=(r, nproc, host, port, args, steps, K, q))
             for r in range(nproc)]
    for p in procs:
        p.start()
    res = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            res.append(q.get(timeout=0.5))
        except _queue.Empty:
            pass
        if len(res) >= nproc:
            break
        if not any(p.is_alive() for p in procs):
            break  # all workers exited without reporting enough
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join()
    if len(res) < nproc or any(r.get("error") for r in res):
        return float("nan")
    # collect every timed repeat across all ranks (for average + scatter plots)
    all_times = [t for r in res for t in (r.get("times") or [])]
    return all_times if all_times else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cars", type=int, default=1250)
    ap.add_argument("--routes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--nproc", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--steps", nargs="+", type=int, default=[1000, 10000])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 5, 7, 10],
                    help="freeze-field sync period K to sweep")
    ap.add_argument("--dt", type=float, default=0.01)
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
    ap.add_argument("--sparse", action="store_true",
                    help="use CSR (cuSPARSE) column blocks instead of dense")
    ap.add_argument("--out", default="multigpu_scaling")
    args = ap.parse_args()

    out_dir = REPO_ROOT / "benchmark_results" / "traffic_realistic"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.out}.csv"
    # incremental write + resume: each config is appended as it completes so
    # partial progress survives worker OOM / crashes on shared GPUs, and a
    # re-run skips configs already present.
    fresh = not csv_path.exists()
    done = set()
    t1 = {}
    if not fresh:
        with open(csv_path, newline="") as rf:
            for r in csv.DictReader(rf):
                done.add((int(r["nproc"]), int(r["steps"]), int(r["K"])))
                try:
                    t = float(r["time_s"])
                    if t == t:
                        t1[(int(r["nproc"]), int(r["steps"]), int(r["K"]))] = t
                except (ValueError, KeyError):
                    pass
    f = open(csv_path, "a", newline="")
    w = csv.writer(f)
    if fresh:
        w.writerow(["nproc", "steps", "K", "time_s", "speedup_vs_1gpu",
                    "n_meas", "time_std", "times"])
        f.flush()
    print(f"multi-GPU scaling: cars={args.cars} routes={args.routes} "
          f"nproc={args.nproc} steps={args.steps} Ks={args.ks} "
          f"precision={args.precision} repeats={args.repeats} "
          f"(resume skips {len(done)} done)")
    for steps in args.steps:
        for K in args.ks:
            for nproc in args.nproc:
                if (nproc, steps, K) in done:
                    print(f"  nproc={nproc} steps={steps:>6} K={K:>3}  "
                          f"(already in CSV, skipping)")
                    continue
                t = run_nproc(args, nproc, steps, K, args.host, args.port)
                if isinstance(t, list):                 # per-repeat times
                    times = t
                    t = float(np.median(times))          # final average (median)
                else:
                    times = []
                t1[(nproc, steps, K)] = t
                b = t1.get((1, steps, K))
                speedup = (b / t if (t == t and b is not None and b == b
                                    and b > 0) else float("nan"))
                n_meas = len(times)
                t_std = float(np.std(times)) if n_meas > 1 else 0.0
                w.writerow([nproc, steps, K, f"{t:.4f}", f"{speedup:.3f}",
                            n_meas, f"{t_std:.4f}",
                            ";".join(f"{x:.4f}" for x in times)])
                f.flush()
                print(f"  nproc={nproc} steps={steps:>6} K={K:>3}  "
                      f"wall={t:.4f}s  speedup_vs_1={speedup:.2f}x  "
                      f"n={n_meas} std={t_std:.3f}s", flush=True)
    f.close()
    print(f"\nresults written to {csv_path}")


if __name__ == "__main__":
    main()
