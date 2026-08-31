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
import socket
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


# --------------------------------------------------------------------------- #
# Minimal NCCL rendezvous + comm shim
# --------------------------------------------------------------------------- #
# ``cupyx.distributed.init_process_group`` starts its rendezvous TCP store in
# a *nested* multiprocessing.Process (using the default start method). Inside
# a ``spawn`` worker that default is itself ``spawn``, and pickling the store's
# bound-method target (a TCPStore holding a threading.Lock) raises
# ``TypeError: cannot pickle '_thread.lock' object``; forcing ``fork`` there
# deadlocks on the resource_tracker, and forking the whole worker breaks CUDA.
# Instead we rendezvous over a plain TCP socket (128-byte nccl id, rank 0 is
# the server) and drive the low-level ``cupy.cuda.nccl`` communicator directly
# — no multiprocessing.Process is ever started inside a worker.


class NcclRendezvousComm:
    """cupyx.distributed-compatible NCCL comm with a socket rendezvous.

    Implements just the ops ``cupy_engine.solve_ising_cupy_nccl`` needs
    (``all_to_all``, ``all_gather``, ``all_reduce``) plus ``stop``.
    """

    def __init__(self, world_size: int, rank: int, host: str, port: int):
        import cupy as cp
        from cupy.cuda import nccl

        self.world_size = world_size
        self.rank = rank
        self._nccl = nccl
        self._cp = cp
        self._nccl_dtypes = {
            "b": nccl.NCCL_INT8, "B": nccl.NCCL_UINT8,
            "i": nccl.NCCL_INT32, "I": nccl.NCCL_UINT32,
            "l": nccl.NCCL_INT64, "L": nccl.NCCL_UINT64,
            "q": nccl.NCCL_INT64, "Q": nccl.NCCL_UINT64,
            "e": nccl.NCCL_FLOAT16, "f": nccl.NCCL_FLOAT32,
            "d": nccl.NCCL_FLOAT64,
            "F": nccl.NCCL_FLOAT32, "D": nccl.NCCL_FLOAT64,
        }
        self._nccl_ops = {"sum": nccl.NCCL_SUM, "prod": nccl.NCCL_PROD,
                          "max": nccl.NCCL_MAX, "min": nccl.NCCL_MIN}

        nccl_id = self._rendezvous(host, port)
        self._comm = nccl.NcclCommunicator(world_size, nccl_id, rank)

    # -- rendezvous --------------------------------------------------------- #
    def _rendezvous(self, host, port) -> bytes:
        nccl_id = self._nccl.get_unique_id()          # 128-byte bytes
        if self.rank == 0:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((host, port))
            srv.listen(self.world_size - 1)
            try:
                for _ in range(self.world_size - 1):
                    conn, _ = srv.accept()
                    with conn:
                        conn.sendall(nccl_id)
            finally:
                srv.close()
            return nccl_id
        # every other rank connects and reads the nccl id
        for attempt in range(200):
            try:
                conn = socket.create_connection((host, port), timeout=10)
                break
            except OSError:
                if attempt == 199:
                    raise
                time.sleep(0.05)
        with conn:
            data = b""
            while len(data) < len(nccl_id):
                chunk = conn.recv(len(nccl_id) - len(data))
                if not chunk:
                    raise RuntimeError("rendezvous server closed early")
                data += chunk
        return data

    # -- helpers ------------------------------------------------------------ #
    def _dtype_count(self, array, count=None):
        c = array.dtype.char
        if c not in self._nccl_dtypes:
            raise TypeError(f"Unknown dtype {array.dtype} for NCCL")
        n = array.size if count is None else count
        if c in "FD":
            n *= 2
        return self._nccl_dtypes[c], n

    def _stream(self, stream):
        if stream is None:
            return self._cp.cuda.get_current_stream().ptr
        return stream.ptr

    # -- ops used by solve_ising_cupy_nccl ---------------------------------- #
    def all_reduce(self, in_array, out_array, op="sum", stream=None):
        dtype, count = self._dtype_count(in_array)
        self._comm.allReduce(
            in_array.data.ptr, out_array.data.ptr, count, dtype,
            self._nccl_ops[op], self._stream(stream))

    def all_gather(self, in_array, out_array, count, stream=None):
        dtype, _ = self._dtype_count(in_array, count)
        self._comm.allGather(
            in_array.data.ptr, out_array.data.ptr, count, dtype,
            self._stream(stream))

    def all_to_all(self, in_array, out_array, stream=None):
        st = self._stream(stream)
        nccl = self._nccl
        nccl.groupStart()
        for i in range(self.world_size):
            idtype, icount = self._dtype_count(in_array[i])
            odtype, ocount = self._dtype_count(out_array[i])
            self._comm.send(in_array[i].data.ptr, icount, idtype, i, st)
            self._comm.recv(out_array[i].data.ptr, ocount, odtype, i, st)
        nccl.groupEnd()

    def stop(self):
        try:
            self._comm.destroy()
        except Exception:
            pass


def _init_nccl_comm(world_size, rank, host, port):
    """Create an NCCL communicator with a plain-socket rendezvous."""
    return NcclRendezvousComm(world_size, rank, host, port)


def _worker(rank, nproc, host, port, args, q):
    import cupy as cp
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
    comm = _init_nccl_comm(nproc, rank, host, port)

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
    # equal blocks for the real-NCCL broadcast frame (see _pad_uniform)
    N = J_np.shape[0]
    Npad = ((N + nproc - 1) // nproc) * nproc
    if Npad > N:
        J_pad = np.zeros((Npad, Npad), dtype=np.float32)
        J_pad[:N, :N] = J_np
        J_np = J_pad
        if h_np is not None:
            h_pad = np.zeros((Npad, 1), dtype=np.float32)
            h_pad[:N] = h_np
            h_np = h_pad
    s, e = _partition_columns(Npad, nproc)[rank]
    J_part = cp.asarray(J_np[:, s:e])          # (Npad, part_len) column block
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

    # ``spawn`` workers give each rank a clean interpreter + CUDA context on
    # its assigned GPU (``fork`` breaks with CUDA already initialised in the
    # parent). ``cupyx.distributed``'s own rendezvous TCP store is replaced by
    # a plain-socket rendezvous (see ``_init_nccl_comm``) so no nested
    # multiprocessing.Process is ever started inside a worker.
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
