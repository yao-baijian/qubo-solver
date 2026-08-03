"""Run the TARGET repo's real distributed ConstApprox SimCIM on G1 (4 ranks).

Launched with torch.multiprocessing so each rank is a real module using
torch.distributed all-reduce — the exact reference implementation. Used to
check whether the const-scheme collapse at large K on G1 is genuine
target-repo behaviour or a bug in the qubo-solver port.

Usage:
    python tools/run_target_dist_g1.py
"""
import math
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, r"c:\project\qubo-solver\tools")
import target_repo  # noqa: E402
target_repo.import_target()

from sim.parallel.optimizations import DistIsingOptimization  # noqa: E402
from sim.models import SimplifiedSimCIM                        # noqa: E402
from sim.models.coupler import QuadraticCoupler                # noqa: E402
from sim.scheduler import LinearScheduler                      # noqa: E402
from sim.parallel import (                                     # noqa: E402
    ConstApproxDistributedModelParallel,
    StandardDistributedModelParallel,
)

REPO = Path(r"c:\project\qubo-solver")
GSET = REPO / "benchmarks" / "instances" / "maxcut" / "Gset" / "G1"
WORLD = 4


def load_gset(path):
    with open(path) as f:
        N, _ = [int(x) for x in f.readline().split()]
    data = torch.tensor(
        [list(map(int, l.split()))
         for l in open(path).read().strip().split("\n")[1:]],
        dtype=torch.long,
    )
    u, v = data[:, 0] - 1, data[:, 1] - 1
    w = data[:, 2].float() if data.shape[1] > 2 else torch.ones(data.shape[0])
    J = torch.zeros(N, N)
    J[u, v] = w
    J[v, u] = w
    return J, N


def cut_value(J, spins):
    spins = torch.as_tensor(spins, dtype=torch.float32)
    return 0.25 * (J.sum() - (spins @ J @ spins)).item()


def worker(rank, init_file):
    dist.init_process_group("gloo", init_method=f"file://{init_file}",
                            rank=rank, world_size=WORLD)
    ws = WORLD
    N = 800

    J_adj, _ = load_gset(GSET)
    J = -J_adj / 2.0                 # MaxCut -> Ising
    h = torch.zeros(N, 1)

    part_len = N // ws
    s = rank * part_len
    e = s + part_len if rank < ws - 1 else N
    J_part = J[:, s:e].contiguous()
    h_part = h[s:e]
    xi = 0.5 / math.sqrt(J.square().sum().item() / (N - 1))

    for scheme, cls in [("standard", StandardDistributedModelParallel),
                        ("constK10", ConstApproxDistributedModelParallel)]:
        # fresh optimization per model (the wrapper replaces optimization.J)
        opt = DistIsingOptimization(J_part, h_part)
        model = SimplifiedSimCIM(opt, QuadraticCoupler(), xi=xi,
                                 A_init=1e-3, As=70.0, dt=0.1)
        model = cls(model, time_intvl=10) if scheme == "constK10" else cls(model)

        sched = LinearScheduler(start=0, end=1, span=0.5)
        model.attach_scheduler(sched, "pump")
        sched.scheduling(3000)
        for _ in range(3000):
            sched.step()
            model.step()

        spins = model.ising_state.flatten()
        if rank == 0:
            n_plus = int((spins > 0).sum())
            print(f"[target] {scheme:<10} cut={cut_value(J_adj, spins):7.1f} "
                  f"+1 spins={n_plus}/{N}", flush=True)
        dist.barrier()

    if rank == 0:
        print("[target] done")
    dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(delete=False, suffix=".store") as f:
        init_file = f.name.replace("\\", "/")
    mp.start_processes(worker, args=(init_file,), nprocs=WORLD, join=True)
    os.unlink(init_file)
