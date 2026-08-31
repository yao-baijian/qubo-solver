#!/usr/bin/env python
"""Pick the least-contended GPUs and print a CUDA_VISIBLE_DEVICES mapping.

On a shared server other users occupy GPUs unpredictably. This queries
``nvidia-smi`` and orders the devices so the **freest** card (most free
memory, lowest utilization) is device 0 — the one used by the nproc=1
baseline, which is the most memory-heavy single process.

Usage::

    python tools/pick_freest_gpu.py                      # all 6, sorted
    python tools/pick_freest_gpu.py -n 4                 # best 4
    python tools/pick_freest_gpu.py -n 6 --min-free 12   # need >=12 GB free each
    CUDA_VISIBLE_DEVICES=$(python tools/pick_freest_gpu.py -n 6 --echo) \\
        python tools/benchmark_distcim_multigpu.py ...

Exit code is 1 (and no mapping is echoed) if no GPU meets the constraints,
so a shell can skip the run rather than launch into an OOM.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def query_gpus():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.decode().strip().splitlines():
        idx, free, used, util = [x.strip() for x in line.split(",")]
        gpus.append(dict(idx=int(idx), free_mb=int(free), used_mb=int(used),
                         util=int(util)))
    return gpus


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=6, help="how many GPUs to pick")
    ap.add_argument("--min-free", type=float, default=0.0,
                    help="require at least this many GB free on every picked GPU")
    ap.add_argument("--max-util", type=int, default=100,
                    help="skip GPUs with utilization above this %%")
    ap.add_argument("--echo", action="store_true",
                    help="print only the comma-separated mapping (for bash)")
    args = ap.parse_args()

    gpus = query_gpus()
    # rank: most free memory first, then lowest utilization
    ok = [g for g in gpus
          if g["free_mb"] >= args.min_free * 1024 and g["util"] <= args.max_util]
    ok.sort(key=lambda g: (-g["free_mb"], g["util"]))
    picked = ok[:args.n]

    if len(picked) < min(args.n, len(gpus)):
        print(f"ERROR: only {len(picked)}/{args.n} GPUs meet min-free="
              f"{args.min_free}GB max-util={args.max_util}%",
              file=sys.stderr)
        for g in gpus:
            print(f"  GPU {g['idx']}: free={g['free_mb']/1024:.1f}GB "
                  f"util={g['util']}%", file=sys.stderr)
        sys.exit(1)

    mapping = ",".join(str(g["idx"]) for g in picked)
    if args.echo:
        print(mapping)
        return
    print(f"recommended CUDA_VISIBLE_DEVICES={mapping}")
    print(f"{'rank':>5} {'GPU':>4} {'free GB':>8} {'util %':>7}")
    for i, g in enumerate(picked):
        print(f"{i:>5} {g['idx']:>4} {g['free_mb']/1024:>8.1f} {g['util']:>7}")


if __name__ == "__main__":
    main()
