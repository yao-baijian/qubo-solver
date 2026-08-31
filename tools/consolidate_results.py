"""Consolidate a benchmark CSV: keep the LAST row per (nproc, steps, K),
drop NaN-time rows (failed configs), recompute speedup from the kept rows."""
from __future__ import annotations
import csv
import sys
from collections import OrderedDict

path = sys.argv[1]
rows = OrderedDict()
with open(path, newline="") as f:
    for r in csv.reader(f):
        if r and r[0] == "nproc":
            header = r
            continue
        try:
            n, s, k, t = int(r[0]), int(r[1]), int(r[2]), float(r[3])
        except (ValueError, IndexError):
            continue
        if t != t:           # NaN time -> failed, drop
            continue
        # keep extra columns (n_meas, time_std, times) if present
        extra = list(r[5:]) if len(r) > 5 else []
        rows[(n, s, k)] = (n, s, k, t, extra)

# recompute speedups vs the nproc=1 baseline
with open(path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header or ["nproc", "steps", "K", "time_s", "speedup_vs_1gpu",
                          "n_meas", "time_std", "times"])
    for (n, s, k), (_, _, _, t, extra) in rows.items():
        b = rows.get((1, s, k))
        sp = (b[3] / t) if b and t > 0 else float("nan")
        row = [n, s, k, f"{t:.4f}",
               f"{sp:.3f}" if sp == sp else "nan"] + list(extra)
        w.writerow(row)
print(f"consolidated {path}: {len(rows)} valid configs")
