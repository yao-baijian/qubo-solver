"""Plot multi-GPU distCIM speedup curves from benchmark CSVs.

Reads the CSVs produced by ``benchmark_distcim_multigpu.py``
(columns: ``nproc, steps, K, time_s, speedup_vs_1gpu``) and renders, for each
CSV, a figure of **speedup vs number of GPUs** with one line per K and one
subplot per step count, plus the ideal-linear reference line.

Usage::

    python tools/plot_multigpu_speedup.py --csv benchmark_results/traffic_realistic/multigpu_speedup_5000.csv \\
        --out benchmark_results/traffic_realistic/speedup_5000_dense.png --title "N=5004, dense"
    # several CSVs -> one PNG each (dense + sparse comparison)
    python tools/plot_multigpu_speedup.py \\
        --csv .../multigpu_speedup_10000.csv .../multigpu_speedup_10000_sparse.csv \\
        --outdir benchmark_results/traffic_realistic
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                times = [float(x) for x in r["times"].split(";") if x]
            except (KeyError, ValueError):
                times = []
            rows.append({
                "nproc": int(r["nproc"]),
                "steps": int(r["steps"]),
                "K": int(r["K"]),
                "time_s": float(r["time_s"]),
                "speedup": float(r["speedup_vs_1gpu"]),
                "times": times,
            })
    return rows


def _aggregate(times, agg):
    """Aggregate per-repeat times into one representative time.

    ``agg`` is one of ``median`` (robust typical), ``min`` (fastest single
    repeat) or ``fast`` (mean of the fastest ``nfast`` repeats — the "fast
    tail", robust to a single noisy repeat yet biased toward the machine's
    true capability on a shared/noisy GPU).  Falls back to the stored
    ``time_s`` median when no per-repeat data is available.
    """
    if not times:
        return None
    t = sorted(times)
    if agg == "median":
        return float(np.median(t))
    if agg == "min":
        return float(t[0])
    nfast = min(2, len(t))            # mean of the fastest 2 repeats
    return float(np.mean(t[:nfast]))


def _scatter_around_avg(ax, n, base_time, times, c):
    """Plot each per-repeat measurement as a speedup point jittered around n."""
    if not times or not base_time or base_time != base_time:
        return
    xs = n + (np.random.default_rng(1234).uniform(-0.16, 0.16, len(times))
              if len(times) > 1 else 0.0)
    sp = [base_time / t for t in times if t and t == t]
    ax.scatter(xs[:len(sp)], sp, s=18, color=c, alpha=0.35,
               edgecolors="none", zorder=2)


def plot_csv(csv_path, out_path, title, agg="fast"):
    rows = load_csv(csv_path)
    steps_list = sorted({r["steps"] for r in rows})
    ks_list = sorted({r["K"] for r in rows})
    nprocs = sorted({r["nproc"] for r in rows})

    ncols = len(steps_list)
    fig, axes = plt.subplots(1, ncols, figsize=(6.0 * ncols, 5.0), squeeze=False)
    colors = plt.cm.viridis([0.1, 0.35, 0.6, 0.85]) if len(ks_list) > 1 else ["tab:blue"]

    for ax, steps in zip(axes[0], steps_list):
        ax.axhline(1.0, color="gray", lw=0.8, ls=":", label="1 GPU baseline")
        ax.plot(nprocs, nprocs, "--", color="lightgray", lw=1.2,
                label="ideal linear")
        for k, c in zip(ks_list, colors):
            sp = []
            base = None
            for n in nprocs:
                match = [r for r in rows
                         if r["steps"] == steps and r["K"] == k and r["nproc"] == n]
                if not match:
                    sp.append(float("nan"))
                    continue
                r = match[0]
                # representative time: fast-tail aggregate (or min/median)
                ag = _aggregate(r["times"], agg) or r["time_s"]
                if base is None:
                    base = ag                      # nproc=1 baseline
                sp.append(base / ag)               # speedup vs same baseline
                # individual data points around the representative point
                _scatter_around_avg(ax, n, base, r["times"], c)
            ax.plot(nprocs, sp, "o-", color=c, lw=1.6, ms=5, label=f"K={k}",
                    zorder=3)
        ax.set_title(f"steps = {steps}")
        ax.set_xlabel("number of GPUs")
        ax.set_ylabel(f"speedup vs 1 GPU ({agg} = marker, dots = repeats)")
        ax.set_xticks(nprocs)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")

    # markdown summary table
    print(f"\n## {title} ({csv_path})")
    print("| nproc | steps | K | time_s | speedup |")
    print("|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: (x["steps"], x["K"], x["nproc"])):
        print(f"| {r['nproc']} | {r['steps']} | {r['K']} | {r['time_s']:.4f} | "
              f"{r['speedup']:.3f} |")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True,
                    help="benchmark CSV(s) to plot")
    ap.add_argument("--outdir", default=str(
        REPO_ROOT / "benchmark_results" / "traffic_realistic"))
    ap.add_argument("--out", nargs="+", default=None,
                    help="explicit output PNG path(s), one per CSV")
    ap.add_argument("--title", nargs="+", default=None,
                    help="title(s), one per CSV")
    ap.add_argument("--agg", default="fast", choices=["fast", "min", "median"],
                    help="how to reduce per-repeat times to the representative "
                         "point: 'fast' (default) = mean of the fastest 2 "
                         "repeats, 'min' = fastest repeat, 'median' = median")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for i, csv_path in enumerate(args.csv):
        p = Path(csv_path)
        out = (Path(args.out[i]) if args.out and i < len(args.out)
               else outdir / (p.stem + ".png"))
        title = (args.title[i] if args.title and i < len(args.title)
                 else p.stem)
        plot_csv(p, out, title, agg=args.agg)


if __name__ == "__main__":
    main()
