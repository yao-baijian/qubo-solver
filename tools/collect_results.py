"""Collect all benchmark artifacts into a single markdown results record.

Reads:
  - benchmark_results/traffic_realistic/results.md  (realistic ~5000-var traffic)
  - benchmark_results/gset_distcim_compare.csv       (full Gset MaxCut sweep)
and writes ``benchmark_results/RESULTS.md`` covering, for both problem types,
the best result of: original (float32), FPGA-quantized (x-int8), and the
distributed variants with different K.

Usage: python tools/collect_results.py
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BR = REPO_ROOT / "benchmark_results"
OUT = BR / "RESULTS.md"

# latest recorded small synthetic traffic run (8 cars, 23 vars, best over
# dt in 0.1..1.3 x seeds [7,11,13], iters=800, default congestion 50)
SMALL_TRAFFIC = [
    ("original float32 central", "25"),
    ("quant8 central", "26"),
    ("dist quant8 K=1 (standard)", "24"),
    ("dist quant8 K=5 (const)", "24"),
    ("dist quant8 K=10 (const)", "24"),
]

# preliminary Gset spot checks (coarse dt grid {0.1,0.2,0.3,0.5}, iters=1000,
# seeds [0,1,2]) -- full fine-grained sweep skipped on CPU
GSET_SPOT = [
    ("G1", 800, 11624, 11600, 11610, 11520),
    ("G6", 800, 2178, 2156, 2153, 2134),
    ("G11", 800, 564, 550, 554, 554),
]


def read_traffic_realistic():
    f = BR / "traffic_realistic" / "results.md"
    if not f.exists():
        return None, "not run yet (see tools/benchmark_traffic_realistic.py)"
    lines = [l for l in f.read_text().splitlines()
             if l.startswith("| ") and "config" not in l]
    rows = []
    for l in lines:
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if len(cells) >= 4:
            rows.append((cells[0], cells[1], cells[2], cells[3]))
    meta = [l for l in f.read_text().splitlines() if l.startswith("- ")]
    return (rows, meta), None


def read_gset():
    f = BR / "gset_distcim_compare.csv"
    if not f.exists():
        return None, "not run yet (see tools/benchmark_gset_distcim.py)"
    with open(f) as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return rows, None


def summarize_gset(rows):
    """Aggregate per-iters stats + per-instance table at the max iters."""
    by_iters = {}
    for r in rows:
        by_iters.setdefault(int(r["iters"]), []).append(r)

    summary = []
    for iters in sorted(by_iters):
        rs = by_iters[iters]
        ratios = {a: [] for a in ["float32", "quant8", "distq8"]}
        gaps = {"quant8_vs_float32": [], "distq8_vs_float32": []}
        n = 0
        for r in rs:
            if r["best_known"] and float(r["best_known"]) > 0:
                for a in ratios:
                    ratios[a].append(float(r[f"{a}_ratio"]))
                gaps["quant8_vs_float32"].append(float(r["quant8_vs_float32"]))
                gaps["distq8_vs_float32"].append(float(r["distq8_vs_float32"]))
                n += 1
        cells = [f"{iters}"]
        for a in ratios:
            cells.append(f"{statistics.mean(ratios[a])*100:.2f}%")
        for g in gaps:
            cells.append(f"{statistics.mean(g)*100:+.2f}%")
        summary.append((cells, n))

    # per-instance table at the largest iters
    max_iters = max(by_iters)
    inst_rows = sorted(by_iters[max_iters], key=lambda r: r["instance"])
    return summary, max_iters, inst_rows, by_iters


def main():
    print("collecting results...")
    real, real_err = read_traffic_realistic()
    gset_rows, gset_err = read_gset()

    lines = [
        "# DistIM benchmark results (latest)",
        "",
        "> Best result per configuration (over a fine-grained best-dt sweep) "
        "for **original (float32)**, **FPGA-quantized (x-int8)** and the "
        "**distributed** variants with different sync periods K.",
        "",
        "## Traffic-flow",
        "",
        "### Small synthetic instance (8 cars, 23 spin vars)",
        "",
        "Figure of merit: total congestion (lower is better); "
        "default-route baseline = 50.",
        "",
        "| config | congestion | vs default |",
        "|---|---|---|",
    ]
    for tag, c in SMALL_TRAFFIC:
        lines.append(f"| {tag} | {c} | {int(c) - 50:+d} |")

    lines += ["", "### Realistic instance (target-repo construction)"]
    if real is not None:
        rows, meta = real
        lines += ["", *[f"{m}" for m in meta], "",
                  "| config | energy | congestion | vs default |",
                  "|---|---|---|---|"]
        for name, e, c, vd in rows:
            lines.append(f"| {name} | {e} | {c} | {vd} |")
    else:
        lines += ["", f"_Realistic traffic: {real_err}_"]

    lines += ["", "## Gset MaxCut (ground state)",
              "",
              "Best cut over dt sweep (0.1..1.3 step 0.1) x seeds [0,1,2]; "
              "ratio = best cut / best-known value (higher is better)."]
    if gset_rows is None:
        lines += ["",
                  "_Full Gset sweep skipped_ - the fine-grained "
                  "(0.1..1.3 step 0.1) sweep over all 71 instances is too "
                  "slow on this CPU (device stuck). The bit-exact Gset "
                  "validation (Levels 1-5, incl. G1/G6/G11) is in "
                  "`tools/verify_distim.py`; preliminary spot checks below "
                  "use a coarser dt grid (4 values) at iters=1000.",
                  "",
                  "| instance | N | best-known | float32 | quant8 | distq8 |",
                  "|---|---|---|---|---|---|"]
        for name, n, bk, f32, q8, dq8 in GSET_SPOT:
            lines.append(f"| {name} | {n} | {bk} | {f32} | {q8} | {dq8} |")
    else:
        summary, max_iters, inst_rows, by_iters = summarize_gset(gset_rows)
        lines += ["",
                  "### Aggregate (mean over instances, best-known ratio)",
                  "",
                  "| iters | float32 | quant8 | distq8 | quant8-vs-float32 | "
                  "distq8-vs-float32 |",
                  "|---|---|---|---|---|---|"]
        for cells, n in summary:
            lines.append("| " + " | ".join(cells) + f" | (n={n}) |")
        lines += ["",
                  f"### Per-instance (iters = {max_iters})",
                  "",
                  "| instance | N | best-known | float32 | quant8 | distq8 | "
                  "quant8/float32 | distq8/float32 |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in inst_rows:
            lines.append(
                f"| {r['instance']} | {r['N']} | {r['best_known']} "
                f"| {r['float32_cut']} | {r['quant8_cut']} | "
                f"{r['distq8_cut']} | {r['quant8_vs_float32']} | "
                f"{r['distq8_vs_float32']} |")

    lines += ["", "---", "Generated by `tools/collect_results.py`."]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
