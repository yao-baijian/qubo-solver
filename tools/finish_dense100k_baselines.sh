#!/usr/bin/env bash
# Run the missing dense 100k baselines (nproc=1), auto-retrying until all of
# {steps=3000,5000} x {K=1,5,10} baselines are recorded. Each attempt launches
# on the freshest free GPU (>= 30 GB), since a neighbor grabs cards constantly.
set -u
cd /home/byao/qubo-solver
export PATH=/home/byao/qubo-solver/.venv/bin:$PATH
LOG=/tmp/dense100k_baselines.log
CSV=benchmark_results/traffic_realistic/multigpu_speedup_100000.csv

baseline_missing() {
    python -c "
import csv
rows=list(csv.DictReader(open('$CSV')))
by={(int(r['nproc']),int(r['steps']),int(r['K'])) for r in rows}
missing=[(s,k) for s in [3000,5000] for k in [1,5,10] if (1,s,k) not in by]
for s,k in missing: print(f'{s} {k}')
"
}

attempt=0
while true; do
    missing=$(baseline_missing)
    echo "=== attempt $attempt: baselines missing ===" | tee -a "$LOG"
    echo "$missing" | tee -a "$LOG"
    if [ -z "$missing" ]; then
        echo "=== BASELINES DONE ===" | tee -a "$LOG"
        break
    fi
    # pick a GPU with >= 30 GB free; fall back to gpu1
    DEV=$(python tools/pick_freest_gpu.py -n 1 --min-free 30 --echo 2>/dev/null) \
        || DEV="1"
    echo "attempt $attempt: CUDA_VISIBLE_DEVICES=$DEV" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$DEV" python -u tools/benchmark_distcim_multigpu.py \
        --cars 33332 --routes 3 --seed 7 --nproc 1 \
        --steps 3000 5000 --ks 1 5 10 --dt 0.01 --precision fp32 \
        --repeats 3 --out multigpu_speedup_100000 >> "$LOG" 2>&1
    sleep 15
    attempt=$((attempt + 1))
done
