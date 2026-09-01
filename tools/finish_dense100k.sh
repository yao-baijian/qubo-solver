#!/usr/bin/env bash
# Finish the dense 100k sweep: retry until all 54 configs are in the CSV.
# Each attempt picks a freest GPU for the (memory-heavy) 1-GPU baseline, since
# a neighbor unpredictably grabs cards on this shared server. Resume skips
# configs already recorded.
set -u
cd /home/byao/qubo-solver
export PATH=/home/byao/qubo-solver/.venv/bin:$PATH
LOG=/tmp/dense100k_driver.log
CSV=benchmark_results/traffic_realistic/multigpu_speedup_100000.csv

count_missing() {
    python -c "
import csv
rows=list(csv.DictReader(open('$CSV')))
by={(int(r['nproc']),int(r['steps']),int(r['K'])) for r in rows}
missing=[(n,s,k) for s in [1000,3000,5000] for k in [1,5,10] for n in [1,2,3,4,5,6] if (n,s,k) not in by]
print(len(missing))
"
}

attempt=0
while true; do
    miss=$(count_missing)
    echo "=== attempt $attempt: $miss missing ===" | tee -a "$LOG"
    if [ "$miss" = "0" ]; then
        echo "=== DONE: all 54 dense 100k configs present ===" | tee -a "$LOG"
        break
    fi
    # pick a GPU with >= 40 GB free for the baseline (device 0); fall back
    DEV=$(python tools/pick_freest_gpu.py -n 6 --min-free 35 --echo 2>/dev/null) \
        || DEV="0,1,2,3,4,5"
    echo "attempt $attempt: CUDA_VISIBLE_DEVICES=$DEV" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$DEV" python -u tools/benchmark_distcim_multigpu.py \
        --cars 33332 --routes 3 --seed 7 --nproc 1 2 3 4 5 6 \
        --steps 1000 3000 5000 --ks 1 5 10 --dt 0.01 --precision fp32 \
        --repeats 3 --out multigpu_speedup_100000 >> "$LOG" 2>&1
    # wait a little before retrying so a contended card has a chance to free
    sleep 20
    attempt=$((attempt + 1))
done
