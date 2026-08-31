#!/usr/bin/env bash
# Tighten fast-tail estimates on the headline configs with repeats=5.
# Waits for the main repeats=3 chain (/tmp/repeats_all.log "ALL DONE") so it
# never contends with it for GPUs, then re-runs the best (steps, K) at each
# big size across all nproc 1..6, writing *new* CSVs so the resume logic
# doesn't skip them.
set -u
cd /home/byao/qubo-solver
export PATH=/home/byao/qubo-solver/.venv/bin:$PATH
LOG=/tmp/repeats5.log
echo "=== wait for main chain ===" | tee -a "$LOG"
while ! grep -aq "ALL DONE" /tmp/repeats_all.log 2>/dev/null; do
    sleep 120
done
echo "=== main chain done, starting r5 $(date) ===" | tee -a "$LOG"

run() {
    echo "##### $* #####" | tee -a "$LOG"
    eval "$*" >> "$LOG" 2>&1
    echo "##### done: $* (exit $?) at $(date) #####" | tee -a "$LOG"
}

# device 0 -> freest GPU (picked live; errors -> fall back to 3,4,5,0,1,2)
DEV=$(python tools/pick_freest_gpu.py -n 6 --min-free 12 --echo 2>/dev/null) \
    || DEV="3,4,5,0,1,2"
echo "using CUDA_VISIBLE_DEVICES=$DEV" | tee -a "$LOG"
export CUDA_VISIBLE_DEVICES="$DEV"

# headline (steps, K) with the best speedup at each size/mode
run "python -u tools/benchmark_distcim_multigpu.py --cars 16668 --routes 3 --seed 7 --nproc 1 2 3 4 5 6 --steps 3000 --ks 10 --dt 0.01 --precision fp32 --repeats 5 --out multigpu_speedup_50000_headline"
run "python -u tools/benchmark_distcim_multigpu.py --cars 16668 --routes 3 --seed 7 --nproc 1 2 3 4 5 6 --steps 3000 --ks 5 --dt 0.01 --precision fp32 --repeats 5 --sparse --out multigpu_speedup_50000_sparse_headline"
run "python -u tools/benchmark_distcim_multigpu.py --cars 33332 --routes 3 --seed 7 --nproc 1 2 3 4 5 6 --steps 1000 --ks 10 --dt 0.01 --precision fp32 --repeats 5 --sparse --out multigpu_speedup_100000_sparse_headline"

echo "=== R5 ALL DONE $(date) ===" | tee -a "$LOG"
