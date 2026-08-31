#!/usr/bin/env bash
# Re-run all multi-GPU speedup sweeps with per-repeat recording (repeats=3).
# Sequential to avoid GPU contention skewing timings. Incremental CSV + resume.
set -u
cd /home/byao/qubo-solver
export PATH=/home/byao/qubo-solver/.venv/bin:$PATH
LOG=/tmp/repeats_all.log
echo "=== start $(date) ===" | tee -a "$LOG"

run() {
    echo "##### $* #####" | tee -a "$LOG"
    eval "$*" >> "$LOG" 2>&1
    echo "##### done: $* (exit $?) at $(date) #####" | tee -a "$LOG"
}

# device 0 -> GPU 2 (freest ~34GB) so 1-GPU baselines run on a quiet card
# dense (5k/10k use K 1,2,5,10; 50k uses K 1,5,10 as before)
run "CUDA_VISIBLE_DEVICES=2,3,4,5,0,1 python -u tools/benchmark_distcim_multigpu.py --cars 1668 --routes 3 --seed 7 --nproc 1 2 3 4 5 6 --steps 1000 3000 5000 --ks 1 2 5 10 --dt 0.01 --precision fp32 --repeats 3 --out multigpu_speedup_5000"
run "CUDA_VISIBLE_DEVICES=2,3,4,5,0,1 python -u tools/benchmark_distcim_multigpu.py --cars 3332 --routes 3 --seed 7 --nproc 1 2 3 4 5 6 --steps 1000 3000 5000 --ks 1 2 5 10 --dt 0.01 --precision fp32 --repeats 3 --out multigpu_speedup_10000"
run "CUDA_VISIBLE_DEVICES=2,3,4,5,0,1 python -u tools/benchmark_distcim_multigpu.py --cars 16668 --routes 3 --seed 7 --nproc 1 2 3 4 5 6 --steps 1000 3000 5000 --ks 1 5 10 --dt 0.01 --precision fp32 --repeats 3 --out multigpu_speedup_50000"

# sparse
run "CUDA_VISIBLE_DEVICES=2,3,4,5,0,1 python -u tools/benchmark_distcim_multigpu.py --cars 1668 --routes 3 --seed 7 --nproc 1 2 3 4 5 6 --steps 1000 3000 5000 --ks 1 5 10 --dt 0.01 --precision fp32 --repeats 3 --sparse --out multigpu_speedup_5000_sparse"
run "CUDA_VISIBLE_DEVICES=2,3,4,5,0,1 python -u tools/benchmark_distcim_multigpu.py --cars 3332 --routes 3 --seed 7 --nproc 1 2 3 4 5 6 --steps 1000 3000 5000 --ks 1 5 10 --dt 0.01 --precision fp32 --repeats 3 --sparse --out multigpu_speedup_10000_sparse"
run "CUDA_VISIBLE_DEVICES=2,3,4,5,0,1 python -u tools/benchmark_distcim_multigpu.py --cars 16668 --routes 3 --seed 7 --nproc 1 2 3 4 5 6 --steps 1000 3000 5000 --ks 1 5 10 --dt 0.01 --precision fp32 --repeats 3 --sparse --out multigpu_speedup_50000_sparse"
run "CUDA_VISIBLE_DEVICES=2,3,4,5,0,1 python -u tools/benchmark_distcim_multigpu.py --cars 33332 --routes 3 --seed 7 --nproc 1 2 3 4 5 6 --steps 1000 3000 5000 --ks 1 5 10 --dt 0.01 --precision fp32 --repeats 3 --sparse --out multigpu_speedup_100000_sparse"

echo "=== ALL DONE $(date) ===" | tee -a "$LOG"
