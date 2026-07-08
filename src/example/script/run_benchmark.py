import os
import subprocess
import re
import time
import statistics

BENCHMARK_DIR = "./tools/pysgen/allbenchmarks/"
OUTPUT_DIR = "./output/"
QUBO_SAT_PATH = "./build/fp"
MINISAT_PATH = "./tools/minisat/build/minisat"

TEST_CASES = [
    "sgen-base-s32-g4-0.cnf",
    # add case here
]

def parse_time(output):
    match = re.search(r'CPU time\s*:\s*([\d.]+)\s*s', output)
    if match:
        return float(match.group(1))
    
    match = re.search(r'CPU time:\s*([\d.]+)\s*ms', output)
    if match:
        return float(match.group(1)) / 1000
    
    print("ERROR, cannot parse time")
    return None

def run_minisat(cnf_file, output_file):
    cmd = [MINISAT_PATH, cnf_file, output_file]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    cpu_time = parse_time(result.stdout)
    sat_status = parse_minisat_result(result.stdout)
    return cpu_time, sat_status

def run_qubo_sat(cnf_file, hessian, run_count=10):

    total_time = 0.0
    total_pre_rate = 0.0
    total_post_rate = 0.0

    for i in range(run_count):
        cmd = [QUBO_SAT_PATH, "sat", cnf_file, hessian, "single", "1", "1", "0"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        cpu_time = parse_time(result.stdout)
        pre_fix_rate, post_fix_rate = parse_qubo_sat_success_rate(result.stdout)

        total_time += cpu_time
        total_pre_rate += pre_fix_rate
        total_post_rate += post_fix_rate

    
    avg_time = total_time / run_count
    avg_pre_rate = total_pre_rate / run_count
    avg_post_rate = total_post_rate / run_count

    return avg_time, avg_pre_rate, avg_post_rate

def parse_qubo_sat_success_rate(output):
    if isinstance(output, bytes):
        output = output.decode('utf-8')
    
    # 匹配修复前的解决率
    pre_fix_match = re.search(
        r'\[Solution report\] Total clauses: \d+, Successed clauses num: \d+, Precent: ([\d.]+)',
        output
    )
    # 匹配修复后的解决率
    post_fix_match = re.search(
        r'\[Solution report\] After fixing total clauses: \d+, Successed clauses num: \d+, Precent: ([\d.]+)',
        output
    )
    
    if pre_fix_match and post_fix_match:
        return float(pre_fix_match.group(1)), float(post_fix_match.group(1))
    return None, None

def parse_minisat_result(output):
    if isinstance(output, bytes):
        output = output.decode('utf-8')
    
    # 匹配 SATISFIABLE 或 UNSATISFIABLE
    match = re.search(r'^(SATISFIABLE|UNSATISFIABLE)$', output, re.MULTILINE)
    if match:
        return match.group(1)
    return None

def compare_with_minisat():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    results = []
    
    for test_case in TEST_CASES:
        cnf_path = os.path.join(BENCHMARK_DIR, test_case)
        if not os.path.exists(cnf_path):
            print(f"testcase {cnf_path} non-exist")
            continue
            
        output_path = os.path.join(OUTPUT_DIR, f"{test_case}.out")
        
        print(f"\nTESTING: {test_case}")
        
        minisat_time, sat_status = run_minisat(cnf_path, output_path)
        print(f"MiniSat runtime: {minisat_time:.6f} s, Result: {sat_status}")
        
        qubo_sat_time, pre_rate, post_rate = run_qubo_sat(cnf_path, "auxiliary", 1)
        print(f"QUBOSAT runtime: {qubo_sat_time:.6f} s")
        
        results.append({
            "test_case": test_case,
            "minisat_time": minisat_time,
            "qubo_sat_time": qubo_sat_time
        })
    
def compare_hessian(run_count = 10):

    print("Run count: ", run_count)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    results = []
    
    for test_case in TEST_CASES:

        cnf_path = os.path.join(BENCHMARK_DIR, test_case)
        if not os.path.exists(cnf_path):
            print(f"testcase {cnf_path} non-exist")
            continue
            
        output_path = os.path.join(OUTPUT_DIR, f"{test_case}.out")
        
        print(f"\nTESTING: {test_case}")
        
        aux_qubo_sat_time, aux_pre_rate, aux_post_rate = run_qubo_sat(cnf_path, "auxiliary", run_count)
        print(f"QUBOSAT auxiliary runtime: {aux_qubo_sat_time:.6f} s")
        print(f"Pre-fix success rate: {aux_pre_rate:.4f}, Post-fix success rate: {aux_post_rate:.4f}")
        
        hes_qubo_sat_time, hes_pre_rate, hes_post_rate = run_qubo_sat(cnf_path, "hessian", run_count)
        print(f"QUBOSAT hessian runtime: {hes_qubo_sat_time:.6f} s")
        print(f"Pre-fix success rate: {hes_pre_rate:.4f}, Post-fix success rate: {hes_post_rate:.4f}")

if __name__ == "__main__":
    # compare_with_minisat()
    compare_hessian(10)