"""
Performance Benchmark: Speed-Memory Pareto Frontier
Comparing: Autograd, PGF (Proposed), and Gradient Checkpointing.
Evaluates O(1) memory scaling and computational efficiency.
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from pgf_mamba import FrechetMambaOperator
import numpy as np
import time
import csv

def autograd_grad(model, u):
    """Reference Autograd gradient computation."""
    u = u.detach().requires_grad_(True)
    y = model.forward(u).sum()
    y.backward()
    return u.grad.detach()

def checkpointed_autograd_grad(model, u, segments=4):
    """Memory-efficient Autograd via Gradient Checkpointing."""
    u = u.detach().requires_grad_(True)

    def checkpointed_forward(u_seg):
        return model.forward(u_seg)

    if segments > 1:
        u_segments = torch.chunk(u, segments, dim=0)
        segment_outputs = [checkpoint.checkpoint(checkpointed_forward, seg, use_reentrant=False) for seg in u_segments]
        y = torch.cat(segment_outputs, dim=0).sum()
    else:
        y = model.forward(u).sum()

    y.backward()
    return u.grad.detach()

def run_benchmark_cycle(L, seed, model):
    """Single benchmark cycle for time and peak memory metrics."""
    torch.manual_seed(seed)
    u = torch.randn(L, 4) * 0.1
    du = torch.randn_like(u) * 1e-6

    if torch.cuda.is_available():
        u, du, model = u.cuda(), du.cuda(), model.cuda()
    model.eval()

    def measure_method(grad_fn, *args):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated() / 1024 / 1024
        
        start_t = time.perf_counter()
        grad_fn(*args)
        torch.cuda.synchronize()
        latency = (time.perf_counter() - start_t) * 1000
        
        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        return latency, peak_mem - base_mem

    # 1. Autograd
    ag_time, ag_mem = measure_method(autograd_grad, model, u)
    
    # 2. PGF (Proposed)
    pgf_time, pgf_mem = measure_method(model.pgf_forward, u, du, 32)
    
    # 3. Checkpointing
    segments = max(2, int(np.sqrt(L / 100))) if L > 100 else 1
    cp_time, cp_mem = measure_method(checkpointed_autograd_grad, model, u, segments)

    return {
        'L': L, 'ag_t': ag_time, 'ag_m': ag_mem,
        'pgf_t': pgf_time, 'pgf_m': pgf_mem,
        'cp_t': cp_time, 'cp_m': cp_mem,
        'speedup': ag_time / pgf_time
    }

def perform_full_analysis():
    """Execute complete benchmark suite across L values."""
    print("[INFO] Starting Speed-Memory Pareto Analysis")
    L_values = [100, 500, 1000, 5000, 10000]
    seeds = [1, 2, 3]
    model = FrechetMambaOperator(d_model=4, d_state=2, init_window=1)
    
    csv_file = 'result/bench_performance.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['L', 'ag_time', 'ag_mem', 'pgf_time', 'pgf_mem', 'cp_time', 'cp_mem', 'speedup'])

        for L in L_values:
            print(f"[RUN] Benchmarking L={L}...")
            cycle_results = [run_benchmark_cycle(L, s, model) for s in seeds]
            
            # Average results
            avg_res = {k: np.mean([r[k] for r in cycle_results]) for k in cycle_results[0].keys()}
            writer.writerow([L, avg_res['ag_t'], avg_res['ag_m'], avg_res['pgf_t'], avg_res['pgf_m'], avg_res['cp_t'], avg_res['cp_m'], avg_res['speedup']])

    print(f"[INFO] Analysis complete. Results saved to {csv_file}")

if __name__ == "__main__":
    perform_full_analysis()
