import torch
import numpy as np
import time
import csv
import gc
import psutil
import os
from pgf_mamba import FrechetMambaOperator

def get_memory_usage():
    """Get current GPU/CPU memory usage statistics."""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_memory_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        return gpu_memory, gpu_memory_peak
    else:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024, 0

def autograd_grad(model, u):
    """Reference Autograd gradient computation."""
    u = u.detach().requires_grad_(True)
    y = model.forward(u).sum()
    y.backward()
    return u.grad.detach()

def test_single_length(L, d_model=256, d_state=16, directions_per_L=1, seeds=[1]):
    """Evaluate performance and precision for a given sequence length L."""
    print(f"[BENCH] Sequence Length L={L}, Dimension D={d_model}")

    model = FrechetMambaOperator(d_model=d_model, d_state=d_state)
    if torch.cuda.is_available():
        model = model.cuda()

    results = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        u = torch.randn(L, d_model) * 0.1
        if torch.cuda.is_available():
            u = u.cuda()

        # --- Baseline: Autograd ---
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            mem_before_ag, _ = get_memory_usage()
            start_time = time.time()
            grad_ag = autograd_grad(model, u)
            autograd_time = time.time() - start_time
            _, autograd_memory_peak = get_memory_usage()
            autograd_memory_current = autograd_memory_peak - mem_before_ag
            autograd_available = True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [STATUS] Autograd: OOM")
                autograd_available = False
                autograd_time, autograd_memory_peak = float('inf'), float('inf')
            else: raise e

        # --- Proposed: PGF (Streaming Mode) ---
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            start_time = time.time()
            mem_before, _ = get_memory_usage()

            for dir_idx in range(directions_per_L):
                rand_dir = torch.randn_like(u)
                rand_dir = rand_dir / (torch.norm(rand_dir) + 1e-12)

                jvp_accumulator = 0.0
                def streaming_callback(y_blk, dy_blk, start, end):
                    nonlocal jvp_accumulator
                    jvp_accumulator += torch.sum(dy_blk).item()

                model.pgf_forward(u, rand_dir, block_size=64, streaming=True, callback=streaming_callback)
                frechet_time = time.time() - start_time
                _, frechet_memory_peak = get_memory_usage()
                frechet_memory_current = frechet_memory_peak - mem_before

                if autograd_available:
                    ag_projection = torch.sum(grad_ag * rand_dir)
                    abs_error = abs(jvp_accumulator - ag_projection.item())
                    rel_error = abs_error / (abs(ag_projection.item()) + 1e-12)
                    speedup = autograd_time / frechet_time
                    print(f"  [STATUS] PGF: Success | Speedup: {speedup:.1f}x | RelErr: {rel_error:.2e}")
                else:
                    print(f"  [STATUS] PGF: Success | Memory: {frechet_memory_peak:.1f}MB")

                results.append({
                    'L': L, 'seed': seed, 'direction': dir_idx,
                    'autograd_time': autograd_time, 'autograd_memory_peak': autograd_memory_peak,
                    'frechet_time': frechet_time, 'frechet_memory_peak': frechet_memory_peak,
                    'rel_error': rel_error if autograd_available else float('nan'),
                    'autograd_available': autograd_available
                })
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [STATUS] PGF: OOM")
                results.append({'L': L, 'frechet_time': float('inf')})
            else: raise e

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return results

def main():
    print("=== Phase Gradient Flow (PGF) Stress Benchmark ===")
    L_start, L_step, d_model = 10000, 5000, 256
    csv_filename = "result/bench_scaling.csv"

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['L', 'seed', 'direction', 'ag_time', 'ag_mem', 'pgf_time', 'pgf_mem', 'rel_err', 'ag_ok'])

    L = L_start
    while True:
        results = test_single_length(L, d_model=d_model)
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            for r in results:
                if r.get('frechet_time') != float('inf'):
                    writer.writerow([
                        r['L'], r['seed'], r['direction'],
                        r['autograd_time'], r['autograd_memory_peak'],
                        r['frechet_time'], r['frechet_memory_peak'],
                        r['rel_error'], r['autograd_available']
                    ])
        
        if any(r.get('frechet_time') == float('inf') for r in results):
            print(f"[INFO] PGF reached hardware limit at L={L}")
            break
        L += L_step

    print(f"[INFO] Benchmark completed. Results saved to {csv_filename}")

if __name__ == "__main__":
    main()
