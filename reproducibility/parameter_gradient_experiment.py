import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import gc
from pgf_mamba import FrechetMambaOperator
from torch.func import jvp

os.makedirs("result", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 10,
    "font.size": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.2,
})

def get_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0

def run_parameter_gradient_benchmark():
    """
    Appendix A.4: O(1) Memory Verification and Gradient Numerical Fidelity.
    """
    print("=" * 60)
    print("TRAINING FEASIBILITY: MEMORY SCALING & NUMERICAL ACCURACY")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = 128
    L_list = [1000, 5000, 10000, 20000]
    
    results = []
    
    for L in L_list:
        print(f"\nBenchmarking L = {L}...")
        
        # Initialize model and data
        model = FrechetMambaOperator(d_model=D).to(device)
        model.eval()
        # Use small inputs and perturbations to ensure numerical safety
        with torch.no_grad():
            model.A_log.data.fill_(-4.0) # Increase damping for numerical stability
            
        u = torch.randn(L, D, device=device) * 0.01
        du = torch.randn(L, D, device=device) * 0.01
        
        # --- 1. Autograd Reference (JVP) ---
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        
        u_ag = u.clone().detach().requires_grad_(True)
        # Use torch.func.jvp for exact tangent-flow reference
        def f(x): return model.forward(x)
        _, dy_ag = jvp(f, (u_ag,), (du,))
        
        # Record peak memory (Backpropagation mode)
        torch.cuda.reset_peak_memory_stats()
        y_tmp = model.forward(u_ag)
        y_tmp.sum().backward()
        mem_ag = get_gpu_memory()

        # --- 2. PGF (Streaming Mode) ---
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        
        dy_pgf_list = []
        def callback(y_blk, dy_blk, start, end):
            dy_pgf_list.append(dy_blk.detach())
            
        model.pgf_forward(u, du, block_size=128, streaming=True, callback=callback)
        mem_pgf = get_gpu_memory()
        
        # Concatenate PGF results for verification
        dy_pgf = torch.cat(dy_pgf_list, dim=0)
        
        # Compute relative error (with epsilon protection)
        norm_ag = torch.norm(dy_ag)
        if norm_ag < 1e-12:
            rel_error_val = 0.0 if torch.norm(dy_pgf) < 1e-12 else float('inf')
        else:
            rel_error = torch.norm(dy_pgf - dy_ag) / norm_ag
            rel_error_val = rel_error.item()
        
        results.append({
            'L': L, 
            'Backprop_Mem': mem_ag, 
            'PGF_Mem': mem_pgf,
            'Rel_Error': rel_error_val
        })
        
        print(f"  [Memory] Backprop: {mem_ag:.2f} MB, PGF: {mem_pgf:.2f} MB")
        print(f"  [Accuracy] Relative Error: {rel_error_val:.2e}")
        
        del model, u, du, u_ag, dy_ag, dy_pgf, dy_pgf_list
        torch.cuda.empty_cache()
        gc.collect()

    # Save results and generate plot
    df = pd.DataFrame(results)
    df.to_csv("result/parameter_gradient_benchmark.csv", index=False)
    
    fig, ax1 = plt.subplots(figsize=(7, 5))
    
    ax1.plot(df['L'], df['Backprop_Mem'], 's-', color='#D6404E', label='Backprop $O(L)$ Memory')
    ax1.plot(df['L'], df['PGF_Mem'], 'o-', color='#2A7BB6', label='PGF $O(1)$ Memory')
    ax1.set_xlabel('Sequence Length ($L$)', fontweight='bold')
    ax1.set_ylabel('Peak GPU Memory (MB)', fontweight='bold')
    ax1.legend(loc='upper left')
    
    # Secondary axis for relative error (optional)
    print(f"\nMean Relative Error across all L: {df['Rel_Error'].mean():.2e}")
    
    plt.tight_layout()
    plt.savefig("result/FigA4_TrainingFeasibility.pdf", format='pdf')
    print("\n[COMPLETE] Figure and Results saved.")

def get_peak_memory_stats():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 / 1024

if __name__ == "__main__":
    run_parameter_gradient_benchmark()
