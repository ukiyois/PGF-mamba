import torch
import torch.nn as nn
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import gc
from pgf_mamba import FrechetMambaOperator

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
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

class DenseRTRLSim(nn.Module):
    """
    Simulates a Dense RTRL update to demonstrate O(N^2) or O(N^3) complexity
    for state-to-input sensitivity when ignoring diagonal structure.
    """
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # Pre-allocate large dense matrices to simulate non-diagonal state transition
        self.A_dense = nn.Parameter(torch.randn(d_model, d_state, d_state))
        self.B_dense = nn.Parameter(torch.randn(d_model, d_state))

    def forward_step_with_rtrl(self, u_t, h_prev, dh_du_prev):
        # h_t = A h_{t-1} + B u_t
        # dh_t/du = A * dh_{t-1}/du + B (if we track full Jacobian)
        # Note: This is O(D * N^2) operations per step
        h_curr = torch.einsum('dij,dj->di', self.A_dense, h_prev) + self.B_dense * u_t.unsqueeze(-1)
        
        # RTRL Jacobian update: dh/du = A * (dh/du)_prev + identity-like terms
        # This is the bottleneck that PGF avoids via diagonal isomorphism
        dh_du_curr = torch.einsum('dij,djk->dik', self.A_dense, dh_du_prev) 
        # (Simplified for complexity simulation)
        return h_curr, dh_du_curr

def run_complexity_benchmark():
    print("=" * 60)
    print("EMPIRICAL COMPLEXITY COLLAPSE: PGF VS DENSE RTRL")
    print("=" * 60)
    
    # Range of state dimensions (N)
    N_list = [8, 16, 32, 64, 128, 256]
    D = 64  # Fixed model dimension
    L = 100 # Short sequence for per-step profiling
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = []
    
    for N in N_list:
        print(f"Benchmarking d_state (N) = {N}...")
        
        # 1. Benchmark PGF (O(N))
        model_pgf = FrechetMambaOperator(d_model=D, d_state=N).to(device)
        u = torch.randn(L, D, device=device)
        du = torch.randn(L, D, device=device)
        
        torch.cuda.synchronize()
        t0 = time.time()
        # Warmup
        model_pgf.pgf_forward(u, du, block_size=L)
        torch.cuda.synchronize()
        
        t0 = time.time()
        for _ in range(5):
            model_pgf.pgf_forward(u, du, block_size=L)
        torch.cuda.synchronize()
        pgf_time = (time.time() - t0) / 5.0
        
        # 2. Benchmark Dense RTRL (O(N^2))
        # We simulate the scaling of tracking full state Jacobians
        model_rtrl = DenseRTRLSim(D, N).to(device)
        h = torch.zeros(D, N, device=device)
        dh_du = torch.zeros(D, N, N, device=device) # Full dense Jacobian per channel
        
        torch.cuda.synchronize()
        t0 = time.time()
        # Single pass simulation
        for t in range(L):
            h, dh_du = model_rtrl.forward_step_with_rtrl(u[t], h, dh_du)
        torch.cuda.synchronize()
        
        t0 = time.time()
        for _ in range(5):
            h_tmp, dh_tmp = h, dh_du
            for t in range(L):
                h_tmp, dh_tmp = model_rtrl.forward_step_with_rtrl(u[t], h_tmp, dh_tmp)
        torch.cuda.synchronize()
        rtrl_time = (time.time() - t0) / 5.0
        
        results.append({
            'N': N,
            'PGF_Time': pgf_time,
            'RTRL_Time': rtrl_time,
            'Speedup': rtrl_time / pgf_time
        })
        print(f"  PGF: {pgf_time:.4f}s, RTRL: {rtrl_time:.4f}s | Speedup: {rtrl_time/pgf_time:.1f}x")
        
        del model_pgf, model_rtrl
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv("result/complexity_collapse.csv", index=False)
    
    plot_complexity_collapse(df)

def plot_complexity_collapse(df):
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Time Scaling (Log-Log plot to show polynomial degree)
    ax1.loglog(df['N'], df['RTRL_Time'], marker='s', color='#D6404E', 
               linewidth=2, label='Naive RTRL (Dense Simulation $O(N^2)$)')
    ax1.loglog(df['N'], df['PGF_Time'], marker='o', color='#2A7BB6', 
               linewidth=2, label='PGF (Ours, Diagonal $O(N)$)')
    
    ax1.set_xlabel('State Dimension ($N$)', fontweight='bold')
    ax1.set_ylabel('Execution Time (s)', fontweight='bold')
    ax1.legend(loc='upper left')
    
    # Highlight the "Collapse" gap
    ax1.fill_between(df['N'], df['PGF_Time'], df['RTRL_Time'], color='gray', alpha=0.1)
    ax1.text(df['N'].iloc[-1], df['RTRL_Time'].iloc[-1], ' Complexity Explosion', 
             va='bottom', ha='right', color='#D6404E', fontsize=8, fontweight='bold')
    ax1.text(df['N'].iloc[-1], df['PGF_Time'].iloc[-1], ' Linear Scaling', 
             va='top', ha='right', color='#2A7BB6', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig("result/FigA3_ComplexityCollapse.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("result/FigA3_ComplexityCollapse.png", dpi=300, bbox_inches='tight')
    print(f"[SAVED] Figure to result/FigA3_ComplexityCollapse.pdf")

if __name__ == "__main__":
    run_complexity_benchmark()

