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

# ICML standard plotting settings
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

def autograd_jvp(model, u, du):
    """Autograd JVP for reference."""
    u = u.detach().requires_grad_(True)
    y = model.forward(u)
    _, dy_ag = jvp(lambda x: model.forward(x), (u,), (du,))
    return dy_ag.detach()

def test_stiffness_regime(A_log_value, L=10000, D=256, d_state=16, seed=42, device="cuda"):
    """
    Test PGF vs Autograd under extreme stiffness.
    
    Args:
        A_log_value: Value for A_log parameter (smaller = stiffer)
        L: Sequence length
        D: Model dimension
        d_state: State dimension
        seed: Random seed for data generation
    """
    model = FrechetMambaOperator(d_model=D, d_state=d_state).to(device)
    model.eval()
    
    # Set extreme stiffness via A_log
    with torch.no_grad():
        model.A_log.data.fill_(A_log_value)
    
    # Generate test data
    torch.manual_seed(seed)
    u = torch.randn(L, D, device=device) * 0.01
    du = torch.randn(L, D, device=device) * 0.01
    
    # Compute PGF
    try:
        dy_pgf_list = []
        def callback(y_blk, dy_blk, start, end):
            dy_pgf_list.append(dy_blk.detach())
        
        model.pgf_forward(u, du, block_size=128, streaming=True, callback=callback)
        dy_pgf = torch.cat(dy_pgf_list, dim=0).to(device)
        pgf_success = True
    except Exception as e:
        print(f"  PGF failed at A_log={A_log_value:.2f}: {e}")
        dy_pgf = None
        pgf_success = False
    
    # Compute Autograd reference
    try:
        dy_ag = autograd_jvp(model, u, du).to(device)
        ag_success = True
    except Exception as e:
        print(f"  Autograd failed at A_log={A_log_value:.2f}: {e}")
        dy_ag = None
        ag_success = False
    
    # Compute relative error if both succeeded
    if pgf_success and ag_success:
        rel_error = torch.norm(dy_pgf - dy_ag) / (torch.norm(dy_ag) + 1e-12)
        rel_error_val = rel_error.item()
    else:
        rel_error_val = float('inf') if not pgf_success else float('nan')
    
    # Cleanup
    del u, du, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'A_log': A_log_value,
        'stiffness': -np.exp(A_log_value),  # Actual A value (negative)
        'L': L,
        'seed': seed,
        'pgf_success': pgf_success,
        'ag_success': ag_success,
        'rel_error': rel_error_val,
    }

def run_stiffness_experiment():
    """Run extreme stiffness stress test."""
    print("=" * 60)
    print("EXTREME STIFFNESS STABILITY TEST")
    print("=" * 60)
    
    # Test range: from extremely stiff (A_log = -8) to normal (A_log = -1)
    # Smaller A_log = stiffer system (A = -exp(A_log) becomes very negative)
    A_log_values = np.linspace(-8.0, -1.0, 15)  # 15 points
    
    L_values = [1000, 2000, 5000, 10000]
    seeds = [1, 2, 3, 4, 5]
    D = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    total_runs = len(A_log_values) * len(L_values) * len(seeds)
    current_run = 0
    
    results = []
    for A_log in A_log_values:
        for L in L_values:
            for seed in seeds:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] Testing A_log={A_log:.2f}, L={L}, seed={seed}...")
                result = test_stiffness_regime(A_log, L=L, D=D, seed=seed, device=device)
                results.append(result)
                print(f"  PGF: {'OK' if result['pgf_success'] else 'FAIL'}, "
                      f"Autograd: {'OK' if result['ag_success'] else 'FAIL'}, "
                      f"Rel Error: {result['rel_error']:.2e}")
    
    # Save to CSV (store all individual run results, do not average)
    df = pd.DataFrame(results)
    csv_path = "result/stiffness_stability.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] Results to {csv_path}")
    print(f"  Total records: {len(df)}")
    print(f"  Columns: {', '.join(df.columns.tolist())}")
    print(f"  Unique (A_log, L, seed) combinations: {len(df.groupby(['A_log', 'L', 'seed']))}")
    
    # Generate visualization
    plot_stiffness_results(df)
    
    return df

def plot_stiffness_results(df):
    """Plot stiffness stability comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Filter valid results
    valid = df[(df['pgf_success']) & (df['ag_success']) & (df['rel_error'] < 1e3)]
    
    # Plot 1: Relative Error vs Stiffness
    ax1.semilogy(valid['A_log'], valid['rel_error'], 
                 marker='o', color='#2A7BB6', linewidth=2, markersize=6, label='PGF vs Autograd')
    ax1.axhline(y=1e-6, color='#D6404E', linestyle='--', linewidth=1.5, label='Machine Precision Threshold')
    ax1.set_xlabel('Stiffness Parameter ($\\log A$)', fontweight='bold')
    ax1.set_ylabel('Relative Error (log scale)', fontweight='bold')
    ax1.set_title('Numerical Stability Under Extreme Stiffness', fontweight='bold')
    ax1.legend(frameon=True, fontsize=8)
    ax1.grid(True, alpha=0.2)
    
    # Plot 2: Success Rate
    pgf_success = df['pgf_success'].sum()
    ag_success = df['ag_success'].sum()
    total = len(df)
    
    methods = ['PGF (Ours)', 'Autograd']
    success_counts = [pgf_success, ag_success]
    colors = ['#2A7BB6', '#D6404E']
    
    bars = ax2.bar(methods, success_counts, color=colors, alpha=0.8, width=0.6)
    ax2.set_ylabel('Successful Runs', fontweight='bold')
    ax2.set_title('Robustness Comparison', fontweight='bold')
    ax2.set_ylim(0, total + 1)
    
    # Add count labels on bars
    for bar, count in zip(bars, success_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{count}/{total}', ha='center', va='bottom', fontweight='bold')
    
    ax2.grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout()
    plt.savefig("result/FigA1_StiffnessStability.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("result/FigA1_StiffnessStability.png", dpi=300, bbox_inches='tight')
    print(f"[SAVED] Figure to result/FigA1_StiffnessStability.pdf")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"PGF Success Rate: {pgf_success}/{total} ({100*pgf_success/total:.1f}%)")
    print(f"Autograd Success Rate: {ag_success}/{total} ({100*ag_success/total:.1f}%)")
    if len(valid) > 0:
        print(f"Mean Relative Error (valid runs): {valid['rel_error'].mean():.2e}")
        print(f"Max Relative Error (valid runs): {valid['rel_error'].max():.2e}")
        print(f"Error < 1e-6: {(valid['rel_error'] < 1e-6).sum()}/{len(valid)} runs")

if __name__ == "__main__":
    df_results = run_stiffness_experiment()
    print("\n[COMPLETE] Extreme stiffness experiment finished.")

