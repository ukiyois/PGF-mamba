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
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def test_precision_regime(dtype, L=5000, D=64, seed=42, device="cuda"):
    """
    Test PGF vs Autograd under different precision formats.
    """
    # Map string to torch dtype
    torch_dtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }[dtype]

    # Initialize model in full precision then cast
    model = FrechetMambaOperator(d_model=D).to(device)
    model.eval()
    
    # Set high damping to ensure stability is not the bottleneck
    with torch.no_grad():
        model.A_log.data.fill_(-4.0)
    
    model = model.to(torch_dtype)

    # Generate test data in target precision
    torch.manual_seed(seed)
    u = torch.randn(L, D, device=device, dtype=torch_dtype) * 0.01
    du = torch.randn(L, D, device=device, dtype=torch_dtype) * 0.01

    # 1. PGF Forward
    try:
        dy_pgf_list = []
        def callback(y_blk, dy_blk, start, end):
            dy_pgf_list.append(dy_blk.detach())
        
        model.pgf_forward(u, du, block_size=128, streaming=True, callback=callback)
        dy_pgf = torch.cat(dy_pgf_list, dim=0).to(device)
        pgf_success = True
    except Exception as e:
        print(f"  PGF Error ({dtype}): {e}")
        dy_pgf = None
        pgf_success = False

    # 2. Autograd JVP (Reference in same precision)
    try:
        u_ag = u.detach().requires_grad_(True)
        def f(x): return model.forward(x)
        _, dy_ag = jvp(f, (u_ag,), (du,))
        dy_ag = dy_ag.detach().to(device)
        ag_success = True
    except Exception as e:
        print(f"  Autograd Error ({dtype}): {e}")
        dy_ag = None
        ag_success = False

    # 3. Compute relative error (cast to float32 for metric calculation)
    if pgf_success and ag_success:
        dy_pgf_f32 = dy_pgf.to(torch.float32)
        dy_ag_f32 = dy_ag.to(torch.float32)
        rel_error = torch.norm(dy_pgf_f32 - dy_ag_f32) / (torch.norm(dy_ag_f32) + 1e-12)
        rel_error_val = rel_error.item()
    else:
        rel_error_val = float('nan')

    # Cleanup
    del u, du, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'dtype': dtype,
        'L': L,
        'seed': seed,
        'rel_error': rel_error_val,
        'success': pgf_success and ag_success
    }

def run_precision_experiment():
    print("=" * 60)
    print("MIXED PRECISION ROBUSTNESS TEST (APPENDIX A.2)")
    print("=" * 60)
    
    dtypes = ['float32', 'bfloat16', 'float16']
    L_values = [2000, 5000, 10000]
    seeds = [1, 2, 3, 4, 5]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = []
    total_runs = len(dtypes) * len(L_values) * len(seeds)
    current_run = 0

    for dtype in dtypes:
        for L in L_values:
            for seed in seeds:
                current_run += 1
                print(f"[{current_run}/{total_runs}] Testing {dtype} | L={L} | Seed={seed}...")
                res = test_precision_regime(dtype, L=L, seed=seed, device=device)
                results.append(res)
                if not np.isnan(res['rel_error']):
                    print(f"  Rel Error: {res['rel_error']:.2e}")
                else:
                    print(f"  FAILED")

    df = pd.DataFrame(results)
    csv_path = "result/precision_robustness.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] Results to {csv_path}")
    
    # Visualization
    plot_precision_results(df)

def plot_precision_results(df):
    plt.figure(figsize=(6, 4.5))
    
    # Calculate statistics across seeds and L values for each dtype
    summary = df.groupby('dtype')['rel_error'].agg(['mean', 'std']).reset_index()
    
    # Sort order
    order = {'float32': 0, 'bfloat16': 1, 'float16': 2}
    summary['order'] = summary['dtype'].map(order)
    summary = summary.sort_values('order')

    colors = ['#2A7BB6', '#92C5DE', '#D6404E']
    
    # Use standard error for cleaner bar chart or just std
    bars = plt.bar(summary['dtype'], summary['mean'], yerr=summary['std'], 
                   color=colors, alpha=0.8, capsize=8, edgecolor='black', linewidth=1)

    plt.yscale('log')
    plt.ylabel('Relative Error (PGF vs Autograd)', fontweight='bold')
    plt.xlabel('Computational Precision Format', fontweight='bold')
    # No title for ICML
    
    # Add labels on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.3,
                f'{height:.2e}', ha='center', va='bottom', 
                fontsize=8, fontweight='bold')

    plt.grid(True, which="both", ls="-", alpha=0.1)
    plt.tight_layout()
    plt.savefig("result/FigA2_PrecisionRobustness.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("result/FigA2_PrecisionRobustness.png", dpi=300, bbox_inches='tight')
    print(f"[SAVED] Figure to result/FigA2_PrecisionRobustness.pdf")

if __name__ == "__main__":
    run_precision_experiment()

