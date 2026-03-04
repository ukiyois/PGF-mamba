# Phase Gradient Flow (PGF): Reproducibility Suite



## Prerequisites

- Python 3.10+
- PyTorch 2.1.0+ (requires `torch.func`)
- NumPy, Pandas, Matplotlib, SciPy, psutil

## Core Scripts & Reproducibility

| Script | Purpose | Paper Reference |
| :--- | :--- | :--- |
| `pgf_mamba.py` | Core library implementing the TOSE algorithm. | Methodology |
| `train_singlemamba.py` | PGF vs Autograd training comparison (single-layer Mamba). Generates Figure 5. | Figure 5 |
| `bench_scaling.py` | Scalability stress test data collection (RTX 5090). Generates CSV data for Figure 1. | Figure 1 (data only) |
| - | Numerical stability landscape data. Data source: `result/verify_fidelity.csv`. | Figure 2 (data only) |
| `bench_performance.py` | Hardware efficiency benchmarking data collection (RTX 5060). Generates CSV data for Figure 3. | Figure 3 (data only) |
| `sensitivity.py` | Ghost Pulse detection in 128k sequences. | Figure 4 |
| `run_longtext_experiments.py` | Real-world long-text PGF sensitivity (Wiki/C4-Tech). | Appendix H |
| `stiffness_experiment.py` | Stability under extreme selective stiffness. | Appendix A |
| `precision_robustness_experiment.py` | Mixed precision and Length-Invariance analysis. | Appendix B |
| `complexity_collapse_experiment.py` | Complexity class verification (O(N^4) to O(N)). | Appendix C |

**Note:** 
- `bench_scaling.py` and `bench_performance.py` generate CSV data files only. The actual figures (Figure 1 and Figure 3) are created by separate visualization scripts or manually from the CSV data.
- Figure 2 data is stored in `result/verify_fidelity.csv`. The 3D surface plot (Figure 2) is generated separately from this CSV data.
- Appendix A/B/C figures are rendered to `result/FigA1_StiffnessStability.pdf`, `result/FigA2_PrecisionRobustness.pdf`, and `result/FigA3_ComplexityCollapse.pdf`, respectively.
- Real-world long-text sensitivity figures are exported directly by `run_longtext_experiments.py` to `result/FigA4_WikiLong.pdf` and `result/FigA5_C4Tech.pdf` .

## Running the Experiments

Experimental results (CSV and Plots) are saved to the `../result/` directory.

```bash
# Example: Reproduce Stability analysis
python stiffness_experiment.py

# Example: Reproduce PGF vs Autograd training comparison (Figure 5)
python train_singlemamba.py

# Example: Reproduce Speed-Memory Pareto Frontier
python bench_performance.py
```

## Hardware Note
Benchmarks were executed on NVIDIA RTX 5090 and RTX 5060 Laptop GPUs. Metrics may vary based on hardware specifications and CUDA versions.


