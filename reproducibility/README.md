# Phase Gradient Flow (PGF): Reproducibility Suite

This directory contains the core implementation of Phase Gradient Flow (PGF) and the experimental scripts used to generate the results presented in the arxiv submission: "Breaking the Memory Wall: Exact Analytical Differentiation via Tiled Operator-Space Evolution".

## Prerequisites

- Python 3.10+
- PyTorch 2.1.0+ (requires `torch.func`)
- NumPy, Pandas, Matplotlib, SciPy, psutil

## Core Scripts & Reproducibility

| Script | Purpose | Paper Reference |
| :--- | :--- | :--- |
| `pgf_mamba.py` | Core library implementing the TOSE algorithm. | Methodology |
| `bench_scaling.py` | Scalability stress test data collection (RTX 5090). Generates CSV data for Figure 1. | Figure 1 (data only) |
| - | Numerical stability landscape data. Data source: `result/verify_fidelity.csv`. | Figure 2 (data only) |
| `bench_performance.py` | Hardware efficiency benchmarking data collection (RTX 5060). Generates CSV data for Figure 3. | Figure 3 (data only) |
| `sensitivity.py` | Ghost Pulse detection in 128k sequences. | Figure 4 |
| `stiffness_experiment.py` | Stability under extreme selective stiffness. | Appendix A.1 |
| `precision_robustness_experiment.py` | Mixed precision and Length-Invariance analysis. | Appendix A.2 |
| `complexity_collapse_experiment.py` | Complexity class verification (O(N^4) to O(N)). | Appendix A.3 |
| `parameter_gradient_experiment.py` | VRAM scaling and parameter gradient fidelity. | Appendix A.4 |

**Note:** 
- `bench_scaling.py` and `bench_performance.py` generate CSV data files only. The actual figures (Figure 1 and Figure 3) are created by separate visualization scripts or manually from the CSV data.
- Figure 2 data is stored in `result/verify_fidelity.csv`. The 3D surface plot (Figure 2) is generated separately from this CSV data.

## Running the Experiments

Experimental results (CSV and Plots) are saved to the `../result/` directory.

```bash
# Example: Reproduce Stability analysis
python stiffness_experiment.py

# Example: Reproduce Speed-Memory Pareto Frontier
python bench_performance.py
```

## Hardware Note
Benchmarks were executed on NVIDIA RTX 5090 and RTX 5060 Laptop GPUs. Metrics may vary based on hardware specifications and CUDA versions.


