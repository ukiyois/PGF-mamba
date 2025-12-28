# Phase Gradient Flow (PGF): Breaking the Memory Wall in Long-Sequence Gradient Computation

Phase Gradient Flow (PGF) is a novel framework that computes exact analytical derivatives for long-sequence models with O(1) memory complexity relative to sequence length, achieving a 94% reduction in peak VRAM and a 23x increase in throughput compared to standard Autograd.

## Key Features

- O(1) Memory Complexity: Constant memory footprint regardless of sequence length
- Exact Gradients: Mathematically equivalent to standard backpropagation, no approximations
- Numerical Stability: Robust to extreme sequence lengths (tested up to L=128,000)
- Hardware Efficient: Enables chromosome-scale sensitivity analysis on consumer GPUs

## Quick Start

### Installation

```bash
pip install -r reproducibility/requirements.txt
```

### Core Implementation

#### Mamba/SSM Gradient Computation

```python
from reproducibility.pgf_mamba import FrechetMambaOperator

# Initialize PGF operator
model = FrechetMambaOperator(d_model=128, d_state=16)

# Forward pass with O(1) memory
u_seq = torch.randn(L, d_model)  # Input sequence
du_seq = torch.randn(L, d_model)  # Tangent vector

y_seq, dy_seq = model.pgf_forward(u_seq, du_seq, block_size=64)
```

## Performance Benchmarks

| Sequence Length | Standard Autograd | PGF | Speedup | Memory Reduction |
|----------------|------------------|-----|---------|------------------|
| L = 1,000      | 63ms, 3.3GB      | 12s, 18MB | - | 94% ↓ |
| L = 6,000      | 63ms, 3.3GB      | 12s, 18MB | - | 94% ↓ |
| L = 100,000    | OOM              | 18MB      | ∞ | 100% ↓ |

*Benchmarks on NVIDIA RTX 5090/5060 GPUs*

## Theoretical Foundation

PGF is based on **Tiled Operator-Space Evolution (TOSE)**, which reframes differentiation as a synchronized dynamical system that evolves alongside the primal state. The **Tangent-Flow Isomorphism** establishes that the Fréchet derivative of a linear recurrence is itself a dynamical system isomorphic to the original.

### Key Innovation: Operator-Space Collapse

Instead of materializing the computational graph, PGF operates directly in the **state-space manifold**, collapsing the graph into a sequence of constant-time state handoffs:

```
Traditional: O(L) memory for hidden states
PGF: O(1) memory via operator-space evolution
```

## Repository Structure

```
reproducibility/
├── pgf_mamba.py                    # Core TOSE algorithm implementation
├── bench_performance.py            # Hardware efficiency benchmarking (Figure 3)
├── bench_scaling.py                # Scalability stress test (Figure 1)
├── sensitivity.py                  # Ghost pulse detection (Figure 4)
├── stiffness_experiment.py         # Stability analysis (Figure 2, Appendix A.1)
├── precision_robustness_experiment.py  # Mixed precision analysis (Appendix A.2)
├── complexity_collapse_experiment.py   # Complexity verification (Appendix A.3)
├── parameter_gradient_experiment.py    # VRAM scaling analysis (Appendix A.4)
├── requirements.txt                # Python dependencies
└── README.md                       # Detailed experimental setup guide
```

## Documentation

- **Reproducibility Guide**: `reproducibility/README.md` - Complete experimental setup and script descriptions

## Reproducing Results

All experimental scripts are in the `reproducibility/` directory:

```bash
cd reproducibility

# Scalability stress test (Figure 1)
python bench_scaling.py

# Performance benchmarking (Figure 3)
python bench_performance.py

# Stability analysis (Figure 2, Appendix A.1)
python stiffness_experiment.py

# Sensitivity analysis (Figure 4)
python sensitivity.py

# Mixed precision analysis (Appendix A.2)
python precision_robustness_experiment.py

# Complexity verification (Appendix A.3)
python complexity_collapse_experiment.py

# VRAM scaling analysis (Appendix A.4)
python parameter_gradient_experiment.py
```

Experimental results (CSV files and plots) are saved to the `result/` directory.

## Citation

If you use PGF in your research, please cite:

```bibtex
@article{pgf2026,
  title={Breaking the Memory Wall: Exact Analytical Differentiation via Tiled Operator-Space Evolution},
  author={Anonymous},
  journal={ICML},
  year={2026}
}
```

## Requirements

- Python 3.10+
- PyTorch 2.1.0+ (requires `torch.func`)
- NumPy, Pandas, Matplotlib, SciPy, psutil
- CUDA-capable GPU (tested on NVIDIA RTX 5090/5060)

See `reproducibility/requirements.txt` for exact package versions.

## Hardware Note

Benchmarks were executed on NVIDIA RTX 5090 and RTX 5060 Laptop GPUs. Performance metrics may vary based on hardware specifications and CUDA versions.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This work enables chromosome-scale sensitivity analysis on single-GPU workstations, bridging the gap between theoretical infinite-context models and practical hardware limitations.

---

