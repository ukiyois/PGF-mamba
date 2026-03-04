import argparse
import os
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pgf_mamba import FrechetMambaOperator


def load_token_tensor(path: str) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Token file {path} not found.")
    tensor = torch.load(path, map_location="cpu")
    if tensor.dim() != 1:
        raise ValueError(f"Token tensor must be 1-D, got shape {tensor.shape}")
    return tensor


def build_embedder(tokens: torch.Tensor, d_model: int) -> torch.nn.Embedding:
    vocab_est = int(tokens.max().item()) + 1 if tokens.numel() else 65536
    vocab_size = max(vocab_est, 1 << 16)
    embed = torch.nn.Embedding(vocab_size, d_model)
    with torch.no_grad():
        embed.weight.uniform_(-0.02, 0.02)
    return embed


def measure_time_mem(fn, device: str):
    def wrapper():
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device=device)
        start = time.perf_counter()
        result = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device=device)
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        else:
            peak_mem = 0.0
        duration = time.perf_counter() - start
        return result, duration, peak_mem

    return wrapper


def run_pgf_chunk(model, embed, tokens_chunk, block_size, device):
    u = torch.tanh(embed(tokens_chunk)) * 0.1
    u = u.to(device)
    du = torch.randn_like(u) * 1e-3
    outputs, dy = model.pgf_forward(u, du, block_size=block_size, streaming=False)
    outputs_cpu = outputs.detach().cpu()
    dy_cpu = dy.detach().cpu()
    del u, du, outputs, dy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs_cpu, dy_cpu


def pgf_sensitivity_run(model, tokens, embed, window_size, num_windows, block_size, device):
    stats = []
    total_len = tokens.size(0)
    win_idx = 0
    while win_idx * window_size < total_len:
        if 0 < num_windows <= win_idx:
            break
        start = win_idx * window_size
        end = min(start + window_size, total_len)
        if end - start == 0:
            break

        tokens_chunk = tokens[start:end]

        def call():
            return run_pgf_chunk(model, embed, tokens_chunk, block_size, device)

        (outputs, dy), win_time, win_mem = measure_time_mem(call, device)()

        sens_norm = torch.norm(dy, dim=-1)
        out_norm = torch.norm(outputs, dim=-1)
        stats.append(
            {
                "window": win_idx,
                "start": start,
                "end": end,
                "time_s": win_time,
                "peak_mem_mb": win_mem,
                "sens_norm_mean": sens_norm.mean().item(),
                "sens_norm_max": sens_norm.max().item(),
                "output_norm_mean": out_norm.mean().item(),
            }
        )
        win_idx += 1

    return stats


def stabilize_mamba(model):
    with torch.no_grad():
        if hasattr(model, "A_log"):
            model.A_log.uniform_(-3.0, -1.0)
        if hasattr(model, "B_proj"):
            nn.init.normal_(model.B_proj.weight, mean=0.0, std=0.01)
            if model.B_proj.bias is not None:
                nn.init.normal_(model.B_proj.bias, mean=0.0, std=0.001)
        if hasattr(model, "C_proj"):
            nn.init.normal_(model.C_proj.weight, mean=0.0, std=0.01)
            if model.C_proj.bias is not None:
                nn.init.normal_(model.C_proj.bias, mean=0.0, std=0.001)
        if hasattr(model, "dt_proj"):
            nn.init.normal_(model.dt_proj.weight, mean=0.0, std=0.001)
            nn.init.constant_(model.dt_proj.bias, -1.0)
        if hasattr(model, "D"):
            model.D.fill_(1.0)


def plot_stats(stats, save_path):
    if not stats:
        print("No stats to plot; skipping figure.")
        return

    windows = [s["window"] for s in stats]
    times = [s["time_s"] for s in stats]
    mems = [s["peak_mem_mb"] for s in stats]
    sens_mean = [s["sens_norm_mean"] for s in stats]
    sens_max = [s["sens_norm_max"] for s in stats]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 12,
        "font.size": 11,
    })

    fig, ax1 = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1[0].plot(windows, times, marker="o", color="#1f77b4", label="Latency (s)")
    ax1[0].set_ylabel("Latency (s)")
    ax1_2 = ax1[0].twinx()
    ax1_2.plot(windows, mems, marker="s", color="#d62728", label="Peak VRAM (MB)")
    ax1_2.set_ylabel("Peak VRAM (MB)")
    ax1_2.tick_params(axis="y", colors="black")
    lines, labels = ax1[0].get_legend_handles_labels()
    lines2, labels2 = ax1_2.get_legend_handles_labels()
    ax1[0].legend(lines + lines2, labels + labels2, loc="upper right")
    ax1[0].set_title("PGF Runtime and Memory per Window")
    ax1[0].grid(alpha=0.2)

    ax1[1].plot(windows, sens_mean, marker="o", color="#2ca02c", label="Mean Sensitivity")
    ax1[1].fill_between(windows, sens_mean, sens_max, color="#2ca02c", alpha=0.2, label="Max Sensitivity")
    ax1[1].set_ylabel("Sensitivity Norm")
    ax1[1].set_xlabel("Window Index")
    ax1[1].set_title("Input Sensitivity Across Windows")
    ax1[1].grid(alpha=0.2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] Saved figure to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="PGF long-text experiments")
    parser.add_argument(
        "--data",
        type=str,
        default="data/real_world/wiki_long_131072.pt",
        help="Path to token tensor (.pt).",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on.")
    parser.add_argument("--block-size", type=int, default=256, help="TOSE block size.")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension.")
    parser.add_argument("--window-size", type=int, default=16384, help="Tokens per window.")
    parser.add_argument("--num-windows", type=int, default=0, help="Number of windows to evaluate (0 = entire text).")
    parser.add_argument(
        "--plot-path",
        type=str,
        default="",
        help="Where to save the summary figure (PDF). Leave empty to auto-name.",
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    tokens = load_token_tensor(args.data)
    embed = build_embedder(tokens, args.d_model)
    model = FrechetMambaOperator(d_model=args.d_model, d_state=16).to(device)
    stabilize_mamba(model)

    print("=== Sensitivity Run (PGF, chunked) ===")
    stats = pgf_sensitivity_run(
        model,
        tokens,
        embed,
        window_size=args.window_size,
        num_windows=args.num_windows,
        block_size=args.block_size,
        device=device,
    )
    if not stats:
        print("No windows processed (sequence too short?).")
    for entry in stats:
        print(entry)
    plot_path = args.plot_path or infer_plot_path(args.data)
    if plot_path:
        plot_stats(stats, plot_path)


def infer_plot_path(data_path: str) -> str:
    base = os.path.basename(data_path).lower()
    if "c4" in base:
        fname = "FigA5_C4Tech.pdf"
    else:
        fname = "FigA4_WikiLong.pdf"
    return os.path.join("result", fname)


if __name__ == "__main__":
    main()

