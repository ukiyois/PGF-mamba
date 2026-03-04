import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import io
import time
import multiprocessing as mp


def phi_1(z):
    return torch.expm1(z) / (z + 1e-9)

def phi_1_out(z, out):
    torch.expm1(z, out=out)
    out.div_(z + 1e-9)
    return out

def phi_1_prime(z):
    eps = 1e-9
    mask = torch.abs(z) < eps
    z_safe = torch.where(mask, torch.ones_like(z) * eps, z)
    res = ((z_safe - 1) * torch.exp(z_safe) + 1) / (z_safe ** 2)
    return torch.where(mask, torch.ones_like(res) * 0.5, res)



def _scan_linear_recursive(a, b, h0, out=None):
    L = a.shape[0]
    if out is None:
        out = torch.empty_like(a)
    h = h0
    for t in range(L):
        h = a[t] * h + b[t]
        out[t] = h
    return out

def _scan_linear_reverse_recursive(a_fwd, b_fwd, h_next, out=None):
    L = a_fwd.shape[0]
    z = h_next
    if out is None:
        out = torch.empty_like(a_fwd)
    for t in range(L - 1, -1, -1):
        z = a_fwd[t] * z + b_fwd[t]
        out[t] = z
    return out

# ====================  Mamba Layer  ====================

class SingleLayerMambaPGF(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float() * 0.1))
        self.D = nn.Parameter(torch.ones(d_model))
        
        self.dt_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def _project(self, u):
        W = torch.cat([self.dt_proj.weight, self.B_proj.weight, self.C_proj.weight], dim=0)
        b = torch.cat([self.dt_proj.bias, self.B_proj.bias, self.C_proj.bias], dim=0)
        out = F.linear(u, W, b)
        dt_input = out[:, :self.d_model]
        B_vec = out[:, self.d_model:self.d_model + self.d_state]
        C_vec = out[:, self.d_model + self.d_state:]
        return dt_input, B_vec, C_vec

    def _get_buffers(self, block_size, device, dtype):
        key = (block_size, str(device), str(dtype))
        cache = getattr(self, "_buffer_cache", {})
        if key not in cache:
            D, N = self.d_model, self.d_state
            cache[key] = {
                "h_traj": torch.empty((block_size, D, N), device=device, dtype=dtype),
                "a_fwd": torch.empty((block_size, D, N), device=device, dtype=dtype),
                "delta_h": torch.empty((block_size, D, N), device=device, dtype=dtype),
                "h_prev": torch.empty((block_size, D, N), device=device, dtype=dtype),
            }
            self._buffer_cache = cache
        return cache[key]

    def _pass1_block(self, u_block, target_block, h0, A, buffers):

        dt_input, B_vec, C_vec = self._project(u_block)
        dt = F.softplus(dt_input)
        dtA = dt.unsqueeze(-1) * A.unsqueeze(0)
        dA = torch.exp(dtA)
        dB = phi_1(dtA) * B_vec.unsqueeze(1)
        b = dB * u_block.unsqueeze(-1)
        h_traj = buffers["h_traj"][:u_block.shape[0]]
        _scan_linear_recursive(dA, b, h0, out=h_traj)
        h_traj = h_traj.detach()
        y_raw = (C_vec.unsqueeze(1) * h_traj).sum(dim=-1) + self.D * u_block
        y_pred = y_raw * self.scale
        diff = y_pred - target_block
        loss_sum = (diff * diff).sum()
        return h_traj[-1], loss_sum

    def _pass2_block(
        self,
        u_block,
        target_block,
        h0_block,
        A_local,
        delta_h_next,
        dA_next,
        scale_const,
        buffers,
    ):

        dt_input, B_vec, C_vec = self._project(u_block)
        dt_input = dt_input.detach()
        B_vec = B_vec.detach()
        C_vec = C_vec.detach()
        dt = F.softplus(dt_input)
        dtA = dt.unsqueeze(-1) * A_local.unsqueeze(0)
        dA = torch.exp(dtA)
        dB = phi_1(dtA) * B_vec.unsqueeze(1)
        b = dB * u_block.unsqueeze(-1)
        h_traj = buffers["h_traj"][:u_block.shape[0]]
        _scan_linear_recursive(dA, b, h0_block, out=h_traj)
        h_traj = h_traj.detach()
        y_raw = (C_vec.unsqueeze(1) * h_traj.detach()).sum(dim=-1) + self.D * u_block
        y_pred = y_raw * self.scale

        delta_in = scale_const * (y_pred - target_block) * self.scale
        delta_in = delta_in.detach()
        grad_scale = (scale_const * (y_pred - target_block) * y_raw).sum()

        C_contrib = delta_in.unsqueeze(-1) * C_vec.unsqueeze(1)
        a_fwd = buffers["a_fwd"][:u_block.shape[0]]
        a_fwd[:-1].copy_(dA[1:])
        a_fwd[-1].copy_(dA_next)
        delta_h_block = buffers["delta_h"][:u_block.shape[0]]
        _scan_linear_reverse_recursive(a_fwd, C_contrib, delta_h_next, out=delta_h_block)

        h_prev = buffers["h_prev"][:u_block.shape[0]]
        h_prev[0].copy_(h0_block)
        h_prev[1:].copy_(h_traj[:-1])
        J_Alog_local = (
            dtA * dA * h_prev +
            phi_1_prime(dtA) * dtA * B_vec.unsqueeze(1) * u_block.unsqueeze(-1)
        )
        grad_A_log = (delta_h_block * J_Alog_local).sum(dim=(0, 1))

        J_dt_local = (
            A_local.unsqueeze(0) * dA * h_prev +
            phi_1_prime(dtA) * A_local.unsqueeze(0) * B_vec.unsqueeze(1) * u_block.unsqueeze(-1)
        )
        g_dt = (delta_h_block * J_dt_local).sum(dim=-1)
        grad_dt_input_block = g_dt * torch.sigmoid(dt_input)

        J_B_local = phi_1(dtA) * u_block.unsqueeze(-1)
        grad_B_vec_block = (delta_h_block * J_B_local).sum(dim=1)

        C_vec_grad_block = (delta_in.unsqueeze(-1) * h_traj.detach()).sum(dim=1)
        grad_D = (delta_in * u_block).sum(dim=0)

        grad_stack = torch.cat([grad_dt_input_block, grad_B_vec_block, C_vec_grad_block], dim=1)
        grad_w_stack = grad_stack.t() @ u_block
        grad_dt_w = grad_w_stack[:self.d_model]
        grad_B_w = grad_w_stack[self.d_model:self.d_model + self.d_state]
        grad_C_w = grad_w_stack[self.d_model + self.d_state:]

        grad_b_stack = grad_stack.sum(dim=0)
        grad_dt_b = grad_b_stack[:self.d_model]
        grad_B_b = grad_b_stack[self.d_model:self.d_model + self.d_state]
        grad_C_b = grad_b_stack[self.d_model + self.d_state:]

        delta_h_next = delta_h_block[0].detach()
        dA_next = dA[0].detach()

        return (
            grad_A_log,
            grad_D,
            grad_scale,
            grad_dt_w,
            grad_dt_b,
            grad_B_w,
            grad_B_b,
            grad_C_w,
            grad_C_b,
            delta_h_next,
            dA_next,
        )

    def forward_standard(self, u):
        L, D = u.shape
        N = self.d_state
        A = -torch.exp(self.A_log)
        
        dt_input, B_vec, C_vec = self._project(u)
        dt = F.softplus(dt_input)
        
        dtA = dt.unsqueeze(-1) * A.unsqueeze(0) # (L, D, N)
        dA = torch.exp(dtA)
        dB = phi_1(dtA) * B_vec.unsqueeze(1)    # (L, D, N)
        
        h = torch.zeros(D, N, device=u.device)
        y = []
        for t in range(L):
            h = dA[t] * h + dB[t] * u[t].unsqueeze(-1)
            yt = (C_vec[t].unsqueeze(0) * h).sum(dim=-1) + self.D * u[t]
            y.append(yt)
        
        y = torch.stack(y) * self.scale
        return y

    def pgf_train_step(self, u, target_y, optimizer, block_size):

        L, D = u.shape
        N = self.d_state
        device = next(self.parameters()).device
        A = -torch.exp(self.A_log)
        num_blocks = (L + block_size - 1) // block_size

        # ==================== PASS 1: Block Forward (TOSE) ====================
        pass1_cache = []
        loss_sum = torch.tensor(0.0, device=device, dtype=u.dtype)
        scale_const = 2.0 / (L * D)
        h0 = torch.zeros(D, N, device=device)
        buffers = self._get_buffers(block_size, device, u.dtype)

        with torch.no_grad():
            for block_idx in range(num_blocks):
                start = block_idx * block_size
                end = min(start + block_size, L)
                u_block = u[start:end].to(device)
                target_block = target_y[start:end].to(device)

                h_last, loss_block = self._pass1_block(u_block, target_block, h0, A, buffers)
                loss_sum += loss_block

                pass1_cache.append({
                    "h0": h0.detach().cpu(),
                    "start": start,
                    "end": end,
                })
                h0 = h_last.detach().clone()

        loss = loss_sum / (L * D)

        # ==================== PASS 2: Adjoint Injection ====================
        grad_A_log = torch.zeros_like(self.A_log)
        grad_D = torch.zeros_like(self.D)
        grad_scale = torch.zeros_like(self.scale)

        grad_dt_w = torch.zeros_like(self.dt_proj.weight)
        grad_dt_b = torch.zeros_like(self.dt_proj.bias)
        grad_B_w = torch.zeros_like(self.B_proj.weight)
        grad_B_b = torch.zeros_like(self.B_proj.bias)
        grad_C_w = torch.zeros_like(self.C_proj.weight)
        grad_C_b = torch.zeros_like(self.C_proj.bias)

        delta_h_next = torch.zeros(D, N, device=device)
        dA_next = torch.ones(D, N, device=device)

        for cache in reversed(pass1_cache):
            start = cache["start"]
            end = cache["end"]
            u_block = u[start:end].to(device)
            target_block = target_y[start:end].to(device)
            h0_block = cache["h0"].to(device)
            A_local = A.detach()
            (
                grad_A_log_block,
                grad_D_block,
                grad_scale_block,
                grad_dt_w_block,
                grad_dt_b_block,
                grad_B_w_block,
                grad_B_b_block,
                grad_C_w_block,
                grad_C_b_block,
                delta_h_next,
                dA_next,
            ) = self._pass2_block(
                u_block,
                target_block,
                h0_block,
                A_local,
                delta_h_next,
                dA_next,
                scale_const,
                buffers,
            )

            grad_A_log += grad_A_log_block
            grad_D += grad_D_block
            grad_scale += grad_scale_block
            grad_dt_w += grad_dt_w_block
            grad_dt_b += grad_dt_b_block
            grad_B_w += grad_B_w_block
            grad_B_b += grad_B_b_block
            grad_C_w += grad_C_w_block
            grad_C_b += grad_C_b_block

        optimizer.zero_grad()
        self.A_log.grad = grad_A_log
        self.D.grad = grad_D
        self.scale.grad = grad_scale
        self.dt_proj.weight.grad = grad_dt_w
        self.dt_proj.bias.grad = grad_dt_b
        self.B_proj.weight.grad = grad_B_w
        self.B_proj.bias.grad = grad_B_b
        self.C_proj.weight.grad = grad_C_w
        self.C_proj.bias.grad = grad_C_b

        optimizer.step()

        return loss.item()


class SequenceToSequenceDataset:
    def __init__(self, num_samples, seq_length, d_model, seed=42):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.d_model = d_model
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.sequences = []
        self.targets = []
        for _ in range(num_samples):
            x = torch.randn(seq_length, d_model)
            y_target = torch.randn(seq_length, d_model)
            self.sequences.append(x)
            self.targets.append(y_target)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def train_with_pgf(model, data_batches, optimizer, device, block_size):
    """PGFtraing"""
    model.train()
    total_loss = 0.0
    peak_memory = 0.0
    for x, y_target in data_batches:
        x = x.to(device)
        y_target = y_target.to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        loss = model.pgf_train_step(x, y_target, optimizer, block_size)
        total_loss += loss
        if device.type == "cuda":
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / 1024**2)
    return total_loss / len(data_batches), peak_memory

def train_with_autograd(model, data_batches, optimizer, device):
    model.train()
    total_loss = 0.0
    peak_memory = 0.0
    for x, y_target in data_batches:
        x = x.to(device)
        y_target = y_target.to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad()
        y_pred = model.forward_standard(x)
        loss = F.mse_loss(y_pred, y_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if device.type == "cuda":
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / 1024**2)
    return total_loss / len(data_batches), peak_memory


def _run_training_process(mode, model_state_bytes, opt_state_bytes, config, result_queue):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SequenceToSequenceDataset(config["num_samples"], config["L"], config["D"], seed=config["seed"])
    data_batches = [(dataset[i][0], dataset[i][1]) for i in range(len(dataset))]

    model = SingleLayerMambaPGF(config["D"], config["N"]).to(device)
    model_state = torch.load(io.BytesIO(model_state_bytes), map_location="cpu")
    model.load_state_dict(model_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    optimizer_state = torch.load(io.BytesIO(opt_state_bytes), map_location="cpu")
    optimizer.load_state_dict(optimizer_state)

    losses = []
    memories = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    for epoch in range(config["num_epochs"]):
        if mode == "pgf":
            loss, mem = train_with_pgf(model, data_batches, optimizer, device, config["block_size"])
        else:
            loss, mem = train_with_autograd(model, data_batches, optimizer, device)
        losses.append(loss)
        memories.append(mem)
        print(
            f"Epoch {epoch+1}/{config['num_epochs']}: "
            f"{mode.upper()} Loss={losses[-1]:.6f}, {mode.upper()} Mem={mem:.2f} MB"
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.time() - start_time
    result_queue.put({
        "mode": mode,
        "losses": losses,
        "memories": memories,
        "total_time": total_time,
    })


def main():
    L_list = [128, 256, 512, 1024, 2048, 4096]
    D = 128
    N = 16
    num_samples = 100
    num_epochs = 20
    lr = 1e-4
    seed = 1
    block_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"Device: {device}")
    print(f"Configuration: D={D}, N={N}, block={block_size}")
    print(f"Epochs: {num_epochs}, Learning rate: {lr}, Seed: {seed}")

    model_init = SingleLayerMambaPGF(D, N)
    optimizer_init = torch.optim.Adam(model_init.parameters(), lr=lr)
    model_state_bytes = io.BytesIO()
    opt_state_bytes = io.BytesIO()
    torch.save(model_init.state_dict(), model_state_bytes)
    torch.save(optimizer_init.state_dict(), opt_state_bytes)

    all_results = {}
    for L in L_list:
        config = {
            "L": L,
            "D": D,
            "N": N,
            "num_samples": num_samples,
            "num_epochs": num_epochs,
            "lr": lr,
            "seed": seed,
            "block_size": block_size,
        }

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        print("\n" + "="*60)
        print(f"Training PGF vs Autograd (separate processes) | L={L}")
        print("="*60)

        pgf_proc = ctx.Process(
            target=_run_training_process,
            args=("pgf", model_state_bytes.getvalue(), opt_state_bytes.getvalue(), config, result_queue),
        )
        pgf_proc.start()
        pgf_proc.join()

        auto_proc = ctx.Process(
            target=_run_training_process,
            args=("auto", model_state_bytes.getvalue(), opt_state_bytes.getvalue(), config, result_queue),
        )
        auto_proc.start()
        auto_proc.join()

        results = {}
        while not result_queue.empty():
            item = result_queue.get()
            results[item["mode"]] = item
        all_results[L] = results

    Path("result").mkdir(exist_ok=True)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Loss curves for different L
    epochs = range(1, num_epochs + 1)
    loss_handles = []
    loss_labels = []
    for L in L_list:
        pgf_losses = all_results[L]["pgf"]["losses"]
        auto_losses = all_results[L]["auto"]["losses"]
        h1, = ax_left.plot(epochs, pgf_losses, 'o-', linewidth=2, alpha=0.6, label=f'PGF L={L}')
        h2, = ax_left.plot(epochs, auto_losses, 's-', linewidth=2, alpha=0.6, label=f'Auto L={L}')
        loss_handles.extend([h1, h2])
        loss_labels.extend([h1.get_label(), h2.get_label()])
    ax_left.set_xlabel('Epoch', fontsize=12)
    ax_left.set_ylabel('Loss', fontsize=12)
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(loss_handles, loss_labels, loc='upper right')

    # Right: Memory lines + latency bars
    L_vals = list(L_list)
    pgf_mems = [max(all_results[L]["pgf"]["memories"]) for L in L_vals]
    auto_mems = [max(all_results[L]["auto"]["memories"]) for L in L_vals]
    pgf_time = [all_results[L]["pgf"]["total_time"] / num_epochs for L in L_vals]
    auto_time = [all_results[L]["auto"]["total_time"] / num_epochs for L in L_vals]

    x = np.arange(len(L_vals))
    width = 0.35
    ax_right.bar(x - width / 2, pgf_time, width, color='#1f77b4', alpha=0.25)
    ax_right.bar(x + width / 2, auto_time, width, color='#ff7f0e', alpha=0.25)
    ax_right.set_xlabel('Sequence Length (L)', fontsize=12)
    ax_right.set_ylabel('Latency (Seconds)', fontsize=12)
    ax_right.set_xticks(x)
    ax_right.set_xticklabels([str(v) for v in L_vals])

    ax_mem = ax_right.twinx()
    line_m1 = ax_mem.plot(x, pgf_mems, 'o-', color='tab:blue', label='PGF Mem', linewidth=2, alpha=0.7)
    line_m2 = ax_mem.plot(x, auto_mems, 's-', color='tab:red', label='Auto Mem', linewidth=2, alpha=0.7)
    ax_mem.set_ylabel('Peak Memory (MB)', fontsize=12)

    # Right legend (latency + memory)
    latency_handles = [
        Patch(facecolor='#1f77b4', alpha=0.25, label='PGF Latency'),
        Patch(facecolor='#ff7f0e', alpha=0.25, label='Auto Latency'),
    ]
    right_handles = latency_handles + line_m1 + line_m2
    right_labels = [h.get_label() for h in right_handles]
    ax_right.legend(right_handles, right_labels, loc='upper left')

    plt.tight_layout()
    plt.savefig('result/Fig5_TrainingComparison.pdf', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to result/Fig5_TrainingComparison.pdf")

if __name__ == "__main__":
    main()
