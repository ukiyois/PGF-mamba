import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp

# --- Discretization Operator ---
def mamba_discretization(z):
    """
    Zero-Order Hold (ZOH) discretization for SSM.
    Equivalent to the phi_1 function in Mamba architecture.
    """
    return torch.expm1(z) / (z + 1e-9)

# --- Section 3.2: Numerical Armor (Log-Shifting Stability) ---
def tiled_parallel_scan(dA, Bu, h0):
    """
    Tiled Parallel Scan (TPS) implementation with Log-shifting Stabilizer.
    Corresponds to Proposition 2 and Section 3.2 in the paper.
    """
    # Log-shifting stabilizer to prevent underflow
    log_dA = torch.log(dA + 1e-12)
    Lambda = torch.cumsum(log_dA, dim=0)
    m, _ = torch.max(Lambda, dim=0, keepdim=True)

    shifted_Bu = Bu * torch.exp(m - Lambda)
    C_t = torch.cumsum(shifted_Bu, dim=0)

    h_init_term = h0.unsqueeze(0) * torch.exp(Lambda)
    h_input_term = torch.exp(Lambda - m) * C_t

    return h_init_term + h_input_term

class FrechetMambaOperator(nn.Module):
    def __init__(self, d_model, d_state=16, init_window=3):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.init_window = init_window

        # Parameter initialization
        A_init = torch.randn(self.d_state) * 0.1 - 1.0
        self.A_log = nn.Parameter(torch.log(-A_init))
        self.dt_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, self.d_state)
        self.C_proj = nn.Linear(d_model, self.d_state)
        self.D = nn.Parameter(torch.ones(d_model))
        self.redshift_scale = nn.Parameter(torch.tensor(2.0))

        self.init_estimator = nn.Linear(d_model * init_window, d_model * d_state)

    def _get_params_dict(self):
        return {
            'A_log': self.A_log,
            'dt_w': self.dt_proj.weight, 'dt_b': self.dt_proj.bias,
            'B_w': self.B_proj.weight, 'B_b': self.B_proj.bias,
            'C_w': self.C_proj.weight, 'C_b': self.C_proj.bias,
            'D_w': self.D,
            'scale': self.redshift_scale
        }

    def estimate_initial_state(self, u_seq):
        L, D = u_seq.shape
        if L < self.init_window:
            return torch.zeros(D, self.d_state, device=u_seq.device)
        u_prefix = u_seq[:self.init_window].flatten()
        return self.init_estimator(u_prefix).view(D, self.d_state)


    def pgf_forward(self, u_seq, du_seq, block_size=64, streaming=False, callback=None):
        """
        Phase Gradient Flow (PGF) Forward Pass.
        Implements Tiled Operator-Space Evolution (TOSE) with O(1) memory.
        Ref: Section 2.2 & 2.3 in the paper.
        """
        L, D = u_seq.shape
        params = self._get_params_dict()

        # 1. State Dual-Projection Initialization (Section 2.2)
        h_primal = self.estimate_initial_state(u_seq)
        _, h_tangent = jvp(lambda u: self.estimate_initial_state(u), (u_seq,), (du_seq,))

        if not streaming:
            y_list, dy_list = [], []
        
        num_blocks = (L + block_size - 1) // block_size

        # 2. Section 2.3: Block Manifold Mapping (Psi)
        def manifold_block_mapping(h_initial, u_block):
            """
            Local manifold function Psi: (h_start, u_blk) -> (h_end, y_blk)
            """
            dt = F.softplus(F.linear(u_block, params['dt_w'], params['dt_b']))
            B = F.linear(u_block, params['B_w'], params['B_b'])
            C = F.linear(u_block, params['C_w'], params['C_b'])
            A = -torch.exp(params['A_log'])
            
            dtA = dt.unsqueeze(-1) * A.unsqueeze(0)
            dA = torch.exp(dtA)
            dB = mamba_discretization(dtA) * B.unsqueeze(1)
            
            Bu = dB * u_block.unsqueeze(-1)
            
            # Execute Tiled Parallel Scan (TPS)
            h_history = tiled_parallel_scan(dA, Bu, h_initial)
            
            y_raw = torch.einsum('ln,ldn->ld', C, h_history) + params['D_w'] * u_block
            y_history = F.softplus(y_raw) * torch.clamp(params['scale'], min=0.5)
            
            return h_history[-1], y_history

        # 3. State-Carrying Handoff Loop (TOSE)
        for i in range(num_blocks):
            start = i * block_size
            end = min(start + block_size, L)
            u_blk, du_blk = u_seq[start:end], du_seq[start:end]

            #  Dual Projection (Section 2.3)
            (h_end, y_blk), (h_tangent_end, dy_blk) = jvp(
                manifold_block_mapping,
                (h_primal, u_blk),
                (h_tangent, du_blk)
            )

            # 4. Result Handling
            if streaming and callback:
                callback(y_blk, dy_blk, start, end)
            elif not streaming:
                y_list.append(y_blk)
                dy_list.append(dy_blk)

            # 5. Graph Decoupling via Terminal Handoff (Section 2.3)
            h_primal = h_end.detach()
            h_tangent = h_tangent_end.detach()

        # 6. Final Sequence Assembly
        if streaming:
            return None
        else:
            return torch.cat(y_list, dim=0), torch.cat(dy_list, dim=0)

    def forward(self, u_seq):
        """Standard Mamba forward pass for precision verification."""
        L, D = u_seq.shape
        params = self._get_params_dict()
        h = self.estimate_initial_state(u_seq)
        y_history = []
        for t in range(L):
            y_t, h = self.mamba_atomic_step(u_seq[t], h, params)
            y_history.append(y_t)
        return torch.stack(y_history)

    def mamba_atomic_step(self, u_t, h_prev, params):
        dt = F.softplus(F.linear(u_t, params['dt_w'], params['dt_b']))
        B = F.linear(u_t, params['B_w'], params['B_b'])
        C = F.linear(u_t, params['C_w'], params['C_b'])
        A = -torch.exp(params['A_log'])
        dtA = dt.unsqueeze(-1) * A.unsqueeze(0)
        dA = torch.exp(dtA)
        dB = mamba_discretization(dtA) * B.unsqueeze(0)
        h_curr = dA * h_prev + dB * u_t.unsqueeze(-1)
        y_raw = torch.einsum('n,dn->d', C, h_curr) + params['D_w'] * u_t
        y_final = F.softplus(y_raw) * torch.clamp(params['scale'], min=0.5)
        return y_final, h_curr

        

       