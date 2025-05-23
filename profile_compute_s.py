# profile_compute_s.py
import os
import torch
import numpy as np
from torch.profiler import profile, ProfilerActivity

# 如果你要在 GPU 上跑，就改成 'cuda'
DEVICE = 'cpu'

# —— 載入你的 policy —— 
from agents.policies import NCMultiAgentPolicy
from types import SimpleNamespace

def make_policy():
    n_agent = 10
    neighbor_mask = np.eye(n_agent, dtype=bool)
    policy = NCMultiAgentPolicy(
        n_s=8, n_a=4,
        n_agent=n_agent, n_step=32,
        neighbor_mask=neighbor_mask,
        n_fc=16, n_h=16,
        n_s_ls=None, n_a_ls=None,
        model_config=None,
        identical=True
    ).to(DEVICE)

    # stub 出 _get_dist 以及讓 gat_layer 不為 None
    policy._get_dist = lambda logits: torch.distributions.Categorical(logits=logits)
    policy.gat_layer = SimpleNamespace(dropout=0.0)
    return policy

def profile_compute_s(policy):
    T, N, Do = 32, policy.n_agent, policy.n_s
    # fake batch
    obs_T_N_Do  = torch.randn(T, N, Do, device=DEVICE)
    fps_T_N_Dfp = torch.zeros(T, N, 0, device=DEVICE)
    # 這裡直接用 zeros 當初始 hidden h0
    H = policy.n_h
    h0 = torch.zeros(N, H, device=DEVICE)

    # profile
    with profile(
        activities=[ProfilerActivity.CPU] if DEVICE=='cpu' else [ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof:
        # 重複多次取平均
        for _ in range(10):
            s_flat = policy._compute_s_features_flat(obs_T_N_Do, fps_T_N_Dfp, h0)
    key = "self_cpu_time_total" if DEVICE=='cpu' else "self_cuda_time_total"
    print(prof.key_averages().table(sort_by=key, row_limit=10))

if __name__ == "__main__":
    print("=== Profiling compute_s_features_flat ===")
    policy = make_policy()
    profile_compute_s(policy)
