import numpy as np
import torch
from agents.policies import NCMultiAgentPolicy

def test_eval_action_shapes():
    B, N, Do, Dfp, H = 5, 3, 8, 6, 32
    obs  = torch.randn(B, N, Do)
    fps  = torch.randn(B, N, Dfp)
    acts = torch.zeros(B, N, dtype=torch.long)
    h0   = torch.zeros(N, 2*H)
    neigh = np.ones((N,N)) - np.eye(N)
    policy = NCMultiAgentPolicy(Do, 4, N, 1, neigh, n_fc=64, n_h=H)
    lp,v,e = policy.evaluate_actions_values_and_entropy(obs, fps, acts, h0)
    assert lp.shape == (B, N)
    assert v.shape  == (B, N)
    assert e.shape  == (B, N)
