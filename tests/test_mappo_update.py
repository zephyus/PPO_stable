# tests/test_mappo_update.py
import os
import sys
import types
import numpy as np
import torch
import pytest
from torch.distributions import Categorical
from types import SimpleNamespace

# 把專案根目錄加到 path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

from agents.models import MA2PPO_NC
from agents.policies import NCMultiAgentPolicy

# ... DummyBuffer 如前 ...
class DummyBuffer:
    def __init__(self, policy):
        T, N = 4, policy.n_agent
        obs_dim = policy.n_s
        fp_dim = 0
        self.data = dict(
            obs        = torch.randn(T, N, obs_dim).numpy(),
            actions    = torch.zeros((T, N), dtype=int).numpy(),
            old_logps  = torch.zeros((T, N)).numpy(),
            dones      = torch.zeros((T, N), dtype=bool).numpy(),
            returns    = torch.randn((T, N)).numpy(),
            advs       = torch.randn((T, N)).numpy(),
            old_values = torch.zeros((T, N)).numpy(),
            fps        = torch.zeros((T, N, fp_dim)).numpy(),
            lstm_states= None
        )

    def sample_transition(self, *args, **kwargs):
        return (
            self.data['obs'],
            self.data['actions'],
            self.data['old_logps'],
            self.data['dones'],
            self.data['returns'],
            self.data['advs'],
            self.data['old_values'],
            self.data['fps'],
            self.data['lstm_states']
        )
@pytest.fixture
def dummy_mappo():
    class Cfg:
        def getfloat(self, *a, **k):   return 0.95
        def getint(self,   *a, **k):   return 2
        def getboolean(self, *a, **k): return True
        def get(self, key, default=None):
            return default if default is not None else 'constant'

    n_agent = 3
    neighbor_mask = np.eye(n_agent, dtype=bool)
    distance_mask = np.zeros((n_agent, n_agent), dtype=float)

    agent = MA2PPO_NC(
        n_s_ls=[8]*n_agent,
        n_a_ls=[4]*n_agent,
        neighbor_mask=neighbor_mask,
        distance_mask=distance_mask,
        coop_gamma=0.9,
        total_step=10,
        model_config=Cfg(),
        seed=0,
        use_gpu=False
    )

    # Override policy
    policy = NCMultiAgentPolicy(
        n_s=8, n_a=4,
        n_agent=n_agent,
        n_step=5,
        neighbor_mask=neighbor_mask,
        n_fc=16, n_h=16,
        n_s_ls=None, n_a_ls=None,
        model_config=None,
        identical=True
    )
    policy.to(torch.device('cpu'))

    # 綁定 _get_dist
    def _get_dist(self, logits): return Categorical(logits=logits)
    policy._get_dist = types.MethodType(_get_dist, policy)

    # **關鍵：確保 gat_layer 不是 None，避免 update() 出錯**
    policy.gat_layer = SimpleNamespace(dropout=0.0)

    agent.policy = policy
    agent.trans_buffer = DummyBuffer(policy)
    agent.optimizer = torch.optim.Adam(agent.policy.parameters(), lr=1e-3)
    agent.v_coef = 0.5
    agent.e_coef = 0.01
    agent.clip_epsilon = 0.2
    agent.max_grad_norm = 0.5

    return agent

def test_mappo_update_smoke(dummy_mappo):
    pl, vl, el, tot = dummy_mappo.update(R_bootstrap_agents=None, dt=None)
    assert isinstance(pl, float)
    assert isinstance(vl, float)
    assert isinstance(el, float)
    assert isinstance(tot, float)

    ws_before = [p.clone() for p in dummy_mappo.policy.parameters()]
    dummy_mappo.update(R_bootstrap_agents=None, dt=None)
    changed = any(not torch.allclose(wb, p)
                  for wb, p in zip(ws_before, dummy_mappo.policy.parameters()))
    assert changed, "Policy parameters should have been updated"

def test_mappo_logs_grad_norm(dummy_mappo):
    from unittest.mock import MagicMock

    writer = MagicMock()
    dummy_mappo.update(R_bootstrap_agents=None, dt=None,
                       summary_writer=writer, global_step=1)

    logged_tags = [c.args[0] for c in writer.add_scalar.call_args_list]
    assert 'train/grad_norm_before_clip' in logged_tags
    assert 'train/grad_norm_after_clip' in logged_tags
