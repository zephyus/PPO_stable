# tests/test_ncmultiagent_policy.py

import os
import sys
import types
from torch.distributions import Categorical

# 将项目根目录加入 import path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
import pytest

# 禁用 GAT 分支（如果没有安装 torch_geometric）
os.environ.setdefault('USE_GAT', '0')

from agents.policies import NCMultiAgentPolicy

@pytest.fixture
def dummy_policy():
    n_s = 8
    n_a = 4
    n_agent = 3
    n_step = 5
    neighbor_mask = torch.eye(n_agent, dtype=torch.bool).cpu().numpy()
    policy = NCMultiAgentPolicy(
        n_s=n_s,
        n_a=n_a,
        n_agent=n_agent,
        n_step=n_step,
        neighbor_mask=neighbor_mask,
        n_fc=16,
        n_h=16,
        n_s_ls=None,
        n_a_ls=None,
        model_config=None,
        identical=True
    )
    policy.to(torch.device('cpu'))

    # 动态绑定 _get_dist 方法
    def _get_dist(self, logits: torch.Tensor):
        return Categorical(logits=logits)
    policy._get_dist = types.MethodType(_get_dist, policy)

    return policy

def test_evaluate_actions_values_and_entropy_shape(dummy_policy):
    B, N, D = 4, dummy_policy.n_agent, dummy_policy.n_s
    o   = torch.randn(B, N, D, device=dummy_policy.device)
    fps = torch.zeros(B, N, 0, device=dummy_policy.device)
    a   = torch.zeros(B, N, dtype=torch.long, device=dummy_policy.device)

    with torch.no_grad():
        logp, values, entropy, new_states = dummy_policy.evaluate_actions_values_and_entropy(o, fps, a)

    assert logp.shape == (B, N)
    assert values.shape == (B, N)
    assert entropy.shape == (B, N)
    # new_states 是 (N, 2*H) 的张量
    assert isinstance(new_states, torch.Tensor)
    expected_shape = (dummy_policy.n_agent, 2 * dummy_policy.n_h)
    assert new_states.shape == expected_shape

# ... 后面两个测试保持不变 ...

def test_forward_outputs(dummy_policy):
    N, Do = dummy_policy.n_agent, dummy_policy.n_s
    ob   = torch.randn(N, Do).cpu().numpy()
    done = torch.zeros(N, dtype=torch.bool).cpu().numpy()
    fp   = torch.zeros((N, 0), dtype=torch.float32).cpu().numpy()

    logits_list, value_list, prob_list = dummy_policy.forward(ob, done, fp)

    assert isinstance(logits_list, list) and len(logits_list) == N
    assert value_list is None
    assert isinstance(prob_list,   list) and len(prob_list)   == N

    for lg in logits_list:
        assert lg.shape[-1] == dummy_policy.n_a
    for p in prob_list:
        assert torch.allclose(p.sum(), torch.tensor(1.0, device=p.device), atol=1e-5)

def test_forward_with_actions(dummy_policy):
    N = dummy_policy.n_agent
    ob   = torch.randn(N, dummy_policy.n_s).cpu().numpy()
    done = torch.zeros(N, dtype=torch.bool).cpu().numpy()
    fp   = torch.zeros((N, 0), dtype=torch.float32).cpu().numpy()
    actions = torch.zeros(N, dtype=torch.long)

    logits_list, value_list, prob_list = dummy_policy.forward(ob, done, fp, actions)

    assert isinstance(logits_list, list) and len(logits_list) == N
    assert isinstance(value_list,  list) and len(value_list)  == N
    assert isinstance(prob_list,   list) and len(prob_list)   == N

def test_critic_input_padding(dummy_policy):
    B = 2
    N = dummy_policy.n_agent
    h = torch.zeros(B, dummy_policy.n_h)
    acts = torch.zeros(B, N, dtype=torch.long)

    expected_dim = dummy_policy.shared_value_head.in_features
    for i in range(N):
        inp = dummy_policy._build_value_input(h, acts, i)
        assert inp.size(1) == expected_dim

def test_backward_computes_loss_and_grad(dummy_policy):
    agent = dummy_policy
    T, N, Do = 3, agent.n_agent, agent.n_s

    obs_t     = torch.randn(T, N, Do).float()
    fps_t     = torch.zeros((T, N, 0), dtype=torch.float32)
    acts_t    = torch.zeros((T, N), dtype=torch.int64)
    dones_t   = torch.zeros(T, dtype=torch.bool)
    returns_t = torch.randn((T, N)).float()
    advs_t    = torch.randn((T, N)).float()

    obs     = obs_t.cpu().numpy()
    fps     = fps_t.cpu().numpy()
    acts    = acts_t.cpu().numpy()
    dones   = dones_t.cpu().numpy()
    returns = returns_t.cpu().numpy()
    advs    = advs_t.cpu().numpy()

    # 清零梯度
    agent.zero_grad = lambda: None
    for p in agent.parameters():
        if p.grad is not None:
            p.grad.zero_()

    pl, vl, el, loss = agent.backward(
        obs=obs,
        fps=fps,
        acts=acts,
        dones=dones,
        Rs=returns,
        Advs=advs,
        e_coef=0.01,
        v_coef=0.5,
        summary_writer=None,
        global_step=None
    )
    assert torch.isfinite(loss), "loss should be finite"
    grads = [p.grad for p in agent.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads), "should have non-zero gradients"
