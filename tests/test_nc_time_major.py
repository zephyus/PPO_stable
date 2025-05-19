import torch
import numpy as np
import sys
import os

# Add project root to path to import agents.policies
# This might need adjustment based on actual project structure and how tests are run
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.policies import NCMultiAgentPolicy

def test_nc_policy_time_major_run_comm_layers():
    T, N, Do, Dfp, H, n_actions = 8, 3, 32, 16, 64, 5
    
    # Mock neighbor_mask (3 agents, all connected to each other for simplicity)
    neighbor_mask = np.ones((N, N), dtype=bool)
    # np.fill_diagonal(neighbor_mask, False) # No self-loops in original meaning of neighbor

    # Mock model_config (can be None or a mock object)
    class MockModelConfig:
        def getint(self, key, default):
            if key == 'gat_num_heads': return 2 # Ensure H is divisible by this
            return default
        def getfloat(self, key, default):
            return default
        def getboolean(self, key, default):
            return default

    model_config = MockModelConfig()
    
    # Adjust H to be divisible by gat_num_heads if GAT processes features of size H
    # GAT input is 3*n_fc, output is 3*n_fc. LSTM input is 3*n_fc. LSTM hidden is H.
    # n_fc needs to be set. Let's assume n_fc = H for simplicity in this test,
    # so GAT input/output is 3*H. LSTM input is 3*H.
    n_fc = H // 2 # Example, ensure 3*n_fc is reasonable
    if (3 * n_fc) % model_config.getint('gat_num_heads', 2) != 0:
        # Adjust n_fc or heads for compatibility
        # For simplicity, let's assume GAT input dim (3*n_fc) is okay with heads.
        pass


    policy = NCMultiAgentPolicy(
        n_s=Do,             # obs_dim per agent
        n_a=n_actions,      # action_dim per agent
        n_agent=N,
        n_step=T,           # rollout length
        neighbor_mask=neighbor_mask,
        n_fc=n_fc,          # fc layer size
        n_h=H,              # LSTM hidden size
        model_config=model_config,
        identical=True
    )
    policy.to(torch.device("cpu")) # Ensure device consistency

    obs   = torch.randn(T, N, Do)
    fps   = torch.randn(T, N, Dfp) # Fingerprint dim, _get_comm_s uses it
    dones = torch.zeros(T, N).bool()
    h0    = torch.zeros(N, 2*H)
    
    print(f"Input shapes: obs({obs.shape}), fps({fps.shape}), dones({dones.shape}), h0({h0.shape})")

    try:
        out, hT = policy._run_comm_layers(obs, dones, fps, h0)
        
        print(f"Output shapes: out({out.shape}), hT({hT.shape})")
        
        assert out.shape == (N, T, H), f"Expected out shape {(N, T, H)}, got {out.shape}"
        assert hT.shape  == (N, 2*H), f"Expected hT shape {(N, 2*H)}, got {hT.shape}"
        
        print("test_nc_policy_time_major_run_comm_layers PASSED")

    except Exception as e:
        print(f"test_nc_policy_time_major_run_comm_layers FAILED: {e}")
        raise

if __name__ == '__main__':
    test_nc_policy_time_major_run_comm_layers()

    # Example for evaluate_actions_values_and_entropy
    # obs_batch_T_N_Do, fps_batch_T_N_Dfp, actions_batch_T_N, initial_lstm_states_N_2H
    actions = torch.randint(0, n_actions, (T,N))
    initial_lstm = torch.randn(N, 2*H)
    
    try:
        log_probs, values, entropy = policy.evaluate_actions_values_and_entropy(
            obs, fps, actions, initial_lstm
        )
        print(f"Evaluate output shapes: log_probs({log_probs.shape}), values({values.shape}), entropy({entropy.shape})")
        assert log_probs.shape == (T,N)
        assert values.shape == (T,N)
        assert entropy.shape == (T,N)
        print("evaluate_actions_values_and_entropy PASSED basic shape check")
    except Exception as e:
        print(f"evaluate_actions_values_and_entropy FAILED: {e}")
        raise

    # Example for forward
    # forward(self, ob_N_Do, done_N, fp_N_Dfp, action=None, out_type='p')
    obs_single_step = np.random.randn(N, Do).astype(np.float32)
    done_single_step = np.zeros(N).astype(bool) # (N,)
    fp_single_step = np.random.randn(N, Dfp).astype(np.float32)
    policy.reset() # Reset states_fw

    try:
        logits_list, value_list, probs_list = policy.forward(obs_single_step, done_single_step, fp_single_step)
        print(f"Forward output types: logits_list (len {len(logits_list)}), value_list (len {len(value_list)}), probs_list (len {len(probs_list)})")
        assert len(logits_list) == N
        assert len(value_list) == N
        assert len(probs_list) == N
        assert logits_list[0].shape == (n_actions,)
        assert value_list[0].shape == tuple() or value_list[0].shape == (1,) # scalar or (1,)
        assert probs_list[0].shape == (n_actions,)
        print("forward PASSED basic shape check")

    except Exception as e:
        print(f"forward FAILED: {e}")
        raise

