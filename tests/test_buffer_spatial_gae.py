# tests/test_buffer_spatial_gae.py

import numpy as np
import pytest # 可選，如果您不使用 pytest 的特定功能，基本的 np.testing.assert_allclose 即可
import sys
import os

# 將項目根目錄添加到 sys.path 以便導入 agents.utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.utils import MultiAgentOnPolicyBuffer

# --- Helper Functions (可選，但有助於保持測試清晰) ---
def _populate_buffer_for_gae_test(buffer, T, N, rewards_data, values_data, dones_data_for_gae):
    """
    Populates the buffer with rewards, values, and dones for GAE testing.
    Other fields (obs, fp, action, etc.) are filled with dummies as they don't affect GAE.
    
    Args:
        dones_data_for_gae (np.ndarray): Shape (T, N). dones_data_for_gae[t, n] is True if s_{t+1} for agent n is terminal.
    """
    dummy_obs_dim = 1
    dummy_fp_dim = 1
    dummy_action_dim = 1 
    dummy_lstm_hidden_dim = 1

    for t in range(T):
        # done_for_add_transition: an older convention was that buffer.dones stored a single bool per step
        # However, our buffer's GAE now expects dones_arr[t] to be (N,)
        # The add_transition's 'done' param might be interpreted as global done or per-agent.
        # Let's assume 'done' in add_transition is a scalar (global done for the step)
        # The sample_transition method correctly handles reshaping this to (T,N) if needed.
        # For clarity, let's pass the per-agent done status for this step.
        current_step_done_per_agent = dones_data_for_gae[t]

        buffer.add_transition(
            ob=np.random.rand(N, dummy_obs_dim).astype(np.float32),
            fp=np.random.rand(N, dummy_fp_dim).astype(np.float32),
            action=np.random.randint(0, dummy_action_dim, size=N),
            reward=rewards_data[t],
            value=values_data[t],
            # MultiAgentOnPolicyBuffer.add_transition takes 'done' for the current step.
            # If it's a scalar, it means the whole environment is done.
            # If it's an array (N,), it's per-agent.
            # The sample_transition correctly converts this.
            # For this test, we are providing rewards/values/dones_data as (T,N)
            # so we can pass dones_data_for_gae[t] directly to simulate per-agent dones.
            done=current_step_done_per_agent, 
            log_prob=np.random.rand(N).astype(np.float32),
            lstm_state=np.random.rand(N, dummy_lstm_hidden_dim * 2).astype(np.float32)
        )

def _manual_gae_calculation(rewards_arr, values_arr, dones_arr, 
                           gamma, gae_lambda, R_bootstrap_agents):
    """
    Manually calculates GAE.
    Args:
        rewards_arr (np.ndarray): Shape (T, N), effective rewards.
        values_arr (np.ndarray): Shape (T, N), V(s_t).
        dones_arr (np.ndarray): Shape (T, N), dones_arr[t,n] is True if s_{t+1} for agent n is terminal.
        R_bootstrap_agents (np.ndarray): Shape (N,), V(s_T) or V(s_{T+1}) for last step.
    """
    T, N = rewards_arr.shape
    advantages = np.zeros_like(rewards_arr, dtype=np.float32)
    last_gae_lam = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        current_rewards_t = rewards_arr[t]
        current_values_t = values_arr[t]
        
        actual_next_values = R_bootstrap_agents if t == T - 1 else values_arr[t + 1]
        done_after_current_step = dones_arr[t] # This is d_{t+1}
        
        next_non_terminal = 1.0 - done_after_current_step.astype(np.float32)
        
        delta = current_rewards_t + gamma * actual_next_values * next_non_terminal - current_values_t
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
    
    return advantages

# --- Test Cases ---

def test_gae_with_coop_gamma_zero():
    """
    Tests GAE when coop_gamma is 0. r_eff should equal rewards_arr.
    """
    T, N = 2, 2
    gamma, gae_lambda = 0.99, 0.95
    coop_gamma = 0.0
    distance_mask = np.array([[0, 1], [1, 0]], dtype=np.int32)
    distance_decay = gamma
    neighbor_dist_thresh = 1

    buffer = MultiAgentOnPolicyBuffer(
        gamma, coop_gamma, distance_mask, gae_lambda, distance_decay, neighbor_dist_thresh
    )

    rewards_data = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32) # (T, N)
    values_data = np.array([[1.0, 1.1], [1.2, 1.3]], dtype=np.float32)  # (T, N)
    # dones_data[t,n] is True if state s_{t+1} for agent n is terminal
    dones_data = np.array([[False, False], [False, True]], dtype=bool) # (T, N) 
    R_bootstrap_agents = np.array([1.4, 0.0], dtype=np.float32) # V(s_T), (N,). Agent 1 done, so V_bootstrap=0

    _populate_buffer_for_gae_test(buffer, T, N, rewards_data, values_data, dones_data)
    
    _, _, _, _, actual_returns, actual_advantages, _, _, _ = buffer.sample_transition(R_bootstrap_agents, dt=None)

    # Expected: Standard GAE using rewards_data directly
    expected_advantages = _manual_gae_calculation(
        rewards_data, values_data, dones_data, gamma, gae_lambda, R_bootstrap_agents
    )
    expected_returns = expected_advantages + values_data
    
    np.testing.assert_allclose(actual_advantages, expected_advantages, rtol=1e-6,
                               err_msg="Advantages mismatch (coop_gamma=0)")
    np.testing.assert_allclose(actual_returns, expected_returns, rtol=1e-6,
                               err_msg="Returns mismatch (coop_gamma=0)")
    print("\ntest_gae_with_coop_gamma_zero PASSED")


def test_gae_with_coop_gamma_one_direct_neighbors():
    """
    Tests GAE with coop_gamma=1, direct neighbors only, no distance decay.
    """
    T, N = 2, 2
    gamma, gae_lambda = 0.99, 0.95
    coop_gamma = 1.0
    # distance_mask[i,j] is distance from j to i (or i to j, assuming symmetric)
    # For W_normalized[i,j] to be influence of j on i:
    distance_mask = np.array([[0, 1], [1, 0]], dtype=np.int32) 
    distance_decay = 1.0 # No decay for distance 1
    neighbor_dist_thresh = 1 # Only direct neighbors (dist=1) and self (dist=0)

    buffer = MultiAgentOnPolicyBuffer(
        gamma, coop_gamma, distance_mask, gae_lambda, distance_decay, neighbor_dist_thresh
    )

    rewards_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    values_data = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    dones_data = np.array([[False, False], [False, False]], dtype=bool)
    R_bootstrap_agents = np.array([0.5, 0.6], dtype=np.float32)

    _populate_buffer_for_gae_test(buffer, T, N, rewards_data, values_data, dones_data)
    _, _, _, _, actual_returns, actual_advantages, _, _, _ = buffer.sample_transition(R_bootstrap_agents, dt=None)

    # Manual W_normalized calculation:
    # dist_matrix = distance_mask
    # weights_unnormalized = (distance_decay ** dist_matrix) * (dist_matrix <= neighbor_dist_thresh)
    # [[1^0 * (0<=1), 1^1 * (1<=1)],  -> [[1, 1],
    #  [1^1 * (1<=1), 1^0 * (0<=1)]]     [1, 1]]
    # weights_sum_per_agent = [[2], [2]]
    # W_normalized_manual = [[0.5, 0.5], [0.5, 0.5]]
    W_normalized_manual = np.array([[0.5, 0.5], [0.5, 0.5]])

    # Manual r_eff_arr calculation:
    # r_eff = (1-coop_gamma)*rewards + coop_gamma * (rewards @ W_normalized.T)
    # With coop_gamma=1, r_eff = rewards @ W_normalized.T
    # W_normalized.T is also [[0.5, 0.5], [0.5, 0.5]]
    # r_eff[t,0] = rewards_data[t,0]*0.5 + rewards_data[t,1]*0.5
    # r_eff[t,1] = rewards_data[t,0]*0.5 + rewards_data[t,1]*0.5
    # Example t=0:
    # r_eff[0,0] = 1.0*0.5 + 2.0*0.5 = 0.5 + 1.0 = 1.5
    # r_eff[0,1] = 1.0*0.5 + 2.0*0.5 = 0.5 + 1.0 = 1.5
    # Example t=1:
    # r_eff[1,0] = 3.0*0.5 + 4.0*0.5 = 1.5 + 2.0 = 3.5
    # r_eff[1,1] = 3.0*0.5 + 4.0*0.5 = 1.5 + 2.0 = 3.5
    expected_r_eff_arr = np.array([[1.5, 1.5], [3.5, 3.5]], dtype=np.float32)
    
    expected_advantages = _manual_gae_calculation(
        expected_r_eff_arr, values_data, dones_data, gamma, gae_lambda, R_bootstrap_agents
    )
    expected_returns = expected_advantages + values_data

    np.testing.assert_allclose(actual_advantages, expected_advantages, rtol=1e-6,
                               err_msg="Advantages mismatch (coop_gamma=1, direct_neighbors)")
    np.testing.assert_allclose(actual_returns, expected_returns, rtol=1e-6,
                               err_msg="Returns mismatch (coop_gamma=1, direct_neighbors)")
    print("\ntest_gae_with_coop_gamma_one_direct_neighbors PASSED")

def test_gae_with_coop_gamma_half_and_decay_N3():
    """
    Tests GAE with N=3, coop_gamma=0.5, distance_decay, and specific thresh.
    """
    T, N = 2, 3
    gamma, gae_lambda = 0.99, 0.95
    coop_gamma = 0.5
    distance_mask = np.array([[0, 1, 2], 
                              [1, 0, 1], 
                              [2, 1, 0]], dtype=np.int32)
    distance_decay = 0.9 # beta
    neighbor_dist_thresh = 1 # Only self (dist 0) and direct neighbors (dist 1)

    buffer = MultiAgentOnPolicyBuffer(
        gamma, coop_gamma, distance_mask, gae_lambda, distance_decay, neighbor_dist_thresh
    )

    rewards_data = np.array([[1,2,3], [4,5,6]], dtype=np.float32)      # (T,N)
    values_data = np.array([[.1,.2,.3], [.4,.5,.6]], dtype=np.float32) # (T,N)
    dones_data = np.array([[False, False, False], [False, False, False]], dtype=bool) # (T,N) <--- MODIFIED HERE
    R_bootstrap_agents = np.array([.7,.8,.9], dtype=np.float32)       # (N,)

    _populate_buffer_for_gae_test(buffer, T, N, rewards_data, values_data, dones_data)
    _, _, _, _, actual_returns, actual_advantages, _, _, _ = buffer.sample_transition(R_bootstrap_agents, dt=None)

    # Manual W_normalized calculation:
    W_unnorm_manual = np.array([
        [1.0, 0.9, 0.0],
        [0.9, 1.0, 0.9],
        [0.0, 0.9, 1.0]
    ])
    W_sum_manual = W_unnorm_manual.sum(axis=1, keepdims=True) 
    W_normalized_manual = W_unnorm_manual / W_sum_manual
    
    neighbor_weighted_manual = np.matmul(rewards_data, W_normalized_manual.T)
    expected_r_eff_arr = (1 - coop_gamma) * rewards_data + coop_gamma * neighbor_weighted_manual
    
    expected_advantages = _manual_gae_calculation(
        expected_r_eff_arr, values_data, dones_data, gamma, gae_lambda, R_bootstrap_agents
    )
    expected_returns = expected_advantages + values_data

    np.testing.assert_allclose(actual_advantages, expected_advantages, rtol=1e-5,
                               err_msg="Advantages mismatch (N=3, coop_gamma=0.5, decay)")
    np.testing.assert_allclose(actual_returns, expected_returns, rtol=1e-5,
                               err_msg="Returns mismatch (N=3, coop_gamma=0.5, decay)")
    print("\ntest_gae_with_coop_gamma_half_and_decay_N3 PASSED")
if __name__ == '__main__':
    print("Running tests for MultiAgentOnPolicyBuffer with Spatial GAE...\n")
    # You can call tests individually for focused debugging:
    test_gae_with_coop_gamma_zero()
    test_gae_with_coop_gamma_one_direct_neighbors()
    test_gae_with_coop_gamma_half_and_decay_N3()
    print("\nAll standalone tests finished. For full pytest integration, run 'pytest'")

    # Or use pytest to run all tests in the file/directory
    # Example: Create a dummy conftest.py in tests/ if needed, then run `pytest` from project root
    # For now, just calling them sequentially.


    # tests/test_buffer_spatial_gae.py (繼續添加以下測試函數)
# ... (請確保之前的 import 和輔助函數已存在) ...

def test_gae_no_neighbors_within_thresh():
    """
    Tests GAE when an agent has no neighbors (including self) within the threshold.
    Its r_eff should only consider its own reward scaled by (1-coop_gamma).
    W_normalized row for that agent should be all zeros.
    """
    T, N = 2, 2
    gamma, gae_lambda = 0.99, 0.95
    coop_gamma = 0.5 # Test with some cooperation
    
    # Agent 0: dist to self=0, dist to agent 1=1
    # Agent 1: dist to self=99, dist to agent 0=99 (effectively isolated by threshold)
    distance_mask = np.array([[0, 1], 
                              [99, 99]], dtype=np.int32) 
    distance_decay = 0.9
    neighbor_dist_thresh = 1 # Agent 0 considers self and agent 1 (if dist=1). Agent 1 considers no one.

    buffer = MultiAgentOnPolicyBuffer(
        gamma, coop_gamma, distance_mask, gae_lambda, distance_decay, neighbor_dist_thresh
    )

    rewards_data = np.array([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32)
    values_data = np.array([[0.1, 1.0], [0.2, 2.0]], dtype=np.float32)
    dones_data = np.array([[False, False], [False, False]], dtype=bool)
    R_bootstrap_agents = np.array([0.3, 3.0], dtype=np.float32)

    _populate_buffer_for_gae_test(buffer, T, N, rewards_data, values_data, dones_data)
    # Store a reference to the internal rewards_arr before it's potentially overwritten by r_eff
    # However, the buffer recalculates rewards_arr from self.rewards each time.
    # We need to manually calculate W_normalized and r_eff to verify.

    _, _, _, _, actual_returns, actual_advantages, _, _, _ = buffer.sample_transition(R_bootstrap_agents, dt=None)
    
    # --- Manual Calculation for this specific case ---
    # For Agent 0 (index 0):
    #   dist_to_self = 0 <= thresh=1. weight_unnorm_00 = 0.9^0 = 1.0
    #   dist_to_agent1 = 1 <= thresh=1. weight_unnorm_01 = 0.9^1 = 0.9
    #   row0_sum = 1.0 + 0.9 = 1.9
    #   W_norm_manual[0,:] = [1.0/1.9, 0.9/1.9] approx [0.5263, 0.4737]
    # For Agent 1 (index 1):
    #   dist_to_self = 99 > thresh=1. weight_unnorm_10 = 0
    #   dist_to_agent0 = 99 > thresh=1. weight_unnorm_11 = 0
    #   row1_sum = 0
    #   W_norm_manual[1,:] = [0.0, 0.0] (due to np.divide where sum is 0)
    W_normalized_manual = np.array([
        [1.0/1.9, 0.9/1.9],
        [0.0, 0.0]
    ])

    neighbor_weighted_manual = np.matmul(rewards_data, W_normalized_manual.T)
    expected_r_eff_arr = (1 - coop_gamma) * rewards_data + coop_gamma * neighbor_weighted_manual
    
    # Verify r_eff for agent 1 (isolated one)
    # expected_r_eff_arr[t, 1] should be (1 - 0.5) * rewards_data[t, 1] + 0.5 * 0 = 0.5 * rewards_data[t,1]
    assert np.allclose(expected_r_eff_arr[:, 1], (1 - coop_gamma) * rewards_data[:, 1]), "r_eff for isolated agent is wrong"

    expected_advantages = _manual_gae_calculation(
        expected_r_eff_arr, values_data, dones_data, gamma, gae_lambda, R_bootstrap_agents
    )
    expected_returns = expected_advantages + values_data
    
    np.testing.assert_allclose(actual_advantages, expected_advantages, rtol=1e-5,
                               err_msg="Advantages mismatch (no_neighbors_within_thresh)")
    np.testing.assert_allclose(actual_returns, expected_returns, rtol=1e-5,
                               err_msg="Returns mismatch (no_neighbors_within_thresh)")
    print("\ntest_gae_no_neighbors_within_thresh PASSED")


def test_gae_distance_decay_zero():
    """
    Tests GAE when distance_decay is 0.
    Only self-reward (dist=0) should contribute to neighbor_weighted_rewards if thresh >= 0.
    """
    T, N = 2, 2
    gamma, gae_lambda = 0.99, 0.95
    coop_gamma = 0.5
    distance_mask = np.array([[0, 1], [1, 0]], dtype=np.int32)
    distance_decay = 0.0 # <--- Key test parameter
    neighbor_dist_thresh = 1 # Allows self (dist 0) and neighbor (dist 1)

    buffer = MultiAgentOnPolicyBuffer(
        gamma, coop_gamma, distance_mask, gae_lambda, distance_decay, neighbor_dist_thresh
    )

    rewards_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    values_data = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    dones_data = np.array([[False, False], [False, False]], dtype=bool)
    R_bootstrap_agents = np.array([0.5, 0.6], dtype=np.float32)

    _populate_buffer_for_gae_test(buffer, T, N, rewards_data, values_data, dones_data)
    _, _, _, _, actual_returns, actual_advantages, _, _, _ = buffer.sample_transition(R_bootstrap_agents, dt=None)

    # Manual W_normalized calculation:
    # distance_decay = 0.0. NumPy's 0**0 = 1. 0**positive = 0.
    # For agent 0: dists [0,1]. thresh=1.
    #   w_unnorm_00 = (0.0**0)*(0<=1) = 1.0 * True = 1.0
    #   w_unnorm_01 = (0.0**1)*(1<=1) = 0.0 * True = 0.0
    #   row0_sum = 1.0. W_norm_manual[0,:] = [1.0, 0.0]
    # For agent 1: dists [1,0]. thresh=1.
    #   w_unnorm_10 = (0.0**1)*(1<=1) = 0.0 * True = 0.0
    #   w_unnorm_11 = (0.0**0)*(0<=1) = 1.0 * True = 1.0
    #   row1_sum = 1.0. W_norm_manual[1,:] = [0.0, 1.0]
    W_normalized_manual = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ]) # Identity matrix

    neighbor_weighted_manual = np.matmul(rewards_data, W_normalized_manual.T) # effectively rewards_data @ I = rewards_data
    # r_eff = (1-coop_gamma)*rewards + coop_gamma*rewards = rewards
    expected_r_eff_arr = (1 - coop_gamma) * rewards_data + coop_gamma * neighbor_weighted_manual
    # This should simplify to rewards_data because neighbor_weighted_manual equals rewards_data here.
    assert np.allclose(expected_r_eff_arr, rewards_data), "r_eff should be original rewards if W_norm is Identity"


    expected_advantages = _manual_gae_calculation(
        expected_r_eff_arr, values_data, dones_data, gamma, gae_lambda, R_bootstrap_agents
    )
    expected_returns = expected_advantages + values_data
    
    np.testing.assert_allclose(actual_advantages, expected_advantages, rtol=1e-6,
                               err_msg="Advantages mismatch (distance_decay=0)")
    np.testing.assert_allclose(actual_returns, expected_returns, rtol=1e-6,
                               err_msg="Returns mismatch (distance_decay=0)")
    print("\ntest_gae_distance_decay_zero PASSED")


def test_gae_larger_T_N_varied_dones():
    """
    Tests GAE with T=4, N=3 and a more varied dones pattern.
    """
    T, N = 4, 3
    gamma, gae_lambda = 0.99, 0.95
    coop_gamma = 0.25
    distance_mask = np.array([[0,1,2],[1,0,1],[2,1,0]], dtype=np.int32)
    distance_decay = 0.9
    neighbor_dist_thresh = 2 # All agents influence each other to some degree

    buffer = MultiAgentOnPolicyBuffer(
        gamma, coop_gamma, distance_mask, gae_lambda, distance_decay, neighbor_dist_thresh
    )

    rewards_data = np.arange(T * N, dtype=np.float32).reshape(T, N) * 0.1
    values_data = np.arange(T * N, dtype=np.float32).reshape(T, N) * 0.05 + 0.5
    # dones_data[t,n] is True if s_{t+1} for agent n is terminal.
    dones_data = np.array([
        [False, True, False], # Agent 1 done after step 0
        [False, True, False], # Agent 1 remains done
        [True,  True, True],  # Agent 0 and 2 done after step 2
        [True,  True, True]   # All remain done (values for V(s_3) will be used by GAE)
    ], dtype=bool)
    R_bootstrap_agents = np.array([0.0, 0.0, 0.0], dtype=np.float32) # V(s_T), all end in terminal states based on dones_data[T-1]

    _populate_buffer_for_gae_test(buffer, T, N, rewards_data, values_data, dones_data)
    _, _, _, _, actual_returns, actual_advantages, _, _, _ = buffer.sample_transition(R_bootstrap_agents, dt=None)

    # Manual W_normalized:
    # dist_matrix = distance_mask
    # beta = 0.9, thresh = 2
    # W_unnorm for row 0 (agent 0): [0.9^0, 0.9^1, 0.9^2] = [1, 0.9, 0.81], sum = 2.71
    # W_unnorm for row 1 (agent 1): [0.9^1, 0.9^0, 0.9^1] = [0.9, 1, 0.9], sum = 2.8
    # W_unnorm for row 2 (agent 2): [0.9^2, 0.9^1, 0.9^0] = [0.81, 0.9, 1], sum = 2.71
    W_unnorm_manual = np.array([
        [1.0, 0.9, 0.81],
        [0.9, 1.0, 0.9],
        [0.81, 0.9, 1.0]
    ])
    W_sum_manual = W_unnorm_manual.sum(axis=1, keepdims=True)
    W_normalized_manual = W_unnorm_manual / W_sum_manual

    neighbor_weighted_manual = np.matmul(rewards_data, W_normalized_manual.T)
    expected_r_eff_arr = (1 - coop_gamma) * rewards_data + coop_gamma * neighbor_weighted_manual

    expected_advantages = _manual_gae_calculation(
        expected_r_eff_arr, values_data, dones_data, gamma, gae_lambda, R_bootstrap_agents
    )
    expected_returns = expected_advantages + values_data
    
    np.testing.assert_allclose(actual_advantages, expected_advantages, rtol=1e-5,
                               err_msg="Advantages mismatch (larger_T_N_varied_dones)")
    np.testing.assert_allclose(actual_returns, expected_returns, rtol=1e-5,
                               err_msg="Returns mismatch (larger_T_N_varied_dones)")
    print("\ntest_gae_larger_T_N_varied_dones PASSED")


if __name__ == '__main__':
    # Assuming previous tests are in the same file and also defined
    print("Running tests for MultiAgentOnPolicyBuffer with Spatial GAE (extended)...\n")
    # from test_buffer_spatial_gae import test_gae_with_coop_gamma_zero # If they are in the same file, no need to import
    # from test_buffer_spatial_gae import test_gae_with_coop_gamma_one_direct_neighbors
    # from test_buffer_spatial_gae import test_gae_with_coop_gamma_half_and_decay_N3
    
    # You would typically run all tests using pytest.
    # For standalone execution of this script, explicitly call the tests.
    
    # If the previous tests are in this file, they might be auto-discovered if named test_*
    # For explicit call:
    # test_gae_with_coop_gamma_zero() # Assumed defined in the same file
    # test_gae_with_coop_gamma_one_direct_neighbors() # Assumed defined
    # test_gae_with_coop_gamma_half_and_decay_N3() # Assumed defined
    
    test_gae_no_neighbors_within_thresh()
    test_gae_distance_decay_zero()
    test_gae_larger_T_N_varied_dones()
    
    print("\nAll specified extended tests finished. For full pytest integration, run 'pytest'")