import numpy as np
import torch
import torch.nn as nn
import logging
import sys  # Import sys for StreamHandler

"""
initializers
"""
def init_layer(layer, method='fc'):
    if method == 'fc':
        torch.nn.init.orthogonal_(layer.weight.data, gain=np.sqrt(2))
        torch.nn.init.constant_(layer.bias.data, 0.0)
    elif method == 'lstm':
        for name, param in layer.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)
            elif 'weight' in name:
                torch.nn.init.orthogonal_(param, gain=np.sqrt(2))
    return layer

"""
layer helpers
"""
def batch_to_seq(data_batch):
    if isinstance(data_batch, torch.Tensor):
        return data_batch.transpose(0, 1)
    elif isinstance(data_batch, np.ndarray):
        return data_batch.swapaxes(0, 1)
    else:
        if isinstance(data_batch[0], torch.Tensor):
            return torch.stack(data_batch, dim=1)
        else:
            return np.stack(data_batch, axis=1)


def run_rnn(rnn_layer, x_seq, done_seq, states):
    if x_seq.dim() == 2 and states.dim() == 1:
        h, c = torch.chunk(states, 2)
        h_new, c_new = rnn_layer(x_seq, (h.unsqueeze(0), c.unsqueeze(0)))
        return h_new.squeeze(0), torch.cat((h_new.squeeze(0), c_new.squeeze(0)))

    if isinstance(rnn_layer, torch.nn.LSTMCell) and x_seq.dim() == 2 and states.dim() == 2:
        h_state, c_state = torch.chunk(states, 2, dim=1)
        new_h, new_c = rnn_layer(x_seq, (h_state, c_state))
        return new_h, torch.cat([new_h, new_c], dim=1)

    if hasattr(rnn_layer, 'flatten_parameters'):
        rnn_layer.flatten_parameters()
    h_states, new_final_states = rnn_layer(x_seq, states)
    return h_states, new_final_states


def one_hot(x, n_class):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(np.asarray(x)).long()
    else:
        x = x.long()
    x_onehot = torch.zeros(x.size(0), n_class, device=x.device)
    x_onehot.scatter_(1, x.unsqueeze(-1), 1)
    return x_onehot


"""
buffers
"""
class OnPolicyBuffer:
    def __init__(self, gamma, coop_gamma, distance_mask, n_step):  # Modified signature
        self.gamma = gamma
        self.coop_gamma = coop_gamma  # May not be used if reward is already appropriately shaped
        self.distance_mask = distance_mask  # May not be used for simple A2C advantage calculation
        self.n_step = n_step  # Rollout length for this buffer

        self.obs = []
        self.nactions = []  # Neighbor actions
        self.actions = []  # Agent's own actions
        self.rewards = []
        self.values = []
        self.dones = []

        self.ptr = 0

    def add_transition(self, ob, naction, action, reward, value, done):  # Modified signature
        self.obs.append(ob)
        # Patch H: Store naction as np.int32
        self.nactions.append(np.asarray(naction, dtype=np.int32))
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.ptr = (self.ptr + 1)  # Simple pointer increment, assumes buffer cleared after sampling

    def sample_transition(self, R_end, dt=None):  # dt is not used for this simple advantage calculation
        """
        Computes returns and advantages for the collected rollout.
        Outputs data in time-major format (T, ...).
        R_end: Bootstrap value V(s_T+1) for the last state if episode didn't end.
        """
        if not self.obs:  # Empty buffer
            return (
                np.empty((0,), dtype=np.object_),  # obs
                np.empty((0,), dtype=np.object_),  # nactions
                np.empty((0,), dtype=np.int32),    # actions
                np.empty((0,), dtype=bool),        # dones
                np.empty((0,), dtype=np.float32),  # Rs
                np.empty((0,), dtype=np.float32)   # Advs
            )

        # Patch I: Handle heterogeneous obs sizes
        try:
            obs_arr = np.array(self.obs, dtype=np.float32)  # (T, obs_dim)
        except ValueError:
            # 異質 size，退回 object array，再由上層自行 pad
            obs_arr = np.array(self.obs, dtype=object)
        try:
            nactions_arr = np.array(self.nactions, dtype=np.float32)  # (T, na_dim)
        except ValueError:
            nactions_arr = np.array(self.nactions, dtype=object)  # Fallback to object array

        actions_arr = np.array(self.actions, dtype=np.int32)  # (T,)
        rewards_arr = np.array(self.rewards, dtype=np.float32)  # (T,)
        values_arr = np.array(self.values, dtype=np.float32)  # (T,)
        dones_arr = np.array(self.dones, dtype=bool)  # (T,)

        T = len(self.rewards)
        Rs = np.zeros_like(rewards_arr)

        current_R = R_end
        for t in reversed(range(T)):
            current_R = rewards_arr[t] + self.gamma * current_R * (1.0 - dones_arr[t])
            Rs[t] = current_R

        Advs = Rs - values_arr

        self.obs, self.nactions, self.actions, self.rewards, self.values, self.dones = [], [], [], [], [], []
        self.ptr = 0

        return obs_arr, nactions_arr, actions_arr, dones_arr, Rs, Advs


class MultiAgentOnPolicyBuffer:
    def __init__(self, gamma, coop_gamma, distance_mask, gae_lambda=0.95,
                 distance_decay=None, neighbor_dist_thresh=1):  # New signature
        self.gamma = gamma
        self.coop_gamma = coop_gamma  # coop_gamma is the new name for alpha
        self.distance_mask = np.asarray(distance_mask)  # Ensure NumPy array, shape (N, N)
        self.gae_lambda = gae_lambda

        # New parameters for spatial reward GAE
        self.distance_decay = distance_decay if distance_decay is not None else self.gamma
        self.neighbor_dist_thresh = neighbor_dist_thresh

        self.obs = []
        self.fps = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.log_probs = []
        self.lstm_states = []

        self.ptr = 0
        self.path_start_idx = 0

    def add_transition(self, ob, fp, action, reward, value, done, log_prob, lstm_state):
        self.obs.append(np.array(ob) if not isinstance(ob, np.ndarray) else ob)
        self.fps.append(np.array(fp) if not isinstance(fp, np.ndarray) else fp)
        self.actions.append(np.array(action) if not isinstance(action, np.ndarray) else action)
        self.rewards.append(np.array(reward) if not isinstance(reward, np.ndarray) else reward)
        self.values.append(np.array(value) if not isinstance(value, np.ndarray) else value)
        self.dones.append(done)
        self.log_probs.append(np.array(log_prob) if not isinstance(log_prob, np.ndarray) else log_prob)
        self.lstm_states.append(np.array(lstm_state) if not isinstance(lstm_state, np.ndarray) else lstm_state)
        self.ptr = (self.ptr + 1)

        if not hasattr(self, 'obs_dim') or self.obs_dim is None:
            if self.obs and self.obs[-1] is not None:
                self.obs_dim = self.obs[-1].shape[-1] if self.obs[-1].ndim > 1 else 1
            if self.fps and self.fps[-1] is not None:
                self.fp_dim = self.fps[-1].shape[-1] if self.fps[-1].ndim > 1 else 1
            if self.lstm_states and self.lstm_states[-1] is not None:
                self.lstm_hidden_dim = self.lstm_states[-1].shape[-1] // 2 if self.lstm_states[-1].ndim > 0 else 1

    def sample_transition(self, R_bootstrap_agents, dt):
        if not self.obs:  # Empty buffer
            dummy_N = R_bootstrap_agents.shape[0] if R_bootstrap_agents is not None and R_bootstrap_agents.ndim > 0 else 1
            dummy_obs_dim = self.obs_dim if hasattr(self, 'obs_dim') and self.obs_dim is not None else 1
            dummy_fp_dim = self.fp_dim if hasattr(self, 'fp_dim') and self.fp_dim is not None else 1
            dummy_lstm_dim = self.lstm_hidden_dim * 2 if hasattr(self, 'lstm_hidden_dim') and self.lstm_hidden_dim is not None else 1

            return (
                np.empty((0, dummy_N, dummy_obs_dim), dtype=np.float32),
                np.empty((0, dummy_N), dtype=np.int32),
                np.empty((0, dummy_N), dtype=np.float32),
                np.empty((0, dummy_N), dtype=bool),
                np.empty((0, dummy_N), dtype=np.float32),
                np.empty((0, dummy_N), dtype=np.float32),
                np.empty((0, dummy_N), dtype=np.float32),
                np.empty((0, dummy_N, dummy_fp_dim), dtype=np.float32),
                np.empty((0, dummy_N, dummy_lstm_dim), dtype=np.float32)
            )

        obs_arr = np.stack(self.obs, axis=0).astype(np.float32)
        fps_arr = np.stack(self.fps, axis=0).astype(np.float32)
        actions_arr = np.stack(self.actions, axis=0).astype(np.int32)
        rewards_arr = np.stack(self.rewards, axis=0).astype(np.float32)
        values_arr = np.stack(self.values, axis=0).astype(np.float32)
        dones_arr_scalar_per_step = np.array(self.dones, dtype=bool)

        T, N = rewards_arr.shape[0], rewards_arr.shape[1]

        if dones_arr_scalar_per_step.ndim == 1:  # Global done signal per step
            dones_arr = np.repeat(dones_arr_scalar_per_step[:, np.newaxis], N, axis=1).astype(bool)
        elif dones_arr_scalar_per_step.ndim == 2 and dones_arr_scalar_per_step.shape[1] == N:  # Per-agent done
            dones_arr = dones_arr_scalar_per_step.astype(bool)
        else:
            raise ValueError(f"Dones array shape {dones_arr_scalar_per_step.shape} is incompatible with T={T}, N={N}")

        log_probs_arr = np.stack(self.log_probs, axis=0).astype(np.float32)
        lstm_states_arr = np.stack(self.lstm_states, axis=0).astype(np.float32)

        rewards_for_gae = rewards_arr  # Default to original rewards

        if self.coop_gamma > 0:
            # STEP-1: Construct Weight Matrix W_normalized (N, N)
            dist_matrix = self.distance_mask  # Shape (N, N)

            # weights_unnormalized[i,j] is beta^dist(i,j) if dist(i,j) <= thresh, else 0
            # This represents the influence of agent j on agent i.
            weights_unnormalized = (self.distance_decay ** dist_matrix) * \
                                   (dist_matrix <= self.neighbor_dist_thresh)

            # Normalize row-wise: sum_j W_normalized[i,j] = 1 for each i
            # weights_sum_per_agent[i] = sum_j weights_unnormalized[i,j]
            weights_sum_per_agent = weights_unnormalized.sum(axis=1, keepdims=True)  # Shape (N, 1)

            W_normalized = np.divide(weights_unnormalized, weights_sum_per_agent,
                                     out=np.zeros_like(weights_unnormalized),
                                     where=weights_sum_per_agent != 0)
            # W_normalized[i,j] is the normalized weight of agent j's reward on agent i's effective reward.

            # STEP-2: Calculate Effective Reward r_eff_arr (T, N)
            neighbor_weighted_rewards = np.matmul(rewards_arr, W_normalized.T)  # (T,N) @ (N,N) -> (T,N)

            r_eff_arr = (1 - self.coop_gamma) * rewards_arr + self.coop_gamma * neighbor_weighted_rewards
            rewards_for_gae = r_eff_arr

        # STEP-3: GAE Calculation (vectorized over agents)
        advantages = np.zeros_like(rewards_for_gae, dtype=np.float32)  # Shape (T, N)
        last_gae_lam = np.zeros(N, dtype=np.float32)  # Shape (N,)

        # R_bootstrap_agents is V(s_{T+1}), shape (N,)
        for t in reversed(range(T)):
            current_rewards_t = rewards_for_gae[t]
            current_values_t = values_arr[t]

            actual_next_values = R_bootstrap_agents if t == T - 1 else values_arr[t + 1]
            done_after_current_step = dones_arr[t]

            next_non_terminal = 1.0 - done_after_current_step.astype(np.float32)

            delta = current_rewards_t + self.gamma * actual_next_values * next_non_terminal - current_values_t
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam

        returns_arr = advantages + values_arr

        self.obs, self.fps, self.actions, self.rewards, self.values, \
        self.dones, self.log_probs, self.lstm_states = [], [], [], [], [], [], [], []
        self.ptr = 0
        self.path_start_idx = 0

        return (obs_arr, actions_arr, log_probs_arr, dones_arr, returns_arr,
                advantages, values_arr, fps_arr, lstm_states_arr)


"""
util functions
"""
class Scheduler:
    def __init__(self, val_init, val_min=0, total_step=0, decay='linear'):
        self.val = val_init
        self.N = float(total_step)
        self.val_min = val_min
        self.decay = decay
        self.n = 0

    def get(self, n_step):
        self.n += n_step
        if self.decay == 'linear':
            return max(self.val_min, self.val * (1 - self.n / self.N))
        else:
            return self.val


def init_log(log_file=None):
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    log_level = logging.INFO

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        stream=sys.stdout
    )

    if log_file:
        logging.info(f"Logging configured. Level: {logging.getLevelName(log_level)}. Outputting to Console and File: {log_file}")
    else:
        logging.info(f"Logging configured. Level: {logging.getLevelName(log_level)}. Outputting to Console (stdout).")

