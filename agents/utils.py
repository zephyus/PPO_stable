import numpy as np
import torch
import torch.nn as nn
import logging
import sys # Import sys for StreamHandler

"""
initializers
"""
def init_layer(layer, layer_type):
    if layer_type == 'fc':
        nn.init.orthogonal_(layer.weight.data)
        nn.init.constant_(layer.bias.data, 0)
    elif layer_type == 'lstm':
        nn.init.orthogonal_(layer.weight_ih.data)
        nn.init.orthogonal_(layer.weight_hh.data)
        nn.init.constant_(layer.bias_ih.data, 0)
        nn.init.constant_(layer.bias_hh.data, 0)

"""
layer helpers
"""
def batch_to_seq(x):
    n_step = x.shape[0]
    if len(x.shape) == 1:
        x = torch.unsqueeze(x, -1)
    return torch.chunk(x, n_step)


def run_rnn(layer, xs, dones, s):
    xs = batch_to_seq(xs)
    # need dones to reset states
    dones = batch_to_seq(dones)
    n_in = int(xs[0].shape[1])
    n_out = int(s.shape[0]) // 2
    s = torch.unsqueeze(s, 0)
    h, c = torch.chunk(s, 2, dim=1)
    h = h.cuda()
    c = c.cuda()
    outputs = []
    for ind, (x, done) in enumerate(zip(xs, dones)):
        c = c * (1-done)
        h = h * (1-done)
        h, c = layer(x, (h, c))
        outputs.append(h)
    s = torch.cat([h, c], dim=1)
    return torch.cat(outputs), torch.squeeze(s)


def one_hot(x, oh_dim, dim=-1):
    oh_shape = list(x.shape)
    if dim == -1:
        oh_shape.append(oh_dim)
    else:
        oh_shape = oh_shape[:dim+1] + [oh_dim] + oh_shape[dim+1:]
    x_oh = torch.zeros(oh_shape)
    x = torch.unsqueeze(x, -1)
    if dim == -1:
        x_oh = x_oh.scatter(dim, x, 1)
    else:
        x_oh = x_oh.scatter(dim+1, x, 1)
    return x_oh


"""
buffers
"""
class TransBuffer:
    def reset(self):
        self.buffer = []

    @property
    def size(self):
        return len(self.buffer)

    def add_transition(self, ob, a, r, *_args, **_kwargs):
        raise NotImplementedError()

    def sample_transition(self, *_args, **_kwargs):
        raise NotImplementedError()


class OnPolicyBuffer(TransBuffer):
    def __init__(self, gamma, alpha, distance_mask, gae_lambda=0.95):
        self.gamma = gamma
        self.alpha = alpha
        self.gae_lambda = gae_lambda              # store GAE lambda
        if alpha > 0:
            self.distance_mask = distance_mask
            self.max_distance = np.max(distance_mask, axis=-1)
        self.reset()

    def reset(self, done=False):
        self.obs = []
        self.acts = []
        self.rs = []
        self.vs = []
        self.adds = []
        self.dones = [done]
        self.log_probs = []                       # new: old policy log‐probs

    def add_transition(self, ob, na, a, r, v, done, log_prob):
        self.obs.append(ob)
        self.adds.append(na)
        self.acts.append(a)
        self.rs.append(r)
        self.vs.append(v)
        self.dones.append(done)
        self.log_probs.append(log_prob)           # new: record log_prob_old

    def _add_R_Adv(self, R_bootstrap):
        # GAE implementation
        Rs_target = []
        Advs_gae = []
        gae = 0.0
        for i in reversed(range(len(self.rs))):
            r_t = self.rs[i]
            v_t = self.vs[i]
            done_tp1 = float(self.dones[i+1])
            v_tp1 = R_bootstrap if i == len(self.rs)-1 else self.vs[i+1]
            delta = r_t + self.gamma * v_tp1 * (1.0 - done_tp1) - v_t
            gae = delta + self.gamma * self.gae_lambda * (1.0 - done_tp1) * gae
            Advs_gae.append(gae)
            Rs_target.append(gae + v_t)
        self.Advs = list(reversed(Advs_gae))
        self.Rs    = list(reversed(Rs_target))

    def sample_transition(self, R_bootstrap, dt=0):
        if self.alpha < 0:
            self._add_R_Adv(R_bootstrap)
        else:
            self._add_s_R_Adv(R_bootstrap)          # if spatial‐reward path is used, adapt to GAE similarly

        obs_batch       = np.array(self.obs,       dtype=np.float32)
        act_batch       = np.array(self.acts,      dtype=np.int32)
        logp_batch      = np.array(self.log_probs, dtype=np.float32)
        return_batch    = np.array(self.Rs,        dtype=np.float32)  # target values V_target
        adv_batch       = np.array(self.Advs,      dtype=np.float32)  # GAE advantages
        value_old_batch = np.array(self.vs,        dtype=np.float32)  # V(s_t) at data‐collection time
        done_batch      = np.array(self.dones[:-1],dtype=np.bool)

        last_done = self.dones[-1]
        self.reset(last_done)

        return obs_batch, act_batch, logp_batch, done_batch, return_batch, adv_batch, value_old_batch


class MultiAgentOnPolicyBuffer(OnPolicyBuffer):
    def __init__(self, gamma, alpha, distance_mask, gae_lambda=0.95):  # add gae_lambda
        super().__init__(gamma, alpha, distance_mask, gae_lambda)

    def _add_R_Adv(self, R_bootstrap_agents):
        # per‐agent GAE
        vs_array = np.array(self.vs)       # shape: (steps, agents)
        rs_array = np.array(self.rs)       # shape: (steps,) or (steps, agents)
        num_steps, num_agents = vs_array.shape
        # if shared rewards, expand to per‐agent
        if rs_array.ndim == 1:
            rs_array = np.tile(rs_array[:, None], (1, num_agents))
        # buffers
        Rs_per_agent = [[] for _ in range(num_agents)]
        Advs_per_agent = [[] for _ in range(num_agents)]
        # GAE loop per agent
        for a in range(num_agents):
            gae = 0.0
            for i in reversed(range(num_steps)):
                r = rs_array[i, a]
                v = vs_array[i, a]
                done_tp1 = float(self.dones[i+1])
                v_tp1 = R_bootstrap_agents[a] if i == num_steps-1 else vs_array[i+1, a]
                delta = r + self.gamma * v_tp1 * (1.0 - done_tp1) - v
                gae = delta + self.gamma * self.gae_lambda * (1.0 - done_tp1) * gae
                Advs_per_agent[a].append(gae)
                Rs_per_agent[a].append(gae + v)
            Advs_per_agent[a].reverse()
            Rs_per_agent[a].reverse()
        # store as arrays (steps, agents)
        self.Advs = np.array(Advs_per_agent).transpose()
        self.Rs   = np.array(Rs_per_agent).transpose()

    def sample_transition(self, R_bootstrap_agents, dt=0):
        # compute GAE returns
        if self.alpha < 0:
            self._add_R_Adv(R_bootstrap_agents)
        else:
            self._add_s_R_Adv(R_bootstrap_agents)  # adapt spatial path similarly if used

        # transpose time‐major to agent‐major
        obs_batch       = np.transpose(np.array(self.obs,       dtype=np.float32), (1, 0, 2))
        act_batch       = np.transpose(np.array(self.acts,      dtype=np.int32),   (1, 0))
        logp_batch      = np.transpose(np.array(self.log_probs, dtype=np.float32), (1, 0))
        value_old_batch = np.transpose(np.array(self.vs,        dtype=np.float32), (1, 0))
        return_batch    = np.transpose(np.array(self.Rs,        dtype=np.float32), (1, 0))
        adv_batch       = np.transpose(np.array(self.Advs,      dtype=np.float32), (1, 0))
        done_batch      = np.array(self.dones[:-1], dtype=np.bool)

        last_done = self.dones[-1]
        self.reset(last_done)

        return obs_batch, act_batch, logp_batch, done_batch, return_batch, adv_batch, value_old_batch

    def _add_st_R_Adv(self, R, dt):
        Rs = []
        Advs = []
        vs = np.array(self.vs)
        for i in range(vs.shape[1]):
            cur_Rs = []
            cur_Advs = []
            cur_R = R[i]
            tdiff = dt
            distance_mask = self.distance_mask[i]
            max_distance = self.max_distance[i]
            for r, v, done in zip(self.rs[::-1], vs[::-1,i], self.dones[:0:-1]):
                cur_R = self.gamma * cur_R * (1.-done)
                if done:
                    tdiff = 0
                # additional spatial rewards
                tmax = min(tdiff, max_distance)
                for t in range(tmax + 1):
                    rt = np.sum(r[distance_mask==t])
                    cur_R += (self.gamma * self.alpha) ** t * rt
                cur_Adv = cur_R - v
                tdiff += 1
                cur_Rs.append(cur_R)
                cur_Advs.append(cur_Adv)
            cur_Rs.reverse()
            cur_Advs.reverse()
            Rs.append(cur_Rs)
            Advs.append(cur_Advs)
        self.Rs = np.array(Rs)
        self.Advs = np.array(Advs)

    def _add_s_R_Adv(self, R):
        Rs = []
        Advs = []
        vs = np.array(self.vs)
        for i in range(vs.shape[1]):
            cur_Rs = []
            cur_Advs = []
            cur_R = R[i]
            distance_mask = self.distance_mask[i]
            max_distance = self.max_distance[i]
            for r, v, done in zip(self.rs[::-1], vs[::-1,i], self.dones[:0:-1]):
                cur_R = self.gamma * cur_R * (1.-done)
                # additional spatial rewards
                for t in range(max_distance + 1):
                    rt = np.sum(r[distance_mask==t])
                    cur_R += (self.alpha ** t) * rt
                cur_Adv = cur_R - v
                cur_Rs.append(cur_R)
                cur_Advs.append(cur_Adv)
            cur_Rs.reverse()
            cur_Advs.reverse()
            Rs.append(cur_Rs)
            Advs.append(cur_Advs)
        self.Rs = np.array(Rs)
        self.Advs = np.array(Advs)

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
    # Configure root logger
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    log_level = logging.INFO # Or logging.DEBUG for more verbose output

    # Remove all existing handlers from the root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Set basic config - this sets up a StreamHandler by default if filename is None
    logging.basicConfig(
        level=log_level,
        format=log_format,
        stream=sys.stdout # Explicitly set stream to stdout
    )

    # If a log file is provided, add a FileHandler as well
    # if log_file:
    #     file_handler = logging.FileHandler(log_file, mode='a')
    #     file_handler.setFormatter(logging.Formatter(log_format))
    #     logging.getLogger().addHandler(file_handler) # Add handler to the root logger

    # Log the configuration
    if log_file:
        logging.info(f"Logging configured. Level: {logging.getLevelName(log_level)}. Outputting to Console and File: {log_file}")
    else:
        logging.info(f"Logging configured. Level: {logging.getLevelName(log_level)}. Outputting to Console (stdout).")

