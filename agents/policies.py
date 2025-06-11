#/agents/policies.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import threading
from agents.utils import batch_to_seq, init_layer, one_hot, run_rnn
from typing import Optional # Added import
# Patch: Optional import for torch_geometric
try:
    from torch_geometric.nn import GATConv
except ImportError:
    GATConv = None
    logging.warning("torch_geometric not found: GATConv will be disabled.")


class Policy(nn.Module):
    def __init__(self, n_a, n_s, n_step, policy_name, agent_name, identical):
        super(Policy, self).__init__()
        self.name = policy_name
        if agent_name is not None:
            # for multi-agent system
            self.name += '_' + str(agent_name)
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step
        self.identical = identical

    def forward(self, ob, *_args, **_kwargs):
        raise NotImplementedError()

    def _init_actor_head(self, n_h, n_a=None):
        if n_a is None:
            n_a = self.n_a
        self.actor_head = nn.Linear(n_h, n_a)
        init_layer(self.actor_head, 'fc')

    def _init_critic_head(self, n_h, n_n=None):
        if n_n is None:
            n_n = int(self.n_n)
        if n_n:
            if self.identical:
                n_na_sparse = self.n_a * n_n
            else:
                n_na_sparse = sum(self.na_dim_ls)
            n_h += n_na_sparse
        self.critic_head = nn.Linear(n_h, 1)
        init_layer(self.critic_head, 'fc')

    def _run_critic_head(self, h, na, n_n=None):
        if n_n is None:
            n_n = int(self.n_n)
        if n_n:
            na = torch.from_numpy(na).long()
            if self.identical:
                na_sparse = one_hot(na, self.n_a)
                na_sparse = na_sparse.view(-1, self.n_a * n_n)
            else:
                na_sparse = []
                na_ls = torch.chunk(na, n_n, dim=1)
                for na_val, na_dim in zip(na_ls, self.na_dim_ls):
                    na_sparse.append(torch.squeeze(one_hot(na_val, na_dim), dim=1))
                na_sparse = torch.cat(na_sparse, dim=1)
            h = torch.cat([h, na_sparse.to(h.device)], dim=1)
        return self.critic_head(h).squeeze()

    def _run_loss(self, actor_dist, e_coef, v_coef, vs, As, Rs, Advs):
        As = As.to(vs.device)
        Advs = Advs.to(vs.device)
        Rs = Rs.to(vs.device)
        log_probs = actor_dist.log_prob(As)
        policy_loss = -(log_probs * Advs).mean()
        entropy_loss = -(actor_dist.entropy()).mean() * e_coef
        value_loss = (Rs - vs).pow(2).mean() * v_coef
        return policy_loss, value_loss, entropy_loss

    def _update_tensorboard(self, summary_writer, global_step):
        summary_writer.add_scalar('loss/{}_entropy_loss'.format(self.name), self.entropy_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_policy_loss'.format(self.name), self.policy_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_value_loss'.format(self.name), self.value_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_total_loss'.format(self.name), self.loss,
                                  global_step=global_step)


class LstmPolicy(Policy):
    def __init__(self, n_s, n_a, n_n, n_step, n_fc=64, n_lstm=64, name=None,
                 na_dim_ls=None, identical=True):
        super(LstmPolicy, self).__init__(n_a, n_s, n_step, 'lstm', name, identical)
        if not self.identical:
            self.na_dim_ls = na_dim_ls
        self.n_lstm = n_lstm
        self.n_fc = n_fc
        self.n_n = n_n
        self._init_net()
        self._reset()

    def backward(self, obs, nactions, acts, dones, Rs, Advs,
                 e_coef, v_coef, summary_writer=None, global_step=None):
        obs = torch.from_numpy(obs).float()
        dones = torch.from_numpy(dones).float()
        obs = obs.to(self.fc_layer.weight.device)
        dones = dones.to(self.fc_layer.weight.device)
        xs = self._encode_ob(obs)
        hs, new_states = run_rnn(self.lstm_layer, xs, dones, self.states_bw)
        self.states_bw = new_states.detach()
        actor_dist = torch.distributions.categorical.Categorical(logits=F.log_softmax(self.actor_head(hs), dim=1))
        vs = self._run_critic_head(hs, nactions)
        self.policy_loss, self.value_loss, self.entropy_loss = \
            self._run_loss(actor_dist, e_coef, v_coef, vs,
                           torch.from_numpy(acts).long(),
                           torch.from_numpy(Rs).float(),
                           torch.from_numpy(Advs).float())
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)
        # return scalar losses for logging
        return (self.policy_loss,
                self.value_loss,
                self.entropy_loss,
                self.loss)

    def forward(self, ob, done, naction=None, out_type='p'):
        ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float().to(self.fc_layer.weight.device)
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float().to(self.fc_layer.weight.device)
        x = self._encode_ob(ob)
        h, new_states = run_rnn(self.lstm_layer, x, done, self.states_fw)
        if out_type.startswith('p'):
            self.states_fw = new_states.detach()
            return F.softmax(self.actor_head(h), dim=1).squeeze().cpu().detach().numpy()
        else:
            return self._run_critic_head(h, np.array([naction])).cpu().detach().numpy()

    def _encode_ob(self, ob):
        return F.relu(self.fc_layer(ob))

    def _init_net(self):
        self.fc_layer = nn.Linear(self.n_s, self.n_fc)
        init_layer(self.fc_layer, 'fc')
        self.lstm_layer = nn.LSTMCell(self.n_fc, self.n_lstm)
        init_layer(self.lstm_layer, 'lstm')
        self._init_actor_head(self.n_lstm)
        self._init_critic_head(self.n_lstm)

    def _reset(self):
        if hasattr(self, 'fc_layer') and hasattr(self.fc_layer, 'weight'): # Check if fc_layer and its weight exist
            current_device = self.fc_layer.weight.device
        else:
            # Fallback, should be avoided if __init__ is correct
            logging.warning(f"Could not infer device for LstmPolicy {self.name} during _reset. Defaulting to CPU.")
            current_device = torch.device("cpu")

        self.states_fw = torch.zeros(self.n_lstm * 2, device=current_device)
        self.states_bw = torch.zeros(self.n_lstm * 2, device=current_device)


class FPPolicy(LstmPolicy):
    def __init__(self, n_s, n_a, n_n, n_step, n_fc=64, n_lstm=64, name=None,
                 na_dim_ls=None, identical=True):
        super(FPPolicy, self).__init__(n_s, n_a, n_n, n_step, n_fc, n_lstm, name,
                         na_dim_ls, identical)

    def _init_net(self):
        if self.identical:
            self.n_x = self.n_s - self.n_n * self.n_a
        else:
            self.n_x = int(self.n_s - sum(self.na_dim_ls))
        self.fc_x_layer = nn.Linear(self.n_x, self.n_fc)
        init_layer(self.fc_x_layer, 'fc')
        n_h = self.n_fc
        if self.n_n:
            self.fc_p_layer = nn.Linear(self.n_s-self.n_x, self.n_fc)
            init_layer(self.fc_p_layer, 'fc')
            n_h += self.n_fc
        self.lstm_layer = nn.LSTMCell(n_h, self.n_lstm)
        init_layer(self.lstm_layer, 'lstm')
        self._init_actor_head(self.n_lstm)
        self._init_critic_head(self.n_lstm)

    def _encode_ob(self, ob):
        x = F.relu(self.fc_x_layer(ob[:, :self.n_x]))
        if self.n_n:
            p = F.relu(self.fc_p_layer(ob[:, self.n_x:]))
            x = torch.cat([x, p], dim=1)
        return x


class NCMultiAgentPolicy(Policy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, model_config=None, identical=True):
        super(NCMultiAgentPolicy, self).__init__(n_a, n_s, n_step, 'nc', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self.model_config = model_config
        # Patch: self.use_gat initialization moved before _init_net
        use_gat_env_value = os.getenv('USE_GAT', '1')
        logging.info(
            f"DEBUG: os.getenv('USE_GAT', '1') returned: '{use_gat_env_value}' "
            f"(type: {type(use_gat_env_value)})"
        )
        # self.use_gat will be True if GATConv is available AND environment variable is '1'
        self.use_gat = (GATConv is not None) and (use_gat_env_value == '1')
        # persist this flag in checkpoints to detect mismatches when loading
        self.register_buffer('use_gat_flag', torch.tensor(int(self.use_gat), dtype=torch.uint8))
        if GATConv is None and use_gat_env_value == '1':
            logging.warning("USE_GAT is '1' but torch_geometric is not installed. GAT will be disabled.")
        
        logging.info(f"DEBUG: self.use_gat evaluated to: {self.use_gat}")

        self._init_net() # self.use_gat is now set before this call

        self._reset()
        self.zero_pad = nn.Parameter(torch.zeros(1, 2*self.n_fc), requires_grad=False)
        self.latest_attention_scores = None

    @property
    def dev(self):
        """Current device inferred from parameters."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def load_state_dict(self, state_dict, strict=True):
        if 'use_gat_flag' in state_dict:
            loaded = bool(state_dict['use_gat_flag'])
            if loaded != self.use_gat:
                raise ValueError(
                    f"use_gat mismatch: checkpoint expects {loaded} but model is configured with {self.use_gat}"
                )
        return super().load_state_dict(state_dict, strict)


    def backward(self, obs, fps, acts, dones, Rs, Advs,
                 e_coef, v_coef, summary_writer=None, global_step=None):
        obs = torch.from_numpy(obs).float().to(self.dev, non_blocking=True)
        fps = torch.from_numpy(fps).float().to(self.dev, non_blocking=True)
        
        dones_np = np.asarray(dones)
        if dones_np.ndim == 1:
            dones_T_N_np = np.repeat(dones_np[:, np.newaxis], self.n_agent, axis=1)
        else:
            dones_T_N_np = dones_np
        dones_T_N = torch.from_numpy(dones_T_N_np).bool().to(obs.device)

        acts_T_N = torch.from_numpy(acts).long().to(obs.device)

        hs_N_T_H, new_states_N_2H = self._run_comm_layers(obs, dones_T_N, fps, self.states_bw)
        self.states_bw = new_states_N_2H.detach()
        
        ps = self._run_actor_heads(hs_N_T_H)

        # Patch-04: Replace vs_placeholder with actual values from PPO value heads
        vs_list_N_of_T = []
        for i in range(self.n_agent):
            critic_input_i = self._build_value_input(hs_N_T_H[i], acts_T_N, i)
            head = self.ppo_value_heads[0] if self.identical else self.ppo_value_heads[i]
            val_i_T = head(critic_input_i).squeeze(-1)
            vs_list_N_of_T.append(val_i_T)
        vs_T_N = torch.stack(vs_list_N_of_T, dim=1)

        self.policy_loss = 0
        self.value_loss = 0
        self.entropy_loss = 0
        Rs_T_N = torch.from_numpy(Rs).float().to(obs.device)
        Advs_T_N = torch.from_numpy(Advs).float().to(obs.device)

        for i in range(self.n_agent):
            actor_dist_i = torch.distributions.categorical.Categorical(logits=ps[i])
            policy_loss_i, value_loss_i, entropy_loss_i = \
                self._run_loss(actor_dist_i, e_coef, v_coef, vs_T_N[:, i], 
                    acts_T_N[:, i], Rs_T_N[:, i], Advs_T_N[:, i])
            self.policy_loss += policy_loss_i
            self.value_loss += value_loss_i
            self.entropy_loss += entropy_loss_i
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)
        return (self.policy_loss,
                self.value_loss,
                self.entropy_loss,
                self.loss)

    def forward(self, ob_N_Do, done_N, fp_N_Dfp, neighbor_actions_N=None, action=None, out_type='p'):
        """Run actor (and optionally critic) for a single timestep.

        Parameters
        ----------
        ob_N_Do : np.ndarray
            Observation array with shape ``(N, obs_dim)``.
        done_N : np.ndarray
            Done flags for each agent.
        fp_N_Dfp : np.ndarray
            Fingerprint features for each agent.
        neighbor_actions_N : optional, array-like or Tensor
            If provided, should contain all agents' actions with shape ``(N,)``.
            When present, value predictions will be returned; otherwise the
            second element of the returned tuple will be ``None``.

        Returns
        -------
        List[Tensor]
            Actor logits for each agent.
        Optional[List[Tensor]]
            Value estimates if ``neighbor_actions_N`` is given, else ``None``.
        List[Tensor]
            Action probabilities for each agent.
        """
        device = self.actor_heads[0].weight.device

        obs_T1_N_Do = torch.from_numpy(ob_N_Do).float().unsqueeze(0).to(device)
        fps_T1_N_Dfp = torch.from_numpy(fp_N_Dfp).float().unsqueeze(0).to(device)

        dones_T1_N = torch.from_numpy(done_N).bool().unsqueeze(0).to(device)

        h_states_N_T1_H, new_states_N_2H = self._run_comm_layers(
            obs_T1_N_Do, dones_T1_N, fps_T1_N_Dfp, self.states_fw
        )

        h_states_N_H = h_states_N_T1_H.squeeze(1)

        actor_logits_list_N_of_A = []
        for i in range(self.n_agent):
            head = self.actor_heads[0] if self.identical else self.actor_heads[i]
            actor_logits_list_N_of_A.append(head(h_states_N_H[i, :].unsqueeze(0)))

        value_list_N_of_1 = None
        if neighbor_actions_N is not None:
            act_tensor = torch.as_tensor(neighbor_actions_N, dtype=torch.long, device=device).unsqueeze(0)
            value_list_N_of_1 = []
            for i in range(self.n_agent):
                h_i = h_states_N_H[i, :].unsqueeze(0)  # (1, H)
                critic_input_i = self._build_value_input(h_i, act_tensor, i)
                head = self.ppo_value_heads[0] if self.identical else self.ppo_value_heads[i]
                v_i = head(critic_input_i).squeeze(-1)
                value_list_N_of_1.append(v_i)

        probs_list_N_of_A = [F.softmax(logits.squeeze(0), dim=-1) for logits in actor_logits_list_N_of_A]

        self.states_fw = new_states_N_2H.detach().to(device)

        actor_logits_squeezed = [lg.squeeze(0) for lg in actor_logits_list_N_of_A]
        if value_list_N_of_1 is not None:
            value_list_squeezed = [val.squeeze(0) if val.numel() == 1 else val for val in value_list_N_of_1]
        else:
            value_list_squeezed = None

        return actor_logits_squeezed, value_list_squeezed, probs_list_N_of_A
    
    def evaluate_actions_values_and_entropy(
        self,
        obs_B_N_D: torch.Tensor,          # (B, N, obs_dim)
        fps_B_N_Dfp: Optional[torch.Tensor], # (B, N, fp_dim)  # Changed type hint
        actions_B_N:    torch.Tensor,     # (B, N, act_dim 或離散 index)
        lstm_states=None                  # 選填: ((1, B*N, H), (1, B*N, H))
    ):
        """
        一次向量化計算整個 minibatch = B × N 個 agent-steps 的
        - log_prob(action)
        - state value
        - entropy
        以及 LSTM 回傳的新隱狀態。

        傳入/傳出 shape 與舊版保持一致，但不再對 B 做 Python 迴圈。
        """
        B, N, _ = obs_B_N_D.shape                       # 取 batch 與 agent 數

        actions_B_N = actions_B_N.to(obs_B_N_D.device).long()

        # --- 1) 先拿到 (N,B,H) → 轉 (B,N,H) ---
        h_N_B_H, lstm_states_new = self._run_comm_layers(
                obs_B_N_D,
                dones_T_N=None, # Will be auto-filled by _run_comm_layers
                fps_T_N_Dfp=fps_B_N_Dfp, # Pass fps_B_N_Dfp here
                initial_states_N_2H=lstm_states
        )
        h_B_N_H = h_N_B_H.transpose(0, 1)  # (B,N,H)

        # --- 2) identical==True 的 logits/metrics 批量處理 ---
        if self.identical:
            logits_B_N_A = self.actor_heads[0](h_B_N_H)        # (B,N,A)
            critic_inputs_list = [
                self._build_value_input(h_B_N_H[:, i, :], actions_B_N, i)
                for i in range(self.n_agent)
            ]
            values_cols = [
                self.ppo_value_heads[0](ci).squeeze(-1) for ci in critic_inputs_list
            ]
            values_B_N = torch.stack(values_cols, dim=1)

            flat_logits  = logits_B_N_A.reshape(-1, logits_B_N_A.size(-1)) # (B*N, A)
            flat_actions = actions_B_N.reshape(-1) # (B*N)
            flat_dist    = self._get_dist(flat_logits)

            logp_B_N     = flat_dist.log_prob(flat_actions).view(B, N) # (B,N)
            entropy_B_N  = flat_dist.entropy().view(B, N) # (B,N)

        # --- 3) heterogeneous 分支保持 list，但按 B 維度批處理 ---
        else:
            logits_list, values_list = [], []
            for i, (actor_h, value_h) in enumerate(zip(self.actor_heads, self.ppo_value_heads)):
                logits_i_B_A = actor_h(h_B_N_H[:, i, :])            # (B,A_i)

                critic_input_i = self._build_value_input(h_B_N_H[:, i, :], actions_B_N, i)
                values_i_B = value_h(critic_input_i).squeeze(-1) # (B)
                logits_list.append(logits_i_B_A)
                values_list.append(values_i_B)

            logits_B_N_A = logits_list # List of (B,A_i) tensors
            values_B_N   = torch.stack(values_list, dim=1)          # (B,N)

            logp_cols, ent_cols = [], []
            for i in range(N): # N is self.n_agent or obs_B_N_D.shape[1]
                # logits_B_N_A[i] is (B,A_i)
                dist_i     = self._get_dist(logits_B_N_A[i])
                # actions_B_N[:, i] is (B)
                logp_cols.append(dist_i.log_prob(actions_B_N[:, i])) # (B)
                ent_cols.append(dist_i.entropy()) # (B)
            logp_B_N    = torch.stack(logp_cols, dim=1) # (B,N)
            entropy_B_N = torch.stack(ent_cols, dim=1) # (B,N)

        # --- 4) 最終回傳 ---
        return logp_B_N, values_B_N, entropy_B_N, lstm_states_new

    def _get_comm_s(self, i, n_n, x, h, p):
        device = x.device
        js = self.neighbor_index_ls[i].to(device)   # <- Use cached neighbor indices
        m_i = torch.index_select(h, 0, js).view(1, self.n_h * n_n)
        p_i_indexed = torch.index_select(p, 0, js)
        # ── ensure at least 1-D when Do==1 ──
        x_src = x.unsqueeze(-1) if x.dim() == 1 else x
        nx_i_indexed = torch.index_select(x_src, 0, js)

        # ── unified fingerprint concatenation ──
        if self.identical:
            # flatten all neighbors into one fingerprint vector
            p_cat_for_fc = p_i_indexed.view(1, -1)           # (1, n_n*Da)
            nx_i         = nx_i_indexed.view(1, -1)          # (1, n_n*Do′)
            x_i          = x_src[i].unsqueeze(0)             # (1, Do′)
        else:
            p_i_ls, nx_i_ls = [], []
            for j_loop in range(n_n):
                p_vec = p_i_indexed[j_loop]
                if p_vec.dim() == 0: p_vec = p_vec.unsqueeze(0)
                nx_vec = nx_i_indexed[j_loop]
                if nx_vec.dim() == 0: nx_vec = nx_vec.unsqueeze(0)
                p_i_ls.append(p_vec.narrow(0, 0, self.na_ls_ls[i][j_loop]))
                nx_i_ls.append(nx_vec.narrow(0, 0, self.ns_ls_ls[i][j_loop]))
            # build concatenated fingerprint or mark empty
            p_cat_for_fc = (torch.cat(p_i_ls, dim=0).unsqueeze(0) 
                            if p_i_ls else None)
            nx_i = (torch.cat(nx_i_ls).unsqueeze(0) 
                    if nx_i_ls else torch.zeros(1, 0, device=device))
            x_i_raw = x_src[i]
            if x_i_raw.dim() == 0: x_i_raw = x_i_raw.unsqueeze(0)
            x_i = x_i_raw.narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)

        fc_x_input = torch.cat([x_i, nx_i], dim=1)
        current_n_ns = fc_x_input.size(1)
        fc_x = self._get_fc_x(i, n_n, current_n_ns)
        s_x = F.relu(fc_x(fc_x_input))

        # ── project fingerprints safely ──
        if n_n > 0 and self.fc_p_layers[i] is not None and p_cat_for_fc is not None:
            in_dim = self.fc_p_layers[i].in_features
            if p_cat_for_fc.size(1) > in_dim:
                logging.warning(
                    f"Agent {i}: fingerprint dim {p_cat_for_fc.size(1)} exceeds {in_dim}, truncating"
                )
                p_cat_for_fc = p_cat_for_fc[:, :in_dim]
            elif p_cat_for_fc.size(1) < in_dim:
                pad = torch.zeros(1, in_dim - p_cat_for_fc.size(1), device=device)
                p_cat_for_fc = torch.cat([p_cat_for_fc, pad], dim=1)
            p_proj = F.relu(self.fc_p_layers[i](p_cat_for_fc.to(device)))
        else:
            # no neighbors or no valid fingerprint → zeros
            p_proj = torch.zeros(1, self.n_fc, device=device)

        # --- project neighbor hidden‐message as before (with safety for None layer) ---
        if self.fc_m_layers[i] is not None:
            m_proj = F.relu(self.fc_m_layers[i](m_i.to(device)))
        else:
            m_proj = torch.zeros(1, self.n_fc, device=device)

        # Finally, concat [s_x | p_proj | m_proj] → size = 3*n_fc
        return torch.cat([s_x.to(device), p_proj, m_proj], dim=1)

    def _get_neighbor_dim(self, i_agent):
        n_n = int(np.sum(self.neighbor_mask[i_agent]))
        if self.identical:
            return n_n, self.n_s * (n_n+1), self.n_a * n_n, [self.n_s] * n_n, [self.n_a] * n_n
        else:
            ns_ls = []
            na_ls = []
            for j in np.where(self.neighbor_mask[i_agent])[0]:
                ns_ls.append(self.n_s_ls[j])
                na_ls.append(self.n_a_ls[j])
            return n_n, self.n_s_ls[i_agent] + sum(ns_ls), sum(na_ls), ns_ls, na_ls

    def _build_value_input(self, h_tensor, actions_tensor, agent_id):
        """Construct critic input by concatenating one-hot neighbor actions.

        Parameters
        ----------
        h_tensor : torch.Tensor
            Hidden states for agent ``agent_id`` with shape ``(B_or_T, H)``.
        actions_tensor : torch.Tensor or None
            Tensor containing all agents' actions with shape ``(B_or_T, N)``. If
            ``None`` a zero placeholder will be used for neighbor actions.
        agent_id : int
            Which agent's neighbor configuration to use.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B_or_T, H + n_na)`` suitable for the
            corresponding value head.
        """

        _, _, n_na, _, _ = self._get_neighbor_dim(agent_id)
        if n_na == 0:
            return h_tensor

        device = h_tensor.device
        dtype = h_tensor.dtype
        if self.identical:
            target_dim = self.ppo_value_heads[0].in_features - self.n_h
        else:
            target_dim = n_na

        if actions_tensor is None:
            pad = torch.zeros(h_tensor.size(0), target_dim, device=device, dtype=dtype)
            return torch.cat([h_tensor, pad], dim=1)

        neighbor_indices = self.neighbor_index_ls[agent_id].to(actions_tensor.device)
        if neighbor_indices.numel() == 0:
            pad = torch.zeros(h_tensor.size(0), target_dim, device=actions_tensor.device, dtype=dtype)
            return torch.cat([h_tensor, pad], dim=1)

        if self.identical:
            neighbor_actions = actions_tensor[:, neighbor_indices]
            neighbor_actions = neighbor_actions.clamp(0, self.n_a - 1)
            one_hot = F.one_hot(neighbor_actions, num_classes=self.n_a).to(dtype)
            flat = one_hot.view(h_tensor.size(0), -1)
            if flat.size(1) > target_dim:
                logging.warning(
                    f"[Agent {agent_id}] one-hot overflow {flat.size(1)}>{target_dim}, truncating."
                )
                flat = flat[:, :target_dim]
            elif flat.size(1) < target_dim:
                pad_dim = target_dim - flat.size(1)
                padding = torch.zeros(h_tensor.size(0), pad_dim, device=device, dtype=dtype)
                flat = torch.cat([flat, padding], dim=1)
            assert flat.size(1) == target_dim
            return torch.cat([h_tensor, flat], dim=1)
        else:
            segs = []
            for k, n_id in enumerate(neighbor_indices):
                act = actions_tensor[:, n_id]
                n_dim = self.na_ls_ls[agent_id][k]
                act = act.clamp(0, n_dim - 1)
                seg = F.one_hot(act, num_classes=n_dim).to(dtype)
                segs.append(seg)
            cat = (
                torch.cat(segs, dim=1)
                if segs
                else torch.zeros(h_tensor.size(0), 0, device=actions_tensor.device, dtype=dtype)
            )
            if cat.size(1) > target_dim:
                logging.warning(
                    f"[Agent {agent_id}] one-hot overflow {cat.size(1)}>{target_dim}, truncating."
                )
                cat = cat[:, :target_dim]
            elif cat.size(1) < target_dim:
                pad = torch.zeros(h_tensor.size(0), target_dim - cat.size(1), device=device, dtype=dtype)
                cat = torch.cat([cat, pad], dim=1)
            assert cat.size(1) == target_dim
            return torch.cat([h_tensor, cat], dim=1)

    def _init_actor_head(self, n_a):
        actor_head = nn.Linear(self.n_h, n_a)
        init_layer(actor_head, 'fc')
        self.actor_heads.append(actor_head)

    def _init_net(self):
        self._fc_x_lock = threading.Lock()
        self.fc_x_layers = nn.ModuleDict()
        self.fc_p_layers = nn.ModuleList()
        self.fc_m_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.ppo_value_heads = nn.ModuleList()
        self.ns_ls_ls = []
        self.na_ls_ls = []
        self.n_n_ls = []

        self.lstm_layer = nn.LSTM(input_size=3 * self.n_fc, hidden_size=self.n_h, num_layers=1)
        init_layer(self.lstm_layer, 'lstm')

        # Patch: Conditional GAT initialization
        self.gat_layer = None
        if self.use_gat and GATConv is not None: # Check both self.use_gat and GATConv availability
            self.gat_num_heads = self.model_config.getint('gat_num_heads', 8) if self.model_config else 8
            gat_input_dim = 3 * self.n_fc
            assert gat_input_dim % self.gat_num_heads == 0, \
                f"GAT input dim ({gat_input_dim}) not divisible by num_heads ({self.gat_num_heads})"
            gat_dropout_init = self.model_config.getfloat('gat_dropout_init', 0.2) if self.model_config else 0.1
            self.gat_layer = GATConv(
                in_channels=gat_input_dim,
                out_channels=gat_input_dim // self.gat_num_heads,
                heads=self.gat_num_heads,
                concat=True,
                dropout=gat_dropout_init
            )
            adj_matrix = torch.tensor(self.neighbor_mask, dtype=torch.float32)
            adj_matrix = torch.maximum(
                adj_matrix,
                torch.eye(adj_matrix.size(0), dtype=torch.float32)
            )
            edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()
            self.register_buffer('edge_index', edge_index.long())

            self.use_layer_norm = True
            self.use_residual = True
            if self.model_config:
                self.use_layer_norm = self.model_config.getboolean('gat_use_layer_norm', True)
                self.use_residual = self.model_config.getboolean('gat_use_residual', True)
            if self.use_layer_norm:
                self.pre_gat_ln = nn.LayerNorm(3 * self.n_fc)
            self.gat_output_projection = nn.Linear(3 * self.n_fc, 3 * self.n_fc, bias=False)
            nn.init.xavier_uniform_(self.gat_output_projection.weight.data, gain=1.414)
        elif self.use_gat and GATConv is None: # Log if GAT was intended but module missing
            logging.warning(f"[{self.name}] GAT is configured (use_gat=True) but GAT layer object is None (likely due to import error or config).")


        max_n_na = 0
        if self.identical:
            # compute maximum neighbour-action dimension among all agents
            dims = [self._get_neighbor_dim(i)[2] for i in range(self.n_agent)]
            if dims:
                max_n_na = max(dims)
            self.shared_value_head = nn.Linear(self.n_h + max_n_na, 1)
            init_layer(self.shared_value_head, 'fc')
            self._init_actor_head(self.n_a)
            self.ppo_value_heads = nn.ModuleList([self.shared_value_head])

        for i in range(self.n_agent):
            n_n, n_ns, n_na, ns_ls, na_ls = self._get_neighbor_dim(i)
            self.ns_ls_ls.append(ns_ls)
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            if n_n:
                max_dim = n_na
                if self.identical:
                    max_dim = max_n_na
                fc_p_layer = nn.Linear(max_dim, self.n_fc)
                init_layer(fc_p_layer, 'fc')
                fc_m_layer = nn.Linear(self.n_h * n_n, self.n_fc)
                init_layer(fc_m_layer, 'fc')
                self.fc_m_layers.append(fc_m_layer)
                self.fc_p_layers.append(fc_p_layer)
            else:
                self.fc_m_layers.append(None)
                self.fc_p_layers.append(None)

            if not self.identical:
                n_a = self.n_a_ls[i]
                self._init_actor_head(n_a)
                ppo_v = nn.Linear(self.n_h + n_na, 1)
                init_layer(ppo_v, 'fc')
                self.ppo_value_heads.append(ppo_v)

        if not list(self.parameters()):
            logging.warning(f"[{self.name}] cannot infer device in _init_net.")

        # Cache neighbor indices as a list of LongTensors
        self.neighbor_index_ls = []
        for idx, mask in enumerate(self.neighbor_mask):
            tensor = torch.from_numpy(np.where(mask)[0]).long()
            self.register_buffer(f'neighbor_index_{idx}', tensor)
            self.neighbor_index_ls.append(tensor)

    def _reset(self):
        current_device = self.dev
        self.states_fw = torch.zeros(self.n_agent, self.n_h * 2, device=current_device)
        self.states_bw = torch.zeros(self.n_agent, self.n_h * 2, device=current_device)

    def _run_actor_heads(self, hs, detach=False):
        logits_list = []
        for i in range(self.n_agent):
            head = self.actor_heads[0] if self.identical else self.actor_heads[i]
            logits_i = head(hs[i])
            if detach:
                logits_list.append(logits_i.cpu().detach().numpy())
            else:
                logits_list.append(logits_i)
        return logits_list

    def _compute_s_features_flat(self, obs_T_N_Do, fps_T_N_Dfp, h_for_comm_N_H):
        """
        直接一次產生 (T*N, 3*n_fc)；不再對 T 做 for-loop。
        identical == True 可 100% 去掉時間迴圈；
        identical == False 先保留 per-agent, 再進階優化。
        """
        T, N, Do = obs_T_N_Do.shape
        device   = obs_T_N_Do.device
        H        = self.n_h
        n_fc     = self.n_fc

        # ---- Step 1. 展平成 (T*N, Do | Dfp | H) ----
        obs_flat = obs_T_N_Do.reshape(T*N, Do)                 # (TN, Do)
        fps_dim  = fps_T_N_Dfp.size(-1) if fps_T_N_Dfp is not None else 0
        if fps_dim:
            fps_flat = fps_T_N_Dfp.reshape(T*N, fps_dim)       # (TN, Dfp)
        h_repeat = h_for_comm_N_H.unsqueeze(0).expand(T, N, H).reshape(T*N, H)  # (TN,H)

        # ---- Step 2. 依 Agent 分批組合 s 向量 ----
        s_cat_list = []
        for i in range(N):          # 只剩 N 次迴圈
            n_n = self.n_n_ls[i]
            idx_nei = self.neighbor_index_ls[i].to(device)     # (n_n,)

            # gather neighbor h, obs, fp  —— 利用 flatten trick
            if n_n:
                # 先算展平 index:  t_idx* N + nei_j
                if not hasattr(self, '_arange_cache'):
                    self._arange_cache = {}
                key = (T, device)
                if key not in self._arange_cache:
                    self._arange_cache[key] = torch.arange(T, device=device).unsqueeze(1)
                base  = self._arange_cache[key] * N       # (T,1)
                idx_flat = (base + idx_nei.unsqueeze(0)).reshape(-1)          # (T*n_n,)
                # → 取鄰居 hidden
                m_i_flat  = h_repeat[idx_flat].reshape(T, n_n*H)              # (T, n_n*H)
            else:
                m_i_flat  = torch.zeros(T, 0, device=device)

            # self.identical 判斷
            if self.identical:
                x_i_flat  = obs_T_N_Do[:, i, :].reshape(T, Do)                # (T,Do)
                if n_n:
                    nx_i_flat = obs_flat[idx_flat].reshape(T, n_n*Do)             # (T, n_n*Do)
                    if fps_dim:
                        p_i_flat  = fps_flat[idx_flat].reshape(T, n_n*fps_dim)    # (T,n_n*Dfp)
                    else:
                        p_i_flat = torch.zeros(T, 0, device=device)
                else:
                    nx_i_flat = torch.zeros(T, 0, device=device)
                    p_i_flat = torch.zeros(T, 0, device=device)
                fc_x_in   = torch.cat([x_i_flat, nx_i_flat], dim=1)           # (T, Do + n_n*Do)
            else:  # -------- heterogeneous ----------
                # a) 取自身觀測前 ns_i 維
                ns_i   = self.n_s_ls[i]
                x_i_raw = obs_T_N_Do[:, i, :ns_i].reshape(T, ns_i)

                # b) 逐鄰居 slice fingerprint / obs
                p_seg_ls, nx_seg_ls = [], []
                for j_nei, nei_id in enumerate(idx_nei):
                    na_j = self.na_ls_ls[i][j_nei]   # 该邻居 action 維數
                    ns_j = self.ns_ls_ls[i][j_nei]   # 该邻居 obs   維數

                    idx_flat = (base + nei_id).reshape(-1)        # (T,)
                    # fingerprint 取前 na_j 維
                    if fps_dim:
                        fp_j = fps_flat[idx_flat][:, :na_j]       # (T,na_j)
                        p_seg_ls.append(fp_j)
                    # neighbor obs 取前 ns_j 維
                    nx_j = obs_flat[idx_flat][:, :ns_j]           # (T,ns_j)
                    nx_seg_ls.append(nx_j)

                # c) Concatenate segments（若無鄰居則空）
                p_i_flat  = torch.cat(p_seg_ls, dim=1) if p_seg_ls else torch.zeros(T, 0, device=device)
                nx_i_flat = torch.cat(nx_seg_ls, dim=1) if nx_seg_ls else torch.zeros(T, 0, device=device)

                # d) feed fc_x
                fc_x_in  = torch.cat([x_i_raw, nx_i_flat], dim=1)     # (T, n_ns)

            # ---- 線性層運算 (一次性 T) ----
            fc_x = self._get_fc_x(i, n_n, fc_x_in.size(1))
            s_x  = F.relu(fc_x(fc_x_in))                                      # (T,n_fc)

            # fingerprint / hidden‐msg 投影
            if n_n and self.fc_p_layers[i] is not None and fps_dim:
                in_dim = self.fc_p_layers[i].in_features
                if p_i_flat.size(1) > in_dim:
                    logging.warning(
                        f"Agent {i}: fingerprint dim {p_i_flat.size(1)} exceeds {in_dim}, truncating"
                    )
                    p_i_flat = p_i_flat[:, :in_dim]
                elif p_i_flat.size(1) < in_dim:
                    pad = torch.zeros(T, in_dim - p_i_flat.size(1), device=device)
                    p_i_flat = torch.cat([p_i_flat, pad], dim=1)
                p_proj = F.relu(self.fc_p_layers[i](p_i_flat))
            else:
                p_proj = torch.zeros(T, n_fc, device=device)

            if self.fc_m_layers[i] is not None and n_n:
                m_proj = F.relu(self.fc_m_layers[i](m_i_flat))
            else:
                m_proj = torch.zeros(T, n_fc, device=device)

            s_all_T_3fc = torch.cat([s_x, p_proj, m_proj], dim=1)             # (T,3*n_fc)
            s_cat_list.append(s_all_T_3fc)

        # ---- Step 3. 拼回 (T,N,3*n_fc) → flatten (T*N,3*n_fc) ----
        s_T_N_3fc = torch.stack(s_cat_list, dim=1)           # (T,N,3*n_fc)
        return s_T_N_3fc.reshape(T*N, 3*n_fc)


    def _apply_gat(self, s_flat_TN_D):
        s_block_input = s_flat_TN_D
        # Patch: Update GAT condition
        if self.use_gat and self.gat_layer is not None:
            if s_flat_TN_D.numel() == 0:
                return s_flat_TN_D
            if self.use_layer_norm and hasattr(self, 'pre_gat_ln'):
                s_input_for_gat = self.pre_gat_ln(s_block_input)
            else:
                s_input_for_gat = s_block_input

            num_nodes_per_graph = self.n_agent
            num_graphs = s_flat_TN_D.shape[0] // num_nodes_per_graph

            if s_flat_TN_D.shape[0] % num_nodes_per_graph != 0:
                raise ValueError("s_flat_TN_D cannot be reshaped to (T,N,D) for batched GAT processing.")

            if not hasattr(self, '_edge_index_cache'):
                self._edge_index_cache = {}
            if num_graphs not in self._edge_index_cache:
                edge_index_list = [
                    self.edge_index + i * num_nodes_per_graph for i in range(num_graphs)
                ]
                self._edge_index_cache[num_graphs] = torch.cat(edge_index_list, dim=1)
            batched_edge_index = self._edge_index_cache[num_graphs].to(s_input_for_gat.device)

            s_after_gat = self.gat_layer(s_input_for_gat, batched_edge_index)

            if hasattr(self, 'gat_output_projection'):
                s_projected = self.gat_output_projection(s_after_gat)
            else:
                s_projected = s_after_gat

            if self.use_residual:
                s_all_processed_flat = s_block_input + s_projected
            else:
                s_all_processed_flat = s_projected
        else:
            s_all_processed_flat = s_block_input
            # Patch: Update logging condition
            if self.use_gat and self.gat_layer is None: # Check if GAT was intended but layer is missing
                logging.error(f"[{self.name}] GAT layer specified (use_gat=True) but GAT layer object is None (likely due to import error or config).")
        return s_all_processed_flat

    def _run_comm_layers(self,
                         obs_T_N_Do,
                         dones_T_N=None,
                         fps_T_N_Dfp=None,
                         initial_states_N_2H=None):
        # ─── 新增：搬到 device ───────────────────────────
        obs_T_N_Do = obs_T_N_Do.to(self.dev)
        if dones_T_N is not None:
            dones_T_N = dones_T_N.to(self.dev)
        if fps_T_N_Dfp is not None:
            fps_T_N_Dfp = fps_T_N_Dfp.to(self.dev)
        if initial_states_N_2H is not None:
            initial_states_N_2H = initial_states_N_2H.to(self.dev)
        # ── 原有填補 default 的程式碼可保留或調整順序 ──
        if dones_T_N is None:
            # All-False → 代表持續序列
            dones_T_N = torch.zeros(
                obs_T_N_Do.size(0),
                obs_T_N_Do.size(1),
                dtype=torch.bool,
                device=self.dev
            )
        if fps_T_N_Dfp is None:
            # 若無鄰居 fingerprint，可用 0 占位；shape 保持 (T,N,0) 即可
            # Ensure the dimension for fingerprints (Dfp) is 0 if not provided.
            # The _get_comm_s method might expect a specific shape for fps,
            # so if it's critical, this might need adjustment based on its usage.
            # For now, assuming a 0-dim fingerprint if None.
            fps_T_N_Dfp = torch.zeros(
                obs_T_N_Do.size(0),
                obs_T_N_Do.size(1),
                0, # Assuming 0 features for fingerprint if None
                device=self.dev
            )
        if initial_states_N_2H is None:
            H = self.n_h
            N_agents = obs_T_N_Do.size(1)
            initial_states_N_2H = torch.zeros(
                N_agents, 2*H, device=self.dev
            )
        # ────────────────────────────────────────────────
        
        T, N, _ = obs_T_N_Do.shape

        h0, c0 = torch.chunk(initial_states_N_2H, 2, dim=1)

        if T > 0:
            initial_done_mask_N = (1.0 - dones_T_N[0].float()).unsqueeze(-1)
            h0 = h0 * initial_done_mask_N
            c0 = c0 * initial_done_mask_N

        s_intermediate_flat = self._compute_s_features_flat(obs_T_N_Do, fps_T_N_Dfp, h0)

        s_after_gat_flat = self._apply_gat(s_intermediate_flat)

        s_for_lstm = s_after_gat_flat.reshape(T, N, -1)

        h0_lstm = h0.unsqueeze(0)
        c0_lstm = c0.unsqueeze(0)

        lstm_out, (h_T, c_T) = self.lstm_layer(s_for_lstm, (h0_lstm, c0_lstm))

        outputs_N_T_H = lstm_out.transpose(0, 1)

        new_final_states_N_2H = torch.cat([h_T.squeeze(0), c_T.squeeze(0)], dim=1)

        return outputs_N_T_H, new_final_states_N_2H

    def _get_fc_x(self, agent_id: int, n_n: int, n_ns: int) -> nn.Linear:
        key = f'agent_{agent_id}_nn_{n_n}_in{n_ns}'
        with self._fc_x_lock:
            if key not in self.fc_x_layers:
                logging.info(f"Creating fc_x layer: {key} (in={n_ns})")
                layer = nn.Linear(n_ns, self.n_fc)
                init_layer(layer, 'fc')
                layer = layer.to(self.dev)
                self.fc_x_layers[key] = layer
            else:
                assert self.fc_x_layers[key].in_features == n_ns, \
                    f"fc_x[{key}] expects {self.fc_x_layers[key].in_features}, got {n_ns}"
            return self.fc_x_layers[key]


class NCLMMultiAgentPolicy(NCMultiAgentPolicy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, groups=0, identical=True, model_config=None):
        super(NCLMMultiAgentPolicy, self).__init__(n_s, n_a, n_agent, n_step, neighbor_mask, n_fc, n_h,
                                                  n_s_ls, n_a_ls, model_config, identical)
        self.groups = groups


class ConsensusPolicy(NCMultiAgentPolicy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, model_config=None, identical=True):
        super(ConsensusPolicy, self).__init__(n_s, n_a, n_agent, n_step, neighbor_mask, n_fc, n_h,
                                             n_s_ls, n_a_ls, model_config, identical)
        self.zero_pad = nn.Parameter(torch.zeros(1, 2 * self.n_fc), requires_grad=False)

    def consensus_update(self):
        consensus_update = []
        with torch.no_grad():
            for i in range(self.n_agent):
                mean_wts = self._get_critic_wts(i)
                for param, wt in zip(self.lstm_layers[i].parameters(), mean_wts):
                    param.copy_(wt)

    def _get_critic_wts(self, i_agent):
        wts = []
        for wt in self.lstm_layers[i_agent].parameters():
            wts.append(wt.detach())
        neighbors = list(np.where(self.neighbor_mask[i_agent] == 1)[0])
        for j in neighbors:
            for k, wt in enumerate(self.lstm_layers[j].parameters()):
                wts[k] += wt.detach()
        n = 1 + len(neighbors)
        for k in range(len(wts)):
            wts[k] /= n
        return wts


class CommNetMultiAgentPolicy(NCMultiAgentPolicy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'cnet', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()

    def _get_comm_s(self, i, n_n, x, h, p):
        device = h.device
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long().to(device)
        m_i = torch.index_select(h, 0, js).mean(dim=0, keepdim=True)
        nx_i = torch.index_select(x, 0, js)
        if self.identical:
            nx_i = nx_i.view(1, self.n_s * n_n)
            x_i = x[i].unsqueeze(0)
        else:
            nx_i_ls = []
            for j in range(n_n):
                nx_i_ls.append(nx_i[j].narrow(0, 0, self.ns_ls_ls[i][j]))
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)
            x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
        fc_x_input = torch.cat([x_i, nx_i], dim=1)
        current_n_ns = fc_x_input.size(1)
        fc_x = self._get_fc_x(i, n_n, current_n_ns)
        return F.relu(fc_x(fc_x_input)) + self.fc_m_layers[i](m_i)


class DIALMultiAgentPolicy(NCMultiAgentPolicy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'dial', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()

    def _get_comm_s(self, i, n_n, x, h, p):
        device = h.device
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long().to(device)
        m_i = torch.index_select(h, 0, js).view(1, self.n_h * n_n).to(device)
        nx_i = torch.index_select(x, 0, js).to(device)
        if self.identical:
            nx_i = nx_i.view(1, self.n_s * n_n)
            x_i = x[i].unsqueeze(0)
        else:
            nx_i_ls = []
            for j in range(n_n):
                nx_i_ls.append(nx_i[j].narrow(0, 0, self.ns_ls_ls[i][j]))
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)
            x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
        a_i = one_hot(p[i].argmax().unsqueeze(0).cpu(), self.n_fc).to(device)
        fc_x_input = torch.cat([x_i, nx_i], dim=1)
        current_n_ns = fc_x_input.size(1)
        fc_x = self._get_fc_x(i, n_n, current_n_ns)
        return F.relu(fc_x(fc_x_input)) + F.relu(self.fc_m_layers[i](m_i)) + a_i
