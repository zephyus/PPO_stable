#/agents/policies.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from agents.utils import batch_to_seq, init_layer, one_hot, run_rnn
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
        logging.info(f"DEBUG: os.getenv('USE_GAT', '1') returned: '{use_gat_env_value}' (type: {type(use_gat_env_value)}")
        # self.use_gat will be True if GATConv is available AND environment variable is '1'
        self.use_gat = (GATConv is not None) and (use_gat_env_value == '1')
        if GATConv is None and use_gat_env_value == '1':
            logging.warning("USE_GAT is '1' but torch_geometric is not installed. GAT will be disabled.")
        
        logging.info(f"DEBUG: self.use_gat evaluated to: {self.use_gat}")

        self._init_net() # self.use_gat is now set before this call
        self._reset()
        self.zero_pad = nn.Parameter(torch.zeros(1, 2*self.n_fc), requires_grad=False)
        self.latest_attention_scores = None

        # capture device placeholder; real device set when model.to(device) is called
        # This needs to be after _init_net() as parameters are created there
        if list(self.parameters()):
             self.device = next(self.parameters()).device
        else:
             # If no parameters (e.g. GAT disabled and no other layers yet)
             self.device = torch.device("cpu") # Default to CPU
             logging.warning(f"[{self.name}] No parameters found after _init_net. Defaulting device to CPU.")

    def backward(self, obs, fps, acts, dones, Rs, Advs,
                 e_coef, v_coef, summary_writer=None, global_step=None):
        obs = torch.from_numpy(obs).float() 
        fps = torch.from_numpy(fps).float()
        
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
            val_i_T = self.ppo_value_heads[i](hs_N_T_H[i]).squeeze(-1)
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

    def forward(self, ob_N_Do, done_N, fp_N_Dfp, action=None, out_type='p'):
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
            actor_logits_list_N_of_A.append(self.actor_heads[i](h_states_N_H[i, :].unsqueeze(0)))

        value_list_N_of_1 = []
        for i in range(self.n_agent):
            v_i = self.ppo_value_heads[i](h_states_N_H[i, :].unsqueeze(0)).squeeze(-1)
            value_list_N_of_1.append(v_i)

        probs_list_N_of_A = [F.softmax(logits.squeeze(0), dim=-1) for logits in actor_logits_list_N_of_A]

        self.states_fw = new_states_N_2H.detach()

        actor_logits_squeezed = [lg.squeeze(0) for lg in actor_logits_list_N_of_A]
        value_list_squeezed = [val.squeeze(0) if val.numel() == 1 else val for val in value_list_N_of_1]

        return actor_logits_squeezed, value_list_squeezed, probs_list_N_of_A
    

    def evaluate_actions_values_and_entropy(
            self, obs_batch, fps_batch, actions_batch, initial_states_batch):
        """
        Phase-0-plus：  
        - 仍然外層 loop B，確保每 sample 用自己的 LSTM state  
        - 把 Actor / Value head 批次化（identical 時一行解決）
        回傳 (B,N) 的 logp, value, entropy
        """
        # shape guard ----------------------------------------------------------
        if obs_batch.shape[0] == self.n_agent and obs_batch.shape[1] != self.n_agent:
            obs_batch  = obs_batch .permute(1,0,2).contiguous()
            fps_batch  = fps_batch .permute(1,0,2).contiguous()
            actions_batch = actions_batch.permute(1,0).contiguous()

        B, N, _ = obs_batch.shape
        dev = obs_batch.device
        if initial_states_batch.dim() == 2:          # (N,2H) → broadcast
            initial_states_batch = initial_states_batch.unsqueeze(0).expand(B, -1, -1)

        zeros_done = torch.zeros(1, N, dtype=torch.bool, device=dev)

        logp   = torch.zeros(B, N, device=dev)
        ent    = torch.zeros(B, N, device=dev)
        values = torch.zeros(B, N, device=dev)

        # --------  COMM-LSTM 每 sample 跑一次  --------
        for b in range(B):
            h_N_T1_H, _ = self._run_comm_layers(
                obs_batch[b:b+1],          # (1,N,Do)
                zeros_done,                # (1,N)
                fps_batch[b:b+1],          # (1,N,Dfp)
                initial_states_batch[b]    # (N,2H)  ← 關鍵：各用各的
            )
            h_N_H = h_N_T1_H.squeeze(1)     # (N,H)

            if self.identical:
                logits_N_A = self.actor_heads[0](h_N_H)              # (N,A)
                dist       = torch.distributions.Categorical(logits=logits_N_A)
                logp[b]    = dist.log_prob(actions_batch[b])
                ent [b]    = dist.entropy()
                values[b]  = self.ppo_value_heads[0](h_N_H).squeeze(-1)
            else:
                for i in range(N):                                    # 28 次而已
                    h_i = h_N_H[i].unsqueeze(0)
                    dist_i = torch.distributions.Categorical(
                                logits=self.actor_heads[i](h_i))
                    logp  [b, i] = dist_i.log_prob(actions_batch[b, i])
                    ent   [b, i] = dist_i.entropy()
                    values[b, i] = self.ppo_value_heads[i](h_i).squeeze()

        return logp, values, ent


    def _get_comm_s(self, i, n_n, x, h, p):
        device = x.device
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long().to(device)
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
            if p_cat_for_fc.size(1) == self.fc_p_layers[i].in_features:
                p_proj = F.relu(self.fc_p_layers[i](p_cat_for_fc.to(device)))
            else:
                logging.warning(
                    f"Agent {i}: fc_p_layer.in_features={self.fc_p_layers[i].in_features}, "
                    f"got {p_cat_for_fc.size(1)} → zeros fallback."
                )
                p_proj = torch.zeros(1, self.n_fc, device=device)
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

    def _init_actor_head(self, n_a):
        actor_head = nn.Linear(self.n_h, n_a)
        init_layer(actor_head, 'fc')
        self.actor_heads.append(actor_head)

    def _init_critic_head(self, n_na):
        critic_head = nn.Linear(self.n_h + n_na, 1)
        init_layer(critic_head, 'fc')
        self.critic_heads.append(critic_head)

    def _init_net(self):
        self.fc_x_layers = nn.ModuleDict()
        self.fc_p_layers = nn.ModuleList()
        self.fc_m_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
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
            adj_matrix = adj_matrix + torch.eye(adj_matrix.size(0), dtype=torch.float32)
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
            logging.warning(f"[{self.name}] GAT is configured (use_gat=True) but GATConv module is not available. GAT layers will not be initialized.")


        for i in range(self.n_agent):
            n_n, n_ns, n_na, ns_ls, na_ls = self._get_neighbor_dim(i)
            self.ns_ls_ls.append(ns_ls)
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            if n_n:
                fc_p_layer = nn.Linear(n_na, self.n_fc)
                init_layer(fc_p_layer, 'fc')
                fc_m_layer = nn.Linear(self.n_h * n_n, self.n_fc)
                init_layer(fc_m_layer, 'fc')
                self.fc_m_layers.append(fc_m_layer)
                self.fc_p_layers.append(fc_p_layer)
            else:
                self.fc_m_layers.append(None)
                self.fc_p_layers.append(None)

            n_a = self.n_a if self.identical else self.n_a_ls[i]
            self._init_actor_head(n_a)
            self._init_critic_head(n_na)

            ppo_v = nn.Linear(self.n_h, 1)
            init_layer(ppo_v, 'fc')
            self.ppo_value_heads.append(ppo_v)

        if list(self.parameters()):
            self.device = next(self.parameters()).device
        else:
            logging.warning(f"[{self.name}] cannot infer device in _init_net.")

    def _reset(self):
        if not hasattr(self, 'device') or self.device is None:
            logging.error(f"Device not properly set for NCMultiAgentPolicy instance {self.name} during _reset. Attempting to infer or defaulting to CPU.")
            try:
                current_device = next(self.parameters()).device
                logging.info(f"Inferred device {current_device} for {self.name} in _reset.")
                self.device = current_device
            except StopIteration:
                logging.error(f"Could not infer device for {self.name} from parameters in _reset. Defaulting to CPU.")
                current_device = torch.device("cpu")
                self.device = current_device
        else:
            current_device = self.device

        self.states_fw = torch.zeros(self.n_agent, self.n_h * 2, device=current_device)
        self.states_bw = torch.zeros(self.n_agent, self.n_h * 2, device=current_device)

    def _run_actor_heads(self, hs, detach=False):
        logits_list = []
        for i in range(self.n_agent):
            logits_i = self.actor_heads[i](hs[i])
            if detach:
                logits_list.append(logits_i.cpu().detach().numpy())
            else:
                logits_list.append(logits_i)
        return logits_list

    def _compute_s_features_flat(self, obs_T_N_Do, fps_T_N_Dfp, h_for_comm_N_H):
        T, N, _ = obs_T_N_Do.shape
        device = obs_T_N_Do.device

        s_sequence_pre_gat_list = []
        for t in range(T):
            obs_t = obs_T_N_Do[t]
            fps_t = fps_T_N_Dfp[t]

            s_list_for_gat_t = []
            for i in range(N):
                s_agent_i_t = self._get_comm_s(i, self.n_n_ls[i], obs_t, h_for_comm_N_H, fps_t)
                s_list_for_gat_t.append(s_agent_i_t.squeeze(0))
            s_all_raw_t = torch.stack(s_list_for_gat_t, dim=0)
            s_sequence_pre_gat_list.append(s_all_raw_t)

        s_sequence_pre_gat = torch.stack(s_sequence_pre_gat_list, dim=0)
        return s_sequence_pre_gat.reshape(T * N, -1)

    def _apply_gat(self, s_flat_TN_D):
        s_block_input = s_flat_TN_D
        # Patch: Update GAT condition
        if self.use_gat and self.gat_layer is not None:
            if self.use_layer_norm and hasattr(self, 'pre_gat_ln'):
                s_input_for_gat = self.pre_gat_ln(s_block_input)
            else:
                s_input_for_gat = s_block_input

            num_nodes_per_graph = self.n_agent
            num_graphs = s_flat_TN_D.shape[0] // num_nodes_per_graph

            if s_flat_TN_D.shape[0] % num_nodes_per_graph != 0:
                raise ValueError("s_flat_TN_D cannot be reshaped to (T,N,D) for batched GAT processing.")

            edge_index_list = []
            for i in range(num_graphs):
                edge_index_list.append(self.edge_index + i * num_nodes_per_graph)
            batched_edge_index = torch.cat(edge_index_list, dim=1).to(s_input_for_gat.device)

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

    def _run_comm_layers(self, obs_T_N_Do, dones_T_N, fps_T_N_Dfp, initial_states_N_2H):
        T, N, _ = obs_T_N_Do.shape
        device = obs_T_N_Do.device

        h0, c0 = torch.chunk(initial_states_N_2H.to(device), 2, dim=1)

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
        if key not in self.fc_x_layers:
            logging.info(f"Creating fc_x layer: {key} (in={n_ns})")
            layer = nn.Linear(n_ns, self.n_fc)
            init_layer(layer, 'fc')
            layer = layer.to(self.zero_pad.device)
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
