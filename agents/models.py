"""
/agents/models.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from agents.utils import OnPolicyBuffer, MultiAgentOnPolicyBuffer, Scheduler
from agents.policies import (LstmPolicy, FPPolicy, ConsensusPolicy, NCMultiAgentPolicy,
                             CommNetMultiAgentPolicy, DIALMultiAgentPolicy, NCLMMultiAgentPolicy)
import logging
import numpy as np


class IA2C:
    """
    The basic IA2C implementation with decentralized actor and centralized critic,
    limited to neighborhood area only.
    """
    # Decentralized: each agent decides its action based on its local observation
    # Critics are centralized but limited to a neighborhood area; each agent's critic can access information from neighboring agents to estimate the value function.
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=True):
        self.name = 'ia2c'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def add_transition(self, ob, naction, action, reward,
                       value, done, log_prob, lstm_state_rollout):
        if self.reward_norm > 0:
            reward = reward / self.reward_norm
        if self.reward_clip > 0:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        for i in range(self.n_agent):
            self.trans_buffer[i].add_transition(
                ob[i],                  # local observation
                naction[i],             # neighbor action (fingerprint) if any
                action[i],              # chosen action
                reward if np.isscalar(reward) else reward[i], # reward can be scalar (global) or per-agent
                value[i],               # V_old(s_t)
                done if np.isscalar(done) else done[i] # done can be scalar (global) or per-agent
            )

    def backward(self, Rends, dt, summary_writer=None, global_step=None):
        self.optimizer.zero_grad()
        losses_to_return = None
        for i in range(self.n_agent):
            obs, nas, acts, dones, Rs, Advs = self.trans_buffer[i].sample_transition(Rends[i], dt)
            # call backward and save agent0 losses
            current_losses = self.policy[i].backward(
                obs, nas, acts, dones, Rs, Advs,
                self.e_coef, self.v_coef,
                summary_writer=summary_writer if i == 0 else None,
                global_step=global_step if i == 0 else None
            )
            if i == 0:
                losses_to_return = current_losses
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.lr_decay != 'constant':
            self._update_lr(global_step)  # pass global_step
        return losses_to_return

    def forward(self, obs, done, nactions=None, out_type='p'):
        out = []
        if nactions is None:
            nactions = [None] * self.n_agent
        for i in range(self.n_agent): 
            cur_out = self.policy[i](obs[i], done, nactions[i], out_type)
            out.append(cur_out)
        return out

    def load(self, model_dir, global_step=None, train_mode=True):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if global_step is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        tokens = file.split('.')[0].split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = file
                            save_step = cur_step
            else:
                save_file = 'checkpoint-{:d}.pt'.format(global_step)
        if save_file is not None:
            file_path = model_dir + save_file
            checkpoint = torch.load(file_path)
            logging.info('Checkpoint loaded: {}'.format(file_path))
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            if train_mode:
                #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.policy.train()
            else:
                self.policy.eval()
            return True
        logging.error('Can not find checkpoint for {}'.format(model_dir))
        return False

    def reset(self):
        for i in range(self.n_agent):
            self.policy[i]._reset()

    def save(self, model_dir, global_step):
        file_path = model_dir + 'checkpoint-{:d}.pt'.format(global_step)
        torch.save({'global_step': global_step,
                    'model_state_dict': self.policy.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                    file_path)
        logging.info('Checkpoint saved: {}'.format(file_path))

    def _init_algo(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                   total_step, seed, use_gpu, model_config):
        # init params
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.identical_agent = False
        #基本上所有的Agents自己都有一個action space, 意指所有的actions都在這個集合裡面，假如size都一樣，代表全部都是一樣的action
        #這樣所有agents就可以分享同一組參數，加快學習。
        if (max(self.n_a_ls) == min(self.n_a_ls)):
            # note for identical IA2C, n_s_ls may have varient dims
            self.identical_agent = True
            self.n_s = n_s_ls[0]
            self.n_a = n_a_ls[0]
        else:
        #如果不一樣代表不是完全一致，全部的action dimension跟state dimension都設成最高的那個
            self.n_s = max(self.n_s_ls)
            self.n_a = max(self.n_a_ls)
        #這會在這會在init_policy中產生差異，基本上如果是identical，就會全員跑進一個比較簡單的LSTM中，如果比較難，則每一個agent
        #會被個別訓練，等等會看到
        self.neighbor_mask = neighbor_mask
        self.n_agent = len(self.neighbor_mask)
        #下面這兩個好像是預處理等等reward要用的東西，兩個都存在model_config裡面是一個key
        #反正就是超參數 哈哈
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        #下面三個很直觀
        self.n_step = model_config.getint('batch_size')
        self.n_fc = model_config.getint('num_fc')
        self.n_lstm = model_config.getint('num_lstm')
        self.model_config = model_config
        # init torch
        logging.debug(torch.version.cuda)
        logging.debug(type(torch))
        #if(use_gpu):
        #    print("USEGPU")
        #if(torch.cuda.is_available()):
        #    print("CUDAISAVAIL")
        if use_gpu and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            self.device = torch.device("cuda:0")
            logging.info('Use gpu for pytorch...')
        else:
            torch.manual_seed(seed)
            torch.set_num_threads(1)
            self.device = torch.device("cpu")
            logging.info('Use cpu for pytorch...')
        #先叫出一個policy，然後把它丟進GPU or CPU去跑 🆒
        self.policy = self._init_policy()
        self.policy.to(self.device)
        # --- 重建 hidden states 在正確裝置 ---
        if hasattr(self.policy, "_reset"):
            self.policy._reset()
        
        # init exp buffer and lr scheduler for training
        if total_step:
            self.total_step = total_step
            #初始化training要的東西
            self._init_train(model_config, distance_mask, coop_gamma)

    def _init_policy(self):
        policy = []
        for i in range(self.n_agent):
            n_n = np.sum(self.neighbor_mask[i])
            if self.identical_agent:
                #很簡單，policy全都一樣就是直接丟進去
                local_policy = LstmPolicy(self.n_s_ls[i], self.n_a_ls[i], n_n, self.n_step,
                                          n_fc=self.n_fc, n_lstm=self.n_lstm, name='{:d}'.format(i))
            else:
                #不一樣就麻煩，我們要先知道na_dim_ls是啥子
                #na_dim_ls是一個list (程式裡面ls結尾就是list)，代表第i個鄰居他的action dimension是多少
                na_dim_ls = []
                #下面這個np.where會從neighbor中找出mask是1的，然後自己組成一個新的tuple
                #然後我們要再從tuple裡面抓出第一個，就可以抓到鄰居。
                #e.g. Suppose self.neighbor_mask[i] = [0, 1, 0, 1, 0]
                #np.where(self.neighbor_mask[i] == 1)  # Returns (array([1, 3]),)
                #np.where(self.neighbor_mask[i] == 1)[0]  # Returns array([1, 3])
                for j in np.where(self.neighbor_mask[i] == 1)[0]:
                    na_dim_ls.append(self.n_a_ls[j]) #這個迴圈就會把mask==1的鄰居的動作的次元load進na_dim_ls這個list中
                local_policy = LstmPolicy(self.n_s_ls[i], self.n_a_ls[i], n_n, self.n_step,
                                          n_fc=self.n_fc, n_lstm=self.n_lstm, name='{:d}'.format(i),
                                          na_dim_ls=na_dim_ls, identical=False) #這邊就會多一個na_dim_ls參數才可以確保lstm知道鄰居的動作次元
                # local_policy.to(self.device)
            policy.append(local_policy) #然後local的load進global的大list中
        return nn.ModuleList(policy) #模組化

    def _init_scheduler(self, model_config):
        # init lr scheduler
        self.lr_init = model_config.getfloat('lr_init')
        self.lr_decay = model_config.get('lr_decay')
        if self.lr_decay == 'constant':
            self.lr_scheduler = Scheduler(self.lr_init, decay=self.lr_decay)
        else:
            lr_min = model_config.getfloat('lr_min')
            self.lr_scheduler = Scheduler(self.lr_init, lr_min, self.total_step, decay=self.lr_decay)

    def _init_train(self, model_config, distance_mask, coop_gamma):
        #coopgamma == 0 -> 注重當前利益
        #coopgamma == 1 -> 注重未來利益
        #distance_mask 存的是每一個agent跟當前agent的距離
        # init lr(learning rate我不知道他爲什麼要簡寫好討厭) scheduler
        self._init_scheduler(model_config)
        # init parameters for grad computation
        self.v_coef = model_config.getfloat('value_coef')
        self.e_coef = model_config.getfloat('entropy_coef')
        self.max_grad_norm = model_config.getfloat('max_grad_norm')
        # init optimizer
        alpha = model_config.getfloat('rmsp_alpha') 
        epsilon = model_config.getfloat('rmsp_epsilon')
        self.optimizer = optim.RMSprop(self.policy.parameters(), self.lr_init, 
                                       eps=epsilon, alpha=alpha)
        # init transition buffer
        gamma = model_config.getfloat('gamma')
        self._init_trans_buffer(gamma, distance_mask, coop_gamma)

    def _init_trans_buffer(self, gamma, distance_mask, coop_gamma):
        self.trans_buffer = []
        for i in range(self.n_agent):
            # init replay buffer
            self.trans_buffer.append(OnPolicyBuffer(gamma, coop_gamma, distance_mask[i], self.n_step))

    def _update_lr(self, global_step):  # add global_step parameter
        # TODO: refactor this using optim.lr_scheduler
        cur_lr = self.lr_scheduler.get(global_step)  # use global_step
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr


class IA2C_FP(IA2C):
    """
    In fingerprint IA2C, neighborhood policies (fingerprints) are also included.
    """
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=True):
        self.name = 'ia2c_fp'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma, 
                        total_step, seed, use_gpu, model_config)

    def _init_policy(self):
        policy = []
        for i in range(self.n_agent):
            n_n = np.sum(self.neighbor_mask[i])
            # neighborhood policies are included in local state
            if self.identical_agent:
                n_s1 = int(self.n_s_ls[i] + self.n_a*n_n)
                policy.append(FPPolicy(n_s1, self.n_a, int(n_n), self.n_step, n_fc=self.n_fc,
                                       n_lstm=self.n_lstm, name='{:d}'.format(i)))
            else:
                na_dim_ls = []
                for j in np.where(self.neighbor_mask[i] == 1)[0]:
                    na_dim_ls.append(self.n_a_ls[j])
                n_s1 = int(self.n_s_ls[i] + sum(na_dim_ls))
                policy.append(FPPolicy(n_s1, self.n_a_ls[i], int(n_n), self.n_step, n_fc=self.n_fc,
                                       n_lstm=self.n_lstm, name='{:d}'.format(i),
                                       na_dim_ls=na_dim_ls, identical=False))
        return nn.ModuleList(policy)


class MA2C_NC(IA2C):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0,  use_gpu=True):
        logging.debug("DEBUG: Initializing MA2C_NC model.")
        self.name = 'ma2c_nc'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def add_transition(self, ob, p, action, reward, value, done, log_prob, lstm_state_rollout):  # added log_prob and lstm_state_rollout
        if self.reward_norm > 0:
            reward = reward / self.reward_norm
        if self.reward_clip > 0:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)

        if self.identical_agent:
            self.trans_buffer.add_transition(
                np.array(ob),            # observations for all agents
                np.array(p),             # policy metadata (e.g. fingerprints)
                np.array(action),        # actions chosen
                np.array(reward),        # rewards
                np.array(value),         # V_old(s_t)
                done,                    # done flag
                np.array(log_prob),      # log_prob_old(a_t|s_t)
                lstm_state_rollout       # new: lstm_state_rollout
            )
        else:
            pad_ob, pad_p = self._convert_hetero_states(ob, p)
            self.trans_buffer.add_transition(
                pad_ob,
                pad_p,
                np.array(action),
                np.array(reward),
                np.array(value),
                done,
                np.array(log_prob),
                lstm_state_rollout        # new: lstm_state_rollout
            )

    def backward(self, Rends, dt, summary_writer=None, global_step=None):
        self.optimizer.zero_grad()
        # sample_transition returns: (obs, actions, old_logps, dones, returns,
        #                        advantages, old_values, fps, lstm_states)
        # Corrected unpacking:
        obs, acts, _, dones, Rs, Advs, _, fps, _ = \
            self.trans_buffer.sample_transition(Rends, dt)
        
        # The call to policy.backward uses the correctly unpacked 'fps' and 'acts'.
        # policy.backward expects (obs, fps, acts, dones, Rs, Advs, ...)
        policy_loss, value_loss, entropy_loss, total_loss = \
            self.policy.backward(obs, fps, acts, dones, Rs, Advs,
                                 self.e_coef, self.v_coef,
                                 summary_writer=summary_writer, global_step=global_step)
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.lr_decay != 'constant':
            self._update_lr(global_step)  # pass global_step
        return policy_loss, value_loss, entropy_loss, total_loss

    def forward(self, obs, done, ps, actions=None, out_type='p'):

        if self.identical_agent:
      
            return self.policy.forward(np.array(obs), done, np.array(ps),
                                       actions, out_type)
        else:
            pad_ob, pad_p = self._convert_hetero_states(obs, ps)
        
            return self.policy.forward(pad_ob, done, pad_p,
                                       actions, out_type)

    def reset(self):
        self.policy._reset()

    def _convert_hetero_states(self, ob, p):
        pad_ob = np.zeros((self.n_agent, self.n_s))
        pad_p = np.zeros((self.n_agent, self.n_a))
        for i in range(self.n_agent):
            pad_ob[i, :len(ob[i])] = ob[i]
            pad_p[i, :len(p[i])] = p[i]
        return pad_ob, pad_p

    def _init_policy(self):
        if self.identical_agent:
            return NCMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                      self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                      model_config=self.model_config)
        else:
            return NCMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                      self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                      n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False,
                                      model_config=self.model_config)

    def _init_trans_buffer(self, gamma, distance_mask, coop_gamma):
        self.trans_buffer = MultiAgentOnPolicyBuffer(gamma, coop_gamma, distance_mask)


class MA2C_NCLM(MA2C_NC):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, groups=0, seed=0, use_gpu=True):
        self.name = 'ma2c_nclm'
        self.groups = groups
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def _init_policy(self):
        if self.identical_agent:
            return NCLMMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                      self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm, groups=self.groups,
                                      model_config=self.model_config)
        else:
            return NCLMMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                      self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                      n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, groups=self.groups, identical=False,
                                      model_config=self.model_config)

class IA2C_CU(MA2C_NC):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=True):
        self.name = 'ma2c_cu'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def _init_policy(self):
        if self.identical_agent:
            return ConsensusPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                   self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                   model_config=self.model_config)
        else:
            return ConsensusPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                   self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                   n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False,
                                   model_config=self.model_config)

    def backward(self, Rends, dt, summary_writer=None, global_step=None):
        super(IA2C_CU, self).backward(Rends, dt, summary_writer, global_step)
        self.policy.consensus_update()


class MA2C_CNET(MA2C_NC):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=True):
        self.name = 'ma2c_ic3'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def _init_policy(self):
        if self.identical_agent:
            return CommNetMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                           self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                           model_config=self.model_config)
        else:
            return CommNetMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                           self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                           n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False,
                                           model_config=self.model_config)


class MA2C_DIAL(MA2C_NC):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=True):
        self.name = 'ma2c_dial'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def _init_policy(self):
        if self.identical_agent:
            return DIALMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                        self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                        model_config=self.model_config)
        else:
            return DIALMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                        self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                        n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False,
                                        model_config=self.model_config)


class MA2PPO_NC(MA2C_NC):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask,
                 coop_gamma, total_step, model_config, seed=0, use_gpu=True):
        super().__init__(n_s_ls, n_a_ls, neighbor_mask, distance_mask,
                         coop_gamma, total_step, model_config, seed, use_gpu)
        self.name = 'mappo_nc'
        logging.info(f"Initializing {self.name} model.")

        # PPO-specific hyperparams
        self.gae_lambda       = model_config.getfloat('gae_lambda', 0.95)
        self.ppo_epochs       = model_config.getint('ppo_epochs', 10)
        self.num_minibatches  = model_config.getint('num_minibatches', 4)
        self.clip_epsilon     = model_config.getfloat('clip_epsilon', 0.2)
        # Default value_clip_param to 0.2 (enable) if not in config (Fatal Error #5)
        self.value_clip_param = model_config.getfloat('value_clip_param', 0.2)
        self.normalize_advantage = model_config.getboolean('normalize_advantage', True)

        # re-create multi-agent buffer with GAE lambda
        gamma = model_config.getfloat('gamma')
        self.trans_buffer = MultiAgentOnPolicyBuffer(gamma, coop_gamma, distance_mask, self.gae_lambda)
        logging.info(f"[{self.name}] buffer re-init with GAE λ={self.gae_lambda}")

    def update(self, R_bootstrap_agents, dt, summary_writer=None, global_step=None):
        # sample rollout from buffer (now returns fps and lstm_states)
        # Expects (T, N, ...) format from sample_transition
        obs, actions, old_logps, dones, returns, advs, old_values, fps, lstm_states = \
            self.trans_buffer.sample_transition(R_bootstrap_agents, dt)

        device = self.policy.device
        
        # --- handle empty rollout ---
        if obs.shape[0] == 0: # Check if T (first dimension) is 0
            logging.warning(f"[{self.name}] Empty rollout buffer at step {global_step}, skipping update.")
            return 0.0, 0.0, 0.0, 0.0

        # obs is (T, N, obs_dim), actions is (T,N) etc.
        T = obs.shape[0] 
        # n_agents = obs.shape[1] # Not directly used here, but good to note

        # --- compute minibatch size safely (Fatal Error #4 fix) ---
        # ensure positive num_minibatches
        actual_mb = self.num_minibatches if self.num_minibatches > 0 else 1
        # if not enough steps, use one sample per batch
        if T < actual_mb:
            logging.warning(f"[{self.name}] T={T} < num_minibatches={actual_mb}, forcing mb_size=1.")
            mb_size = 1
        else:
            assert T % actual_mb == 0, \
                f"[{self.name}] T={T} not divisible by num_minibatches={actual_mb}."
            mb_size = T // actual_mb
        # final guard
        if mb_size == 0:
            mb_size = 1
            logging.warning(f"[{self.name}] mb_size computed as 0; forcing mb_size=1.")

        # Patch-03: Remove transpositions. Data from buffer is already (T, N, ...)
        obs_tm   = torch.from_numpy(obs).to(device).float()
        fps_tm   = torch.from_numpy(fps).to(device).float()
        act_tm   = torch.from_numpy(actions).to(device).long()
        oldlp_tm = torch.from_numpy(old_logps).to(device).float()
        ret_tm   = torch.from_numpy(returns).to(device).float()
        adv_tm   = torch.from_numpy(advs).to(device).float()
        val_tm   = torch.from_numpy(old_values).to(device).float()


        if self.normalize_advantage:
            adv_tm = (adv_tm - adv_tm.mean()) / (adv_tm.std() + 1e-8)

        total_steps = T
        sum_pl = sum_vl = sum_el = updates = 0

        # ---------- freeze dropout but 保留 train() ----------
        gat_orig_p  = None
        lstm_orig_p = None
        if getattr(self.policy, 'gat_layer', None) is not None:
            gat_orig_p = float(self.policy.gat_layer.dropout)
            self.policy.gat_layer.dropout = 0.0
        if getattr(self.policy, 'lstm_layer', None) is not None:
            lstm_orig_p = float(self.policy.lstm_layer.dropout)
            self.policy.lstm_layer.dropout = 0.0

        # ----- ❶ 進入確定性前向 -----
        was_training = self.policy.training      # 保留，但不切 eval()

        for _ in range(self.ppo_epochs):
            idx_perm = torch.randperm(total_steps, device=device)
            for start in range(0, total_steps, mb_size):
                mb_idx = idx_perm[start:start+mb_size]
                mb_obs    = obs_tm[mb_idx]    # (mb_size, N, obs_dim)
                mb_fps    = fps_tm[mb_idx]    # (mb_size, N, fp_dim)
                mb_act    = act_tm[mb_idx]
                mb_oldlp  = oldlp_tm[mb_idx]
                mb_ret    = ret_tm[mb_idx]
                mb_adv    = adv_tm[mb_idx]
                mb_val    = val_tm[mb_idx]
                

                # Vectorized call to policy evaluation
                # Pass mb_fps, lstm_states defaults to None
                newlp, newv, ent, _ = self.policy.evaluate_actions_values_and_entropy(
                    mb_obs,          # (mb_size, N, obs_dim)
                    mb_fps,          # (mb_size, N, fp_dim)
                    mb_act           # (mb_size, N)
                )
                # newlp, newv, ent are expected to be (mb_size, N)
                
                ratio = torch.exp(newlp - mb_oldlp.detach())
                surr1 = ratio * mb_adv.detach()
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * mb_adv.detach()
                ploss = -torch.min(surr1, surr2).mean()

                if self.value_clip_param>0:
                    v_clipped = mb_val.detach() + torch.clamp(newv - mb_val.detach(),
                                                             -self.value_clip_param,
                                                             self.value_clip_param)
                    vf1 = (newv - mb_ret.detach()).pow(2)
                    vf2 = (v_clipped - mb_ret.detach()).pow(2)
                    vloss = 0.5 * torch.max(vf1, vf2).mean()
                else:
                    vloss = 0.5 * (newv - mb_ret.detach()).pow(2).mean()

                eloss = -ent.mean()
                loss = ploss + self.v_coef*vloss + self.e_coef*eloss

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm>0:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                sum_pl += ploss.item()
                sum_vl += vloss.item()
                sum_el += eloss.item()
                updates += 1

        # ---------- 恢復原來的 dropout ----------
        if gat_orig_p is not None and getattr(self.policy, 'gat_layer', None) is not None:
            self.policy.gat_layer.dropout = gat_orig_p
        if lstm_orig_p is not None and getattr(self.policy, 'lstm_layer', None) is not None:
            self.policy.lstm_layer.dropout = lstm_orig_p

        if self.lr_decay!='constant' and global_step is not None:
            self._update_lr(global_step)

        if updates>0 and summary_writer and global_step is not None:
            avg_pl = sum_pl/updates
            avg_vl = sum_vl/updates
            avg_el = sum_el/updates
            avg_total = avg_pl + self.v_coef*avg_vl + self.e_coef*avg_el
            summary_writer.add_scalar(f'{self.name}/ppo_actor_loss', avg_pl, global_step)
            summary_writer.add_scalar(f'{self.name}/ppo_critic_loss', avg_vl, global_step)
            summary_writer.add_scalar(f'{self.name}/ppo_entropy_loss', avg_el, global_step)
            summary_writer.add_scalar(f'{self.name}/ppo_total_loss', avg_total, global_step)
            return avg_pl, avg_vl, avg_el, avg_total
        return 0.0,0.0,0.0,0.0
