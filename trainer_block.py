class Trainer():
    #env可以去看.ini檔，裡面就有一整個叫做ENV_CONFIG的地方
    def __init__(self, env, model, global_counter, summary_writer, output_path=None):
        self.cur_step = 0
        self.global_counter = global_counter
        self.env = env
        self.agent = self.env.agent
        self.model = model
        self.n_step = self.model.n_step
        self.summary_writer = summary_writer
        assert self.env.T % self.n_step == 0
        self.data = []   
        self.output_path = output_path
        self.model_dir = output_path.replace('data/', 'model/')
        self.env.train_mode = True
        self.pq = MyQueue(self.model_dir, maxsize=5)
        self.train_results = []
        # read GAT dropout schedule from model_config
        if hasattr(self.model, 'model_config') and self.model.model_config:
            self.gat_dropout_init = self.model.model_config.getfloat('gat_dropout_init', 0.2)
            self.gat_dropout_final = self.model.model_config.getfloat('gat_dropout_final', 0.1)
            self.gat_dropout_decay_steps = self.model.model_config.getfloat('gat_dropout_decay_steps', 500000)
        else:
            logging.warning("GAT dropout config not found in model.model_config. Using defaults (0.2→0.1 over 500k steps).")
            self.gat_dropout_init = 0.2
            self.gat_dropout_final = 0.1
            self.gat_dropout_decay_steps = 500000

    def _log_episode(self, global_step, mean_reward, std_reward, env_stats):
        """
        Log episode-level statistics to console / TensorBoard.

        Args
        ----
        global_step : int              current environment step
        mean_reward : float | np.array mean episode reward
        std_reward  : float | np.array std of episode reward
        env_stats   : dict             any extra stats collected from env
        """
        if hasattr(self, 'summary_writer') and self.summary_writer is not None:
            self.summary_writer.add_scalar('episode/mean_reward', mean_reward, global_step)
            self.summary_writer.add_scalar('episode/std_reward',  std_reward,  global_step)
            # write additional environment stats if provided
            if isinstance(env_stats, dict):
                for k, v in env_stats.items():
                    self.summary_writer.add_scalar(f'episode/{k}', v, global_step)

        # 可選：印到 console 或 logging
        logging.info(f"[Ep @ step {global_step}]  "
                     f"meanR={mean_reward:.3f} ± {std_reward:.3f}  "
                     + " ".join([f"{k}={v:.3f}" for k, v in (env_stats or {}).items()]))

    def _add_summary(self, reward, global_step, is_train=True):
        if is_train:
            self.summary_writer.add_scalar('train_reward', reward, global_step=global_step)
        else:
            self.summary_writer.add_scalar('test_reward', reward, global_step=global_step)
            
    def _add_arrived(self, arrived, global_step, is_train=True):
        if is_train:
            self.summary_writer.add_scalar('arrived', arrived, global_step=global_step)
        else:
            self.summary_writer.add_scalar('arrived', arrived, global_step=global_step)

    def max_min(self):
        try:
            start = min(0, len(self.train_results) - 100)
            r = self.train_results[start:len(self.train_results)]
            return max(r) + min(r)
        except ValueError:
            return 0

    def _get_policy(self, ob, done, prev_actions=None, mode='train'):
        # prepare outputs
        n = self.env.n_agent
        fps = self.env.get_fingerprint()  # get current fingerprint
        actions_np  = np.zeros(n, dtype=np.int32)
        logp_np     = np.zeros(n, dtype=np.float32)
        values_np   = np.zeros(n, dtype=np.float32)
        env_probs   = [None]*n

        if self.agent.startswith('ma2c'):
            logits_list, value_list, probs_list = self.model.forward(ob, done, fps, prev_actions)
            for i in range(n):
                dist = torch.distributions.Categorical(logits=logits_list[i])
                if mode=='train':
                    a_i = dist.sample()
                else:
                    a_i = torch.argmax(logits_list[i], dim=-1)
                actions_np[i]   = a_i.item()
                logp_np[i]      = dist.log_prob(a_i).item()
                values_np[i]    = value_list[i].item()
                env_probs[i]    = probs_list[i].squeeze().cpu().detach().numpy()
        elif self.agent.startswith('mappo'):
            logits_list, value_list, probs_list = self.model.forward(ob, done, fps, prev_actions)
            for i in range(n):
                dist = torch.distributions.Categorical(logits=logits_list[i])
                if mode == 'train':
                    a_i = dist.sample()
                else:
                    a_i = torch.argmax(logits_list[i], dim=-1)
                actions_np[i] = a_i.item()
                logp_np[i]   = dist.log_prob(a_i).item()
                values_np[i] = value_list[i].item()
                env_probs[i] = probs_list[i].squeeze().cpu().detach().numpy()
        elif self.agent=='greedy':
            a = self.model.forward(ob)
            actions_np[:] = np.array(a)
        else:
            raise NotImplementedError(self.agent)

        return fps, env_probs, actions_np, logp_np, values_np


    def explore(self, prev_ob, prev_done):
        ob, done = prev_ob, prev_done
        prev_actions = np.zeros(self.env.n_agent, dtype=np.int64)
        for _ in range(self.n_step):
            # get rollout‐start LSTM state before policy.forward updates it
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'states_fw'):
                lstm_state = self.model.policy.states_fw.cpu().numpy()
            else:
                lstm_state = np.zeros((self.env.n_agent, self.model.policy.n_h * 2))
            fps, env_probs, actions_np, logp_np, values_np = self._get_policy(ob, done, prev_actions, mode='train')
            self.env.update_fingerprint(env_probs)
            next_ob, reward, done, global_reward = self.env.step(actions_np)
            self.episode_rewards.append(global_reward)
            gs = self.global_counter.next()
            self.cur_step += 1

            # pass lstm_state into add_transition
            self.model.add_transition(
                ob, fps, actions_np, reward, values_np, done, logp_np, lstm_state
            )
            prev_actions = actions_np.copy()

            # logging
            self.summary_writer.add_scalar('reward/global_reward', global_reward, gs)
            self.summary_writer.add_scalar('reward/mean_agent_reward', np.mean(reward), gs)
            if self.global_counter.should_log():
                logging.info(f"Step {gs} | r {global_reward:.2f} | a {actions_np} | logp {np.round(logp_np,3)} | v {np.round(values_np,3)} | done {done}")

            if done:
                break
            ob = next_ob

        # bootstrap value for unfinished episodes
        if done:
            Rb = np.zeros(self.env.n_agent)
        else:
            _, _, _, _, Rb = self._get_policy(ob, done, prev_actions, mode='train')
        return ob, done, Rb

    def perform(self, test_ind, gui=False):
        ob = self.env.reset(gui=gui, test_ind=test_ind)
        rewards = []
        # note this done is pre-decision to reset LSTM states!
        done = True
        self.model.reset()
        prev_actions = np.zeros(self.env.n_agent, dtype=np.int64)
        while True:
            if self.agent == 'greedy':
                action = self.model.forward(ob)
            else:
                if self.env.name.startswith('atsc'):
                    policy, action = self._get_policy(ob, done, prev_actions)
                else:
                    policy, action = self._get_policy(ob, done, prev_actions, mode='test')
                self.env.update_fingerprint(policy)
            next_ob, reward, done, global_reward = self.env.step(action)
            rewards.append(global_reward)
            prev_actions = action.copy() if isinstance(action, np.ndarray) else np.array(action)
            if done:
                break
            ob = next_ob
        mean_reward = np.mean(np.array(global_reward))
        std_reward = np.std(np.array(global_reward))
        # mean_reward = np.mean(np.array(rewards))
        # std_reward = np.std(np.array(rewards))
        return mean_reward, std_reward

    #run是最重要的method，他跟環境互動，取得結果，更新模型，然後再登錄到紀錄版上
    def run(self):
        count = 0;
        #這個迴圈會在 global_step 達到預設的最大訓練步數停下
        while not self.global_counter.should_stop():
            np.random.seed(self.env.seed)
            logging.debug(f"round {count}")
            count = count + 1
            #在reset環境的同時也回傳一個observation當作最初的observation
            ob = self.env.reset()
            # note this done is pre-decision to reset LSTM states! <- 會把lstm的internal state初始化
            done = True
            
            self.model.reset()
            self.cur_step = 0 #此episode的step設為0 
            self.episode_rewards = [] #Reward會被存在這個list中
            # ensure env_stats exists even if try block continues or errors
            env_stats = {}
            try:
                while True: #這個while是以episode為單位，一個episode就是一次模擬從頭到尾的過程
                    # --- dynamic GAT dropout update Start ---
                    current_step = self.global_counter.cur_step
                    total_decay_steps = self.gat_dropout_decay_steps
                    init_val = self.gat_dropout_init
                    final_val = self.gat_dropout_final

                    if current_step < total_decay_steps:
                        cosine_decay = 0.5 * (1 + np.cos(np.pi * current_step / total_decay_steps))
                        current_gat_dropout = final_val + (init_val - final_val) * cosine_decay
                    else:
                        current_gat_dropout = final_val

                    current_gat_dropout = max(final_val, float(current_gat_dropout))  # ensure float and not below final_val

                    # 更新 PyG GATConv 層的 dropout 率
                    if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'gat_layer'):
                        gat_conv_layer = self.model.policy.gat_layer
                        new_dropout_rate = float(current_gat_dropout)

                        # 優先嘗試設置 dropout_p (PyG >= 2.0.3 常用)
                        if hasattr(gat_conv_layer, 'dropout_p'):
                            gat_conv_layer.dropout_p = new_dropout_rate
                        elif hasattr(gat_conv_layer, 'dropout'): # 兼容舊版或某些實現的 .dropout 屬性
                            # GATConv 初始化時的 dropout 參數本身就是 p，內部可能存為 self.dropout
                            gat_conv_layer.dropout = new_dropout_rate 
                        else:
                            # 作為最後手段，嘗試修改內部 _dropout (不推薦，但為了覆蓋舊版本)
                            # 注意：PyG 不同版本內部實現可能有差異，直接訪問內部變數 _dropout 風險較高
                            try:
                                # PyG GATConv stores dropout probability as self.dropout, which is then used by F.dropout
                                # If neither dropout_p nor dropout (as a direct setter) works,
                                # this attempts to modify the stored probability.
                                # For many PyG versions, self.dropout is the parameter passed to F.dropout.
                                gat_conv_layer._dropout = new_dropout_rate 
                                logging.debug(f"Set GATConv internal _dropout to {new_dropout_rate}")
                            except AttributeError:
                                logging.warning(f"Could not set dropout for GATConv layer {type(gat_conv_layer)}. Tried 'dropout_p', 'dropout', and '_dropout'.")
                                
                    if self.summary_writer and (current_step % self.global_counter.log_step == 0):
                        self.summary_writer.add_scalar('train/gat_dropout', current_gat_dropout, current_step)

                        # --- *** 新增 Console Log *** ---
                        current_lr = 0.0
                        if hasattr(self.model, 'optimizer') and self.model.optimizer:
                            current_lr = self.model.optimizer.param_groups[0]['lr']
                        logging.info(f"Step: {current_step:<8} | GAT Dropout: {current_gat_dropout:.6f} | LR: {current_lr:.6e}")
                        # --- *** 新增 Console Log 結束 *** ---

                    # --- dynamic GAT dropout update End ---

                    #ob是下一階段的observation，R是這個observation他所產出的action得到的實際分數
                    #在explore裡面還有一個for迴圈，要跑n_step次，n_step就是batch_size，我們可以先定義
                    #整體流程是這樣，explore每次會得到一個批次的結果，然後根據這個批次的結果再調整，直到做到這次模擬結束，然後這一切要做到globa_step結束
                    ob, done, R = self.explore(ob, done) #Explore method主要是要主要是要與環境互動然後得到...環境的結果, 可跳轉到explore method
                    #dt是剩下的time steps
                    dt = self.env.T - self.cur_step
                    global_step = self.global_counter.cur_step
                    logging.debug(f"step: {global_step}")

                    # choose update vs backward
                    if hasattr(self.model, 'update') and callable(self.model.update):
                        losses_tuple = self.model.update(R, dt,
                            summary_writer=self.summary_writer,
                            global_step=global_step)
                    elif hasattr(self.model, 'backward') and callable(self.model.backward):
                        losses_tuple = self.model.backward(R, dt,
                            summary_writer=self.summary_writer,
                            global_step=global_step)
                    else:
                        logging.error(f"Model {type(self.model)} has neither 'update' nor 'backward' method.")
                        losses_tuple = None

                    if losses_tuple and len(losses_tuple) == 4:
                        p_loss, v_loss, e_loss, tot_loss = losses_tuple
                        # handle tensor vs float
                        pl = p_loss.item() if hasattr(p_loss, 'item') else p_loss
                        vl = v_loss.item() if hasattr(v_loss, 'item') else v_loss
                        el = e_loss.item() if hasattr(e_loss, 'item') else e_loss
                        tl = tot_loss.item() if hasattr(tot_loss, 'item') else tot_loss
                        self.summary_writer.add_scalar('loss/policy', pl, global_step)
                        self.summary_writer.add_scalar('loss/value', vl, global_step)
                        self.summary_writer.add_scalar('loss/entropy', el, global_step)
                        self.summary_writer.add_scalar('loss/total', tl, global_step)
                        # learning rate
                        if hasattr(self.model, 'optimizer') and self.model.optimizer:
                            self.summary_writer.add_scalar(
                                'train/learning_rate',
                                self.model.optimizer.param_groups[0]['lr'],
                                global_step)
                        # gradient norm
                        total_norm = 0
                        for p in self.model.policy.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        self.summary_writer.add_scalar('train/grad_norm', total_norm, global_step)
                        # flush occasionally
                        if global_step % 50 == 0:
                            self.summary_writer.flush()
                    else:
                        logging.error(f"Step {global_step}: model.backward() returned {len(losses_tuple)} values, expected 4 for current test.")

                    # termination
                    if done:
                        self.env.terminate()
                        time.sleep(1)
                        # collect environment statistics for this episode
                        env_stats = self.env.get_episode_statistics() \
                            if hasattr(self.env, 'get_episode_statistics') else {}
                        break
            except Exception as e:
                logging.exception("An error occurred during run()")
                self.global_counter.repair()
                continue
            rewards = np.array(self.episode_rewards)
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            # NOTE: for CACC we have to run another testing episode after each
            # training episode since the reward and policy settings are different!
            if not self.env.name.startswith('atsc'):
                self.env.train_mode = False
                mean_reward, std_reward = self.perform(-1)
                self.env.train_mode = True
            self._log_episode(global_step, mean_reward, std_reward, env_stats)

        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')

