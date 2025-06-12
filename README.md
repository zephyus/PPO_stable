
# Multi-Agent Proximal Policy Optimization with GAT Neighborhood Communication (0519_MAPPO)

## 專案概述

本專案旨在實現並驗證一個基於近端策略優化 (PPO) 的多智能體強化學習演算法（MA2PPO_NC）。該演算法的核心特點是智能體間的鄰域通訊機制，此機制通過圖注意力網路 (GAT) 和循環神經網路 (LSTM) 實現。此外，專案還引入了空間獎勵 (Spatial Rewards) 的概念，並將其整合到廣義優勢估計 (GAE) 中，以促進智能體間的合作行為。

主要組件包括：
* **MA2PPO_NC**: 基於 PPO 的多智能體學習框架。
* **NCMultiAgentPolicy**: 實現鄰域通訊的策略網路，可選用 GAT 進行信息聚合。
* **MultiAgentOnPolicyBuffer**: 支持空間獎勵 GAE 計算的經驗回放緩衝區。
* **SUMO 環境集成**: 通過 `test.py` 和相關環境類 (如 `RealNetEnv`, `SmallGridEnv` 等) 與交通模擬器 SUMO 交互 (儘管具體環境細節未完全在此次提供，但從檔案結構和 SUMO 相關設定可推斷)。

## 先決條件

1.  **Docker**: 用於創建和管理一致的運行環境。
2.  **NVIDIA Docker**: 如果您計劃使用 GPU (`--gpus all` 選項)，則需要安裝 NVIDIA Docker toolkit。
3.  **預構建的 Docker 映像**: `best_environment:latest`。您需要確保此映像已經存在於您的本地 Docker 環境中。如果沒有，您需要根據項目依賴自行構建或獲取此映像。
4.  **SUMO**: 交通模擬軟體。SUMO 需要被正確安裝，並且其路徑需要通過 `SUMO_HOME` 環境變量指定。

## 設定與安裝

以下步驟指導如何在 Docker 容器內設定和運行專案。

### 第 1 步：準備 Docker 映像 (如果尚未準備好)
確保您擁有 `best_environment:latest` Docker 映像。此 README 不包含該映像的構建步驟。

### 第 2 步：啟動 Docker 容器
在您的主機終端中執行以下命令，以在背景啟動一個新的 Docker 容器。請將 `/home/russell512/my_deeprl_network_ori_test_0516_multi_head_PyG_GATCon` 替換為您主機上專案的實際路徑。

```bash
docker run \
    --gpus all \
    -d \
    --name 0519_MAPPO_run \
    -v /home/russell512/my_deeprl_network_ori_test_0516_multi_head_PyG_GATCon:/workspace/my_deeprl_network \
    best_environment:latest \
    sleep infinity
````

  * `--gpus all`: 分配所有可用的 GPU 給容器。如果沒有 GPU 或不想使用，可以移除此行。
  * `-d`: 背景運行。
  * `--name 0519_MAPPO_run`: 為容器命名，方便後續管理。
  * `-v /path/to/your/project:/workspace/my_deeprl_network`: 將您主機上的專案目錄掛載到容器的 `/workspace/my_deeprl_network` 路徑。

### 第 3 步：進入運行的容器

```bash
docker exec -it 0519_MAPPO_run /bin/bash
```

### 第 4 步：在 Docker 容器內安裝依賴 (如果映像中尚未包含)

進入 Docker 容器後，執行以下命令來安裝必要的 Python 套件和工具。

```bash
# 安裝基礎 RL 和 PyTorch 相關套件
pip install traci sumolib torch tensorboard

# 安裝 PyTorch Geometric (PyG) 及其依賴
# 注意：以下指令針對 PyTorch 1.9.0 和 CUDA 11.1。如果您的基礎 PyTorch 版本不同，
# 請參考 PyG 官方文件獲取對應的安裝指令。
pip install torch-geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f [https://data.pyg.org/whl/torch-1.9.0+cu111.html](https://data.pyg.org/whl/torch-1.9.0+cu111.html)

# 設定 SUMO_HOME 環境變量
# 這個路徑取決於您的 Docker 映像中 SUMO 的安裝位置。
# 以下路徑 /root/miniconda/envs/py36/share/sumo 是一個示例，請確認其正確性。
export SUMO_HOME="/root/miniconda/envs/py36/share/sumo"
# 建議將此行加入容器內的 ~/.bashrc 文件，以便永久生效。

# 安裝 tmux (用於長時間運行的會話管理)
apt update
apt install -y tmux
```

**注意**: 如果 `pip install torch` 安裝的 PyTorch 版本與 PyG 依賴的 CUDA 版本不匹配，PyG 可能無法正常工作。請確保 PyTorch 版本與 PyG 的兼容性。

## 目錄結構 (容器內 `/workspace/my_deeprl_network/`)

```
.
├── agents/                 # 包含 RL 智能體模型 (models.py)、策略網路 (policies.py) 和工具 (utils.py for buffers)
├── config/                 # 存放 .ini 配置文件
├── envs/                   # 包含環境定義 (例如 small_grid_env.py)
├── tests/                  # 單元測試 (例如 test_buffer_spatial_gae.py)
├── utils.py                # 專案級別的工具函式和類 (如 Trainer, Counter)
├── test.py                 # 主要的訓練和評估腳本
└── ...                     # 其他專案文件和目錄
```

## 組態設定

實驗的行為主要由 `.ini` 配置文件控制。主要的配置文件示例是 `config/config_mappo_nc_net_exp_0519.ini`。

關鍵配置項：

  * **`[ENV_CONFIG]`**:
      * `agent`: 指定要使用的智能體/模型類型 (例如 `mappo_nc`, `ma2c_nc`)。
      * `coop_gamma`: 合作獎勵因子 $\\alpha$，用於空間獎勵 GAE。`0.0` 表示純自私，`1.0` 表示完全納入（加權的）鄰居獎勵。
      * `scenario`: 環境場景。
      * `seed`: 隨機種子。
  * **`[MODEL_CONFIG]`**:
      * `gamma`: 折扣因子 $\\gamma$。
      * `lr_init`: 初始學習率。
      * `entropy_coef`: 熵獎勵係數。
      * `value_coef`: 價值函數損失係數。
      * `batch_size`: 訓練時的批次大小 (在 PPO 中指 rollout buffer 的大小)。
      * **GAT 相關參數**:
          * `gat_dropout_init`, `gat_dropout_final`, `gat_dropout_decay_steps`: GAT dropout 率的動態調整參數。
          * `gat_num_heads`: GAT 的注意力頭數。
          * `gat_use_layer_norm`, `gat_use_residual`: 是否在 GAT 中使用 Layer Normalization 和殘差連接。
      * **PPO 相關超參數**:
          * `gae_lambda`: GAE 的 $\\lambda$ 參數。
          * `ppo_epochs`: 每個 rollout 數據用於策略更新的 epoch 數。
          * `num_minibatches`: 每個 rollout 數據劃分成的 minibatch 數量。
          * `clip_epsilon`: PPO 的裁剪參數 $\\epsilon$。
  * **`[TRAIN_CONFIG]`**:
      * `total_step`: 總訓練步數。
      * `test_interval`: 測試間隔。
      * `log_interval`: 日誌記錄間隔。

此外，環境變量 `USE_GAT` 可以控制是否在 `NCMultiAgentPolicy` 中啟用 GAT 層（`1` 為啟用，`0` 為禁用，預設為 `1`）。

## 運行實驗

### 訓練 MA2PPO\_NC 模型

以下指令假設您已經在 Docker 容器內部，並且位於 `/workspace/my_deeprl_network/` 目錄。

1.  **確保 `PYTHONPATH` (可選，通常在 Docker 中掛載正確即可)**
    如果 Python 無法找到 `agents` 等套件，可以執行：

    ```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    ```

2.  **啟動 `tmux` 會話 (建議用於長時間訓練)**

    ```bash
    tmux new -s training_0519_mappo
    ```

3.  **設置實驗目錄並運行訓練腳本**
    將 `${BASE_DIR_NAME}` 替換為您希望儲存日誌和模型的相對路徑，例如 `real_a1/0519_MAPPO_run`。

    ```bash
    export BASE_DIR_NAME="real_a1/0519_MAPPO_run" # 根據您的實驗命名
    mkdir -p ${BASE_DIR_NAME}/log
    export USE_GAT=1 # 明確啟用 GAT (如果需要)

    python3 test.py \
        --base-dir ${BASE_DIR_NAME} \
        --port 202 \
        train \
        --config-dir config/config_mappo_nc_net_exp_0519.ini \
        > ${BASE_DIR_NAME}/log/0519_MAPPO_run_$(date +%Y%m%d_%H%M%S).log 2>&1
    ```

      * `--base-dir`: 指定實驗結果的根目錄（日誌、模型、數據等將存儲在此目錄下）。
      * `--port`: SUMO 運行的端口號。
      * `train`: 指定執行訓練模式。
      * `--config-dir`: 指定所使用的配置文件。
      * 輸出會重定向到日誌文件。

4.  **從 `tmux` 會話分離**
    按下 `Ctrl + b` 組合鍵，放開，然後馬上再按 `d` 鍵。訓練會在背景繼續運行。

5.  **退出 Docker 容器**
    在容器的命令提示符下，輸入 `exit` 並按 Enter。

### 評估模型 (示例)

`test.py` 腳本也支持評估模式，您可以通過 `python3 test.py evaluate --help` 查看具體參數。

## 監控訓練

您可以使用 TensorBoard 來監控訓練過程中的指標（如獎勵、損失函數等）。

在您的**主機 (非 Docker 容器)** 上打開一個新的終端，並執行：

```bash
# 監控 real_a1 目錄下的所有實驗
tensorboard --logdir=/home/russell512/my_deeprl_network_ori_test_0516_multi_head_PyG_GATCon/real_a1

# 或者，若只想監控此特定實驗（假設 BASE_DIR_NAME 是 real_a1/0519_MAPPO_run）
tensorboard --logdir=/home/russell512/my_deeprl_network_ori_test_0516_multi_head_PyG_GATCon/real_a1/0519_MAPPO_run
```

請將上述路徑替換為您在主機上對應的專案路徑和實驗輸出路徑。然後在瀏覽器中打開 TensorBoard 提供的 URL (通常是 `http://localhost:6006`)。

### 訓練指標解讀

以下是訓練過程中較為重要的 TensorBoard 指標，以及常見的觀察方式：

| 標籤 | 說明 |
| --- | --- |
| `train_reward` | 每回合的平均回報，持續上升通常表示策略正在改善。 |
| `Perf/EpisodeReward_Std` | 回報的標準差，若過大代表策略表現不穩定。 |
| `{name}/ppo_actor_loss` | 策略梯度（Actor）損失，長期趨於 0 通常意味著策略已穩定。 |
| `{name}/ppo_critic_loss` | 值函數（Critic）損失，可觀察其是否逐漸下降。 |
| `{name}/ppo_entropy_loss` | 負熵，值愈大代表策略愈隨機。 |
| `train/grad_norm_before_clip` | 每次更新前的梯度 L2 範數。 |
| `train/grad_norm_after_clip` | 施行 `clip_grad_norm_` 後的梯度範數。 |
| `train/grad_norm_ratio` | `grad_norm_before_clip / max_grad_norm`，若經常大於 1 代表梯度常被裁剪。 |
| `train/clip_factor` | `grad_norm_after_clip / grad_norm_before_clip`，越接近 0 代表裁剪越嚴重。 |

您可以透過觀察 `grad_norm_ratio` 是否大於 1，以及 `clip_factor` 的變化，判斷梯度裁剪對學習的影響。若長期處於裁剪狀態，可考慮調整 `max_grad_norm` 或學習率。

## 重新連接到訓練會話

如果您想查看正在運行的訓練日誌或進行其他操作：

1.  **重新進入 Docker 容器**

    ```bash
    docker exec -it 0519_MAPPO_run /bin/bash
    ```

2.  **重新連接到 `tmux` 會話**

    ```bash
    tmux attach -t training_0519_mappo
    ```

    您現在可以看到訓練腳本的實時輸出。

## 代碼結構概覽

  * **`test.py`**: 主執行腳本，負責解析參數、初始化環境和智能體、啟動訓練或評估流程。
  * **`utils.py`**: 包含專案級別的輔助類和函數，如 `Trainer` (負責訓練循環)、`Tester` (測試)、`Evaluator` (評估)、`Counter` (計步器)、日誌和目錄初始化工具。
  * **`agents/utils.py`**: 包含強化學習特定工具，尤其是 `OnPolicyBuffer` 和 `MultiAgentOnPolicyBuffer` (用於經驗存儲和 GAE 計算，已實現空間獎勵 GAE)。
  * **`agents/policies.py`**: 定義各種策略網路架構，核心是 `NCMultiAgentPolicy`，它整合了 FC 層、可選的 GAT 層和共享的 LSTM 層，以及智能體特定的 Actor/Critic 頭部。
  * **`agents/models.py`**: 實現具体的強化學習演算法模型，如 `MA2PPO_NC`, `MA2C_NC` 等。這些模型將策略網路 (`NCMultiAgentPolicy`) 與特定的學習演算法（如 PPO 或 A2C）結合起來。
  * **`envs/`**: 存放與特定模擬環境（如基於 SUMO 的交通環境）交互的 Python 類。
  * **`config/`**: 存放 `.ini` 格式的配置文件，用於定義實驗的各種參數。

## 單元測試

專案包含單元測試，位於 `tests/` 目錄下。可以使用 `pytest` 來運行：

```bash
# 在 Docker 容器內，位於 /workspace/my_deeprl_network/ 目錄
export PYTHONPATH=$(pwd):$PYTHONPATH # 確保能找到套件
pytest -q tests/
```

目前已針對 `MultiAgentOnPolicyBuffer` 中的空間獎勵 GAE 實現了較為全面的單元測試。

## 未來工作與優化方向

  * **Phase 1 效能優化**:
      * 將 `NCMultiAgentPolicy.evaluate_actions_values_and_entropy` 方法完全向量化，以利用 `_run_comm_layers` 的批處理能力，減少 PPO 更新時的 Python 循環開銷。
  * **代碼清理**: 移除過時的 A2C 相關路徑（如果不再需要）、整理配置文件、更新註解。
  * **進階效能分析**: 使用性能分析工具 (profiling) 找到潛在瓶頸。
  * **混合精度訓練**: 探索使用混合精度訓練以加速並減少 GPU 內存佔用。

