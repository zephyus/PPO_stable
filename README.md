這是一份**針對您 2025/06/17 實驗需求**的全新、**優化版 PPO 訓練流程**指令文件，整合原始邏輯、容器與目錄靈活切換、訓練記錄、背景穩定執行等設計。

---

## ✅ 強化版 PPO 訓練指令流程（使用 `/home/russell512/deeprl_0611_ppo` 與 `config/config_mappo_0617_noMHA.ini`）

### 🔧 前提條件

* 已安裝 Docker + NVIDIA Docker 支援。
* Dockerfile 位於 `~/best_environment/`。
* 要訓練的代碼在：`/home/russell512/deeprl_0611_ppo`。
* 使用的設定檔為：`config/config_mappo_0617_noMHA.ini`。

---

## 🧱 第 1 步：建置 Docker Image（若尚未建置）

```bash
cd ~/best_environment
docker build -t best_environment:latest .
```

---

## 🚀 第 2 步：啟動背景容器

```bash
docker run \
  --gpus all \
  -d \
  --name Trainer_PPO_0617 \
  -v /home/russell512/deeprl_0611_ppo:/workspace/my_deeprl_network \
  best_environment:latest \
  sleep infinity
```

> ✅ *請確認容器名稱「Trainer\_PPO\_0617」未重複。若重複，先刪除：*
> `docker rm Trainer_PPO_0617`

---

## 🧭 第 3 步：進入容器操作

```bash
docker exec -it Trainer_PPO_0617 /bin/bash
```

---

## 🧰 第 4 步：安裝必要套件與環境

```bash
pip install traci sumolib torch

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-1.9.0+cu111.html

pip install torch-geometric

cd /workspace/my_deeprl_network

export SUMO_HOME="/root/miniconda/envs/py36/share/sumo"

apt update && apt install -y tmux
```

---

## 🧠 第 5 步：使用 tmux 啟動訓練（記得先建立 log 資料夾）

```bash
tmux new -s training_ppo_0617
```

**訓練指令（在 tmux 內執行）：**

```bash
export BASE_DIR_NAME="real_a1/0617_PPO_noMHA"
mkdir -p ${BASE_DIR_NAME}/log

export USE_GAT=1

python3 test.py \
  --base-dir ${BASE_DIR_NAME} \
  --port 210 \
  train \
  --config-dir config/config_mappo_0617_noMHA.ini \
  > ${BASE_DIR_NAME}/log/training_0617_PPO_noMHA_$(date +%Y%m%d_%H%M%S).log 2>&1
```

---

## 📴 第 6 步：分離 tmux 並退出容器

```bash
# 在 tmux 裡按下：
Ctrl + b，然後按 d

# 然後輸入：
exit
```

---

## 🔌 第 7 步：安全關閉 SSH，訓練仍會繼續運行

---

## 🔁 第 8 步：日後重新登入觀看訓練

```bash
ssh <your-server>
docker exec -it Trainer_PPO_0617 /bin/bash
tmux attach -t training_ppo_0617
```

---

## 📊 選用：啟動 TensorBoard（查看所有 PPO Log）

```bash
tensorboard --logdir=/home/russell512/deeprl_0611_ppo/real_a1 --port=6010
```

> 瀏覽器開啟：[http://localhost:6010](http://localhost:6010)

---

## 🔍 檢查目前 Python 訓練程序是否還在跑

```bash
ps -eo pid,user,%cpu,%mem,cmd | grep python | grep -v grep
```

---

## 🧠 模型架構備註（紀錄）

> **模型結構：**
> Pre-Norm → GAT → Output Projection → Residual → LSTM

---

如需啟動不同實驗，只需變動以下三個關鍵參數：

1. `--base-dir`：輸出目錄（建議每次訓練都換）
2. `--port`：SUMO 使用埠號（避免衝突）
3. `--config-dir`：配置檔路徑

---

需要我幫你產生多組訓練指令（不同 config / port）也可以告訴我，我可以幫你自動化生成。
