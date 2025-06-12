import torch

checkpoint = torch.load("real_a1/ma2c_nclm/model/checkpoint-100080.pt")

# 讀取其他資訊
optimizer_state = checkpoint['optimizer_state_dict']  # 優化器狀態
glostep = checkpoint['global_step']  # 訓練到第幾個 epoch
modstate = checkpoint['model_state_dict']  # 最後一次的損失值

print(f"Optimizer State: {optimizer_state}")
print(f"Global Step: {glostep}")
#print(f"Model State: {modstate}")
