# Environment Assumptions

This project assumes a fixed multi-agent topology and constant observation and action spaces.

* **Fixed number of agents** – `n_agent` does not change during training or evaluation.
* **Static neighbor mask** – each agent's neighbors are predetermined and remain the same.
* **Observation and action dimensions are constant** – sizes used to build neural networks never change after initialization.
* **Single GPU training** – the code is designed for a single device although tensors automatically follow the model's device via the `dev` property.
* **Optional GAT support** – if `USE_GAT=1` and `torch_geometric` is installed, GAT layers are active; otherwise the model falls back to MLP communication.

These constraints allow the policy and critic networks to allocate all necessary layers during initialization and avoid dynamic rebuilding during runtime.
