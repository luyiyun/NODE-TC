# NODE-TC: Neural ODE for Time-Series Clustering

本项目实现了一个基于神经微分方程（Neural ODE）的时间序列聚类模型（NODE-TC）。该工具包提供了从合成数据生成、数据加载、模型构建到使用 EM（期望最大化）算法进行训练和可视化的完整流程。

## 🛠 环境配置 (Installation)

本项目使用 [uv](https://github.com/astral-sh/uv) 进行极其快速的依赖管理和环境同步。

### 前置要求
- Python 3.10+
- [uv](https://github.com/astral-sh/uv)

### 步骤

1. **克隆项目**
   ```bash
   git clone <your-repo-url>
   cd <your-project-dir>
   ```

2. **安装 uv (如果尚未安装)**
   ```bash
   # MacOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **同步环境依赖**
   使用 `uv sync` 根据 `pyproject.toml` (或 `uv.lock`) 自动创建虚拟环境并安装所有依赖（包括 PyTorch 等）。
   ```bash
   uv sync --dev
   ```

4. **软连接数据路径**
    将数据路径软连接到 `data` 目录，以便多个用户共享同一份数据。
    ```bash
    ln -s /data1/NODETC/data data
    ```

## 📦 核心模块说明 (Usage)

`node_tc` 包主要由以下两个模块组成：

### 1. 数据模拟 (`node_tc.simulate`)

用于生成用于测试和验证模型性能的合成时间序列数据。

*   **`SimulatedDataset.simulate(...)`**: 生成模拟数据集。
    *   `num_patients`: 样本（患者）数量。
    *   `num_clusters`: 真实的聚类簇数。
    *   `obs_dim`: 观测数据的维度。
    *   `latent_dim`: 潜在状态的维度。
    *   `noise_std_per_cluster`: 每个簇的噪声标准差。
    *   `z0`: 初始状态值。
*   **`SimulatedDataset.set_transform(func)`**: 对数据应用预处理变换（如时间归一化）。
*   **`SimulatedDataset.write_csv(path)`**: 将生成的数据保存为 CSV 格式。
*   **`SimulatedDataset.plot(...)`**: 可视化生成的样本轨迹。
*   **`SimulatedDataCollateFunc`**: 配合 PyTorch DataLoader 使用的自定义整理函数，用于处理变长序列或特定格式。

### 2. 模型与训练 (`node_tc.model`)

包含核心的 Neural ODE 模型定义和 EM 训练器。

*   **`NODETC`**: 核心模型类。
    *   `obs_dim`, `latent_dim`: 维度配置。
    *   `num_clusters`: 预设的聚类数量。
    *   `method`: ODE 求解器方法（如 `"rk4"`, `"dopri5"`）。
    *   `options`: 求解器步长等选项。
*   **`EMTrainer`**: 用于训练 NODETC 模型的训练器，通常基于期望最大化（EM）算法。
    *   `model`: 实例化的 NODETC 模型。
    *   `loader`: 数据加载器。
    *   `update_nn_params_epochs_every_round`: 在每一轮 EM 迭代中更新神经网络参数的 Epoch 数。
*   **`EMTrainer.plot_vector_field()`**: 训练完成后，绘制学习到的向量场以分析动力学特征。

## 🚀 快速开始 (Quick Start)

以下代码展示了如何生成数据、构建模型并运行训练：

```python
from dataclasses import replace
import torch.nn as nn
from torch.utils.data import DataLoader
from node_tc.simulate import SimulatedDataset, SimulatedDataCollateFunc
from node_tc.model import NODETC, EMTrainer

# 1. 配置与数据生成
NUM_CLUSTERS = 3
simu_data = SimulatedDataset.simulate(
    num_patients=1000,
    num_clusters=NUM_CLUSTERS,
    obs_dim=2,
    latent_dim=2,
    static_dim=0,
    noise_std_per_cluster=(0.1, 0.1, 0.1),
    seed=42,
    num_time_internval=(5, 11),
    time_interval=(1, 11)
)

# 2. 数据预处理
# 例如：将时间缩放 10 倍
simu_data.set_transform(lambda x: replace(x, t=x.t / 10))

# 3. 创建 DataLoader
loader = DataLoader(
    simu_data,
    batch_size=64,
    shuffle=True,
    collate_fn=SimulatedDataCollateFunc(),
)

# 4. 初始化模型
model = NODETC(
    obs_dim=2,
    latent_dim=2,
    static_dim=0,
    num_clusters=NUM_CLUSTERS,
    activation=nn.GELU,
    method="rk4",
    options={"step_size": 0.1},
)

# 5. 训练 (EM 算法)
trainer = EMTrainer(
    model=model,
    loader=loader,
    num_epochs=20,
    lr=0.001,
    update_nn_params_epochs_every_round=2,
)
trainer.train()

# 6. 结果可视化
fig = trainer.plot_vector_field()
fig.savefig("vector_field.png")
```

## 运行示例

在配置好环境后，直接运行示例脚本：

```bash
uv run main.py
```

程序将生成模拟数据，训练模型，并输出 `simulated_data.png`（原始数据分布）和 `vector_field.png`（学习到的动力学向量场）。