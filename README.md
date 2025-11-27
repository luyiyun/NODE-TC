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

## 运行示例

在配置好环境后，直接运行示例脚本：

```bash
uv run main.py
```

程序将生成模拟数据，训练模型。数据储存在`./data/simulate/`中。