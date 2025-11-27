from __future__ import annotations
from typing import List

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import special_ortho_group
from scipy.linalg import block_diag

from .dataset import SimulatedDataset, SimulatedSample


class LinearMechanisticDynamics:
    """
    定义一个基于线性微分方程的生物机制系统。
    遵循方程: dz/dt = A @ (z - mu)

    物理/医学含义:
    - A (Interaction Matrix): 定义了变量之间的相互作用（例如：A升高导致B降低）。
    A 的特征值决定了系统的稳定性（康复、恶化、震荡）。
    - mu (Homeostasis/Setpoint): 生理稳态点（例如：正常的血糖水平、血压）。
    系统倾向于围绕这个点运动或远离这个点。
    """

    def __init__(self, A: np.ndarray, mu: np.ndarray | None = None):
        self.A = A
        self.dim = A.shape[0]
        # 如果没有指定稳态点，默认为0（虽然在医学上不常见，但作为数学默认值）
        self.mu = mu if mu is not None else np.zeros(self.dim)

    def __call__(self, t: float, z: np.ndarray) -> np.ndarray:
        # dz/dt = A * (current_state - setpoint)
        return self.A @ (z - self.mu)

    def __repr__(self):
        eig = np.linalg.eigvals(self.A)
        return f"Dynamics(dim={self.dim}, max_real_eig={np.max(eig.real):.2f})"


def generate_random_linear_system(
    dim: int,
    behavior_type: str = "mixed",
    rng: int | np.random.Generator | None = None,
) -> LinearMechanisticDynamics:
    """
    程序化生成一个具有特定医学动力学特征的高维线性系统。

    Args:
        dim: 潜在变量的维度（高维支持）。
        behavior_type: 系统的行为模式。
            - 'stable': 模拟康复/稳态回归 (所有特征值实部 < 0)。
            - 'unstable': 模拟疾病恶化/器官衰竭 (存在特征值实部 > 0)。
            - 'oscillatory': 模拟周期性波动 (主要由虚部主导，实部接近0)。
            - 'mixed': 混合模式。
        rng: 随机数生成器。

    Returns:
        LinearMechanisticDynamics 对象
    """
    assert behavior_type in ["stable", "unstable", "oscillatory", "mixed"], (
        f"Unknown behavior_type: {behavior_type}, must be one of "
        "['stable', 'unstable', 'oscillatory', 'mixed']"
    )
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    blocks = []
    remaining_dim = dim
    while remaining_dim > 0:
        # 决定生成复数对(2D)还是实数(1D)
        # 至少剩2维且概率触发时生成复数块
        is_complex = (remaining_dim >= 2) and (rng.random() < 0.6)

        # 定义实部 (决定衰减/增长)
        if behavior_type == "stable":
            real = rng.uniform(-0.5, -0.05)
        elif behavior_type == "unstable":
            real = rng.uniform(0.01, 0.3)
        elif behavior_type == "oscillatory":
            real = rng.uniform(-0.05, 0.05)  # 接近0，主要看虚部
        else:  # mixed
            real = rng.uniform(-0.5, 0.3)

        if is_complex:
            # 生成 2x2 旋转块
            # 特征值: real +/- i*imag
            imag = rng.uniform(0.1, 1.0)  # 震荡频率

            # 构造块 [[a, b], [-b, a]]
            # 注意：scipy solve_ivp 默认定义习惯通常对应顺/逆时针，符号影响相位但不影响频率
            block = np.array([[real, imag], [-imag, real]])
            blocks.append(block)
            remaining_dim -= 2
        else:
            # 生成 1x1 标量块
            block = np.array([[real]])
            blocks.append(block)
            remaining_dim -= 1

    # 1. 构造实分块对角矩阵 J (Jordan-like block diagonal)
    # J 的特征值就是我们生成的那些实数和复数
    J = block_diag(*blocks)
    # print(J)

    # 2. 生成随机基 (Basis) 以混合变量
    # 使用随机正交矩阵 Q，使得 A = Q * J * Q^T
    # 这样生成的 A 是实数矩阵，且特征值与 J 相同
    if dim > 1:
        Q = special_ortho_group.rvs(dim, random_state=rng)
    else:
        Q = np.array([[1.0]])

    # 3. 计算最终的动力学矩阵 A
    A = Q @ J @ Q.T

    # 3. 生成生理稳态点 (Setpoint)
    # 模拟不同的人群可能有不同的“正常值”或“病态目标值”
    mu = rng.uniform(-2, 2, size=dim)

    return LinearMechanisticDynamics(A, mu)


def simulate(
    num_patients: int = 200,
    num_clusters: int = 4,  # 现在可以支持任意数量的簇
    obs_dim: int = 10,  # 观测维度
    static_dim: int = 3,  # 静态变量维度
    z0: int | float | np.ndarray | None = 0.0,
    num_time_internval: tuple[int, int] = (10, 25),  # 观测次数范围
    time_max: float = 10.0,  # 最大时间跨度
    missing_rate: float = 0.0,
    noise_std: float = 0.1,  # 观测噪声
    seed: int = 42,
) -> tuple[SimulatedDataset, List[LinearMechanisticDynamics]]:
    """
    生成具有复杂医学机制的多变量时间序列数据。
    """
    rng = np.random.default_rng(seed)

    # 1. 生成每个簇的动力学机制
    # 我们循环生成不同的动力学方程，确保每个簇代表一种不同的“病理生理过程”
    dynamics_pool = []
    behavior_types = ["stable", "unstable", "oscillatory", "mixed"]

    print(f"正在生成 {num_clusters} 种独特的动力学机制...")
    for k in range(num_clusters):
        # 轮询使用不同的行为模式，确保簇之间差异明显
        b_type = behavior_types[k % len(behavior_types)]
        dyn = generate_random_linear_system(dim=obs_dim, behavior_type=b_type, rng=rng)
        dynamics_pool.append(dyn)
        print(
            f"  Cluster {k}: {b_type} dynamics, max_eig_real={np.linalg.eigvals(dyn.A).real.max():.3f}"
        )

    # # 2. 如果观测维度 != 潜在维度，生成投影矩阵
    # # 比如潜在只有 3 个关键因子驱动 100 个基因表达
    # if obs_dim != latent_dim:
    #     # 使用随机正交投影，保证信息最大程度保留
    #     # shape: (latent_dim, obs_dim)
    #     projection_matrix = rng.normal(0, 1, (latent_dim, obs_dim))
    #     # 归一化以保持尺度
    #     projection_matrix /= np.linalg.norm(projection_matrix, axis=0)
    # else:
    #     projection_matrix = None

    # 3. 分配簇标签
    true_k = rng.choice(num_clusters, num_patients)

    # 4. 生成初始状态 z0
    if z0 is None:
        z0 = rng.normal(0, 1, (num_patients, obs_dim))
        z0 *= (true_k + 1)[:, None]
    elif isinstance(z0, (int, float)):
        z0 = np.full((num_patients, obs_dim), z0)
    elif z0.shape == (obs_dim,):
        z0 = np.tile(z0, (num_patients, 1))
    elif z0.shape == (num_clusters, obs_dim):
        z0 = z0[true_k, :]
    else:
        raise ValueError(
            "z0 must be a scalar, a vector of length obs_dim, or a matrix of shape (num_clusters, obs_dim)"
        )

    # 5. 生成静态变量 (Static Variables)
    # 假设静态变量（年龄、性别、基因）影响病人的初始状态 (z0)
    if static_dim > 0:
        static_vars = rng.normal(0, 1, (num_patients, static_dim))
        # 静态变量对初始状态的映射矩阵
        static_to_z0 = rng.normal(0, 0.5, (static_dim, obs_dim))
        static_effects = static_vars @ static_to_z0
        z0 += static_effects

    # # 让不同簇的初始分布中心略有不同（可选，增强可分性）
    # cluster_centers = rng.normal(0, 2.0, (num_clusters, obs_dim))
    # z0 = z0_base + cluster_centers[true_k] + static_effects

    # 6. 积分生成轨迹
    samples = []

    # 确定每个病人的时间点
    num_time_points = rng.integers(
        num_time_internval[0], num_time_internval[1], size=num_patients
    )

    for i in range(num_patients):
        k_i = true_k[i]
        dyn = dynamics_pool[k_i]
        z0_i = z0[i]
        n_t = num_time_points[i]

        # 随机采样时间点 (不均匀采样是医学数据的常态)
        # 始终包含 t=0
        t_eval = np.sort(rng.uniform(0, time_max, size=n_t - 1))
        t_eval = np.concatenate(([0.0], t_eval))

        # 解微分方程
        # span 稍微比 t_eval 大一点以防边界误差
        sol = solve_ivp(
            fun=dyn, t_span=(0, time_max + 0.1), y0=z0_i, t_eval=t_eval, method="RK45"
        )

        if not sol.success:
            print(f"Warning: Integration failed for sample {i}")
            continue

        traj_z = sol.y.T  # shape (n_t, latent_dim)

        # 投影到观测空间
        # if projection_matrix is not None:
        #     obs_clean = traj_z @ projection_matrix
        # else:
        #     obs_clean = traj_z

        # 添加观测噪声
        obs_noisy = traj_z + rng.normal(0, noise_std, traj_z.shape)

        # 处理缺失值 (Missing Data)
        if missing_rate > 0:
            mask = rng.random(obs_noisy.shape) < missing_rate
            # 保证至少第一个时间点的部分特征存在，或者不做特殊处理（取决于你的模型需求）
            # 这里我们简单地随机mask
            obs_noisy[mask] = np.nan

        samples.append(
            SimulatedSample(
                id=i,
                true_cluster=k_i,
                t=t_eval,
                observations=obs_noisy,
                static_vars=static_vars[i] if static_dim > 0 else None,
            )
        )

    print(f"成功生成 {len(samples)} 条多变量轨迹数据。")
    return SimulatedDataset(samples), dynamics_pool


if __name__ == "__main__":
    # 测试代码
    rng = np.random.default_rng(42)
    dyn = generate_random_linear_system(10, "oscillatory", rng)

    print("矩阵 A 是否为全实数:", np.all(np.isreal(dyn.A)))
    eig_vals = np.linalg.eigvals(dyn.A)
    print("\n特征值 (实部应接近0，虚部存在):")
    for e in eig_vals:
        print(f"{e.real:.4f} + {e.imag:.4f}j")
