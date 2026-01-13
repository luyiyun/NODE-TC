from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .dataset import SimulatedSample, SimulatedDataset


# --- 1. 定义真实的动态系统 (Helper Classes) ---
# 这些类定义了不同簇的潜在轨迹应该遵循的“物理定律”。


class Dynamics:
    def __init__(self, A: np.ndarray):
        self.A = A

    def __call__(self, t: float, z: np.ndarray) -> np.ndarray:
        return np.dot(z, self.A.T)


ALL_DYNAMICS = [
    Dynamics(np.array([[-0.1, -1.0], [1.0, -0.1]])),
    Dynamics(np.array([[0.0, -0.5], [2.0, 0.0]])),
    Dynamics(np.array([[0.2, 0.5], [-0.5, 0.2]])),
]
# spiral_dynamics = Dynamics(np.array([[-0.1, -1.0], [1.0, -0.1]]))
# wave_dynamics = Dynamics(np.array([[0.0, -0.5], [2.0, 0.0]]))
# repulsive_dynamics = Dynamics(np.array([[0.2, 0.5], [-0.5, 0.2]]))


def simulate(
    num_patients: int = 100,
    num_clusters: int = 3,
    obs_dim: int = 2,
    latent_dim: int = 2,
    static_dim: int = 2,
    z0: int | float | np.ndarray | None = None,
    num_time_internval: tuple[int, int] = (11, 31),
    time_interval: tuple[int, int] = (1, 31),
    missing_rate: float = 0.0,
    noise_std_per_cluster: tuple[float, ...] = (0.15, 0.2, 0.25),
    # static_to_z0_effects_mean: tuple[float, ...] | None = (0.5, 0.6, 0.7),
    seed: int = 42,
) -> SimulatedDataset:
    assert 0.0 <= missing_rate < 1.0, "Missing rate must be between 0.0 and 1.0"
    assert len(noise_std_per_cluster) == num_clusters, (
        "Noise std per cluster must have the same length as num_clusters."
    )
    assert num_clusters <= 3, "Requested more clusters than available dynamics."

    rng = np.random.default_rng(seed)
    flag_static = static_dim > 0

    # 如果观测维度和潜在维度不同，需要一个固定的投影矩阵
    if obs_dim != latent_dim:
        projection_matrix = rng.normal(0, 1, (latent_dim, obs_dim))
    else:
        projection_matrix = None

    true_k = rng.choice(num_clusters, num_patients)
    if z0 is None:
        z0 = rng.normal(0, 1, (num_patients, latent_dim))
        z0 *= (true_k + 1)[:, None]
    elif isinstance(z0, (int, float)):
        z0 = np.full((num_patients, latent_dim), z0)
    elif z0.shape == (latent_dim,):
        z0 = np.tile(z0, (num_patients, 1))
    elif z0.shape == (num_clusters, latent_dim):
        z0 = z0[true_k, :]
    else:
        raise ValueError(
            "z0 must be a scalar, a vector of length latent_dim, or a matrix of shape (num_clusters, latent_dim)"
        )

    if flag_static:
        static_to_z0_effects = rng.normal(0, 1, (static_dim, latent_dim))
        static_vars = rng.normal(0, 1, (num_patients, static_dim))
        static_effect = static_vars @ static_to_z0_effects
        z0 = z0 + static_effect

    num_time_points = rng.choice(np.arange(*num_time_internval), num_patients)
    res = []
    for i, (z0_i, k_i, num_t_i) in enumerate(zip(z0, true_k, num_time_points)):
        t = np.sort(
            rng.choice(
                np.arange(*time_interval), num_t_i - 1, replace=False
            )  # -1因为需要将初始值放在第一个
        )
        t = np.r_[0, t]  # 确保第一个时间点为0
        traj_i = solve_ivp(
            lambda t, x: ALL_DYNAMICS[k_i](t, x),
            [0, t[-1]],
            z0_i,
            t_eval=t,
        ).y.T

        # 6. 投影到观测空间并添加高斯噪声
        if projection_matrix is not None:
            true_x_no_noise = traj_i @ projection_matrix
        else:
            true_x_no_noise = traj_i

        obs_i = (
            true_x_no_noise
            + rng.normal(0, 1, (num_t_i, obs_dim)) * noise_std_per_cluster[k_i]
        )

        # 7. 引入数据缺失
        if missing_rate > 0:
            # 创建一个随机掩码，但保留第一个时间点的数据（因为编码器需要它）
            mask = rng.random(obs_i.shape) < missing_rate
            mask[0, :] = False  # 确保第一个观测值不缺失
            obs_i[mask] = float("nan")  # 使用 NaN 表示缺失值

        res.append(
            SimulatedSample(
                id=i,
                true_cluster=k_i,
                static_vars=static_vars[i] if flag_static else None,
                t=t,
                obs=obs_i,
                # true_z=traj_i,
            )
        )

    print("数据生成完毕。")

    return SimulatedDataset(samples=res)
