from typing import List, Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# --- 1. 定义真实的动态系统 (Helper Classes) ---
# 这些类定义了不同簇的潜在轨迹应该遵循的“物理定律”。


class Dynamics:
    def __init__(self, A: np.ndarray):
        self.A = A

    def __call__(self, t: float, z: np.ndarray) -> np.ndarray:
        return np.dot(z, self.A.T)


spiral_dynamics = Dynamics(np.array([[-0.1, -1.0], [1.0, -0.1]]))
wave_dynamics = Dynamics(np.array([[0.0, -0.5], [2.0, 0.0]]))
repulsive_dynamics = Dynamics(np.array([[0.2, 0.5], [-0.5, 0.2]]))


@dataclass
class SimulatedSample:
    """
    一个用于存储模拟数据的类。

    Attributes:
        true_cluster (int): 真实的簇标签。
        static_vars (np.ndarray): 静态协变量。
        t (np.ndarray): 时间序列。
        observations (np.ndarray): 观测数据。
        true_z (np.ndarray): 真实的潜在状态。
    """

    true_cluster: int
    t: np.ndarray
    observations: np.ndarray
    true_z: np.ndarray
    id: int
    static_vars: np.ndarray | None = None

    def plot(
        self, ax: Axes | None = None, color: str = "black", title: str | None = None
    ) -> Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Matplotlib 可以很好地处理 NaN，会在缺失点处断开线条
        # 为了更清晰，我们使用散点图表示观测值
        # 绘制观测点（仅绘制前两个维度）
        ax.scatter(
            self.observations[:, 0],
            self.observations[:, 1],
            color=color,
            alpha=0.8,
            label="Observations (Dim 1&2)",
        )

        # 绘制真实的潜在轨迹（仅绘制前两个维度）
        ax.plot(
            self.true_z[:, 0],
            self.true_z[:, 1],
            "-",
            color="black",
            linewidth=2,
            label="True Latent Trajectory",
        )

        if title is not None:
            ax.set_title(title)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.legend()
        ax.grid(True)
        # ax.set_aspect("equal", "box")

        return ax


@dataclass
class SimulatedDataset:
    """
    一个用于存储多个模拟数据的类。

    Attributes:
        samples (List[SimulatedSample]): 存储多个模拟数据的列表。
    """

    samples: List[SimulatedSample]
    transform: Callable[[SimulatedSample], SimulatedSample] | None = None

    def __post_init__(self):
        self.true_k = np.array([p.true_cluster for p in self.samples])
        self.num_clusters = len(set(self.true_k))

        self.n_static_vars = 0
        for sample in self.samples:
            if sample.static_vars is not None:
                self.n_static_vars = sample.static_vars.shape[0]
                break

        if self.n_static_vars > 0:
            self.static_vars = np.stack(
                [
                    p.static_vars
                    if p.static_vars is not None
                    else np.full((self.n_static_vars,), np.nan)
                    for p in self.samples
                ]
            )

    def __repr__(self) -> str:
        return f"SimulatedDataset(num_patients={len(self.samples)}, num_clusters={self.num_clusters}, obs_dim={self.samples[0].observations.shape[1]}, latent_dim={self.samples[0].true_z.shape[1]})"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SimulatedSample:
        sample = self.samples[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def set_transform(self, transform: Callable[[SimulatedSample], SimulatedSample]):
        self.transform = transform

    @property
    def max_observation_value(self) -> float:
        return np.max([np.max(p.observations) for p in self.samples])

    def plot(self, num_samples_per_cluster: int = 2, seed: int = 42) -> Figure:
        """
        可视化生成的数据。

        Args:
            num_samples_per_cluster (int): 每个簇中随机抽取用于可视化的样本数量。
        """

        rng = np.random.default_rng(seed)

        fig, axs = plt.subplots(
            self.num_clusters,
            num_samples_per_cluster,
            figsize=(6 * num_samples_per_cluster, 5 * self.num_clusters),
            squeeze=False,
        )

        for k in range(self.num_clusters):
            cluster_indices = np.where(self.true_k == k)[0]
            if len(cluster_indices) == 0:
                continue

            sample_indices = rng.choice(
                cluster_indices,
                size=min(num_samples_per_cluster, len(cluster_indices)),
                replace=False,
            )

            for sample_idx, ax in zip(sample_indices, axs[k]):
                patient_data = self.__getitem__(sample_idx)
                patient_data.plot(ax, title=f"Patient ID: {sample_idx}, Cluster: {k}")

        fig.tight_layout()
        return fig

    @classmethod
    def simulate(
        cls,
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
    ) -> "SimulatedDataset":
        assert 0.0 <= missing_rate < 1.0, "Missing rate must be between 0.0 and 1.0"
        assert len(noise_std_per_cluster) == num_clusters, (
            "Noise std per cluster must have the same length as num_clusters."
        )
        assert num_clusters <= 3, "Requested more clusters than available dynamics."

        rng = np.random.default_rng(seed)
        true_dynamics = [spiral_dynamics, wave_dynamics, repulsive_dynamics]
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
                lambda t, x: true_dynamics[k_i](t, x),
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
                    observations=obs_i,
                    true_z=traj_i,
                )
            )

        print("数据生成完毕。")

        return cls(samples=res)

    def write_csv(self, dir: str | Path):
        dir = Path(dir)
        dir.mkdir(exist_ok=True, parents=True)

        indice = np.array([p.id for p in self.samples])
        df_meta = pd.DataFrame(self.true_k[:, None], index=indice, columns=["label"])
        if self.n_static_vars > 0:
            df_static_vars = pd.DataFrame(
                self.static_vars,
                index=indice,
                columns=[f"static_{i}" for i in range(self.n_static_vars)],
            )
            df_meta = pd.concat([df_meta, df_static_vars], axis=1)
        df_meta.to_csv(dir / "meta.csv")

        time_len, obs_arr = [], []
        for sample in self.samples:
            obs_arr.append(
                np.concatenate(
                    [sample.t[:, None], sample.observations, sample.true_z], axis=1
                )
            )
            time_len.append(len(sample.t))
        obs_arr = np.concatenate(obs_arr)
        df_obs = pd.DataFrame(
            obs_arr,
            index=np.repeat(indice, time_len),
            columns=["t"]
            + [f"x{i}" for i in range(sample.observations.shape[1])]
            + [f"z{i}" for i in range(sample.true_z.shape[1])],
        )
        df_obs.to_csv(dir / "observations.csv")

    @classmethod
    def read_csv(cls, dir: str | Path) -> "SimulatedDataset":
        dir = Path(dir)

        meta_fn = dir / "meta.csv"
        df_meta = pd.read_csv(meta_fn, index_col=0)

        df_obs = pd.read_csv(dir / "observations.csv", index_col=0)
        samples = []
        for ind, df_i in df_obs.groupby(lambda ind: ind):
            assert isinstance(ind, int)

            t_i = df_i["t"].to_numpy()
            obs_i = df_i.filter(regex=r"^x\d+$").to_numpy()
            z_i = df_i.filter(regex=r"^z\d+$").to_numpy()

            df_meta_i = df_meta.loc[ind]
            assert isinstance(df_meta_i, pd.Series)
            if "static_0" in df_meta_i.index:
                static_vars_i = df_meta_i.filter(regex=r"^static_\d+$").to_numpy()
            else:
                static_vars_i = None

            samples.append(
                SimulatedSample(
                    true_cluster=int(df_meta_i["label"]),
                    t=t_i,
                    observations=obs_i,
                    true_z=z_i,
                    static_vars=static_vars_i,
                    id=int(ind),
                )
            )

        return cls(samples=samples)


class SimulatedDataCollateFunc:
    def __call__(self, batch: List[SimulatedSample]) -> dict[str, torch.Tensor]:
        t = np.sort(np.unique(np.concatenate([p.t for p in batch])))
        obs = np.zeros((len(batch), len(t), batch[0].observations.shape[1]))
        mask = np.zeros((len(batch), len(t)))
        for i, sample in enumerate(batch):
            ind = np.searchsorted(t, sample.t)
            obs[i, ind, :] = sample.observations
            mask[i, ind] = 1.0

        res = {
            "t": torch.tensor(t, dtype=torch.float32),
            "observations": torch.tensor(obs, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "true_cluster": torch.tensor(
                [p.true_cluster for p in batch], dtype=torch.long
            ),
        }

        if batch[0].static_vars is not None:
            res["static_vars"] = torch.tensor(
                [p.static_vars for p in batch], dtype=torch.float32
            )
        if batch[0].id is not None:
            res["id"] = torch.tensor([p.id for p in batch], dtype=torch.long)

        return res


# --- 3. 示例用法 ---
if __name__ == "__main__":
    # 示例1：生成无缺失的数据并可视化
    print("--- 示例 1: 生成无缺失的数据 ---")
    simu_data = SimulatedDataset.simulate(num_patients=50, missing_rate=0.0)
    fig = simu_data.plot(num_samples_per_cluster=3)
    fig.savefig("simulated_data.png", dpi=300)
