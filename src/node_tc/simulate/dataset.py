from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


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
    # true_z: np.ndarray
    id: int
    static_vars: np.ndarray | None = None


@dataclass
class SimulatedDataset:
    """
    一个用于存储多个模拟数据的类。

    Attributes:
        samples (List[SimulatedSample]): 存储多个模拟数据的列表。
    """

    samples: list[SimulatedSample]

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
        return f"SimulatedDataset(num_patients={len(self.samples)}, num_clusters={self.num_clusters}, obs_dim={self.samples[0].observations.shape[1]}"


def write_simulate_data_to_csv(dir: str | Path, dataset: SimulatedDataset):
    dir = Path(dir)
    dir.mkdir(exist_ok=True, parents=True)

    indice = np.array([p.id for p in dataset.samples])
    df_meta = pd.DataFrame(dataset.true_k[:, None], index=indice, columns=["label"])
    if dataset.n_static_vars > 0:
        df_static_vars = pd.DataFrame(
            dataset.static_vars,
            index=indice,
            columns=[f"static_{i}" for i in range(dataset.n_static_vars)],
        )
        df_meta = pd.concat([df_meta, df_static_vars], axis=1)
    df_meta.to_csv(dir / "meta.csv")

    time_len, obs_arr = [], []
    for sample in dataset.samples:
        obs_arr.append(np.concatenate([sample.t[:, None], sample.observations], axis=1))
        time_len.append(len(sample.t))
    obs_arr = np.concatenate(obs_arr)
    df_obs = pd.DataFrame(
        obs_arr,
        index=np.repeat(indice, time_len),
        columns=["t"] + [f"x{i}" for i in range(sample.observations.shape[1])],
    )
    df_obs.to_csv(dir / "observations.csv")


def read_simulate_data_from_csv(dir: str | Path) -> SimulatedDataset:
    dir = Path(dir)
    meta_fn = dir / "meta.csv"
    df_meta = pd.read_csv(meta_fn, index_col=0)
    df_obs = pd.read_csv(dir / "observations.csv", index_col=0)
    samples = []
    for ind, df_i in df_obs.groupby(lambda ind: ind):
        assert isinstance(ind, int)

        t_i = df_i["t"].to_numpy()
        obs_i = df_i.filter(regex=r"^x\d+$").to_numpy()

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
                static_vars=static_vars_i,
                id=int(ind),
            )
        )

    return SimulatedDataset(samples=samples)


class SimulatedDatasetForTorch(Dataset):
    def __init__(
        self,
        samples: list[SimulatedSample] | SimulatedDataset,
        transform: Callable[[dict[str, np.ndarray | int]], dict[str, np.ndarray | int]]
        | None = None,
    ):
        if isinstance(samples, SimulatedDataset):
            samples = samples.samples
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | int]:
        sample = self.samples[index]
        res = {
            "id": sample.id,
            "t": sample.t,
            "x": sample.observations,
            "y": sample.true_cluster,
        }
        if sample.static_vars is not None:
            res["z"] = sample.static_vars

        if self.transform is not None:
            res = self.transform(res)

        return res
