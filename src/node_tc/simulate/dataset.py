from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypedDict, NotRequired

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SimulatedSample:
    """
    一个用于存储模拟数据的类。

    Attributes:
        true_cluster (int): 真实的簇标签。
        static_vars (np.ndarray): 静态协变量。
        t (np.ndarray): 时间序列。
        obs (np.ndarray): 观测数据。
        t_ (np.ndarray): 真实的时间序列。
        obs_ (np.ndarray): 真实的潜在状态。
    """

    true_cluster: int
    t: np.ndarray
    obs: np.ndarray
    id: int
    static_vars: np.ndarray | None
    t_: np.ndarray
    obs_: np.ndarray


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
        return f"SimulatedDataset(num_patients={len(self.samples)}, num_clusters={self.num_clusters}, obs_dim={self.samples[0].obs.shape[1]}"


    def to_csv(self, dir: str | Path) -> None:
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

        obs_arr, obs__arr = [], []
        for sample in self.samples:
            obs_arr.append(np.concatenate([sample.t[:, None], sample.obs], axis=1))
            obs__arr.append(np.concatenate([sample.t_[:, None], sample.obs_], axis=1) )
        obs_arr = np.concatenate(obs_arr)
        obs__arr = np.concatenate(obs__arr)
        pd.DataFrame(
            obs_arr,
            index=np.repeat(indice, [len(sample.t) for sample in self.samples]),
            columns=["t"] + [f"x{i}" for i in range(sample.obs.shape[1])],
        ).to_csv(dir / "observations.csv")
        pd.DataFrame(
            obs__arr,
            index=np.repeat(indice, [len(sample.t_) for sample in self.samples]),
            columns=["t"] + [f"z{i}" for i in range(sample.obs_.shape[1])],
        ).to_csv(dir / "true_observations.csv")

    @classmethod
    def read_csv(cls, dir: str | Path) -> SimulatedDataset:
        dir = Path(dir)
        meta_fn = dir / "meta.csv"
        df_meta = pd.read_csv(meta_fn, index_col=0)
        df_obs = pd.read_csv(dir / "observations.csv", index_col=0)
        df_obs_ = pd.read_csv(dir / "true_observations.csv", index_col=0)
        samples = []
        for ind, df_i in df_obs.groupby(lambda ind: ind):
            assert isinstance(ind, int)

            df_i = df_i.sort_values("t")
            t_i = df_i["t"].to_numpy()
            obs_i = df_i.filter(regex=r"^x\d+$").to_numpy()

            df_obs__i = df_obs_[df_obs_.index == ind].sort_values("t")
            t__i = df_obs__i["t"].to_numpy()
            obs__i = df_obs__i.filter(regex=r"^z\d+$").to_numpy()

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
                    obs=obs_i,
                    static_vars=static_vars_i,
                    id=int(ind),
                    t_=t__i,
                    obs_=obs__i,
                )
            )

        return SimulatedDataset(samples=samples)


class SimuItem(TypedDict):
    id: int
    t: torch.Tensor
    x: torch.Tensor
    y: int
    z: NotRequired[torch.Tensor]


class SimulatedDatasetForTorch(Dataset):
    def __init__(
        self,
        samples: list[SimulatedSample] | SimulatedDataset,
        transform: Callable[[SimuItem], SimuItem]
        | None = None,
    ):
        if isinstance(samples, SimulatedDataset):
            samples = samples.samples
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SimuItem:
        sample = self.samples[index]

        # NOTE: 先转换成float32。因为在同一个batch中，很多t可能非常接近，
        # 后面有一个步骤是先对t取交集然后排序，需要保证做这一步时也是float32，
        # 否则会出现问题（取交集时是不同的时间点，但是转换成float32进行训练
        # 时就是一样的时间点了）
        res: SimuItem = {
            "id": sample.id,
            "t": torch.tensor(sample.t, dtype=torch.float32),
            "x": torch.tensor(sample.obs, dtype=torch.float32),
            "y": sample.true_cluster,
        }
        if sample.static_vars is not None:
            res["z"] = torch.tensor(sample.static_vars, dtype=torch.float32)

        if self.transform is not None:
            res = self.transform(res)

        return res
