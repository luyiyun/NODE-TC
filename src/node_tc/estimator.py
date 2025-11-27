import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from matplotlib.figure import Figure

from .simulate.dataset import SimulatedDataset, SimulatedDatasetForTorch
from .model import NODETC
from .trainer import EMTrainer


class TrajectoryDatasetCollateFunc:
    def __init__(
        self,
        time_key: str = "t",
        obs_key: str = "x",
        id_key: str = "id",
        label_key: str | None = None,
        static_vars_key: str | None = None,
    ) -> None:
        self.time_key = time_key
        self.obs_key = obs_key
        self.label_key = label_key
        self.static_vars_key = static_vars_key
        self.id_key = id_key

    def __call__(
        self, batch: list[dict[str, np.ndarray | int]]
    ) -> dict[str, torch.Tensor]:
        obs_i = batch[0][self.obs_key]
        assert isinstance(obs_i, np.ndarray), f"obs_i is not a numpy array: {obs_i}"

        t = np.sort(
            np.unique(np.concatenate([sample[self.time_key] for sample in batch]))
        )
        obs = np.zeros((len(batch), len(t), obs_i.shape[1]))
        mask = np.zeros((len(batch), len(t)))
        for i, sample in enumerate(batch):
            ind = np.searchsorted(t, sample[self.time_key])
            obs[i, ind, :] = sample[self.obs_key]
            mask[i, ind] = 1.0

        res = {
            "t": torch.tensor(t, dtype=torch.float32),
            "x": torch.tensor(obs, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "id": torch.tensor(
                [sample[self.id_key] for sample in batch], dtype=torch.long
            ),
        }

        if self.label_key is not None:
            if self.label_key not in batch[0]:
                raise ValueError(f"label_key {self.label_key} not in batch[0]")
            res["y"] = torch.tensor(
                [sample[self.label_key] for sample in batch], dtype=torch.long
            )
        if self.static_vars_key is not None:
            if self.static_vars_key not in batch[0]:
                raise ValueError(
                    f"static_vars_key {self.static_vars_key} not in batch[0]"
                )
            res["z"] = torch.tensor(
                np.stack([sample[self.static_vars_key] for sample in batch], axis=0),
                dtype=torch.float32,
            )

        return res


class NODETrajectoryCluster:
    def __init__(
        self,
        num_clusters: int,
        batch_size: int = 64,
        num_workers: int = 0,
        learning_rate: float = 1e-3,
        num_epochs: int = 20,
        bn: bool = False,
        adjoint: bool = False,
        device: str | torch.device = "cuda",
        update_nn_params_epochs_every_round: int = 1,
    ) -> None:
        self.num_clusters = num_clusters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.bn = bn
        self.adjoint = adjoint
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.update_nn_params_epochs_every_round = update_nn_params_epochs_every_round

    def fit(
        self,
        dataset: Dataset | SimulatedDataset | SimulatedDatasetForTorch,
        time_key: str = "t",
        obs_key: str = "x",
        id_key: str = "id",
        label_key: str | None = None,
        static_vars_key: str | None = None,
    ) -> None:
        """训练模型"""

        if isinstance(dataset, SimulatedDataset):
            dataset = SimulatedDatasetForTorch(dataset)

        loader = DataLoader(
            dataset,  # type: ignore
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=TrajectoryDatasetCollateFunc(
                time_key=time_key,
                obs_key=obs_key,
                id_key=id_key,
                label_key=label_key,
                static_vars_key=static_vars_key,
            ),
        )

        batch1 = next(iter(loader))
        obs_dim = batch1["x"].shape[-1]
        static_dim = 0 if static_vars_key is None else batch1["z"].shape[-1]

        # 初始化模型
        self.model_ = NODETC(
            obs_dim=obs_dim,
            latent_dim=obs_dim,
            static_dim=static_dim,
            num_clusters=self.num_clusters,
            bn=self.bn,
            adjoint=self.adjoint,
            init_state_encoder=static_dim > 0,
            activation=torch.nn.GELU,
        )

        # 初始化训练器
        self.trainer_ = EMTrainer(
            model=self.model_,
            loader=loader,
            num_epochs=self.num_epochs,
            lr=self.learning_rate,
            device=self.device,
            update_nn_params_epochs_every_round=self.update_nn_params_epochs_every_round,
        )

        # 训练模型, 记录训练历史
        history = self.trainer_.train()
        self.history_ = pd.DataFrame.from_records(history)

        # 得到最终属于各自集群的概率
        responsibilities = self.trainer_.e_step(loader, False)
        self.responsibilities_ = responsibilities.cpu().numpy()

    def plot_vector_field(self) -> Figure:
        return self.trainer_.plot_vector_field()
