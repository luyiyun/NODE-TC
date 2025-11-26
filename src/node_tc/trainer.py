from typing import overload, Literal
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from .model import NODETC


def compute_entropy(
    probabilities: torch.Tensor, normalize: bool = True
) -> torch.Tensor:
    """计算熵"""
    res = -torch.sum(probabilities * torch.log(probabilities + 1e-6), dim=1)
    if normalize:
        res = res / torch.log(
            torch.tensor(
                probabilities.shape[1], dtype=torch.float32, device=probabilities.device
            )
        )
    return res


class EMTrainer:
    def __init__(
        self,
        model: NODETC,
        loader: DataLoader,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        lr: float = 1e-2,
        num_epochs: int = 50,
        update_nn_params_epochs_every_round: int = 1,
    ):
        self.model = model
        self.loader = loader
        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        self.update_nn_params_epochs_every_round = update_nn_params_epochs_every_round

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self._obs_start: np.ndarray | None = None
        self._obs_end: np.ndarray | None = None

    @overload
    def e_step(
        self, loader: DataLoader, return_true_clusters: Literal[False]
    ) -> torch.Tensor:
        pass

    @overload
    def e_step(
        self, loader: DataLoader, return_true_clusters: Literal[True]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def e_step(
        self, loader: DataLoader, return_true_clusters: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        responsibilities, sid, true_clusters = [], [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="E-Step:", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                llk = self.model.get_log_likelihoods(batch)
                # 联合对数概率 log(pi_k * p(X_i|k))
                joint_log_prob = llk + self.model.log_prior  # type:ignore
                # 边际对数概率 log(sum_k pi_k * p(X_i|k))
                log_marginal_prob = torch.logsumexp(joint_log_prob, dim=1, keepdim=True)
                # 后验对数概率 log(w_ik)
                log_respon = joint_log_prob - log_marginal_prob
                respon = torch.exp(log_respon)

                responsibilities.append(respon)
                sid.append(batch["id"])
                if return_true_clusters:
                    true_clusters.append(batch["y"])

        responsibilities = torch.cat(responsibilities, dim=0)
        sid = torch.cat(sid, dim=0)
        indice = torch.argsort(sid)
        responsibilities = responsibilities[indice]
        if return_true_clusters:
            true_clusters = torch.cat(true_clusters, dim=0)
            true_clusters = true_clusters[indice]
            return responsibilities, true_clusters

        return responsibilities

    def update_nn_params(
        self, loader: DataLoader, responsibilities: torch.Tensor
    ) -> float:
        self.model.train()
        loss_epoch = 0.0
        for batch in tqdm(loader, desc="M-Step(Update NN params):", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            llk = self.model.get_log_likelihoods(batch)
            respon_k = responsibilities[batch["id"]]
            # 计算损失函数
            loss = self.model.compute_loss(respon_k.detach(), llk)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_epoch += loss.item()
        return loss_epoch / len(loader)

    def update_pi(self, responsibilities: torch.Tensor):
        # M-Step: 更新模型参数
        # 参数1：簇权重 (logits形式，通过softmax保证和为1)
        pi_new = torch.mean(responsibilities, dim=0)
        self.model.log_prior.data = torch.log(pi_new + 1e-6)

    def update_cov(self, loader: DataLoader, responsibilities: torch.Tensor):
        # 参数2：协方差 (对角阵，学习log(sigma^2))
        residue_sum, weight_sum = (
            torch.tensor(0.0, dtype=torch.float32, device=self.device),
            torch.tensor(0.0, dtype=torch.float32, device=self.device),
        )
        with torch.no_grad():
            for batch in tqdm(loader, desc="M-Step(Update cov):", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                residue_i = self.model.get_residue(batch)
                respon_i = responsibilities[batch["id"]]
                respon_i_weight = respon_i[..., None] * batch["mask"][:, None, :]
                residue_sum = residue_sum + (
                    residue_i.pow(2) * respon_i_weight[..., None]
                ).sum(dim=(0, 2))
                weight_sum = weight_sum + respon_i_weight.sum(dim=(0, 2))

        var = residue_sum / weight_sum[:, None]
        self.model.log_vars.data = torch.log(var.clamp(min=1e-6))

    def find_observation_range(self):
        obs_start_list, obs_end_list = [], []
        with torch.no_grad():
            for batch in tqdm(
                self.loader, desc="Finding observation range:", leave=False
            ):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                obs = batch["x"]  # (B, T, D)
                mask = batch["mask"]  # (B, T)

                i, j = torch.nonzero(mask == 1.0, as_tuple=True)
                obs_start_list.append(obs[i, j].min(dim=0).values)
                obs_end_list.append(obs[i, j].max(dim=0).values)

        self._obs_start = (
            torch.stack(obs_start_list, dim=0).min(dim=0)[0].detach().cpu().numpy()
        )
        self._obs_end = (
            torch.stack(obs_end_list, dim=0).max(dim=0)[0].detach().cpu().numpy()
        )

    def train(self) -> list[dict[str, float | int]]:
        self.find_observation_range()

        best_ari = 0.0
        best_model = None
        best_epoch = -1

        history = []
        responsibilities = self.e_step(self.loader, False)  # initial responsibilities
        for epoch in tqdm(range(self.num_epochs), desc="Training: "):
            self.update_pi(responsibilities)
            self.update_cov(self.loader, responsibilities)

            for _ in range(self.update_nn_params_epochs_every_round):
                loss = self.update_nn_params(self.loader, responsibilities)
                tqdm.write(f"Epoch {epoch + 1} | Loss: {loss:.4f}")

            responsibilities, true_clusters = self.e_step(
                self.loader, True
            )  # update responsibilities

            pred_clusters = torch.argmax(responsibilities, dim=1).cpu().detach().numpy()
            true_clusters = true_clusters.cpu().detach().numpy()
            ari = adjusted_rand_score(true_clusters, pred_clusters)
            entropy = compute_entropy(responsibilities).mean().item()
            tqdm.write(
                f"Epoch {epoch + 1} | Adjusted Rand Index: {ari:.4f}, Entropy: {entropy:.4f}"
            )
            history.append({"epoch": epoch + 1, "ari": ari, "entropy": entropy})

            if best_ari <= ari:
                best_model = deepcopy(self.model.state_dict())
                best_ari = ari
                best_epoch = epoch

        print(f"Best ARI: {best_ari:.4f} at epoch {best_epoch}")
        if best_model is not None:
            self.model.load_state_dict(best_model)

        print("\n训练完成!")
        return history

    def plot_vector_field(
        self,
        start: np.ndarray | None = None,
        end: np.ndarray | None = None,
        n_steps: int = 20,
    ) -> Figure:
        start = start or self._obs_start
        end = end or self._obs_end

        assert start is not None and end is not None, "observation range not found"
        assert start.shape == end.shape == (2,), "start and end must be 2D vectors"

        # 创建一个包含两个子图的画布
        fig, axs = plt.subplots(
            1, self.model.num_clusters, figsize=(6 * self.model.num_clusters, 6)
        )

        # --- 子图 2: 学习到的向量场 ---
        x_range = np.linspace(start[0], end[0], n_steps)
        y_range = np.linspace(start[1], end[1], n_steps)
        X, Y = np.meshgrid(x_range, y_range)
        grid_points = torch.tensor(
            np.stack([X.flatten(), Y.flatten()], axis=-1),
            dtype=torch.float32,
            device=self.device,
        )

        for i in range(self.model.num_clusters):
            net = self.model.ode_funcs[i]
            dz_dt = net(0, grid_points)
            u = dz_dt[:, 0].reshape(n_steps, n_steps).detach().cpu().numpy()
            v = dz_dt[:, 1].reshape(n_steps, n_steps).detach().cpu().numpy()

            ax = axs[i]
            ax.streamplot(
                X,
                Y,
                u,
                v,
                color=("red", 0.8),
                linewidth=0.7,
                density=1.0,
                broken_streamlines=True,
                arrowstyle="->",
            )

            ax.set_xlabel("Z_1")
            ax.set_ylabel("Z_2")
            ax.set_title(f"Cluster {i + 1}")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(start[0], end[0])
            ax.set_ylim(start[1], end[1])
            ax.grid(True)

        fig.tight_layout()
        return fig
