from typing import overload, Literal, Type
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint, odeint
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(
        self,
        inpt_dim: int,
        oupt_dim: int = 1,
        hiddens: list[int] = [64, 64],
        activation: Type[nn.Module] = nn.Tanh,
        bn: bool = False,
        dropout: float = 0.0,
    ):
        super(MLP, self).__init__()

        net = []
        for i, o in zip([inpt_dim] + hiddens[:-1], hiddens):
            net.append(nn.Linear(i, o))
            if bn:
                net.append(nn.BatchNorm1d(o))
            net.append(activation())
            if dropout > 0:
                net.append(nn.Dropout(dropout))
        net.append(nn.Linear(hiddens[-1], oupt_dim))
        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ODEFunc(nn.Module):
    """学习ODE动态的神经网络 f(z, t; theta)"""

    def __init__(
        self,
        inpt_dim: int,
        oupt_dim: int = 1,
        hiddens: list[int] = [64, 64],
        activation: Type[nn.Module] = nn.Tanh,
        bn: bool = False,
        dropout: float = 0.0,
        autonomous: bool = True,
    ):
        super(ODEFunc, self).__init__()

        self.autonomous = autonomous
        self.net = MLP(
            inpt_dim if self.autonomous else inpt_dim + 1,
            oupt_dim,
            hiddens,
            activation,
            bn,
            dropout,
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.autonomous:
            return self.net(z)

        return self.net(torch.cat([z, t.unsqueeze(1)], dim=1))


class InitStateEncoder(nn.Module):
    """根据静态变量和首次观测推断初始状态 z0"""

    def __init__(
        self,
        obs_dim: int,
        static_dim: int,
        latent_dim: int,
        hiddens: list[int] = [64, 64],
        activation: Type[nn.Module] = nn.Tanh,
        bn: bool = False,
        dropout: float = 0.0,
    ):
        super(InitStateEncoder, self).__init__()
        self.net = MLP(
            obs_dim + static_dim,
            latent_dim,
            hiddens=hiddens,
            activation=activation,
            bn=bn,
            dropout=dropout,
        )

    def forward(
        self, first_obs: torch.Tensor, static_vars: torch.Tensor
    ) -> torch.Tensor:
        input_data = torch.cat([first_obs, static_vars], dim=1)
        return self.net(input_data)


class NODETC(nn.Module):
    """动态混合神经ODE模型"""

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        static_dim: int,
        num_clusters: int,
        hiddens: list[int] = [64, 64],
        activation: Type[nn.Module] = nn.Tanh,
        bn: bool = False,
        dropout: float = 0.0,
        autonomous: bool = True,
        adjoint: bool = True,
        init_state_encoder: bool = False,
        method: str | None = None,
        options: dict | None = None,
    ):
        super(NODETC, self).__init__()
        self.num_clusters = num_clusters
        self.obs_dim = obs_dim
        self.adjoint = adjoint
        self.init_state_encoder = init_state_encoder
        self.method = method
        self.options = options

        # 为每个簇创建一个独立的ODE函数
        self.ode_funcs = nn.ModuleList(
            [
                ODEFunc(
                    latent_dim, latent_dim, hiddens, activation, bn, dropout, autonomous
                )
                for _ in range(num_clusters)
            ]
        )

        # 创建一个共享的编码器
        if self.init_state_encoder:
            self.encoder = InitStateEncoder(
                obs_dim, static_dim, latent_dim, hiddens, activation, bn, dropout
            )
        else:
            self.init_state = nn.Parameter(torch.randn(latent_dim))

        # 如果观测维度和潜在维度不同，需要一个解码器
        if obs_dim != latent_dim:
            self.decoder = nn.Linear(latent_dim, obs_dim)
        else:
            self.decoder = nn.Identity()

        # 模型参数
        # 簇权重 (logits形式，通过softmax保证和为1)
        self.register_buffer(
            "log_prior", torch.log_softmax(torch.zeros(num_clusters), dim=0)
        )

        # 每个簇的协方差 (这里简化为对角阵，学习log(sigma^2))
        self.register_buffer("log_vars", torch.randn(num_clusters, obs_dim))

    def forward(self, patient_data: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        t = patient_data["t"]  # (T,)
        obs = patient_data["observations"]  # (B, T, D)

        if self.init_state_encoder:
            z0 = obs[:, 0]  # (B, D)
            static = patient_data["static_vars"]  # (B, D)
            z0 = self.encoder(z0, static)
        else:
            z0 = self.init_state.repeat(obs.shape[0], 1)

        preds = []
        for k in range(self.num_clusters):
            # 1. 求解ODE得到潜在轨迹
            if self.adjoint:
                pred_z = odeint_adjoint(
                    self.ode_funcs[k], z0, t, method=self.method, options=self.options
                )  # (T, B, D)
            else:
                pred_z = odeint(
                    self.ode_funcs[k], z0, t, method=self.method, options=self.options
                )  # (T, B, D)
            assert torch.is_tensor(pred_z), "ODE solver must return a tensor"
            pred_z = pred_z.permute(1, 0, 2)  # (B, T, D)

            # 2. 解码到观测空间
            pred_i = self.decoder(pred_z)
            preds.append(pred_i)

        return preds

    def get_residue(self, patient_data: dict[str, torch.Tensor]) -> torch.Tensor:
        obs = patient_data["observations"]  # (B, T, D)
        mask = patient_data["mask"]  # (B, T)
        preds = self.forward(patient_data)

        residue = []
        for pred_obs in preds:
            # 3. 计算残差
            residue_k = (obs - pred_obs) * mask[..., None]
            residue.append(residue_k)

        return torch.stack(residue, dim=1)  # (B, K, T, D)

    def get_log_likelihoods(
        self, patient_data: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """计算单个患者数据在每个簇下的对数似然"""
        obs = patient_data["observations"]  # (B, T, D)
        mask = patient_data["mask"]  # (B, T)

        preds = self.forward(patient_data)

        log_likelihoods_k = []
        for k, pred_obs in enumerate(preds):
            # 3. 计算高斯对数似然
            # 使用torch.distributions简化计算
            dist = torch.distributions.Normal(
                loc=pred_obs,
                scale=torch.exp(0.5 * self.log_vars[k]),  # type:ignore
            )
            # 假设各维度独立，对数似然是各维度和各时间点之和
            log_p = dist.log_prob(obs).mul(mask[..., None]).sum(dim=(1, 2))
            # log_p = log_p / (mask.sum(dim=1) * self.obs_dim)
            log_likelihoods_k.append(log_p)

        return torch.stack(log_likelihoods_k, dim=1)  # (B, K)

    def compute_loss(
        self, responsibilities: torch.Tensor, log_likelihoods: torch.Tensor
    ) -> torch.Tensor:
        """计算M-step的损失函数 (Q函数)"""
        # 加权的对数似然
        weighted_log_likelihood = (responsibilities * log_likelihoods).sum(dim=1).mean()

        # EM算法的目标是最大化Q函数，等价于最小化其负值
        return -weighted_log_likelihood


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
        batch_size: int = 128,
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
    def e_step(self, return_true_clusters: Literal[False]) -> torch.Tensor:
        pass

    @overload
    def e_step(
        self, return_true_clusters: Literal[True]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def e_step(
        self, return_true_clusters: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        responsibilities, sid, true_clusters = [], [], []
        with torch.no_grad():
            for batch in tqdm(self.loader, desc="E-Step:", leave=False):
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
                    true_clusters.append(batch["true_cluster"])

        responsibilities = torch.cat(responsibilities, dim=0)
        sid = torch.cat(sid, dim=0)
        indice = torch.argsort(sid)
        responsibilities = responsibilities[indice]
        if return_true_clusters:
            true_clusters = torch.cat(true_clusters, dim=0)
            true_clusters = true_clusters[indice]
            return responsibilities, true_clusters

        return responsibilities

    def update_nn_params(self, responsibilities: torch.Tensor) -> float:
        self.model.train()
        loss_epoch = 0.0
        for batch in tqdm(self.loader, desc="M-Step(Update NN params):", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            llk = self.model.get_log_likelihoods(batch)
            respon_k = responsibilities[batch["id"]]
            # 计算损失函数
            loss = self.model.compute_loss(respon_k.detach(), llk)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_epoch += loss.item()
        return loss_epoch / len(self.loader)

    def update_pi(self, responsibilities: torch.Tensor):
        # M-Step: 更新模型参数
        # 参数1：簇权重 (logits形式，通过softmax保证和为1)
        pi_new = torch.mean(responsibilities, dim=0)
        self.model.log_prior.data = torch.log(pi_new + 1e-6)

    def update_cov(self, responsibilities: torch.Tensor):
        # 参数2：协方差 (对角阵，学习log(sigma^2))
        residue_sum, weight_sum = (
            torch.tensor(0.0, dtype=torch.float32, device=self.device),
            torch.tensor(0.0, dtype=torch.float32, device=self.device),
        )
        with torch.no_grad():
            for batch in tqdm(self.loader, desc="M-Step(Update cov):", leave=False):
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
                obs = batch["observations"]  # (B, T, D)
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

    def train(self):
        self.find_observation_range()

        best_ari = 0.0
        best_model = None
        best_epoch = -1

        responsibilities = self.e_step(False)  # initial responsibilities
        for epoch in tqdm(range(self.num_epochs), desc="Training: "):
            self.update_pi(responsibilities)
            self.update_cov(responsibilities)

            for _ in range(self.update_nn_params_epochs_every_round):
                loss = self.update_nn_params(responsibilities)
                tqdm.write(f"Epoch {epoch + 1} | Loss: {loss:.4f}")

            responsibilities, true_clusters = self.e_step(
                True
            )  # update responsibilities

            pred_clusters = torch.argmax(responsibilities, dim=1).cpu().detach().numpy()
            true_clusters = true_clusters.cpu().detach().numpy()
            ari = adjusted_rand_score(true_clusters, pred_clusters)
            entropy = compute_entropy(responsibilities).mean().item()
            tqdm.write(
                f"Epoch {epoch + 1} | Adjusted Rand Index: {ari:.4f}, Entropy: {entropy:.4f}"
            )

            if best_ari <= ari:
                best_model = deepcopy(self.model.state_dict())
                best_ari = ari
                best_epoch = epoch

        print(f"Best ARI: {best_ari:.4f} at epoch {best_epoch}")
        if best_model is not None:
            self.model.load_state_dict(best_model)

        print("\n训练完成!")
        return self.model

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
