from typing import Callable

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.integrate import solve_ivp
# from matplotlib.axes import Axes

from .estimator import NODETrajectoryCluster


class TrajectoriesPlotter:
    def __init__(
        self,
        t: np.ndarray,
        observations: np.ndarray,
        trajectories: Callable[[np.ndarray], np.ndarray] | None = None,
        trajectories_dfdt: Callable[[float, np.ndarray], np.ndarray] | None = None,
        nodetc_estimator: NODETrajectoryCluster | None = None,
        nodetc_estimator_k: int | None = None,
        num_intervals: int = 100,
    ) -> None:
        assert t.ndim == 1, f"t should be 1D array, got shape {t.shape}"
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)
        assert observations.ndim == 2, (
            f"observations should be 2D array, got shape {observations.shape}"
        )

        if (
            trajectories is not None
            or trajectories_dfdt is not None
            or nodetc_estimator is not None
        ):
            assert (
                (trajectories is not None)
                + (trajectories_dfdt is not None)
                + (nodetc_estimator is not None)
            ) == 1, (
                "Only one of trajectories, trajectories_dfdt, or nodetc_estimator should be provided"
            )

            t_min, t_max = np.min(t), np.max(t)
            t_eval = np.linspace(t_min, t_max, num_intervals)
            if trajectories is not None:
                x = trajectories(t_eval)
            elif trajectories_dfdt is not None:
                y0 = observations[t == t_min][0]
                x = solve_ivp(
                    trajectories_dfdt,
                    [t_min, t_max],
                    y0,
                    t_eval=t_eval,
                ).y.T
            elif nodetc_estimator is not None:
                assert nodetc_estimator_k is not None, (
                    "nodetc_estimator_k must be provided when using nodetc_estimator"
                )
                y0 = torch.tensor(
                    observations[t == t_min][0],
                    dtype=torch.float32,
                    device=nodetc_estimator.device,
                )
                t_eval = torch.tensor(
                    t_eval, dtype=torch.float32, device=nodetc_estimator.device
                )
                with torch.no_grad():
                    nodetc_estimator.model_.eval()
                    x = (
                        nodetc_estimator.model_.solve_ivp(
                            nodetc_estimator_k, y0, t_eval
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
            assert x.shape[1] == observations.shape[1], (
                "trajectories and observations must have the same number of dimensions"
            )

        self.t = t
        self.observations = observations
        self.ndim = self.observations.shape[1]
        self.flag_has_trajectories = (
            trajectories is not None
            or trajectories_dfdt is not None
            or nodetc_estimator is not None
        )
        if self.flag_has_trajectories:
            self.t_eval = t_eval
            self.x = x

    def plot_trajectories(self, fig: Figure | None = None) -> Figure:
        n_row = int(np.sqrt(self.ndim))
        n_col = int(np.ceil(self.ndim / n_row))

        if fig is None:
            fig = plt.figure(figsize=(n_col * 4, n_row * 3))

        axs = fig.subplots(ncols=n_col, nrows=n_row)
        axs = axs.flatten()

        for i in range(self.ndim):
            ax = axs[i]
            ax.plot(self.t, self.observations[:, i], "o", label=f"Dim {i + 1}")
            if self.flag_has_trajectories:
                ax.plot(self.t_eval, self.x[:, i], "-b", label=f"Dim {i + 1}")
            ax.grid(True)

        fig.tight_layout()
        return fig

    def plot_trajectories_2d(self, fig: Figure | None = None) -> Figure:
        if fig is None:
            fig = plt.figure(figsize=(3 * self.ndim, 3 * self.ndim))

        axs = fig.subplots(ncols=self.ndim, nrows=self.ndim, squeeze=False)
        for i in range(self.ndim):
            for j in range(self.ndim):
                if i == j:
                    axs[i, j].axis("off")
                    continue
                ax = axs[i, j]
                ax.scatter(
                    self.observations[:, j],
                    self.observations[:, i],
                    color="black",
                    label="Observations",
                )
                if self.flag_has_trajectories:
                    ax.plot(
                        self.x[:, j],
                        self.x[:, i],
                        "-r",
                        label="Trajectory",
                    )
                ax.set_xlabel(f"Dim {j + 1}")
                ax.set_ylabel(f"Dim {i + 1}")
                ax.grid(True)

        fig.tight_layout()
        return fig


# class VectorFieldPlotter:
#     def __init__(
#         self,
#         ndim: int,
#         trajectories_dfdt: Callable[[float, np.ndarray], np.ndarray] | None = None,
#         nodetc_estimator: NODETrajectoryCluster | None = None,
#         nodetc_estimator_k: int | None = None,
#         observations: np.ndarray | None = None,
#         field_limit: tuple[np.ndarray, np.ndarray] | None = None,
#         num_intervals: int = 100,
#     ) -> None:
#         if observations is not None:
#             if observations.ndim == 1:
#                 observations = observations.reshape(-1, 1)
#             assert observations.ndim == 2, (
#                 f"observations should be 2D array, got shape {observations.shape}"
#             )

#         assert (
#             (trajectories_dfdt is not None) + (nodetc_estimator is not None)
#         ) == 1, (
#             "Only one of trajectories, trajectories_dfdt, or nodetc_estimator should be provided"
#         )

#         # 计算向量场
#         if field_limit is None:
#             if nodetc_estimator is not None:
#                 start = nodetc_estimator.trainer_._obs_start
#                 end = nodetc_estimator.trainer_._obs_end
#                 assert start is not None and end is not None, (
#                     "Cannot determine field limits from nodetc_estimator"
#                 )
#             elif observations is not None:
#                 start, end = np.min(observations), np.max(observations)
#             else:
#                 start, end = np.ones(ndim) * -5, np.ones(ndim) * 5
#         else:
#             start, end = field_limit
#         xy_range = np.linspace(start, end, num=num_intervals)
#         assert xy_range.shape[1] == ndim, (
#             f"xy_range and ndim must have the same dimensions, got {xy_range.shape[1]} and {ndim}"
#         )

#         self.ndim = ndim
#         self.xy_range = xy_range
#         self.trajectories_dfdt = trajectories_dfdt
#         self.nodetc_estimator = nodetc_estimator
#         self.nodetc_estimator_k = nodetc_estimator_k
#         self.observations = observations
#         # self.observations = observations
#         # self.ndim = self.observations.shape[1]
#         # self.flag_has_trajectories = (
#         #     trajectories is not None or trajectories_dfdt is not None
#         # )
#         # if self.flag_has_trajectories:
#         #     self.t_eval = t_eval
#         #     self.x = x

#     def plot(self, fig: Figure | None = None) -> Figure:
#         if fig is None:
#             fig = plt.figure(figsize=(3 * self.ndim, 3 * self.ndim))

#         axs = fig.subplots(ncols=self.ndim, nrows=self.ndim, squeeze=False)
#         for i in range(self.ndim):
#             for j in range(self.ndim):
#                 if i == j:
#                     continue
#                 ax = axs[i, j]

#                 X, Y = np.meshgrid(self.xy_range[:, i], self.xy_range[:, j])


#                 grid_points = torch.tensor(
#                     np.stack([X.flatten(), Y.flatten()], axis=-1),
#                     dtype=torch.float32,
#                     device=self.device,
#                 )
