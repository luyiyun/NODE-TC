from dataclasses import replace

# import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from node_tc.simulate import SimulatedDataset, SimulatedDataCollateFunc
from node_tc.model import NODETC, EMTrainer


def main():
    # --- 参数设置 ---
    NUM_PATIENTS = 1000
    NUM_CLUSTERS = 3
    OBS_DIM = 2  # 观测维度
    LATENT_DIM = 2  # 潜在维度
    STATIC_DIM = 0  # 静态变量维度
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001

    # 1. 生成数据
    simu_data = SimulatedDataset.simulate(
        num_patients=NUM_PATIENTS,
        num_clusters=NUM_CLUSTERS,
        obs_dim=OBS_DIM,
        latent_dim=LATENT_DIM,
        static_dim=STATIC_DIM,
        missing_rate=0.0,
        noise_std_per_cluster=(0.1, 0.1, 0.1),
        seed=42,
        num_time_internval=(5, 11),
        time_interval=(1, 11),
        z0=1.0,
    )
    simu_data.write_csv("./data/simulate/example1/")

    # obs_all = []
    # for sample in simu_data.samples:
    #     for obs in sample.observations:
    #         obs_all.append(obs)
    # obs_all = np.concatenate(obs_all, axis=0)
    # obs_mean, obs_std = np.mean(obs_all), np.std(obs_all)
    simu_data.set_transform(
        lambda x: replace(
            x,
            t=x.t / 10,  # observations=(x.observations - obs_mean) / obs_std
        )
    )
    fig = simu_data.plot(num_samples_per_cluster=3, seed=42)
    fig.savefig("simulated_data.png")

    loader = DataLoader(
        simu_data,  # type: ignore
        batch_size=64,
        shuffle=True,
        collate_fn=SimulatedDataCollateFunc(),
    )
    batch = next(iter(loader))
    print(",".join(f"{k}:{tuple(v.shape)}" for k, v in batch.items()))

    model = NODETC(
        obs_dim=OBS_DIM,
        latent_dim=LATENT_DIM,
        static_dim=STATIC_DIM,
        num_clusters=NUM_CLUSTERS,
        bn=False,
        adjoint=False,
        init_state_encoder=False,
        activation=nn.GELU,
        method="rk4",
        options={"step_size": 0.1},
    )
    trainer = EMTrainer(
        model=model,
        loader=loader,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        update_nn_params_epochs_every_round=2,
    )
    trainer.train()

    fig = trainer.plot_vector_field()
    fig.savefig("vector_field.png")


if __name__ == "__main__":
    main()
