from node_tc.simulate import (
    simulate,
    write_simulate_data_to_csv,
    SimulatedDatasetForTorch,
)
from node_tc import NODETrajectoryCluster


def main():
    # --- 参数设置 ---
    NUM_PATIENTS = 1000
    NUM_CLUSTERS = 3
    OBS_DIM = 2  # 观测维度
    LATENT_DIM = 2  # 潜在维度
    STATIC_DIM = 3  # 静态变量维度
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001

    # 1. 生成数据
    simu_data = simulate(
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
    write_simulate_data_to_csv("./data/simulate/example1/", simu_data)

    # fig = simu_data.plot(num_samples_per_cluster=3, seed=42)
    # fig.savefig("simulated_data.png")

    def transform(x):
        x["t"] = x["t"] / 10
        return x

    simu_data_torch = SimulatedDatasetForTorch(simu_data, transform)

    model = NODETrajectoryCluster(
        num_clusters=NUM_CLUSTERS,
        batch_size=64,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        bn=False,
        adjoint=False,
        update_nn_params_epochs_every_round=2,
    )

    model.fit(
        simu_data_torch,
        time_key="t",
        obs_key="x",
        id_key="id",
        label_key="y",
        static_vars_key="z",
    )

    fig = model.plot_vector_field()
    fig.savefig("vector_field.png")


if __name__ == "__main__":
    main()
