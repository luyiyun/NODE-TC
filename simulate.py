from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from node_tc.simulate import linear
from node_tc.simulate.dataset import write_simulate_data_to_csv
from node_tc.plot import TrajectoriesPlotter


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="./data/simulate/example1/", help="数据保存路径"
    )
    parser.add_argument(
        "--num_patients", type=int, default=1000, help="生成数据集的病人数量"
    )
    parser.add_argument(
        "--num_clusters", type=int, default=3, help="生成数据集的集群数量"
    )
    parser.add_argument("--obs_dim", type=int, default=5, help="生成数据集的观测维度")
    parser.add_argument(
        "--static_dim", type=int, default=3, help="生成数据集的静态变量维度"
    )
    parser.add_argument(
        "--missing_rate", type=float, default=0.0, help="生成数据集的缺失率"
    )
    parser.add_argument(
        "--noise_std", type=float, default=0.1, help="生成数据集的噪声标准差"
    )
    parser.add_argument("--seed", type=int, default=42, help="生成数据集的随机种子")
    parser.add_argument(
        "--num_time_internval",
        type=int,
        nargs=2,
        default=[5, 11],
        help="生成数据集的时间区间",
    )
    parser.add_argument(
        "--plot", action="store_true", help="是否绘制生成数据集的轨迹图"
    )
    args = parser.parse_args()

    data_dir = Path(
        args.data_dir.rstrip("/") + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    print(f"数据保存路径: {data_dir}")

    simu_data, dynamics = linear.simulate(
        num_patients=args.num_patients,
        num_clusters=args.num_clusters,
        obs_dim=args.obs_dim,
        static_dim=args.static_dim,
        missing_rate=args.missing_rate,
        noise_std=args.noise_std,
        seed=args.seed,
        num_time_internval=args.num_time_internval,
    )
    write_simulate_data_to_csv(data_dir, simu_data)

    if args.plot:
        sample = simu_data.samples[0]
        plotter = TrajectoriesPlotter(
            t=sample.t,
            observations=sample.observations,
            trajectories_dfdt=dynamics[sample.true_cluster],
        )
        fig = plotter.plot_trajectories()
        fig.savefig(data_dir / "sampled_trajectory.png")
        fig = plotter.plot_trajectories_2d()
        fig.savefig(data_dir / "sampled_trajectory_2d.png")


if __name__ == "__main__":
    main()
