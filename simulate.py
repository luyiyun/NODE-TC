from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import json

from node_tc.simulate import linear
from node_tc.plot import TrajectoriesPlotter


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/simulate/example1/",
        help="数据保存路径, 默认为./data/simulate/example1/",
    )
    parser.add_argument(
        "--num_patients",
        type=int,
        default=1000,
        help="生成数据集的病人数量, 默认为1000",
    )
    parser.add_argument(
        "--num_clusters", type=int, default=3, help="生成数据集的集群数量, 默认为3"
    )
    parser.add_argument(
        "--obs_dim", type=int, default=5, help="生成数据集的观测维度, 默认为5"
    )
    parser.add_argument(
        "--static_dim", type=int, default=3, help="生成数据集的静态变量维度, 默认为3"
    )
    parser.add_argument(
        "--missing_rate", type=float, default=0.0, help="生成数据集的缺失率, 默认为0.0"
    )
    parser.add_argument(
        "--noise_std", type=float, default=0.1, help="生成数据集的噪声标准差, 默认为0.1"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="生成数据集的随机种子, 默认为42"
    )
    parser.add_argument(
        "--num_time_interval",
        type=int,
        nargs=2,
        default=[5, 11],
        help="生成数据集的时间区间, 默认为[5, 11]",
    )
    parser.add_argument(
        "--plot", action="store_true", help="是否绘制生成数据集的轨迹图, 默认不绘制"
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
        num_time_interval=args.num_time_interval,
    )
    simu_data.to_csv(data_dir)

    with open(data_dir / "args.json", "w") as f:
        json.dump(vars(args), f)

    if args.plot:
        print("绘制生成数据集的轨迹图...")
        sample = simu_data.samples[0]
        plotter = TrajectoriesPlotter(
            t=sample.t,
            observations=sample.obs,
            trajectories_dfdt=dynamics[sample.true_cluster],
        )
        fig = plotter.plot_trajectories()
        fig.savefig(data_dir / "sampled_trajectory.png")
        fig = plotter.plot_trajectories_2d()
        fig.savefig(data_dir / "sampled_trajectory_2d.png")


if __name__ == "__main__":
    main()
