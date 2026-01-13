from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
import json

from node_tc.simulate.dataset import SimulatedDataset, SimulatedDatasetForTorch
from node_tc import NODETrajectoryCluster


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/example1/",
        help="结果保存目录, 默认是./results/example1_{datetime.now().strftime('%Y%m%d_%H%M%S')}/",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/simulate/example1_20251128_194457/",
        help="数据集目录, 默认是./data/simulate/example1_20251128_194457/",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["simulate", "real"],
        default="simulate",
        help="数据类型, 默认是simulate",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=3,
        help="聚类数量, 默认是3",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="学习率, 默认是0.001",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="训练轮数, 默认是100",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="批次大小, 默认是64",
    )
    parser.add_argument(
        "--bn",
        action="store_true",
        help="是否使用批归一化, 默认是False",
    )
    parser.add_argument(
        "--adjoint",
        action="store_true",
        help="是否使用adjoint方法, 默认是False",
    )
    parser.add_argument(
        "--update_nn_params_epochs_every_round",
        type=int,
        default=2,
        help="每轮更新神经网络参数的轮数, 默认是2",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    save_dir = Path(
        args.save_dir.rstrip("/") + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    )

    # --- 数据集加载 ---
    if args.data_type == "simulate":
        simu_data = SimulatedDataset.read_csv(data_dir)

        with open(data_dir / "args.json", "r") as f:
            simu_args = json.load(f)
        max_t = simu_args["num_time_interval"][1] - 1

        def transform(x):
            x["t"] = x["t"] / max_t
            return x

        dataset = SimulatedDatasetForTorch(simu_data, transform)
    else:
        raise NotImplementedError("暂不支持真实数据集")

    # --- 模型训练 ---
    model = NODETrajectoryCluster(
        num_clusters=args.num_clusters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        bn=args.bn,
        adjoint=args.adjoint,
        update_nn_params_epochs_every_round=args.update_nn_params_epochs_every_round,
    )

    model.fit(
        dataset,
        time_key="t",
        obs_key="x",
        id_key="id",
        label_key="y",
        static_vars_key="z",
    )

    # --- 结果保存 ---
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f)
    model.save_model(save_dir / "model.pt")
    model.save_history(save_dir / "history.csv")


if __name__ == "__main__":
    main()
