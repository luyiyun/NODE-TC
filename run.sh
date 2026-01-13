# cluster 3
# uv run simulate.py --data_dir ./data/simulate/example --num_clusters 3 --obs_dim 5 --static_dim 3 --noise_std 0.1 --missing_rate 0.1 --num_time_interval 5 11
# uv run train.py --data_dir ./data/simulate/example_20260113_120246 --num_clusters 3 --num_epochs 100 --learning_rate 0.001 --batch_size 64 --update_nn_params_epochs_every_round 2 --save_dir ./results/example

# cluster 5
# uv run simulate.py --data_dir ./data/simulate/example_5 --num_clusters 5 --obs_dim 5 --static_dim 3 --noise_std 0.1 --missing_rate 0 --num_time_interval 5 11
uv run train.py --data_dir ./data/simulate/example_5_20260113_172343 --num_clusters 5 --num_epochs 100 --learning_rate 0.001 --batch_size 64 --update_nn_params_epochs_every_round 2 --save_dir ./results/example_5