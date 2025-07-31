#!/bin/bash


# uv run src/cfg_train_expert.py --config.level-paths "worlds/l/hard_lunar_lander.json"  --config.wandb_name "hard_lunar_lander_cfg_a4o1_0730"
# uv run src/cfg_train_expert.py --config.level-paths "worlds/l/hard_lunar_lander.json"  --config.wandb_name "hard_lunar_lander_cfg_a4o1_0730_run2"
# uv run src/cfg_train_expert.py --config.level-paths "worlds/l/hard_lunar_lander.json"  --config.wandb_name "hard_lunar_lander_cfg_a4o2_0730"


# uv run src/cfg_generate_data_dr.py --config.run-path ./logs-expert/hard_lunar_lander_cfg_a4o1_0730_run2



uv run src/cfg_train_flow.py --config.run-path ./logs-expert/hard_lunar_lander_cfg_a4o1_0730_run2/ --config.level-paths "worlds/l/hard_lunar_lander.json" --config.wandb_name "LL_cfg_a4o1_0730_run2"