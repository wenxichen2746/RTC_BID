#!/bin/bash

# levels=(
#     "worlds/l/catapult.json"
#     "worlds/l/cartpole_thrust.json"
#     "worlds/l/hard_lunar_lander.json"
#     "worlds/l/mjc_half_cheetah.json"
#     "worlds/l/mjc_swimmer.json"
#     "worlds/l/mjc_walker.json"
#     "worlds/l/h17_unicycle.json"
#     "worlds/l/chain_lander.json"
#     "worlds/l/catcher_v3.json"
#     "worlds/l/trampoline.json"
#     "worlds/l/car_launch.json"
#     "worlds/l/grasp_easy.json"
# )

# for level in "${levels[@]}"; do
#     echo "Running training on level: $level"
#     uv run src/train_expert.py --config.level-paths "$level"
# done

# uv run src/train_expert_dr.py --config.level-paths "worlds/l/hard_lunar_lander.json"  --config.env_target_x 1.0 --config.wandb_name "hard_lunar_lander_t1"
# uv run src/train_expert_dr.py --config.level-paths "worlds/l/hard_lunar_lander.json"  --config.env_target_x 2.0 --config.wandb_name "hard_lunar_lander_t2"
# uv run src/train_expert_dr.py --config.level-paths "worlds/l/hard_lunar_lander.json"  --config.env_target_x 3.0 --config.wandb_name "hard_lunar_lander_t3"
# uv run src/train_expert_dr.py --config.level-paths "worlds/l/hard_lunar_lander.json"  --config.env_target_x 4.0 --config.wandb_name "hard_lunar_lander_t4"

#new dr script
# uv run src/train_expert_dr.py --config.level-paths "worlds/l/hard_lunar_lander.json"  --config.wandb_name "hard_lunar_lander_t1-4_r2"
#0730 dr+history of actions and obs
uv run src/cfg_train_expert.py --config.level-paths "worlds/l/hard_lunar_lander.json"  --config.wandb_name "hard_lunar_lander_cfg_a4o1_0730"