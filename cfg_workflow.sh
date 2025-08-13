#!/bin/bash


# uv run src/cfg_train_expert.py --config.level-paths "worlds/l/hard_lunar_lander.json"  --config.wandb_name "hard_lunar_lander_cfg_a4o1_0730"
# uv run src/cfg_train_expert.py --config.level-paths "worlds/l/hard_lunar_lander.json"  --config.wandb_name "hard_lunar_lander_cfg_a4o1_0730_run2"
# uv run src/cfg_train_expert.py --config.level-paths "worlds/l/hard_lunar_lander.json"  --config.wandb_name "hard_lunar_lander_cfg_a4o2_0730"


# uv run src/cfg_generate_data_dr.py --config.run-path ./logs-expert/hard_lunar_lander_cfg_a4o1_0730_run2



# uv run src/cfg_train_flow.py --config.run-path ./logs-expert/hard_lunar_lander_cfg_a4o1_0730_run2/ --config.level-paths "worlds/l/hard_lunar_lander.json" --config.wandb_name "LL_cfg_a4o1_0730_run2"

# uv run src/cfg_eval_flow_single_LLvel.py --run_path ./logs-bc/LL_cfg_a4o1_0730_run2 --output-dir ./logs-eval-cfg/LL_cfg_a4o1_0730_run2_eval0731 --level-paths "worlds/l/hard_lunar_lander.json"


#0804

# uv run src/cfg_generate_data_dr.py --config.run-path ./logs-expert/hard_lunar_lander_cfg_a4o1_0730_run2
# uv run src/cfg_train_flow.py --config.run-path ./logs-expert/hard_lunar_lander_cfg_a4o1_0730_run2/ --config.level-paths "worlds/l/hard_lunar_lander.json" --config.wandb_name "LL_cfg_a4o1_0804"
# uv run src/cfg_eval_flow_single_LLvel.py --run_path ./logs-bc/LL_cfg_a4o1_0804 --output-dir ./logs-eval-cfg/LL_cfg_a4o1_0804 --level-paths "worlds/l/hard_lunar_lander.json"
# uv run src/cfg_eval_flow_single_LLvel.py --run_path ./logs-bc/LL_cfg_a4o1_0804 --output-dir ./logs-eval-cfg/LL_cfg_a4o1_0812 --level-paths "worlds/l/hard_lunar_lander.json"


#0806
# uv run src/cfg_train_expert.py --config.level-paths "worlds/c/grasp_elavated.json"  --config.wandb_name "grasp_elavated_a4o1"
# uv run src/cfg_generate_data_dr.py --config.run-path ./logs-expert/grasp_elavated_a4o1 --config.level-path "worlds/c/grasp_elavated.json"
# uv run src/cfg_train_flow.py --config.run-path ./logs-expert/grasp_elavated_a4o1 --config.level-paths "worlds/c/grasp_elavated.json" --config.wandb_name "GraspE_cfg_a4o1_0807"
# uv run src/cfg_eval_flow_single_LLvel.py --run_path ./logs-bc/GraspE_cfg_a4o1_0807 --level-paths "worlds/c/grasp_elavated.json" --output-dir ./logs-eval-cfg/GraspE_cfg_a4o1_0812

# uv run src/cfg_train_expert.py --config.level-paths "worlds/c/reach_avoid.json"  --config.wandb_name "reach_avoid_a4o1"

uv run src/cfg_train_flow.py --config.run-path ./logs-expert/hard_lunar_lander_cfg_a4o1_0730_run2/ --config.level-paths "worlds/l/hard_lunar_lander.json" --config.wandb_name "LL_cfg_a4o1_0812_dropa3o0"
uv run src/cfg_train_flow.py --config.run-path ./logs-expert/grasp_elavated_a4o1 --config.level-paths "worlds/c/grasp_elavated.json" --config.wandb_name "GraspE_cfg_a4o1_0812_dropa3o0"

