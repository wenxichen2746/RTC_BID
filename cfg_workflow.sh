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

#0812
# uv run src/cfg_train_flow.py --config.run-path ./logs-expert/hard_lunar_lander_cfg_a4o1_0730_run2/ --config.level-paths "worlds/l/hard_lunar_lander.json" --config.wandb_name "LL_cfg_a4o1_0812_dropa3o0"
# uv run src/cfg_train_flow.py --config.run-path ./logs-expert/grasp_elavated_a4o1 --config.level-paths "worlds/c/grasp_elavated.json" --config.wandb_name "GraspE_cfg_a4o1_0812_dropa3o0"
# uv run src/cfg_eval_flow_single_LLvel.py --run_path ./logs-bc/GraspE_cfg_a4o1_0812_dropa3o0 --level-paths "worlds/c/grasp_elavated.json" --output-dir ./logs-eval-cfg/GraspE_cfg_dropa3o0_0816
# uv run src/cfg_eval_flow_single_LLvel.py --run_path ./logs-bc/LL_cfg_a4o1_0812_dropa3o0 --level-paths "worlds/l/hard_lunar_lander.json" --output-dir ./logs-eval-cfg/LL_cfg_dropa3o0_0816
# uv run src/cfg_eval_flow.py --run_path ./logs-bc/GraspE_cfg_a4o1_0807 --level-paths "worlds/c/grasp_elavated.json" --output-dir ./logs-eval-cfg/GraspE_cfg_dropa3o0_0817
# uv run src/cfg_eval_flow.py --run_path ./logs-bc/LL_cfg_a4o1_0804 --level-paths "worlds/l/hard_lunar_lander.json" --output-dir ./logs-eval-cfg/LL_cfg_dropa3o0_0817

# uv run src/cfg_eval_flow.py --run_path ./logs-bc/LL_cfg_a4o1_0804 --level-paths "worlds/l/hard_lunar_lander.json" --output-dir ./logs-eval-cfg/LL_cfg_dropa3o0_0818
# uv run src/cfg_eval_flow.py --run_path ./logs-bc/GraspE_cfg_a4o1_0807 --level-paths "worlds/c/grasp_elavated.json" --output-dir ./logs-eval-cfg/GraspE_cfg_dropa3o0_0818

# uv run src/cfg_eval_flow.py --run_path ./logs-bc/LL_cfg_a4o1_0804 --level-paths "worlds/l/hard_lunar_lander.json" --output-dir ./logs-eval-cfg/LL_cfg_0819_static
# uv run src/cfg_eval_flow.py --run_path ./logs-bc/GraspE_cfg_a4o1_0807 --level-paths "worlds/c/grasp_elavated.json" --output-dir ./logs-eval-cfg/GraspE_cfg_0819_static

# uv run src/cfg_train_expert.py --config.level-paths "worlds/c/place_can_easy.json"  --config.wandb_name "place_can_easy_a4o1_dropa3o3"
# uv run src/cfg_generate_data_dr.py --config.run-path ./logs-expert/place_can_easy_a4o1_dropa3o3 --config.level-path "worlds/c/place_can_easy.json"
# uv run src/cfg_train_flow.py --config.run-path ./logs-expert/place_can_easy_a4o1_dropa3o3 --config.level-paths "worlds/c/place_can_easy.json" --config.wandb_name "place_can_easy_a4o1_dropa3o3_0823"

# uv run src/cfg_train_expert.py --config.level-paths "worlds/c/toss_bin.json"  --config.wandb_name "toss_bin_a4o1_0823" --config.num_seeds 12 --config.seed 77
# uv run src/cfg_generate_data_dr.py --config.run-path ./logs-expert/toss_bin_a4o1_0823 --config.level-path "worlds/c/toss_bin.json"
# uv run src/cfg_train_flow.py --config.run-path ./logs-expert/toss_bin_a4o1_0823 --config.level-paths "worlds/c/toss_bin.json" --config.wandb_name "toss_bin_a4o1_0823_a4o1_dropa3o3_0823"


# uv run src/cfg_eval_flow.py --run_path ./logs-bc/toss_bin_a4o1_0823_a4o1_dropa3o3_0823 --level-paths "worlds/c/toss_bin.json" --output-dir ./logs-eval-cfg/08_24_toss_bin
# uv run src/cfg_eval_flow.py --run_path ./logs-bc/place_can_easy_a4o1_dropa3o3_0823 --level-paths "worlds/c/place_can_easy.json" --output-dir ./logs-eval-cfg/08_24_place_can_easy
# uv run src/cfg_eval_flow.py --run_path ./logs-bc/LL_cfg_a4o1_0804 --level-paths "worlds/l/hard_lunar_lander.json" --output-dir ./logs-eval-cfg/08_24_LL
# uv run src/cfg_eval_flow.py --run_path ./logs-bc/GraspE_cfg_a4o1_0807 --level-paths "worlds/c/grasp_elavated.json" --output-dir ./logs-eval-cfg/08_24_GraspE


# XLA_PYTHON_CLIENT_PREALLOCATE=false \
# XLA_PYTHON_CLIENT_ALLOCATOR=platform \
# uv run --with nvidia-ml-py3 src/cfg_eval_flow_batched.py \
#   --run_path ./logs-bc/place_can_easy_a4o1_dropa3o3_0823 \
#   --output-dir ./logs-eval-cfg/08_24_place_can_easy \
#   --level-paths worlds/c/place_can_easy.json


# XLA_PYTHON_CLIENT_PREALLOCATE=false \
# uv run src/cfg_eval_flow.py \
#   --run_path ./logs-bc/LL_cfg_a4o1_0804 \
#   --output-dir ./logs-eval-cfg/08_30_LL \
#   --level-paths worlds/l/hard_lunar_lander.json

# XLA_PYTHON_CLIENT_PREALLOCATE=false \
# uv run src/cfg_eval_flow.py \
#   --run_path ./logs-bc/GraspE_cfg_a4o1_0807 \
#   --output-dir ./logs-eval-cfg/08_30_GraspE \
#   --level-paths worlds/c/grasp_elavated.json

# XLA_PYTHON_CLIENT_PREALLOCATE=false \
# uv run src/cfg_eval_flow.py \
#   --run_path ./logs-bc/place_can_easy_a4o1_dropa3o3_0823 \
#   --output-dir ./logs-eval-cfg/08_30_place_can_easy \
#   --level-paths worlds/c/place_can_easy.json

# XLA_PYTHON_CLIENT_PREALLOCATE=false \
# uv run src/cfg_eval_flow.py \
#   --run_path ./logs-bc/toss_bin_a4o1_0823_a4o1_dropa3o3_0823 \
#   --output-dir ./logs-eval-cfg/08_30_toss_bin \
#   --level-paths worlds/c/toss_bin.json



# 0907 runs add reward preference to expert training
# uv run src/cfg_train_expert.py --config.level-paths "worlds/l/hard_lunar_lander.json"  --config.wandb_name "0908_hard_lunar_lander"
# uv run src/cfg_train_expert.py --config.level-paths "worlds/c/grasp_elavated.json"  --config.wandb_name "0908_grasp_elavated"
# uv run src/cfg_train_expert.py --config.level-paths "worlds/c/place_can_easy.json"  --config.wandb_name "0908_place_can_easy"
# uv run src/cfg_train_expert.py --config.level-paths "worlds/c/toss_bin.json"  --config.wandb_name "0908_toss_bin" 

# uv run src/cfg_generate_data_dr.py --config.run-path ./logs-expert/0908_hard_lunar_lander --config.level-path "worlds/l/hard_lunar_lander.json"
# uv run src/cfg_generate_data_dr.py --config.run-path ./logs-expert/0908_place_can_easy --config.level-path "worlds/c/place_can_easy.json"
# uv run src/cfg_generate_data_dr.py --config.run-path ./logs-expert/0908_toss_bin --config.level-path "worlds/c/toss_bin.json"


# uv run src/cfg_train_flow.py --config.run-path ./logs-expert/0908_hard_lunar_lander/ --config.level-paths "worlds/l/hard_lunar_lander.json" --config.wandb_name "0908_hard_lunar_lander"
# uv run src/cfg_train_flow.py --config.run-path ./logs-expert/0908_toss_bin --config.level-paths "worlds/c/toss_bin.json" --config.wandb_name "0908_toss_bin"
# uv run src/cfg_train_flow.py --config.run-path ./logs-expert/0908_place_can_easy --config.level-paths "worlds/c/place_can_easy.json" --config.wandb_name "0908_place_can_easy"


# uv run src/cfg_eval_flow.py --run_path ./logs-bc/0908_toss_bin --level-paths "worlds/c/toss_bin.json" --output-dir ./logs-eval-cfg/0908_toss_bin
# # uv run src/cfg_eval_flow.py \
# #   --run_path ./logs-bc/toss_bin_a4o1_0823_a4o1_dropa3o3_0823 \
# #   --output-dir ./logs-eval-cfg/08_23_nodiversereward_toss_bin \
# #   --level-paths worlds/c/toss_bin.json

# uv run src/cfg_eval_flow.py --run_path ./logs-bc/0908_place_can_easy --level-paths "worlds/c/place_can_easy.json" --output-dir ./logs-eval-cfg/0908_place_can_easy
# # uv run src/cfg_eval_flow.py \
# #   --run_path ./logs-bc/place_can_easy_a4o1_dropa3o3_0823 \
# #   --output-dir ./logs-eval-cfg/08_23_nodiversereward_place_can_easy \
# #   --level-paths worlds/c/place_can_easy.json

# uv run src/cfg_eval_flow.py --run_path ./logs-bc/0908_hard_lunar_lander --level-paths "worlds/l/hard_lunar_lander.json" --output-dir ./logs-eval-cfg/0908_hard_lunar_lander
# # uv run src/cfg_eval_flow.py \
# #   --run_path ./logs-bc/LL_cfg_a4o1_0804 \
# #   --output-dir ./logs-eval-cfg/08_23_nodiversereward_LL \
# #   --level-paths worlds/l/hard_lunar_lander.json
# # uv run src/cfg_eval_flow.py --run_path ./logs-bc/GraspE_cfg_a4o1_0807 --level-paths "worlds/c/grasp_elavated.json" --output-dir ./logs-eval-cfg/08_24_GraspE

uv run src/cfg_train_expert.py --config.level-paths "worlds/c/grasp_elavated.json"  --config.wandb_name "0908_grasp_elavated"
uv run src/cfg_generate_data_dr.py --config.run-path ./logs-expert/0908_grasp_elavated --config.level-path "worlds/c/grasp_elavated.json"
uv run src/cfg_train_flow.py --config.run-path ./logs-expert/0908_grasp_elavated/ --config.level-paths "worlds/c/grasp_elavated.json" --config.wandb_name "0908_grasp_elavated"
uv run src/cfg_eval_flow.py --run_path ./logs-bc/0908_grasp_elavated --level-paths "worlds/c/grasp_elavated.json" --output-dir ./logs-eval-cfg/0908_grasp_elavated
