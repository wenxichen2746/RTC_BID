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

# uv run src/cfg_train_expert.py --config.level-paths "worlds/c/grasp_elavated.json"  --config.wandb_name "0908_grasp_elavated"
# uv run src/cfg_generate_data_dr.py --config.run-path ./logs-expert/0908_grasp_elavated --config.level-path "worlds/c/grasp_elavated.json"
# uv run src/cfg_train_flow.py --config.run-path ./logs-expert/0908_grasp_elavated/ --config.level-paths "worlds/c/grasp_elavated.json" --config.wandb_name "0908_grasp_elavated"
# uv run src/cfg_eval_flow.py --run_path ./logs-bc/0908_grasp_elavated --level-paths "worlds/c/grasp_elavated.json" --output-dir ./logs-eval-cfg/0908_grasp_elavated



# uv run src/cfg_train_expert.py --config.level-paths "worlds/c/drone.json"  --config.wandb_name "0926_drone"
# uv run src/cfg_generate_data_dr.py --config.run-path ./logs-expert/0926_drone --config.level-path "worlds/c/drone.json"
# uv run src/cfg_train_flow.py --config.run-path ./logs-expert/0926_drone/ --config.level-paths "worlds/c/drone.json" --config.wandb_name "0926_drone"


# uv run src/cfg_eval_flow.py --run_path ./logs-bc/0908_grasp_elavated --level-paths "worlds/c/grasp_elavated.json" --output-dir ./logs-eval-cfg/0908_grasp_elavated_sup
# uv run src/cfg_eval_flow.py --run_path ./logs-bc/0926_drone --level-paths "worlds/c/drone.json" --output-dir ./logs-eval-cfg/0930_drone
# uv run src/cfg_eval_flow.py --run_path ./logs-bc/0908_hard_lunar_lander --level-paths "worlds/l/hard_lunar_lander.json" --output-dir ./logs-eval-cfg/0930_hard_lunar_lander
# uv run src/cfg_eval_flow.py --run_path ./logs-bc/0908_place_can_easy --level-paths "worlds/c/place_can_easy.json" --output-dir ./logs-eval-cfg/0930_place_can_easy
# uv run src/cfg_eval_flow.py --run_path ./logs-bc/0908_toss_bin --level-paths "worlds/c/toss_bin.json" --output-dir ./logs-eval-cfg/0930_toss_bin

# -------- 1001 batch pipelines --------
# set -euo pipefail

# ENV_BATCH_1001=(
#   "grasp_easy worlds/l/grasp_easy.json"
#   # "chain_lander worlds/l/chain_lander.json"
#   # "car_launch worlds/l/car_launch.json"
#   # "trampoline worlds/l/trampoline.json"
#   # "insert_key worlds/c/insert_key.json"
# # #   "whip worlds/c/whip.json"
#   "catapult worlds/l/catapult.json"
# )

# echo "===== Step 1: Train experts (1001 batch) ====="
# for entry in "${ENV_BATCH_1001[@]}"; do
#   IFS=' ' read -r ENV_NAME LEVEL_PATH <<< "${entry}"
#   RUN_NAME="1001_${ENV_NAME}_diversereward"
#   echo "--- [Step 1] ${RUN_NAME}"
#   uv run src/cfg_train_expert.py     --config.level-paths "${LEVEL_PATH}"     --config.wandb_name "${RUN_NAME}"
# done

# echo "
# ===== Step 2: Generate domain-randomized data (1001 batch) ====="
# for entry in "${ENV_BATCH_1001[@]}"; do
#   IFS=' ' read -r ENV_NAME LEVEL_PATH <<< "${entry}"
#   RUN_NAME="1001_${ENV_NAME}_diversereward"
#   echo "--- [Step 2] ${RUN_NAME}"
#   uv run src/cfg_generate_data_dr.py     --config.run-path "./logs-expert/${RUN_NAME}"     --config.level-path "${LEVEL_PATH}"
# done

# echo "
# ===== Step 3: Train flows (1001 batch) ====="
# for entry in "${ENV_BATCH_1001[@]}"; do
#   IFS=' ' read -r ENV_NAME LEVEL_PATH <<< "${entry}"
#   RUN_NAME="1001_${ENV_NAME}_diversereward"
#   echo "--- [Step 3] ${RUN_NAME}"
#   uv run src/cfg_train_flow.py     --config.run-path "./logs-expert/${RUN_NAME}"     --config.level-paths "${LEVEL_PATH}"     --config.wandb_name "${RUN_NAME}"
# done

# echo "
# ===== Step 4: Evaluate flows (1001 batch) ====="
# for entry in "${ENV_BATCH_1001[@]}"; do
#   IFS=' ' read -r ENV_NAME LEVEL_PATH <<< "${entry}"
#   RUN_NAME="1001_${ENV_NAME}_diversereward"
#   echo "--- [Step 4] ${RUN_NAME}"
#   uv run src/cfg_eval_flow.py     --run_path "./logs-bc/${RUN_NAME}"     --level-paths "${LEVEL_PATH}"     --output-dir "./logs-eval-cfg/${RUN_NAME}"
# done
# uv run src/cfg_eval_flow.py     --run_path "./logs-bc/${RUN_NAME}"     --level-paths "${LEVEL_PATH}"     --output-dir "./logs-eval-cfg/${RUN_NAME}"

# uv run src/cfg_eval_flow.py --run_path ./logs-bc/run1009 --level-paths "worlds/l/catapult.json" "worlds/c/place_can_easy.json" "worlds/c/toss_bin.json" "worlds/l/grasp_easy.json" --output-dir ./logs-eval-cfg/1009_4envs
# uv run src/cfg_eval_flow.py --run_path ./logs-bc/1009_catapult_diversereward --level-paths "worlds/l/catapult.json" --output-dir ./logs-eval-cfg/1012_catapult
# uv run src/cfg_eval_flow.py --run_path ./logs-bc/0908_place_can_easy --level-paths "worlds/c/place_can_easy.json" --output-dir ./logs-eval-cfg/1012_place_can_easy
# uv run src/cfg_eval_flow.py --run_path ./logs-bc/0908_toss_bin --level-paths "worlds/c/toss_bin.json" --output-dir ./logs-eval-cfg/1012_toss_bin
# uv run src/cfg_eval_flow.py --run_path ./logs-bc/1009_grasp_easy_diversereward --level-paths "worlds/l/grasp_easy.json" --output-dir ./logs-eval-cfg/1012_grasp_easy


# uv run src/cfg_eval_flow_dynamicenv.py --run_path ./logs-bc/0908_place_can_easy --level-paths "worlds/c/place_can_easy.json" --output-dir ./logs-eval-cfg/1009_place_can_easy
# uv run src/cfg_eval_flow_dynamicenv.py --run_path ./logs-bc/0908_toss_bin --level-paths "worlds/c/toss_bin.json" --output-dir ./logs-eval-cfg/1009_toss_bin
# uv run src/cfg_eval_flow_dynamicenv.py --run_path ./logs-bc/1009_grasp_easy_diversereward --level-paths "worlds/l/grasp_easy.json" --output-dir ./logs-eval-cfg/1009_grasp_easy
# uv run src/cfg_eval_flow_dynamicenv.py --run_path ./logs-bc/1009_catapult_diversereward --level-paths "worlds/l/catapult.json" --output-dir ./logs-eval-cfg/1009_catapult



ENV_BATCH_1015=(
  "grasp_easy worlds/l/grasp_easy.json"
  "place_can_easy worlds/c/place_can_easy.json"
  "toss_bin worlds/c/toss_bin.json" 
  "catapult worlds/l/catapult.json"
)
run_with_retry() {
  local retry_count=0
  until "$@"
  do
    retry_count=$((retry_count + 1))
    if [[ ${retry_count} -ge ${MAX_RETRIES} ]]; then
      echo "!!!!!! Command failed after ${MAX_RETRIES} attempts. Moving on. !!!!!!"
      return 1 # Return a failure code
    fi
    echo "****** Command failed. Retrying in ${RETRY_DELAY}s (Attempt ${retry_count}/${MAX_RETRIES}) ******"
    sleep ${RETRY_DELAY}
  done
  return 0 # Return a success code
}




for entry in "${ENV_BATCH_1015[@]}"; do
  IFS=' ' read -r ENV_NAME LEVEL_PATH <<< "${entry}"
  RUN_NAME="1110_${ENV_NAME}_a0o1"
  BC_RUN_NAME="1110_a6o1_${ENV_NAME}"
  echo "--- ${RUN_NAME}  ----"
  
  # echo "===== Step 1: Train experts ====="
  # run_with_retry uv run src/cfg_train_expert.py  --config.act_history_length 0   --config.level-paths "${LEVEL_PATH}"     --config.wandb_name "${RUN_NAME}"
  
  echo "===== Step 2: Generate domain-randomized data  ====="
  uv run src/cfg_generate_data_dr.py  --config.act_history_length 6  --config.run-path "./logs-expert/${RUN_NAME}"     --config.level-path "${LEVEL_PATH}"
  
  echo "===== Step 3: Train flows ====="
  uv run src/cfg_train_flow.py  --config.act_history_length 6  --config.run-path "./logs-expert/${RUN_NAME}" --config.level-paths "${LEVEL_PATH}" \
   --config.wandb_name "${BC_RUN_NAME}"
done

PYTHONPATH=src python3 src/util/merge_bc_policies.py \
  --source-root logs-bc \
  --target-root logs-bc/1110_a6o1_4envs \
  --run-prefix 1110_a6o1_ \
  --iterations 5 35


echo "
===== Step 4: Evaluate flows ====="



uv run src/cfg_eval_flow.py --run_path ./logs-bc/1110_a6o1_4envs \
 --config.act_history_length 6  --level-paths "worlds/l/catapult.json" "worlds/c/place_can_easy.json" "worlds/c/toss_bin.json" "worlds/l/grasp_easy.json" \
  --output-dir ./logs-eval-cfg/1110_4envs_a6 
