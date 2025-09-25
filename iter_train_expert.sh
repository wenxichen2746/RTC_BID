#!/usr/bin/env bash
set -euo pipefail

levels=(
  "worlds/l/catapult.json"
  "worlds/l/cartpole_thrust.json"
  "worlds/l/hard_lunar_lander.json"
  "worlds/l/mjc_half_cheetah.json"
  "worlds/l/mjc_swimmer.json"
  "worlds/l/mjc_walker.json"
  "worlds/l/h17_unicycle.json"
  "worlds/l/chain_lander.json"
  "worlds/l/catcher_v3.json"
  "worlds/l/trampoline.json"
  "worlds/l/car_launch.json"
  "worlds/l/grasp_easy.json"
)

# Iterate levels and run: train expert -> generate data -> train BC -> eval BC
for level in "${levels[@]}"; do
  name="$(basename "$level" .json)"
  echo "\n=== Level: $level | run_name=$name ==="

  # Step 1: Train expert policies (writes to ./logs-expert/${name})
  echo "[1/4] Training expert..."
  uv run src/train_expert.py \
    --config.level-paths "$level" \
    --config.run_name "$name"

  # Step 2: Generate trajectories from best expert checkpoints
  # Data saved to ./logs-expert/${name}/data
  echo "[2/4] Generating data..."
  uv run src/generate_data.py \
    --config.run-path "./logs-expert/${name}" \
    --config.level-paths "$level"

  # Step 3: Train imitation (flow) policies
  # Name BC run the same as expert via WANDB_NAME for consistent local path ./logs-bc/${name}
  echo "[3/4] Training flow (BC)..."
  WANDB_NAME="$name" uv run src/train_flow.py \
    --config.run-path "./logs-expert/${name}" \
    --config.level-paths "$level"

  # Step 4: Evaluate BC policies (default methods; results.csv under ./logs-eval/${name})
  echo "[4/4] Evaluating flow..."
  uv run src/eval_flow.py \
    --run-path "./logs-bc/${name}" \
    --output-dir "./logs-eval/${name}" \
    --level-paths "$level"

  echo "=== Completed pipeline for ${name} ===\n"
done



# uv run src/train_expert_dr.py --config.level-paths "worlds/l/hard_lunar_lander.json"  --config.env_target_x 1.0 --config.wandb_name "hard_lunar_lander_t1"
# uv run src/train_expert_dr.py --config.level-paths "worlds/l/hard_lunar_lander.json"  --config.env_target_x 2.0 --config.wandb_name "hard_lunar_lander_t2"
# uv run src/train_expert_dr.py --config.level-paths "worlds/l/hard_lunar_lander.json"  --config.env_target_x 3.0 --config.wandb_name "hard_lunar_lander_t3"
# uv run src/train_expert_dr.py --config.level-paths "worlds/l/hard_lunar_lander.json"  --config.env_target_x 4.0 --config.wandb_name "hard_lunar_lander_t4"

#new dr script
# uv run src/train_expert_dr.py --config.level-paths "worlds/l/hard_lunar_lander.json"  --config.wandb_name "hard_lunar_lander_t1-4_r2"
#0730 dr+history of actions and obs
# uv run src/cfg_train_expert.py --config.level-paths "worlds/l/hard_lunar_lander.json"  --config.wandb_name "hard_lunar_lander_cfg_a4o1_0730"
