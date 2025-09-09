#!/bin/bash

# uv run tools/trajectory/plot_multi_traj.py --level_name toss_bin --policy_dir "./logs-expert/toss_bin_a4o1_0823" --level_paths "worlds/c"
# uv run tools/trajectory/plot_multi_traj.py --level_name toss_bin --policy_dir "./logs-expert/toss_bin_a4o1_0823" --level_paths "worlds/c" --switch_control_interval 10

# # uv run tools/trajectory/plot_multi_traj.py --level_name place_can_easy --policy_dir "./logs-expert/place_can_easy_a4o1_dropa3o3" --level_paths "worlds/c"
# uv run tools/trajectory/plot_multi_traj.py --level_name place_can_easy --policy_dir "./logs-expert/place_can_easy_a4o1_dropa3o3" --level_paths "worlds/c" --switch_control_interval 10

# # uv run tools/trajectory/plot_multi_traj.py --level_name grasp_elavated --policy_dir "./logs-expert/grasp_elavated_a4o1" --level_paths "worlds/c"
# uv run tools/trajectory/plot_multi_traj.py --level_name grasp_elavated --policy_dir "./logs-expert/grasp_elavated_a4o1" --level_paths "worlds/c" --switch_control_interval 10

# # uv run tools/trajectory/plot_multi_traj.py --level_name hard_lunar_lander --policy_dir "./logs-expert/hard_lunar_lander_cfg_a4o1_0730_run2" --level_paths "worlds/l"
# uv run tools/trajectory/plot_multi_traj.py --level_name hard_lunar_lander --policy_dir "./logs-expert/hard_lunar_lander_cfg_a4o1_0730_run2" --level_paths "worlds/l" --switch_control_interval 10

uv run tools/trajectory/plot_multi_traj.py --level_name toss_bin --policy_dir "./logs-expert/0907_toss_bin" --level_paths "worlds/c" --switch_control_interval 10
uv run tools/trajectory/plot_multi_traj.py --level_name place_can_easy --policy_dir "./logs-expert/0907_place_can_easy" --level_paths "worlds/c" --switch_control_interval 10
uv run tools/trajectory/plot_multi_traj.py --level_name grasp_elavated --policy_dir "./logs-expert/0907_grasp_elavated" --level_paths "worlds/c" --switch_control_interval 10
uv run tools/trajectory/plot_multi_traj.py --level_name hard_lunar_lander --policy_dir "./logs-expert/0907_hard_lunar_lander" --level_paths "worlds/l" --switch_control_interval 10

