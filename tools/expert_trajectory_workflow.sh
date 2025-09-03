#!/bin/bash

uv run tools/trajectory/plot_multi_traj.py --level_name toss_bin --policy_dir "./logs-expert/toss_bin_a4o1_0823" --policy_seeds 0,2,5,7,8,11 --level_paths "worlds/c" 