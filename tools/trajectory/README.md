# Trajectory Visualization Tools

This directory contains tools for visualizing Kinetix environment scenes and trajectories from trained policies.

## Scripts Overview

### `plot_kinetix_scene.py`
Visualizes the Kinetix environment scene layout, showing the static environment structure without any trajectories.

**Usage:**
```bash
python plot_kinetix_scene.py --level_name <environment_name> --level_paths <path_to_env_json>
```

### `plot_single_traj.py`
Plots and saves trajectories from a single environment using one specific seed.

**Usage:**
```bash
python plot_single_traj.py --level_name <environment_name> --policy_seed <seed> --policy_iter_num <iteration> --level_paths <path_to_env_json>
```

### `plot_multi_traj.py`
Plots and saves trajectories from multiple different policies trained under the same Kinetix environment but with different seeds. Useful for comparing policy performance across different random initializations.

**Usage:**
```bash
python plot_multi_traj.py \
  --level_name <environment_name> \
  --policy_seeds "seed1,seed2,seed3" \
  --policy_iter_num <iteration> \
  --level_paths <path_to_env_json>
```

You can pass seeds comma- or space-separated, e.g. `--policy_seeds "0,1,2"` or `--policy_seeds "0 1 2"`.

## Parameters

- **`--policy_iter_num`**: The iteration steps of the train expert (the last folder under the )
- **`--policy_seed`**: The training seed used to train the expert policy
- **`--level_name`**: Name of the JSON file defining the environment (without .json extension)
- **`--level_paths`**: The folder path containing the environment JSON files

## Policy File Structure

The scripts assume policies are saved under the following structure:
```
./logs/logs-expert/{level_name}/seed_{policy_seed}/{policy_iter_num}/policies/worlds_l_{level_name}.pkl
```

## Project Structure

```
logs/
├── logs-data/
├── logs-figures/  
└── logs-expert/             
        └─{level_name}
                └─seed_0
                └─seed_1
                └─...
                └─seed_{policy_seed}
                    └─0
                    └─20
                    └─40
                    └─...
                    └─{policy_iter_num}
                            └─{policies}
                                └─worlds_l_{level_name}.pkl
                            └─{stats}
                            └─{videos}
                                 
src/
└── train_expert.py     

tools/
└── trajectory/
    ├── plot_kinetix_scene.py    # Environment visualization
    ├── plot_single_traj.py      # Single trajectory plotting
    └── plot_multi_traj.py       # Multi-trajectory comparison
```
