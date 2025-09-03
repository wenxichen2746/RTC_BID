import pathlib
import dataclasses
import argparse
import re
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
from traj_utils import (
    generate_trajectory, 
    plot_scene_base, 
    get_green_obj_shape,
    draw_agent_shape_at_position, 
    build_policy_path
)


@dataclasses.dataclass
class Config:
    policy_seeds: Sequence[int] = (0, 1, 3, 4)  # Multiple seeds
    # level_name: str = "car_climbing_ver2"
    level_name: str = "swing_up_hard_ver1"
    policy_dir: str = "./logs/logs-expert"
    policy_iter_num: int = 500
    level_paths: str = "worlds/l"
    sim_seed: int = 4
    sim_max_step: int = 300


def plot_scene_and_multiple_trajectories(json_filepath, traj_dict, save_path=None):
    """Plot the full scene from a JSON file and overlay multiple agent trajectories."""
    fig, ax = plt.subplots(figsize=(12, 12))
    ax, data = plot_scene_base(json_filepath, ax)
    ax.set_title("Trajectories from Multiple Policy Seeds")
    
    green_obj_shape = get_green_obj_shape(data)
    colors = plt.cm.tab20.colors
    
    for i, (seed, traj) in enumerate(traj_dict.items()):
        green_pos = traj['pos']
        if green_pos.size == 0:
            continue
            
        color = colors[i]
        
        # Plot the path line
        ax.plot(green_pos[:, 0], green_pos[:, 1], 
                color=color, linewidth=3, alpha=0.8, label=f'Seed {seed}')
        
        # Mark the end point
        ax.scatter(green_pos[-1, 0], green_pos[-1, 1], 
                   color=color, s=120, marker='X', edgecolors='white', 
                   zorder=5, label=f'_End Seed {seed}')
        
        # Draw green object's shape at the end of trajectory
        green_traj_end = (green_pos[-1, 0], green_pos[-1, 1])
        # Only add a single legend entry for the agent shape
        add_label = 'Agent Final Position' if i == 0 else None
        draw_agent_shape_at_position(ax, green_obj_shape, green_traj_end, traj['rot'][-1], label=add_label)
    
    # Add a single starting point
    first_traj = next(iter(traj_dict.values()))
    if first_traj['pos'].size > 0:
        ax.scatter(first_traj['pos'][0, 0], first_traj['pos'][0, 1], 
                   color='cyan', s=200, marker='o', edgecolors='black', 
                   zorder=6, label='Start')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(-1, 6)
    ax.set_ylim(-0.5, 5.5)
    
    if save_path:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main(config: Config):
    level_path = f"{config.level_paths}/{config.level_name}.json"
    trajs = {}
    
    print(f"Loading level: {level_path}")
    print("-" * 60)
    
    for policy_seed in config.policy_seeds:
        policy_path = build_policy_path(
            config.policy_dir, config.level_name, policy_seed,
            config.policy_iter_num, config.level_paths
        )

        if not pathlib.Path(policy_path).exists():
            print(f"Policy file not found for seed {policy_seed}: {policy_path}")
            continue
        
        print(f"Found policy for seed {policy_seed}: {policy_path}")
        print(f"Generating trajectory for seed {policy_seed}...")
        
        traj = generate_trajectory(policy_path, level_path, 
                                 max_steps=config.sim_max_step, seed=config.sim_seed)
        trajs[policy_seed] = traj
        
        if traj['dones'].size > 0:
             print(f"Seed {policy_seed} - Episode finished: {traj['dones'][-1]} in {traj['steps'][-1]} steps.")
        else:
            print(f"Seed {policy_seed} - No steps were taken.")
        print("-" * 60)
    
    if trajs:
        # Plot trajectories together
        seeds_str = "_".join(map(str, config.policy_seeds))
        figure_save_path = f"./logs/logs-figures/{config.level_name}_seeds_{seeds_str}.png"

        print("Plotting all trajectories...")
        plot_scene_and_multiple_trajectories(level_path, trajs, save_path=figure_save_path)
        
        # Save trajectory data
        output_path = f"./logs/logs-data/traj_data_seeds_{seeds_str}.npz"
        combined_data = {}
        for seed, traj in trajs.items():
            for key, value in traj.items():
                combined_data[f"seed_{seed}_{key}"] = value
        # Ensure output directory exists
        out_pathlib = pathlib.Path(output_path)
        out_pathlib.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_pathlib, **combined_data)
        print(f"All trajectory data saved to: {output_path}")
    else:
        print("No valid trajectories found!")


if __name__ == "__main__":
    # Parse CLI arguments to override defaults in Config
    parser = argparse.ArgumentParser(description="Plot multiple trajectories from different policy seeds.")
    parser.add_argument("--level_name", type=str, default=Config.level_name,
                        help="Environment level name (without .json)")
    parser.add_argument("--policy_dir", type=str, default=Config.policy_dir,
                        help="Base directory containing expert logs")
    parser.add_argument("--policy_iter_num", type=int, default=Config.policy_iter_num,
                        help="Training iteration number to load policies from")
    parser.add_argument("--level_paths", type=str, default=Config.level_paths,
                        help="Folder path to environment JSONs")
    parser.add_argument("--sim_seed", type=int, default=Config.sim_seed,
                        help="Simulation RNG seed")
    parser.add_argument("--sim_max_step", type=int, default=Config.sim_max_step,
                        help="Max simulation steps per rollout")
    parser.add_argument(
        "--policy_seeds",
        type=str,
        default=",".join(map(str, Config.policy_seeds)),
        help="Comma or space-separated list of seeds (e.g., '0,1,2' or '0 1 2')",
    )

    args = parser.parse_args()

    # Accept either comma or whitespace separated seeds
    seeds = tuple(
        int(s) for s in re.split(r"[\s,]+", args.policy_seeds.strip()) if s
    )

    cfg = Config(
        policy_seeds=seeds,
        level_name=args.level_name,
        policy_dir=args.policy_dir,
        policy_iter_num=args.policy_iter_num,
        level_paths=args.level_paths,
        sim_seed=args.sim_seed,
        sim_max_step=args.sim_max_step,
    )

    main(cfg)
