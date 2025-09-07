import pathlib
import dataclasses
import argparse
import re
from typing import Sequence, Dict
import json
import numpy as np
import matplotlib.pyplot as plt
from traj_utils import (
    generate_trajectory, 
    generate_trajectory_swithing,
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
    # Auto selection config
    auto_select: bool = True
    solve_rate_threshold: float = 0.85
    selection_seed: int = 0
    allow_fallback: bool = False  # if True, include best even when below threshold
    switch_control_interval: int = 0
    num_switch_traj: int = 5


def _expected_policy_basename(level_paths: str, level_name: str) -> str:
    level_path = f"{level_paths}/{level_name}.json"
    return level_path.replace("/", "_").replace(".json", "")


def select_best_policies_for_all_seeds(config: Config) -> Dict[int, pathlib.Path]:
    """Scan seeds under policy_dir/level_name and choose best iteration per seed.

    Only returns seeds with an iteration meeting the threshold, unless
    config.allow_fallback is True, in which case it returns the best anyway.
    """
    base = pathlib.Path(config.policy_dir) / config.level_name
    if not base.exists():
        # Fallback: treat policy_dir as already level-specific
        alt = pathlib.Path(config.policy_dir)
        if alt.exists():
            print(f"'{base}' not found; using level directory: {alt}")
            base = alt
        else:
            print(f"Policy base directory not found: {base}")
            return {}

    expected_base = _expected_policy_basename(config.level_paths, config.level_name)
    rng = np.random.default_rng(config.selection_seed)
    result: Dict[int, pathlib.Path] = {}

    for seed_dir in sorted(base.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        # Extract integer seed
        try:
            seed_num = int(seed_dir.name.split("_")[-1])
        except ValueError:
            continue

        # Find numeric iteration subdirs
        iter_dirs = [p for p in seed_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        if not iter_dirs:
            continue

        # Collect stats values per iteration
        stats_vals = []
        valid_iters = []
        for it_dir in iter_dirs:
            stats_path = it_dir / "stats" / f"{expected_base}.json"
            if not stats_path.exists():
                continue
            try:
                with stats_path.open("r") as f:
                    st = json.load(f)
                val = float(st.get("returned_episode_solved", 0.0))
                stats_vals.append(val)
                valid_iters.append(it_dir)
            except Exception:
                continue

        if not valid_iters:
            continue

        stats_arr = np.asarray(stats_vals)
        solved_idxs = np.nonzero(stats_arr >= config.solve_rate_threshold)[0]
        masked = False
        if len(solved_idxs) == 0:
            chosen_idx = int(np.argmax(stats_arr))
            masked = True
        else:
            chosen_idx = int(rng.choice(solved_idxs))

        if masked and not config.allow_fallback:
            print(f"Seed {seed_num}: best solve rate {stats_arr[chosen_idx]:.3f} < threshold; skipping")
            continue

        chosen_dir = valid_iters[chosen_idx]
        policy_path = chosen_dir / "policies" / f"{expected_base}.pkl"
        if policy_path.exists():
            print(
                f"Seed {seed_num}: using iter {chosen_dir.name} (solve {stats_arr[chosen_idx]:.3f})"
                + (" [FALLBACK]" if masked and config.allow_fallback else "")
            )
            result[seed_num] = policy_path
        else:
            print(f"Seed {seed_num}: policy file missing at {policy_path}")

    return dict(sorted(result.items()))


def plot_scene_and_multiple_trajectories(json_filepath, traj_dict, save_path=None, show_seed_labels: bool = True):
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
            
        color = colors[i % len(colors)]
        
        # Plot the path line
        line_label = f'Seed {seed}' if show_seed_labels else '_nolegend_'
        ax.plot(green_pos[:, 0], green_pos[:, 1], 
                color=color, linewidth=3, alpha=0.8, label=line_label)
        
        # Mark the end point
        ax.scatter(green_pos[-1, 0], green_pos[-1, 1], 
                   color=color, s=120, marker='X', edgecolors='white', 
                   zorder=5, label='_nolegend_')
        
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
    
    policy_map: Dict[int, pathlib.Path]
    if config.auto_select:
        print("Auto-selecting best policies per seed based on stats...")
        policy_map = select_best_policies_for_all_seeds(config)
    else:
        policy_map = {}
        for policy_seed in config.policy_seeds:
            policy_path = build_policy_path(
                config.policy_dir, config.level_name, policy_seed,
                config.policy_iter_num, config.level_paths
            )
            p = pathlib.Path(policy_path)
            if not p.exists():
                print(f"Policy file not found for seed {policy_seed}: {policy_path}")
                continue
            policy_map[policy_seed] = p

    for policy_seed, policy_path in policy_map.items():
        print(f"Found policy for seed {policy_seed}: {policy_path}")
        print(f"Generating trajectory for seed {policy_seed}...")
        if config.switch_control_interval and config.switch_control_interval > 0:
            # Build full map of seed->path strings for switching
            pm = {s: str(p) for s, p in policy_map.items()}
            for run_idx in range(config.num_switch_traj):
                traj = generate_trajectory_swithing(
                    pm,
                    level_path,
                    max_steps=config.sim_max_step,
                    seed=config.sim_seed + run_idx,
                    switch_control_interval=config.switch_control_interval,
                    start_seed=policy_seed,
                )
                trajs[(policy_seed, run_idx)] = traj
                if traj['dones'].size > 0:
                    print(f"Seed {policy_seed} run {run_idx} - Episode finished: {traj['dones'][-1]} in {traj['steps'][-1]} steps.")
                else:
                    print(f"Seed {policy_seed} run {run_idx} - No steps were taken.")
                print("-" * 60)
        else:
            traj = generate_trajectory(str(policy_path), level_path, 
                                       max_steps=config.sim_max_step, seed=config.sim_seed)
            trajs[policy_seed] = traj
            if traj['dones'].size > 0:
                print(f"Seed {policy_seed} - Episode finished: {traj['dones'][-1]} in {traj['steps'][-1]} steps.")
            else:
                print(f"Seed {policy_seed} - No steps were taken.")
            print("-" * 60)
    
    if trajs:
        # Plot trajectories together
        used_seeds = list(policy_map.keys())
        seeds_str = "_".join(map(str, used_seeds))
        sci_suffix = f"_sci{config.switch_control_interval}" if config.switch_control_interval and config.switch_control_interval > 0 else ""
        figure_save_path = f"./logs/logs-figures/{config.level_name}_seeds_{seeds_str}{sci_suffix}.png"

        print("Plotting all trajectories...")
        show_labels = not (config.switch_control_interval and config.switch_control_interval > 0)
        plot_scene_and_multiple_trajectories(level_path, trajs, save_path=figure_save_path, show_seed_labels=show_labels)
        
        # Save trajectory data
        output_path = f"./logs/logs-data/traj_data_seeds_{seeds_str}.npz"
        combined_data = {}
        for seed_key, traj in trajs.items():
            if isinstance(seed_key, tuple):
                seed_name = f"{seed_key[0]}_run_{seed_key[1]}"
            else:
                seed_name = str(seed_key)
            for key, value in traj.items():
                combined_data[f"seed_{seed_name}_{key}"] = value
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
    parser.add_argument("--auto_select", action="store_true", help="Auto-select best iteration per seed across all seeds")
    parser.add_argument("--no-auto_select", dest="auto_select", action="store_false")
    parser.set_defaults(auto_select=Config.auto_select)
    parser.add_argument("--solve_rate_threshold", type=float, default=Config.solve_rate_threshold,
                        help="Minimum solve rate to qualify as 'good' policy")
    parser.add_argument("--selection_seed", type=int, default=Config.selection_seed,
                        help="RNG seed for tie-breaking among good iterations")
    parser.add_argument("--allow_fallback", action="store_true", help="Include best iteration even if below threshold")
    parser.add_argument("--switch_control_interval", type=int, default=Config.switch_control_interval,
                        help="If >0, switch controlling policy every N steps among available seeds")
    parser.add_argument("--num_switch_traj", type=int, default=Config.num_switch_traj,
                        help="Number of switching trajectories to generate per starting seed when switching is enabled")

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
        auto_select=args.auto_select,
        solve_rate_threshold=args.solve_rate_threshold,
        selection_seed=args.selection_seed,
        allow_fallback=args.allow_fallback,
        switch_control_interval=args.switch_control_interval,
        num_switch_traj=args.num_switch_traj,
    )

    main(cfg)
