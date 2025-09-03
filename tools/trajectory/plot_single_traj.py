import sys
import pathlib
# Add the parent directory to Python path
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import dataclasses
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
    policy_seed: int = 6
    level_name: str = "swing_up_hard_ver1"
    policy_dir: str = "./logs/logs-expert"
    policy_iter_num: int = 500
    level_paths: str = "worlds/l"
    sim_seed: int = 4
    num_sim_done: int = 1


def plot_scene_and_trajectory(json_filepath, trajectory, save_path=None):
    """Plot the full scene from a JSON file and overlay the agent's trajectory."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax, data = plot_scene_base(json_filepath, ax)
    ax.set_title("Agent Trajectory in 2D Physics Environment")
    
    green_obj_shape = get_green_obj_shape(data)
    green_positions = trajectory['pos']
    
    # Plot trajectory
    ax.plot(green_positions[:, 0], green_positions[:, 1], 'r-', 
            linewidth=2.5, alpha=0.8, label='Trajectory')
    
    # Plot start and end points
    ax.scatter(green_positions[0, 0], green_positions[0, 1], 
               color='cyan', s=150, edgecolors='black', label='Start', zorder=5)
    ax.scatter(green_positions[-1, 0], green_positions[-1, 1], 
               color='magenta', s=150, edgecolors='black', label='End', zorder=5)

    # Draw agent's final position
    green_traj_end = (green_positions[-1, 0], green_positions[-1, 1])
    draw_agent_shape_at_position(ax, green_obj_shape, green_traj_end, trajectory['rot'][-1], label='Agent Final Position')

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
    policy_path = build_policy_path(
        config.policy_dir, config.level_name, config.policy_seed,
        config.policy_iter_num, config.level_paths
    )

    if not pathlib.Path(policy_path).exists():
        print(f"Policy file not found: {policy_path}")
        return
    
    print(f"Found policy: {policy_path}")
    print("Generating trajectory...")
    
    # Generate multiple episodes until we get the desired number of completions
    trajectory = generate_trajectory(policy_path, level_path, 
                                   max_steps=1000, seed=config.sim_seed)
    
    figure_save_path = f"./logs/logs-figures/{config.level_name}_{config.policy_seed}.png"
    print("Plotting scene and trajectory...")
    plot_scene_and_trajectory(level_path, trajectory, save_path=figure_save_path)


if __name__ == "__main__":
    main(Config())
