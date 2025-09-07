import pickle
import json
import jax
import numpy as np
import flax.nnx as nnx
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as wrappers
import kinetix.util.saving as saving
import train_expert
import cfg_train_expert
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pathlib
from typing import Optional, Dict


def load_policy(policy_path: str, obs_dim: int, action_dim: int):
    """Load a trained policy from pickle file."""
    with open(policy_path, "rb") as f:
        state_dict = pickle.load(f)
    
    # Create agent with same architecture as training
    agent = train_expert.Agent(obs_dim, action_dim, 256, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(agent)
    state.replace_by_pure_dict(state_dict)
    agent = nnx.merge(graphdef, state)
    return agent


def get_green_obj_idx(raw_json_data):
    """Find the green object (agent) index and type in the environment."""
    env_state = raw_json_data['env_state']
    
    # Find agent polygon
    green_polygon_idx = None
    for i, poly_data in enumerate(env_state.get('polygon', [])):
        if poly_data.get('active', False) and poly_data.get('role', 0) == 1:
            green_polygon_idx = i
            break

    # Find agent circle
    green_circle_idx = None
    for i, circle_data in enumerate(env_state.get('circle', [])):
        if circle_data.get('active', False) and circle_data.get('role', 0) == 1:
            green_circle_idx = i
            break

    if green_polygon_idx is None and green_circle_idx is None:
        raise ValueError("No active agent (role=1) found in environment")

    green_obj_type = 'polygon' if green_polygon_idx is not None else 'circle'
    green_obj_idx = green_polygon_idx if green_polygon_idx is not None else green_circle_idx

    return green_obj_idx, green_obj_type


def setup_environment():
    """Setup the environment with standard configuration."""
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    
    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)
    env = wrappers.LogWrapper(
        wrappers.AutoReplayWrapper(
            # train_expert.ActionHistoryWrapper(
            #     # train_expert.ObsHistoryWrapper(train_expert.NoisyActionWrapper(env), 4)
            #                 )
            cfg_train_expert.ActObsHistoryWrapper(
                # cfg_train_expert.NoisyActionWrapper(env)
                env
                , act_history_length=4, obs_history_length=1)

        )
    )
    return env, env_params, static_env_params


def get_green_obj_shape(raw_json_data):
    """Extract the shape information of the green object from JSON data."""
    env_state = raw_json_data['env_state']
    
    # Check polygons first
    for poly_data in env_state.get('polygon', []):
        if poly_data.get('active', False) and poly_data.get('role', 0) == 1:
            vertices = np.array([[v['0'], v['1']] for v in poly_data['vertices'].values()])
            return {
                'type': 'polygon',
                'vertices': vertices,
                'rotation': poly_data['rotation']
            }
    
    # Check circles
    for circle_data in env_state.get('circle', []):
        if circle_data.get('active', False) and circle_data.get('role', 0) == 1:
            return {
                'type': 'circle',
                'radius': circle_data['radius']
            }
    
    return None


def generate_trajectory(policy_path: str, level_path: str, max_steps: int = 1000, seed: int = 0):
    """Generate a trajectory using the given policy on the specified level."""
    env, env_params, static_env_params = setup_environment()
    
    level, _, _ = saving.load_from_json_file(level_path)
    
    with open(level_path, 'r') as f:
        raw_json_data = json.load(f)

    rng = jax.random.key(seed)
    obs, env_state = env.reset_to_level(rng, level, env_params)
    
    action_dim = env.action_space(env_params).shape[0]
    obs_dim = obs.shape[0]
    print(f"observation dim is {obs_dim}")
    print(f"action dim is {action_dim}")
    
    agent = load_policy(policy_path, obs_dim, action_dim)
    
    # Generate trajectory
    traj = {
        'pos': [],
        'rot': [],
        'steps': [],
        'dones': [],
    }
    
    done = False
    step_count = 0
    
    while not done and step_count < max_steps:
        rng, key = jax.random.split(rng)
        mean, std = agent.action(obs)
        action_dist = train_expert.make_squashed_normal_diag(mean, std, static_env_params.num_motor_bindings)
        action = action_dist.sample(seed=key)
        rng, key = jax.random.split(rng)
        next_obs, next_env_state, reward, done, info = env.step(key, env_state, action, env_params)

        green_obj_idx, green_obj_type = get_green_obj_idx(raw_json_data)
        if green_obj_type == 'polygon':
            green_obj = env_state.env_state.env_state.env_state.polygon
        else:
            green_obj = env_state.env_state.env_state.env_state.circle
        green_obj_pos = green_obj.position[green_obj_idx]
        green_obj_rot = green_obj.rotation[green_obj_idx]

        traj['pos'].append(np.array(green_obj_pos))
        traj['rot'].append(np.array(green_obj_rot))
        traj['dones'].append(bool(done))
        traj['steps'].append(step_count)
        
        obs = next_obs
        env_state = next_env_state
        step_count += 1
    
    # Convert lists to numpy arrays
    for key in traj:
        traj[key] = np.array(traj[key])
    
    return traj


def generate_trajectory_swithing(
    policy_map: Dict[int, str],
    level_path: str,
    max_steps: int = 1000,
    seed: int = 0,
    switch_control_interval: int = 1,
    start_seed: Optional[int] = None,
):
    """Generate a single trajectory while switching control among multiple policies.

    - Switches controller every `switch_control_interval` steps.
    - If `start_seed` is provided and in `policy_map`, starts with that controller.
    - Returns same fields as `generate_trajectory` plus `controllers` (seed per step).
    """
    if switch_control_interval <= 0:
        # Fallback to single-controller behavior using an arbitrary policy
        any_seed, any_path = next(iter(policy_map.items()))
        return generate_trajectory(any_path, level_path, max_steps=max_steps, seed=seed)

    env, env_params, static_env_params = setup_environment()
    level, _, _ = saving.load_from_json_file(level_path)
    with open(level_path, 'r') as f:
        raw_json_data = json.load(f)

    rng = jax.random.key(seed)
    obs, env_state = env.reset_to_level(rng, level, env_params)

    action_dim = env.action_space(env_params).shape[0]
    obs_dim = obs.shape[0]

    # Load all policies
    agents: Dict[int, object] = {}
    for s, path in policy_map.items():
        agents[s] = load_policy(path, obs_dim, action_dim)

    seeds = list(agents.keys())
    # Choose initial controller
    if start_seed is not None and start_seed in agents:
        controller = start_seed
    else:
        controller = seeds[0]

    traj = {
        'pos': [],
        'rot': [],
        'steps': [],
        'dones': [],
        'controllers': [],
    }

    done = False
    step_count = 0
    steps_since_switch = 0

    while not done and step_count < max_steps:
        # Switch controller if needed (avoid picking same seed when possible)
        if steps_since_switch >= switch_control_interval:
            steps_since_switch = 0
            if len(seeds) > 1:
                # Pick a different controller uniformly from others
                others = [s for s in seeds if s != controller]
                # jax PRNG for reproducibility
                rng, key = jax.random.split(rng)
                idx = int(jax.random.randint(key, (), 0, len(others)))
                controller = others[idx]

        agent = agents[controller]

        rng, key = jax.random.split(rng)
        mean, std = agent.action(obs)
        action_dist = train_expert.make_squashed_normal_diag(mean, std, static_env_params.num_motor_bindings)
        action = action_dist.sample(seed=key)

        rng, key = jax.random.split(rng)
        next_obs, next_env_state, reward, done, info = env.step(key, env_state, action, env_params)
        # print(f"infokeys: {list(info.keys())}")
        green_obj_idx, green_obj_type = get_green_obj_idx(raw_json_data)
        if green_obj_type == 'polygon':
            green_obj = env_state.env_state.env_state.env_state.polygon
        else:
            green_obj = env_state.env_state.env_state.env_state.circle
        green_obj_pos = green_obj.position[green_obj_idx]
        green_obj_rot = green_obj.rotation[green_obj_idx]

        traj['pos'].append(np.array(green_obj_pos))
        traj['rot'].append(np.array(green_obj_rot))
        traj['dones'].append(bool(done))
        traj['steps'].append(step_count)
        traj['controllers'].append(controller)

        obs = next_obs
        env_state = next_env_state
        step_count += 1
        steps_since_switch += 1

    for key in traj:
        traj[key] = np.array(traj[key])
    return traj


def plot_scene_base(json_filepath, ax=None):
    """Plot the base scene (environment objects) from JSON file."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    with open(json_filepath, 'r') as f:
        data = json.load(f)
    env_state = data['env_state']
    
    role_colors = {
        0: '#4c4c4c', 
        1: '#2ca02c', 
        2: '#1f77b4', 
        3: '#8c564b'
    }
    
    ax.set_facecolor('#fdfbf6')
    ax.set_aspect('equal', 'box')
    
    # Plot Polygons
    if 'polygon' in env_state:
        for poly_data in env_state.get('polygon', []):
            if not poly_data.get('active', False): continue
            pos = np.array([poly_data['position']['0'], poly_data['position']['1']])
            angle = poly_data['rotation']
            local_vertices = np.array([[v['0'], v['1']] for v in poly_data['vertices'].values()])
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            world_vertices = (local_vertices @ rotation_matrix.T) + pos
            ax.add_patch(patches.Polygon(world_vertices,
                                          closed=True, 
                                          facecolor=role_colors.get(poly_data.get('role', 0)), 
                                          edgecolor='black', 
                                          linewidth=1.5))
    
    # Plot Circles
    if 'circle' in env_state:
        for circle_data in env_state.get('circle', []):
            if not circle_data.get('active', False): continue
            center = (circle_data['position']['0'], circle_data['position']['1'])
            radius = circle_data['radius']
            ax.add_patch(patches.Circle(center, 
                                        radius, 
                                        facecolor=role_colors.get(circle_data.get('role', 0)), 
                                        edgecolor='black', 
                                        linewidth=1.5))
    
    return ax, data


def draw_agent_shape_at_position(
    ax,
    green_obj_shape,
    position,
    rotation,
    label: Optional[str] = None,
    edgecolor: str = 'purple',
    linewidth: float = 2.0,
):
    """Draw the agent's shape at a specific position and rotation.

    Set `label` for legend entry; pass None to skip legend.
    """
    legend_label = label if label is not None else '_nolegend_'
    if green_obj_shape and green_obj_shape['type'] == 'circle':
        circle = plt.Circle(
            position,
            green_obj_shape['radius'],
            fill=False,
            edgecolor=edgecolor,
            linewidth=linewidth,
            label=legend_label,
        )
        ax.add_patch(circle)
    elif green_obj_shape and green_obj_shape['type'] == 'polygon':
        rotation_matrix = np.array(
            [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]
        )
        world_vertices = (green_obj_shape['vertices'] @ rotation_matrix.T) + position
        polygon = plt.Polygon(
            world_vertices,
            fill=False,
            edgecolor=edgecolor,
            linewidth=linewidth,
            label=legend_label,
        )
        ax.add_patch(polygon)


def build_policy_path(policy_dir: str, level_name: str, policy_seed: int, 
                     policy_iter_num: int, level_paths: str) -> str:
    """Build the policy file path from configuration parameters."""
    level_path = f"{level_paths}/{level_name}.json"
    expected_policy_name = level_path.replace("/", "_").replace(".json", "") + ".pkl"
    return f"{policy_dir}/seed_{policy_seed}/{policy_iter_num}/policies/{expected_policy_name}"
