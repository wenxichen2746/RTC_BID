"""
~/VLA_RTC/real-time-chunking-kinetix$ uv run src/rollout_video.py --level_path "worlds/l/hard_lunar_lander.json" --run_path logs-bc/balmy-morning-11/ --episode 8
--level_path: choose a json file
--run_path: bc policy path, eg: my model is at "real-time-chunking-kinetix/logs-bc/balmy-morning-11/0/policies/worlds_l_hard_lunar_lander.pkl", so my run_path is logs-bc/balmy-morning-11/
--episode: load bc policy of episode 
uv run src/rollout_video.py --level_path "worlds/c/ufo.json" 


"""

import pathlib
import pickle
import dataclasses
import os

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import imageio.v2 as imageio

import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.render.renderer_pixels as renderer_pixels
import model as _model
import train_expert
import kinetix.environment.wrappers as wrappers
import numpy as np  

from train_expert_dr import change_polygon_position
from train_expert_dr import change_polygon_position_and_velocity
from cfg_train_expert import RandomizedResetWrapper,ActObsHistoryWrapper
from util.env_dr import *

class DynamicPolygonDriftWrapper(wrappers.UnderspecifiedEnvWrapper):
    """
    Dynamically moves a specified polygon slightly at each step along a chosen axis (x or y).
    Useful for making the environment dynamic.
    """

    def __init__(
        self,
        env,
        polygon_index: int = 4,
        axis: str = "y",  # "x" or "y"
        drift_per_step: float = 0.01,
        min_pos: float = 0.1,
        max_pos: float = 5.0,
    ):
        super().__init__(env)
        assert axis in ("x", "y"), "axis must be either 'x' or 'y'"
        self.polygon_index = polygon_index
        self.axis = axis
        self.drift_per_step = drift_per_step
        self.min_pos = min_pos
        self.max_pos = max_pos

    def step_env(self, key, state, action, params):
        """Drift polygon before stepping the environment."""
        level = state.level if hasattr(state, "level") else None
        if level is not None:
            # Get current position
            pos = level.polygon.position[self.polygon_index]
            axis_idx = 0 if self.axis == "x" else 1
            new_val = jnp.clip(pos[axis_idx] + self.drift_per_step, self.min_pos, self.max_pos)
            print(f'axis_idx{axis_idx},new_val{new_val}')
            new_pos = pos.at[axis_idx].set(new_val)
            new_positions = level.polygon.position.at[self.polygon_index].set(new_pos)
            new_polygon = replace(level.polygon, position=new_positions)
            new_level = replace(level, polygon=new_polygon)
            state = replace(state, level=new_level)

        return self._env.step_env(key, state, action, params)

    def reset_to_level(self, rng, level, params):
        """Pass reset through without change."""
        return self._env.reset_to_level(rng, level, params)

    def action_space(self, params):
        """Expose the underlying action space."""
        return self._env.action_space(params)



@dataclasses.dataclass(frozen=True)
class Config:
    episode: int = 0
    model: _model.ModelConfig = _model.ModelConfig()
    save_dir: str = "env_video"
    


def rollout_and_save_video(level_path, run_path, config=Config(),load=False):
    # Env setup
    static_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    batched_level = train_expert.load_levels([level_path], static_params, env_params)
    # batched_level = change_polygon_position(batched_level, pos_x=2, pos_y=None, index=4)

    # batched_level = change_polygon_position_and_velocity(batched_level, pos_x=1,vel_x=2, index=10)
    single_level = jax.tree.map(lambda x: x[0], batched_level)  # Unbatch safely


    static_params = static_params.replace(screen_dim=train_expert.SCREEN_DIM)
    print(static_params)
    base_env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_params)
    # base_env = DynamicPolygonDriftWrapper(base_env, polygon_index=4, axis="x", drift_per_step=2, min_pos=1, max_pos=5)
    # base_env = RandomizedResetWrapper(base_env, polygon_index=4)
    # base_env = RandomizedResetWrapper(base_env, polygon_index=8)
    # base_env = RandomizedResetWrapper(base_env, polygon_index=[1,10,11],xy_min=2.5,xy_max=3.5)
    # base_env = RandomizedResetWrapper(base_env, polygon_index=[9,10,11],xy_min=1.5,xy_max=3.5)
    base_env=ActObsHistoryWrapper(base_env, act_history_length=4, obs_history_length=1)
    base_env= DR_static_wrapper(base_env,level_path)
    env = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(
            wrappers.AutoReplayWrapper(train_expert.NoisyActionWrapper(base_env))
        ),
        num=1,
    )
    
    # Policy setup
    level_name = level_path.replace("/", "_").replace(".json", "")


    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), single_level, env_params)[0].shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    rng = jax.random.key(0)
    # policy = _model.FlowPolicy(obs_dim=obs_dim, action_dim=action_dim, config=config.model, rngs=nnx.Rngs(rng))

    policy = _model.FlowPolicyCFG2(
            context_dim=obs_dim,
            action_dim=action_dim,
            config=config.model,
            rngs=nnx.Rngs(rng),
            context_act_index= (679, 703) ,
            context_obs_index= (0, 679),
        )
    graphdef, state = nnx.split(policy)
    if load:
    # if True:
        print(load,'loading')
        log_dirs = sorted(
            [p for p in pathlib.Path(run_path).iterdir() if p.is_dir() and p.name.isdigit()],
            key=lambda p: int(p.name),
        )
        with (log_dirs[config.episode] / "policies" / f"{level_name}.pkl").open("rb") as f:
            state_dict = pickle.load(f)
        state.replace_by_pure_dict(state_dict)
    policy = nnx.merge(graphdef, state)

    def make_render_video(render_pixels):
        def render_single(state):
            while not isinstance(state, kenv_state.EnvState):
                state = state.env_state
            return render_pixels(state).round().astype(jnp.uint8).transpose(1, 0, 2)[::-1]
        return render_single

    render_fn_raw = renderer_pixels.make_render_pixels(env_params, static_params)
    render_fn = make_render_video(render_fn_raw)

    # Init rollout
    rng, key = jax.random.split(rng)
    

    # Setup video frame collection
    frames = []
    for eps in range(10):
        print(f'episode:{eps}')
        t=0
        obs, env_state = env.reset_to_level(key, single_level, env_params)
        while True:
            rng, key = jax.random.split(rng)
            action = policy.action(key, obs, num_steps=5)[0][0, :]
            action = jnp.expand_dims(action, axis=0)

            def unwrap_env_state(state):
                while hasattr(state, "env_state"):
                    state = state.env_state
                return jax.tree.map(lambda x: x[0] if hasattr(x, "__getitem__") and not isinstance(x, dict) else x, state)

            pixels = render_fn(unwrap_env_state(env_state))

            frame = np.array(pixels, dtype=np.uint8)
            frame = np.transpose(frame, (1, 0, 2))  # W x H x C -> H x W x C
            frame = np.rot90(frame, k=-1, axes=(0, 1))  # Rotate 90° clockwise
            frames.append(frame)

            obs, env_state, reward, done, info = env.step(key, env_state, action, env_params)
            # print(info.keys())
            t+=1
            print(f"step{t}")

            if done[0] or t>10:
                break
        

    # Save video
    output_dir = pathlib.Path(config.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{level_name}_episode{config.episode}.mp4"
    imageio.mimwrite(video_path, frames, fps=15)
    print(f"Saved rollout video to: {video_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--level_path", required=True)
    parser.add_argument("--run_path", default=None)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="env_video")
    parser.add_argument("--load", action="store_true", help="Load the model if this flag is passed.")

    args = parser.parse_args()

    rollout_and_save_video(args.level_path, args.run_path, Config(episode=args.episode, save_dir=args.save_dir),load=args.load)
