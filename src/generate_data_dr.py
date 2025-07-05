import dataclasses
import functools
import json
import pathlib
import pickle

import einops
from flax import struct
import flax.nnx as nnx
import flax.serialization
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as wrappers
import numpy as np
import tqdm_loggable.auto as tqdm
import tyro

import train_expert_dr

@dataclasses.dataclass
class Config:
    run_path: str
    level_path: str = "worlds/l/hard_lunar_lander.json"
    seed: int = 0
    num_envs: int = 128
    batch_size: int = 256
    num_steps: int = 1_000_000
    solve_rate_threshold: float = 0.65
    action_sample_std: float | None = None

@struct.dataclass
class Data:
    obs: jax.Array
    action: jax.Array
    done: jax.Array
    solved: jax.Array
    return_: jax.Array
    length: jax.Array

@struct.dataclass
class StepCarry:
    rng: jax.Array
    obs: jax.Array
    env_state: kenv_state.EnvState
    policy_idxs: jax.Array

def main(config: Config):
    num_steps_per_env = (
        (config.num_steps // config.num_envs + config.batch_size - 1) // config.batch_size
    ) * config.batch_size
    print(
        f"Generating {num_steps_per_env * config.num_envs:_} steps with {config.num_envs} environments ({num_steps_per_env} steps per env)"
    )

    static_env_params = kenv_state.StaticEnvParams(**train_expert_dr.LARGE_ENV_PARAMS, frame_skip=train_expert_dr.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    level, _, _ = train_expert_dr.saving.load_from_json_file(config.level_path)

    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)
    env = train_expert_dr.RandomizedResetWrapper(env, polygon_index=4)
    env = train_expert_dr.BatchEnvWrapper(
        wrappers.LogWrapper(
            wrappers.AutoReplayWrapper(
                train_expert_dr.ActionHistoryWrapper(
                    train_expert_dr.ObsHistoryWrapper(train_expert_dr.NoisyActionWrapper(env), 4)
                )
            )
        ),
        config.num_envs,
    )

    # --- Policy selection logic for a single level ---
    level_name = config.level_path.replace("/", "_").replace(".json", "")
    gen = np.random.default_rng(config.seed)
    state_dicts = []
    good_policy_mask = []
    for seed_dir in pathlib.Path(config.run_path).glob("seed_*"):
        log_dirs = list(filter(lambda p: p.is_dir() and p.name.isdigit(), seed_dir.iterdir()))
        level_stats = [json.load((p / "stats" / f"{level_name}.json").open("r")) for p in log_dirs]
        level_stats = jax.tree.map(lambda *x: jnp.stack(x), *level_stats)
        solved_idxs = np.nonzero(level_stats["returned_episode_solved"] >= config.solve_rate_threshold)[0]
        if len(solved_idxs) == 0:
            chosen_idx = np.argmax(level_stats["returned_episode_solved"])
            good_policy_mask.append(False)
        else:
            chosen_idx = gen.choice(solved_idxs)
            good_policy_mask.append(True)
        chosen_log_dir = log_dirs[chosen_idx]
        with open(chosen_log_dir / "policies" / f"{level_name}.pkl", "rb") as f:
            state_dicts.append(pickle.load(f))
        print(
            f"\t{seed_dir.name}: {level_stats['returned_episode_solved'][chosen_idx]:.3f} {'[MASKED]' if not good_policy_mask[-1] else ''}"
        )
    state_dicts = jax.tree.map(lambda *x: jnp.array(x), *state_dicts)
    good_policy_mask = jnp.array(good_policy_mask)
    state_dicts, good_policy_mask = jax.device_put((state_dicts, good_policy_mask))

    def new_policy_idxs(rng: jax.Array, good_policy_mask: jax.Array) -> jax.Array:
        rng, key = jax.random.split(rng)
        randint = jax.random.randint(key, (config.num_envs,), 0, good_policy_mask.sum())
        return jnp.nonzero(good_policy_mask, size=good_policy_mask.shape[0])[0][randint]

    @jax.jit
    def init(rng: jax.Array, level: kenv_state.EnvState, good_policy_mask: jax.Array) -> StepCarry:
        rng, key = jax.random.split(rng)
        obs, env_state = env.reset_to_level(key, level, env_params)
        rng, key = jax.random.split(rng)
        policy_idxs = new_policy_idxs(key, good_policy_mask)
        return StepCarry(rng, obs, env_state, policy_idxs)

    @functools.partial(jax.jit, static_argnums=(3,), donate_argnums=(0,))
    def step_n(carry: StepCarry, state_dict: dict, good_policy_mask: jax.Array, n: int):
        def step(carry: StepCarry, _):
            action_dim = env.action_space(env_params).shape[0]
            obs_dim = carry.obs.shape[1]

            @jax.vmap
            def get_action(key, obs, policy_idx):
                agent = train_expert_dr.Agent(obs_dim, action_dim, 1, rngs=nnx.Rngs(0))
                graphdef, state = nnx.split(agent)
                state.replace_by_pure_dict(jax.tree.map(lambda x: x[policy_idx], state_dict))
                agent = nnx.merge(graphdef, state)
                mean, std = agent.action(obs)
                if config.action_sample_std is not None:
                    std = jnp.full_like(mean, config.action_sample_std)
                action_dist = train_expert_dr.make_squashed_normal_diag(mean, std, static_env_params.num_motor_bindings)
                return action_dist.sample(seed=key)

            rng, key = jax.random.split(carry.rng)
            action = get_action(jax.random.split(key, config.num_envs), carry.obs, carry.policy_idxs)
            rng, key = jax.random.split(rng)
            next_obs, next_env_state, reward, done, info = env.step(key, carry.env_state, action, env_params)
            rng, key = jax.random.split(rng)
            next_policy_idxs = jnp.where(done, new_policy_idxs(key, good_policy_mask), carry.policy_idxs)
            info = {
                k: v
                for k, v in info.items()
                if k in ["returned_episode_returns", "returned_episode_lengths", "returned_episode_solved"]
            }
            return StepCarry(rng, next_obs, next_env_state, next_policy_idxs), Data(
                train_expert_dr.ObsHistoryWrapper.get_original_obs(carry.env_state),
                action,
                done,
                info["returned_episode_solved"],
                info["returned_episode_returns"],
                info["returned_episode_lengths"],
            )
        return jax.lax.scan(step, carry, None, length=n)

    rng = jax.random.key(config.seed)
    carry = init(rng, level, good_policy_mask)
    pbar = tqdm.tqdm(total=num_steps_per_env * config.num_envs, dynamic_ncols=True)
    data = []
    for _ in range(0, num_steps_per_env, config.batch_size):
        carry, result = step_n(carry, state_dicts, good_policy_mask, config.batch_size)
        data.append(jax.device_get(result))
        pbar.update(config.batch_size * config.num_envs)
    pbar.close()
    with jax.default_device(jax.devices("cpu")[0]):
        data: Data = jax.tree.map(
            lambda *x: einops.rearrange(
                jnp.stack(x),
                "num_batch batch_size num_env ... -> (num_batch batch_size) num_env ...",
            ),
            *data,
        )

    print_info = {"num_episodes": data.done.sum()}
    for key in ["return_", "length", "solved"]:
        print_info[key] = (getattr(data, key) * data.done).sum() / print_info["num_episodes"]

    print(f"{level_name}:")
    for k, v in print_info.items():
        print(f"\t{k}: {v:.3f}")

    data_path = pathlib.Path(config.run_path) / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    data_dict = flax.serialization.to_state_dict(data)
    np.savez(data_path / f"{level_name}.npz", **data_dict)

if __name__ == "__main__":
    tyro.cli(main)