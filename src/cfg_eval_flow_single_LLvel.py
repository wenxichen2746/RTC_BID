'''

uv run src/eval_flow_single.py --run_path ./logs-bc/snowy-sky-20/ --output-dir ./logs-eval-dr/hard_lunar_lander --level-paths "worlds/l/hard_lunar_lander.json"
'''

import collections
import dataclasses
import functools
import math
import pathlib
import pickle
from typing import Sequence

import flax.nnx as nnx
import jax
from jax.experimental import shard_map
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as wrappers
import kinetix.render.renderer_pixels as renderer_pixels
import pandas as pd
import tyro
import copy
import model as _model
import train_expert
import cfg_train_expert
from dataclasses import replace

def change_polygon_position_and_velocity(levels, pos_x=None, pos_y=None, vel_x=None, vel_y=None, index=4):
    # levels: pytree of stacked levels (batched)
    batch_size = levels.polygon.position.shape[0]
    new_levels = []

    for batch_idx in range(batch_size):
        level_mod = copy.deepcopy(jax.tree.map(lambda x: x[batch_idx], levels))

        # Get current position and velocity
        current_pos = level_mod.polygon.position[index]
        current_vel = level_mod.polygon.velocity[index]

        # Set new values or keep old ones
        new_pos = jnp.array([
            pos_x if pos_x is not None else current_pos[0],
            pos_y if pos_y is not None else current_pos[1],
        ])
        new_vel = jnp.array([
            vel_x if vel_x is not None else current_vel[0],
            vel_y if vel_y is not None else current_vel[1],
        ])

        # Replace position and velocity
        new_positions = level_mod.polygon.position.at[index].set(new_pos)
        new_velocities = level_mod.polygon.velocity.at[index].set(new_vel)
        new_polygon = replace(level_mod.polygon, position=new_positions, velocity=new_velocities)
        level_mod = replace(level_mod, polygon=new_polygon)
        new_levels.append(level_mod)

    return jax.tree.map(lambda *x: jnp.stack(x), *new_levels)




@dataclasses.dataclass(frozen=True)
class NaiveMethodConfig:
    pass


@dataclasses.dataclass(frozen=True)
class RealtimeMethodConfig:
    prefix_attention_schedule: _model.PrefixAttentionSchedule = "exp"
    max_guidance_weight: float = 5.0


@dataclasses.dataclass(frozen=True)
class BIDMethodConfig:
    n_samples: int = 16
    bid_k: int | None = None


@dataclasses.dataclass(frozen=True)
class CFGMethodConfig:
    # weights in u = (1-2*w1) u(∅,∅) + w2 u(actions,∅) + w3 u(∅,obs)
    w_1: float = 1.0
    w_2: float = 2.0
    w_3: float = 2.0



@dataclasses.dataclass(frozen=True)
class EvalConfig:
    step: int = -1
    weak_step: int | None = None
    num_evals: int = 248#2048
    num_flow_steps: int = 5

    inference_delay: int = 0
    execute_horizon: int = 1
    method: NaiveMethodConfig | RealtimeMethodConfig | BIDMethodConfig | CFGMethodConfig = NaiveMethodConfig()

    model: _model.ModelConfig = _model.ModelConfig()

    obs_history_length: int = 1
    act_history_length: int = 4

def eval(
    config: EvalConfig,
    env: kenv.environment.Environment,
    rng: jax.Array,
    level: kenv_state.EnvState,
    policy: _model.FlowPolicy | _model.FlowPolicyCFG2,
    env_params: kenv_state.EnvParams,
    static_env_params: kenv_state.EnvParams,
    weak_policy: _model.FlowPolicy | _model.FlowPolicyCFG2 | None = None,
):
    env = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(wrappers.AutoReplayWrapper(train_expert.NoisyActionWrapper(env))), config.num_evals
    )
    render_video = train_expert.make_render_video(renderer_pixels.make_render_pixels(env_params, static_env_params))
    assert config.execute_horizon >= config.inference_delay, f"{config.execute_horizon=} {config.inference_delay=}"

    def execute_chunk(carry, _):
        def step(carry, action):
            
            rng, obs, env_state = carry
            rng, key = jax.random.split(rng)


            next_obs, next_env_state, reward, done, info = env.step(key, env_state, action, env_params)
            return (rng, next_obs, next_env_state), (done, env_state, info)

        rng, obs, env_state, action_chunk, n = carry
        rng, key = jax.random.split(rng)
        if isinstance(config.method, NaiveMethodConfig):
            next_action_chunk = policy.action(key, obs, config.num_flow_steps)
        elif isinstance(config.method, RealtimeMethodConfig):
            prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
            assert (
                config.inference_delay <= policy.action_chunk_size
                and prefix_attention_horizon <= policy.action_chunk_size
            ), f"{config.inference_delay=} {prefix_attention_horizon=} {policy.action_chunk_size=}"
            print(
                f"RTC{config.execute_horizon=} {config.inference_delay=} {prefix_attention_horizon=} {policy.action_chunk_size=}"
            )
            next_action_chunk = policy.realtime_action(
                key,
                obs,
                config.num_flow_steps,
                action_chunk,
                config.inference_delay,
                prefix_attention_horizon,
                config.method.prefix_attention_schedule,
                config.method.max_guidance_weight,
            )
        elif isinstance(config.method, BIDMethodConfig):
            prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
            if config.method.bid_k is not None:
                assert weak_policy is not None, "weak_policy is required for BID"
            print('BID')
            next_action_chunk = policy.bid_action(
                key,
                obs,
                config.num_flow_steps,
                action_chunk,
                config.inference_delay,
                prefix_attention_horizon,
                config.method.n_samples,
                bid_k=config.method.bid_k,
                bid_weak_policy=weak_policy if config.method.bid_k is not None else None,
            )
        elif isinstance(config.method, CFGMethodConfig):
            print('CFG')
            next_action_chunk = policy.action_cfg(#inference delay NOT IMPLEMENTED YET
                key,
                obs, #context (obs,actions)
                config.num_flow_steps,
                w1=config.method.w_1,
                w2=config.method.w_2,
                w3=config.method.w_3,
            )
        else:
            raise ValueError(f"Unknown method: {config.method}")

        # we execute `inference_delay` actions from the *previously generated* action chunk, and then the remaining
        # `execute_horizon - inference_delay` actions from the newly generated action chunk
        action_chunk_to_execute = jnp.concatenate(
            [
                action_chunk[:, : config.inference_delay],
                next_action_chunk[:, config.inference_delay : config.execute_horizon],
            ],
            axis=1,
        )
        # throw away the first `execute_horizon` actions from the newly generated action chunk, to align it with the
        # correct frame of reference for the next scan iteration
        next_action_chunk = jnp.concatenate(
            [
                next_action_chunk[:, config.execute_horizon :],
                jnp.zeros((obs.shape[0], config.execute_horizon, policy.action_dim)),
            ],
            axis=1,
        )
        next_n = jnp.concatenate([n[config.execute_horizon :], jnp.zeros(config.execute_horizon, dtype=jnp.int32)])
        (rng, next_obs, next_env_state), (dones, env_states, infos) = jax.lax.scan(
            step, (rng, obs, env_state), action_chunk_to_execute.transpose(1, 0, 2)
        )
        # if config.inference_delay > 0:
        #     infos["match"] = jnp.mean(jnp.abs(fixed_prefix - action_chunk_to_execute))
        return (rng, next_obs, next_env_state, next_action_chunk, next_n), (dones, env_states, infos)

    rng, key = jax.random.split(rng)
    obs, env_state = env.reset_to_level(key, level, env_params)
    
    rng, key = jax.random.split(rng)
    action_chunk = policy.action(key, obs, config.num_flow_steps)  # [batch, horizon, action_dim]
    n = jnp.ones(action_chunk.shape[1], dtype=jnp.int32)
    scan_length = math.ceil(env_params.max_timesteps / config.execute_horizon)
    _, (dones, env_states, infos) = jax.lax.scan(
        execute_chunk,
        (rng, obs, env_state, action_chunk, n),
        None,
        length=scan_length,
    )
    dones, env_states, infos = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), (dones, env_states, infos))
    assert dones.shape[0] >= env_params.max_timesteps, f"{dones.shape=}"
    return_info = {}
    for key in ["returned_episode_returns", "returned_episode_lengths", "returned_episode_solved"]:
        # only consider the first episode of each rollout
        first_done_idx = jnp.argmax(dones, axis=0)
        return_info[key] = infos[key][first_done_idx, jnp.arange(config.num_evals)].mean()
    for key in ["match"]:
        if key in infos:
            return_info[key] = jnp.mean(infos[key])
    video = render_video(jax.tree.map(lambda x: x[:, 0], env_states))
    return return_info, video


def main(
    run_path: str,
    config: EvalConfig = EvalConfig(),
    level_paths: Sequence[str] = (
        "worlds/l/grasp_easy.json",
        "worlds/l/catapult.json",
        "worlds/l/cartpole_thrust.json",
        "worlds/l/hard_lunar_lander.json",
        "worlds/l/mjc_half_cheetah.json",
        "worlds/l/mjc_swimmer.json",
        "worlds/l/mjc_walker.json",
        "worlds/l/h17_unicycle.json",
        "worlds/l/chain_lander.json",
        "worlds/l/catcher_v3.json",
        "worlds/l/trampoline.json",
        "worlds/l/car_launch.json",
    ),
    seed: int = 0,
    output_dir: str | None = "eval_output",
):
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)

    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)

    # load policies from best checkpoints by solve rate
    state_dicts = []
    weak_state_dicts = []
    for level_path in level_paths:
        level_name = level_path.replace("/", "_").replace(".json", "")
        log_dirs = list(filter(lambda p: p.is_dir() and p.name.isdigit(), pathlib.Path(run_path).iterdir()))
        log_dirs = sorted(log_dirs, key=lambda p: int(p.name))
        # load policy
        with (log_dirs[config.step] / "policies" / f"{level_name}.pkl").open("rb") as f:
            state_dicts.append(pickle.load(f))
        if config.weak_step is not None:
            with (log_dirs[config.weak_step] / "policies" / f"{level_name}.pkl").open("rb") as f:
                weak_state_dicts.append(pickle.load(f))
    state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *state_dicts))
    if config.weak_step is not None:
        weak_state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *weak_state_dicts))
    else:
        weak_state_dicts = None

    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)[
        0
    ].shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    pspec = jax.sharding.PartitionSpec("x")
    sharding = jax.sharding.NamedSharding(mesh, pspec)

    #calculate index for masking in CFG
    raw_obs_dim = jax.eval_shape(
        env.reset_to_level, jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params
    )[0].shape[-1]
    action_dim = env.action_space(env_params).shape[0]
    context_act_len = config.act_history_length * action_dim      # 24
    context_obs_len = raw_obs_dim - context_act_len               # 679 - 24 = 655
    context_dim     = raw_obs_dim                                 # 679
    context_obs_index = (0, context_obs_len)                      # (0, 655)
    context_act_index = (context_obs_len, context_obs_len + context_act_len)  # (655, 679)
    print("=== Context/Env Dimension Debug ===")
    print(f"raw_obs_dim:        {raw_obs_dim}")
    print(f"action_dim:         {action_dim}")
    print(f"obs_history_length: {config.obs_history_length}")
    print(f"act_history_length: {config.act_history_length}")
    print(f"context_obs_len:    {context_obs_len}")
    print(f"context_act_len:    {context_act_len}")
    print(f"context_dim:        {context_dim}")
    print(f"context_obs_index:  {context_obs_index}")
    print(f"context_act_index:  {context_act_index}")
    print("===================================")

    # Optional: sanity check—should print (679,)
    sample_obs, *_ = env.reset_to_level(jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)
    print(f"env.reset_to_level() sample RAW obs shape: {sample_obs.shape}")
    assert sample_obs.shape[-1] == context_dim
    # env = cfg_train_expert.ActObsHistoryWrapper(env, obs_history_length=config.obs_history_length, act_history_length=config.act_history_length)


    @functools.partial(jax.jit, static_argnums=(0,), in_shardings=sharding, out_shardings=sharding)
    @functools.partial(shard_map.shard_map, mesh=mesh, in_specs=(None, pspec, pspec, pspec, pspec), out_specs=pspec)
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
    def _eval(config: EvalConfig, rng: jax.Array, level: kenv_state.EnvState, state_dict, weak_state_dict):
        policy = _model.FlowPolicyCFG2(
            context_dim=context_dim,
            action_dim=action_dim,
            config=config.model,
            rngs=nnx.Rngs(rng),
            context_act_index=context_act_index,
            context_obs_index=context_obs_index,
        )
        graphdef, state = nnx.split(policy)
        state.replace_by_pure_dict(state_dict)
        #Add demension check
        trained_obs_len = policy.null_obs_embed.value.shape[0]
        trained_act_len = policy.null_act_embed.value.shape[0]
        assert trained_obs_len == context_obs_len, (trained_obs_len, context_obs_len)
        assert trained_act_len == context_act_len, (trained_act_len, context_act_len)
        policy = nnx.merge(graphdef, state)
        if weak_state_dict is not None:
            graphdef, state = nnx.split(policy)
            state.replace_by_pure_dict(weak_state_dict)
            weak_policy = nnx.merge(graphdef, state)
        else:
            weak_policy = None
        eval_info, _ = eval(config, env, rng, level, policy, env_params, static_env_params, weak_policy)
        return eval_info

    rngs = jax.random.split(jax.random.key(seed), len(level_paths))
    results = collections.defaultdict(list)


    for inference_delay in [1,3,5]:
    # for inference_delay in [0]:
        for execute_horizon in range(max(1, inference_delay), 8 - inference_delay + 1,2):
        # for execute_horizon in [max(1, inference_delay)]:
            # execute_horizon=max(1, inference_delay)
            for vel_target in [0.1,0.3,0.5,0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5]:
                #

                levels = change_polygon_position_and_velocity(levels, pos_x=1,vel_x=vel_target, index=4) #change to vel_y=something here if needed
                
                print(f"{inference_delay=} {execute_horizon=} {vel_target=}")
                c = dataclasses.replace(
                    config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=NaiveMethodConfig()
                )
                if weak_state_dicts is None:
                    out = jax.device_get(_eval(c, rngs, levels, state_dicts, None))
                else:
                    out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
                for i in range(len(level_paths)):
                    for k, v in out.items():
                        results[k].append(v[i])
                    results["delay"].append(inference_delay)
                    results["method"].append("naive")
                    results["level"].append(level_paths[i])
                    results["execute_horizon"].append(execute_horizon)
                    results["env_vel"].append(vel_target)

                c = dataclasses.replace(
                    config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=RealtimeMethodConfig()
                )
                if weak_state_dicts is None:
                    out = jax.device_get(_eval(c, rngs, levels, state_dicts, None))
                else:
                    out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
                for i in range(len(level_paths)):
                    for k, v in out.items():
                        results[k].append(v[i])
                    results["delay"].append(inference_delay)
                    results["method"].append("realtime")
                    results["level"].append(level_paths[i])
                    results["execute_horizon"].append(execute_horizon)
                    results["env_vel"].append(vel_target)

                c = dataclasses.replace(
                    config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=BIDMethodConfig()
                )
                if weak_state_dicts is None:
                    out = jax.device_get(_eval(c, rngs, levels, state_dicts, None))
                else:
                    out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
                for i in range(len(level_paths)):
                    for k, v in out.items():
                        results[k].append(v[i])
                    results["delay"].append(inference_delay)
                    results["method"].append("bid")
                    results["level"].append(level_paths[i])
                    results["execute_horizon"].append(execute_horizon)
                    results["env_vel"].append(vel_target)

                c = dataclasses.replace(
                    config,
                    inference_delay=inference_delay,
                    execute_horizon=execute_horizon,
                    method=RealtimeMethodConfig(prefix_attention_schedule="zeros"),
                )
                if weak_state_dicts is None:
                    out = jax.device_get(_eval(c, rngs, levels, state_dicts, None))
                else:
                    out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
                for i in range(len(level_paths)):
                    for k, v in out.items():
                        results[k].append(v[i])
                    results["delay"].append(inference_delay)
                    results["method"].append("hard_masking")
                    results["level"].append(level_paths[i])
                    results["execute_horizon"].append(execute_horizon)
                    results["env_vel"].append(vel_target)
                
                #CFG
                c = dataclasses.replace(
                    config,
                    inference_delay=inference_delay,
                    execute_horizon=execute_horizon,
                    method=CFGMethodConfig(w_1=1.0, w_2=2.0, w_3=2.0),
                )
                if weak_state_dicts is None:
                    out = jax.device_get(_eval(c, rngs, levels, state_dicts, None))
                else:
                    out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
                for i in range(len(level_paths)):
                    for k, v in out.items():
                        results[k].append(v[i])
                    results["delay"].append(inference_delay)
                    results["method"].append("cfg122")
                    results["level"].append(level_paths[i])
                    results["execute_horizon"].append(execute_horizon)
                    results["env_vel"].append(vel_target)

                c = dataclasses.replace(
                    config,
                    inference_delay=inference_delay,
                    execute_horizon=execute_horizon,
                    method=CFGMethodConfig(w_1=1.0, w_2=0.0, w_3=2.0),
                )
                if weak_state_dicts is None:
                    out = jax.device_get(_eval(c, rngs, levels, state_dicts, None))
                else:
                    out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
                for i in range(len(level_paths)):
                    for k, v in out.items():
                        results[k].append(v[i])
                    results["delay"].append(inference_delay)
                    results["method"].append("cfg102")
                    results["level"].append(level_paths[i])
                    results["execute_horizon"].append(execute_horizon)
                    results["env_vel"].append(vel_target)

                c = dataclasses.replace(
                    config,
                    inference_delay=inference_delay,
                    execute_horizon=execute_horizon,
                    method=CFGMethodConfig(w_1=1.0, w_2=2.0, w_3=0.0),
                )
                if weak_state_dicts is None:
                    out = jax.device_get(_eval(c, rngs, levels, state_dicts, None))
                else:
                    out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
                for i in range(len(level_paths)):
                    for k, v in out.items():
                        results[k].append(v[i])
                    results["delay"].append(inference_delay)
                    results["method"].append("cfg120")
                    results["level"].append(level_paths[i])
                    results["execute_horizon"].append(execute_horizon)
                    results["env_vel"].append(vel_target)


    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(pathlib.Path(output_dir) / "results.csv", index=False)


if __name__ == "__main__":
    tyro.cli(main)
