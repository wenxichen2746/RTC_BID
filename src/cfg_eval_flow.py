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
import time
from datetime import timedelta

from util.env_dr import *
import numpy as np

@dataclasses.dataclass(frozen=True)
class NaiveMethodConfig:
    mask_action: bool = False


@dataclasses.dataclass(frozen=True)
class RealtimeMethodConfig:
    prefix_attention_schedule: _model.PrefixAttentionSchedule = "exp"
    max_guidance_weight: float = 5.0
    mask_action: bool = False


@dataclasses.dataclass(frozen=True)
class BIDMethodConfig:
    n_samples: int = 16
    bid_k: int | None = None
    mask_action: bool = False

@dataclasses.dataclass(frozen=True)
class CFGMethodConfig:
    # weights in u = (1-2*w1) u(∅,∅) + w2 u(actions,∅) + w3 u(∅,obs)
    w_1: float = 1.0
    w_2: float = 2.0
    w_3: float = 2.0
    w_4: float = 0.0

@dataclasses.dataclass(frozen=True)
class CFGCOS_MethodConfig:
    # weights in u = (1-2*w1) u(∅,∅) + w2 u(actions,∅) + w3 u(∅,obs)
    w_a: float = 1.0

@dataclasses.dataclass(frozen=True)
class CFG_BI_COS_MethodConfig:
    #u = u(∅,∅) + cos*w_a * [u(actions,∅)-u(∅,∅)] + w_o * [u(∅,obs)-u(∅,∅) ]​​​
    w_a: float = 1.0
    w_o: float = 1.0

@dataclasses.dataclass(frozen=True)
class EvalConfig:
    step: int = -1
    weak_step: int = 5 #| None = None
    num_evals: int = 2048 #2048
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
    noise_std: float =0.0,
):
    env = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(wrappers.AutoReplayWrapper(cfg_train_expert.NoisyActionWrapper(
            cfg_train_expert.ActObsHistoryWrapper(env, act_history_length=4, obs_history_length=1)
            ,noise_std=noise_std)
            )), config.num_evals
    )
    # env=cfg_train_expert.ActObsHistoryWrapper(env, act_history_length=4, obs_history_length=1)
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

        # default: NaN cos-history and variance for methods that don't produce them
        cos_hist_this = jnp.full((obs.shape[0], config.num_flow_steps), jnp.nan)  # (B, S)
        action_var_this = jnp.full((obs.shape[0], config.num_flow_steps), jnp.nan)  # (B, S)

        if isinstance(config.method, NaiveMethodConfig):
            next_action_chunk = policy.action(key, obs, config.num_flow_steps, mask_action=config.method.mask_action)

        elif isinstance(config.method, RealtimeMethodConfig):
            prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
            next_action_chunk = policy.realtime_action(
                key, obs, config.num_flow_steps, action_chunk,
                config.inference_delay, prefix_attention_horizon,
                config.method.prefix_attention_schedule, config.method.max_guidance_weight,
                mask_action=config.method.mask_action
            )

        elif isinstance(config.method, BIDMethodConfig):
            prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
            next_action_chunk, action_var = policy.bid_action(
                key, obs, config.num_flow_steps, action_chunk,
                config.inference_delay, config.execute_horizon , #<- passing execute_horizon to compare valid overlapping loss
                prefix_attention_horizon,
                config.method.n_samples,
                bid_k=config.method.bid_k,
                bid_weak_policy=weak_policy if config.method.bid_k is not None else None,
                mask_action=config.method.mask_action
            )
            action_var_this = action_var

        elif isinstance(config.method, CFGMethodConfig):
            next_action_chunk = policy.action_cfg(
                key, obs, config.num_flow_steps,
                w1=config.method.w_1, w2=config.method.w_2, w3=config.method.w_3, w4=config.method.w_4
            )

        elif isinstance(config.method, CFGCOS_MethodConfig):
            # NOTE: your action_cfg_cos returns (x_1, cos_history) with cos_history shape (num_steps, B)
            next_action_chunk, cos_hist = policy.action_cfg_cos(
                key, obs, config.num_flow_steps, w_a=config.method.w_a
            )
            cos_hist_this = jnp.transpose(cos_hist, (1, 0))  # -> (B, S)

        elif isinstance(config.method, CFG_BI_COS_MethodConfig):
            # NOTE: your action_cfg_cos returns (x_1, cos_history) with cos_history shape (num_steps, B)
            next_action_chunk, cos_hist = policy.action_cfg_BI_cos(
                key, obs, config.num_flow_steps,w_o=config.method.w_o, w_a=config.method.w_a
            )
            cos_hist_this = jnp.transpose(cos_hist, (1, 0))  # -> (B, S)
        else:
            raise ValueError(f"Unknown method: {config.method}")

        # Compute overlap continuity loss between previous and new chunks over the valid horizon
        valid_horizon = max(0, policy.action_chunk_size - config.execute_horizon)
        def _overlap_loss(prev_chunk, new_chunk):
            # Compare head of carried previous chunk to head of new chunk
            prev_overlap = prev_chunk[:, :valid_horizon, :]
            new_overlap = new_chunk[:, :valid_horizon, :]
            if valid_horizon == 0:
                return jnp.zeros(prev_chunk.shape[0])
            diff = jnp.linalg.norm(new_overlap - prev_overlap, axis=-1)  # (B, valid)
            return jnp.mean(diff, axis=-1)  # (B,)
        backward_loss_per_env = _overlap_loss(action_chunk, next_action_chunk)

        # Cross-chunk distance at the inference boundary
        delay_idx = config.inference_delay if config.inference_delay > 0 else 0
        delay_idx = min(delay_idx, action_chunk.shape[1] - 1)
        prev_boundary_action = action_chunk[:, delay_idx, :]
        next_boundary_action = next_action_chunk[:, delay_idx, :]
        cross_chunk_distance_per_env = jnp.linalg.norm(next_boundary_action - prev_boundary_action, axis=-1)

        # Execute actions
        action_chunk_to_execute = jnp.concatenate(
            [ action_chunk[:, : config.inference_delay],
              next_action_chunk[:, config.inference_delay : config.execute_horizon] ],
            axis=1,
        )
        next_action_chunk = jnp.concatenate(
            [ next_action_chunk[:, config.execute_horizon :],
              jnp.zeros((obs.shape[0], config.execute_horizon, policy.action_dim)) ],
            axis=1,
        )
        next_n = jnp.concatenate([n[config.execute_horizon :], jnp.zeros(config.execute_horizon, dtype=jnp.int32)])

        (rng, next_obs, next_env_state), (dones, env_states, infos) = jax.lax.scan(
            step, (rng, obs, env_state), action_chunk_to_execute.transpose(1, 0, 2)
        )

        # Return cos_hist_this per outer iteration (B, S)
        return (
            rng, next_obs, next_env_state, next_action_chunk, next_n
        ), (dones, env_states, infos, cos_hist_this, action_var_this, backward_loss_per_env, cross_chunk_distance_per_env)

    rng, key = jax.random.split(rng)
    obs, env_state = env.reset_to_level(key, level, env_params)

    rng, key = jax.random.split(rng)
    action_chunk = policy.action(key, obs, config.num_flow_steps)  # [B, horizon, action_dim]
    n = jnp.ones(action_chunk.shape[1], dtype=jnp.int32)

    scan_length = math.ceil(env_params.max_timesteps / config.execute_horizon)
    _, (dones, env_states, infos, cos_hist_iters, action_var_iters, backward_losses, cross_chunk_distances) = jax.lax.scan(
        execute_chunk,
        (rng, obs, env_state, action_chunk, n),
        None,
        length=scan_length,
    )
    # shapes from scan:
    #   dones:           (L, H, B)
    #   env_states:      (L, H, ...)
    #   infos[k]:        (L, H, B) for array-like fields; some keys may be None
    #   cos_hist_iters:  (L, B, S)
    #   action_var_iters: (L, B, S)
    L, H, B = dones.shape
    T = L * H

    # Flatten env-steps (no done/alive masking)
    dones_flat = dones.reshape(T, B)
    env_states = jax.tree.map(lambda x: x.reshape(T, *x.shape[2:]), env_states)

    # Safely flatten infos: keep only array-like with (L, H, ...)
    def _is_array_like(x):
        return hasattr(x, "shape") and hasattr(x, "reshape")

    infos_flat = {}
    for k, v in infos.items():
        if v is None:
            continue
        if not _is_array_like(v):
            continue
        # Expect (L, H, B) or (L, H, *extra)
        if v.shape[0] == L and v.shape[1] == H:
            infos_flat[k] = v.reshape(T, *v.shape[2:])
        # If it’s already flattened to (T, ...), keep as-is
        elif v.shape[0] == T:
            infos_flat[k] = v
        # Otherwise skip silently (debug print optional)
        # else:
        #     print(f"[eval] skipping info '{k}' with shape {v.shape}")

    assert dones_flat.shape[0] >= env_params.max_timesteps, f"{dones_flat.shape=}"

    # Episode summaries with NO masking.
    # For typical log fields that spike at episode end, max over time recovers the episode value.
    def _episode_stat(x):  # x: (T, B)
        return jnp.nanmax(x, axis=0)

    return_info = {}
    for key in ["returned_episode_returns", "returned_episode_lengths", "returned_episode_solved"]:
        if key in infos_flat:
            per_batch = _episode_stat(infos_flat[key])   # (B,)
            # Keep backward-compatible scalar means
            return_info[key] = per_batch.mean()

            # Add mean/std across parallel envs for episode length/return
            if key == "returned_episode_lengths":
                return_info["returned_episode_lengths_mean"] = jnp.nanmean(per_batch)
                return_info["returned_episode_lengths_std"] = jnp.nanstd(per_batch)

            if key == "returned_episode_returns":
                return_info["returned_episode_returns_mean"] = jnp.nanmean(per_batch)
                return_info["returned_episode_returns_std"] = jnp.nanstd(per_batch)

    # If available, also report how long solved episodes took (mean/std over solved envs only)
    if ("returned_episode_lengths" in infos_flat) and ("returned_episode_solved" in infos_flat):
        lens_per_batch = _episode_stat(infos_flat["returned_episode_lengths"])  # (B,)
        solved_mask = _episode_stat(infos_flat["returned_episode_solved"])      # (B,) in {0,1}
        denom = jnp.maximum(1.0, solved_mask.sum())
        solved_len_mean = (lens_per_batch * solved_mask).sum() / denom
        # Compute masked variance safely
        diffs = (lens_per_batch - solved_len_mean) * solved_mask
        solved_len_var = (diffs * diffs).sum() / denom
        return_info["returned_episode_lengths_solved_mean"] = solved_len_mean
        return_info["returned_episode_lengths_solved_std"] = jnp.sqrt(solved_len_var)

    # Optional extra scalar logs
    if "match" in infos_flat:
        return_info["match"] = jnp.nanmean(infos_flat["match"])

    # Backward-overlap loss across all outer steps and envs (lower is better)
    # backward_losses: (L, B)
    return_info["backward_overlap_loss"] = jnp.nanmean(backward_losses)
    return_info["backward_overlap_loss_std"] = jnp.nanstd(backward_losses)

    # Cross-chunk action distance at inference boundary
    return_info["cross_chunk_distance_mean"] = jnp.nanmean(cross_chunk_distances)
    return_info["cross_chunk_distance_std"] = jnp.nanstd(cross_chunk_distances)

    # Cosine artifacts: keep raw per-chunk; aggregate without masking
    cos_step_mean = jnp.nanmean(cos_hist_iters, axis=(0, 1))  # (S,)
    cos_step_std  = jnp.nanstd( cos_hist_iters, axis=(0, 1))  # (S,)

    cos_artifacts = {
        # "per_chunk":    cos_hist_iters,   # (L, B, S)
        "episode_mean": cos_step_mean, # (B, )
        "episode_std":  cos_step_std,  # (B, )
    }

    action_var_mean = jnp.nanmean(action_var_iters, axis=(0, 1))  # (S,)
    action_var_std = jnp.nanstd(action_var_iters, axis=(0, 1))    # (S,)
    variance_artifacts = {
        "episode_mean": action_var_mean,
        "episode_std": action_var_std,
    }

    artifacts = {
        "cosine": cos_artifacts,
        "action_variance": variance_artifacts,
    }

    video = render_video(jax.tree.map(lambda x: x[:, 0], env_states))
    return return_info, video, artifacts




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
    _env=cfg_train_expert.ActObsHistoryWrapper(env, act_history_length=4, obs_history_length=1)
    # load policies from best checkpoints by solve rate
    state_dicts = []
    weak_state_dicts = []

    for level_path in level_paths:
        level_name = level_path.replace("/", "_").replace(".json", "")
        # all numeric dirs in run_path
        log_dirs = [p for p in pathlib.Path(run_path).iterdir() if p.is_dir() and p.name.isdigit()]
        if not log_dirs:
            raise FileNotFoundError(f"No checkpoint dirs found under {run_path}")

        # sort numerically
        log_dirs = sorted(log_dirs, key=lambda p: int(p.name))
        last_dir = log_dirs[-1]  # largest number = "last"

        # --- load last checkpoint ---
        last_ckpt = last_dir / "policies" / f"{level_name}.pkl"
        if not last_ckpt.exists():
            raise FileNotFoundError(f"Missing {last_ckpt}")
        with last_ckpt.open("rb") as f:
            state_dicts.append(pickle.load(f))

        # --- load weak checkpoint (explicit folder name) ---
        if config.weak_step is not None:
            weak_dir = pathlib.Path(run_path) / str(config.weak_step)
            weak_ckpt = weak_dir / "policies" / f"{level_name}.pkl"
            if not weak_ckpt.exists():
                raise FileNotFoundError(f"Missing {weak_ckpt}")
            with weak_ckpt.open("rb") as f:
                weak_state_dicts.append(pickle.load(f))

    # device put
    state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *state_dicts))
    if config.weak_step is not None:
        weak_state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *weak_state_dicts))
    else:
        weak_state_dicts = None


    action_dim = _env.action_space(env_params).shape[0]

    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    pspec = jax.sharding.PartitionSpec("x")
    sharding = jax.sharding.NamedSharding(mesh, pspec)

    #calculate index for masking in CFG
    raw_obs_dim = jax.eval_shape(
        _env.reset_to_level, jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params
    )[0].shape[-1]
    action_dim = _env.action_space(env_params).shape[0]
    context_act_len = config.act_history_length * action_dim      # 24
    context_obs_len = raw_obs_dim - context_act_len               # 679
    context_dim     = raw_obs_dim                                 # 703
    context_obs_index = (0, context_obs_len)                      # (0, 679)
    context_act_index = (context_obs_len, context_obs_len + context_act_len)  # (679, 703)
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
    sample_obs, *_ = _env.reset_to_level(jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)
    # print(f"env.reset_to_level() sample RAW obs shape: {sample_obs.shape}")
    assert sample_obs.shape[-1] == context_dim
    # env = cfg_train_expert.ActObsHistoryWrapper(env, obs_history_length=config.obs_history_length, act_history_length=config.act_history_length)


    rngs = jax.random.split(jax.random.key(seed), len(level_paths))
    results = collections.defaultdict(list)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)


    def test_methods(config,levels,env,vel_target,inference_delay,execute_horizon,test_noise_std):
        @functools.partial(jax.jit, static_argnums=(0,), in_shardings=sharding, out_shardings=sharding)
        @functools.partial(shard_map.shard_map, mesh=mesh, in_specs=(None, pspec, pspec, pspec, pspec), out_specs=pspec)
        @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
        def _eval(config: EvalConfig, rng: jax.Array, level: kenv_state.EnvState, state_dict, weak_state_dict=None):
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
            eval_info, video, artifacts = eval(
                config, env, rng, level, policy, env_params, static_env_params, weak_policy, noise_std=test_noise_std
            )
            return eval_info, video, artifacts
        
        # if 'hard_lunar_lander' in level_paths[0]:   
        #     # print(f'training LL, target moving at vel{vel_target}')
        #     levels = change_polygon_position_and_velocity(levels, pos_x=1,vel_x=vel_target, index=4) #change to vel_y=something here if needed
        # elif 'grasp' in level_paths[0]:
        #     # print(f'training grasp, randomizing target location')
        #     levels = change_polygon_position_and_velocity(levels, pos_x=1,vel_x=vel_target, index=10)
        # elif 'toss_bin' in level_paths[0]:
        #     # print(f'training grasp, randomizing target location')
        #     levels = change_polygon_position_and_velocity(levels, pos_x=None,vel_x=vel_target, index=9)
        #     levels = change_polygon_position_and_velocity(levels, pos_x=None,vel_x=vel_target, index=10)
        #     levels = change_polygon_position_and_velocity(levels, pos_x=None,vel_x=vel_target, index=11)
        # elif 'place_can_easy' in level_paths[0]:
        #     # print(f'training grasp, randomizing target location')
        #     levels = change_polygon_position_and_velocity(levels, pos_x=2,vel_x=vel_target, index=9)
        #     levels = change_polygon_position_and_velocity(levels, pos_x=2.5,vel_x=vel_target, index=10)
        # elif 'drone' in level_paths[0]:
        #     # print(f'training grasp, randomizing target location')
        #     levels = change_polygon_position_and_velocity(levels, pos_x=1,vel_x=vel_target, index=4)
        #     levels = change_polygon_position_and_velocity(levels, pos_x=1,vel_x=vel_target, index=7)
        # else:
        #     if vel_target!=0.0:
        #         print(f'skipping moving target for {level_paths[0]}')
        #         return
            
        #     #raise NotImplementedError("*** Level not recognized DR not implemented **")
        # if vel_target==0.0:
        #     env=DR_static_wrapper(env,level_paths[0])

        def eval_and_record(c,method_name,weak_state_dicts=None):
            out, _, artifacts = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
            cos_art = artifacts.get("cosine") if artifacts is not None else None
            variance_art = artifacts.get("action_variance") if artifacts is not None else None

            for i in range(len(level_paths)):
                for k, v in out.items():
                    results[k].append(v[i])
                results["delay"].append(inference_delay)
                results["method"].append(method_name)
                results["level"].append(level_paths[i])
                results["execute_horizon"].append(execute_horizon)
                results["env_vel"].append(vel_target)
                results["noise_std"].append(test_noise_std)
                if cos_art is not None:
                    ep_mean = np.asarray(cos_art["episode_mean"])  # expected (S,), but be robust
                    ep_std  = np.asarray(cos_art["episode_std"])   # expected (S,)

                    if np.isfinite(ep_mean).any():
                        # Reduce any extra dims to scalars safely
                        def _scalar(x):
                            return np.nanmean(np.asarray(x)).item()

                        # Number of steps S (last dim if shape is (..., S))
                        S = int(ep_mean.shape[-1]) if ep_mean.ndim >= 1 else 1

                        row = {
                            "method": method_name,
                            "execute_horizon": execute_horizon,
                            "env_vel": vel_target,
                            "noise_std": test_noise_std,
                            "cos_overall_mean": float(np.nanmean(ep_mean)),
                            "cos_overall_std":  float(np.nanstd(ep_mean)),
                        }

                        # Add per-step means/stds using robust scalarization
                        for s in range(S):
                            row[f"cos_mean_s{s}"] = _scalar(ep_mean[..., s])
                            row[f"cos_std_s{s}"]  = _scalar(ep_std[..., s])

                        df_cos = pd.DataFrame([row])
                        csv_path = pathlib.Path(output_dir) / "cosine_analysis.csv"
                        header = not csv_path.exists()
                        df_cos.to_csv(csv_path, mode="a", index=False, header=header)

                if variance_art is not None:
                    var_mean = np.asarray(variance_art["episode_mean"])
                    var_std = np.asarray(variance_art["episode_std"])

                    if np.isfinite(var_mean).any():
                        def _scalar(x):
                            return np.nanmean(np.asarray(x)).item()

                        S = int(var_mean.shape[-1]) if var_mean.ndim >= 1 else 1

                        row = {
                            "method": method_name,
                            "execute_horizon": execute_horizon,
                            "env_vel": vel_target,
                            "noise_std": test_noise_std,
                            "variance_overall_mean": float(np.nanmean(var_mean)),
                            "variance_overall_std": float(np.nanstd(var_mean)),
                        }

                        for s in range(S):
                            row[f"variance_mean_s{s}"] = _scalar(var_mean[..., s])
                            row[f"variance_std_s{s}"] = _scalar(var_std[..., s])

                        df_var = pd.DataFrame([row])
                        csv_path = pathlib.Path(output_dir) / "action_variance_analysis.csv"
                        header = not csv_path.exists()
                        df_var.to_csv(csv_path, mode="a", index=False, header=header)

        
        # cfg_coef=[x for x in range(1, 5)]
        cfg_coef=list(np.arange(1, 5.5, 0.5))
        # cfg_coef=[x for x in range(0, 3)]
        # for w_a in cfg_coef:
        #     c = dataclasses.replace(
        #         config,
        #         inference_delay=inference_delay,
        #         execute_horizon=execute_horizon,
        #         method=CFGCOS_MethodConfig(w_a=w_a),#u=u(a|o)+ cos_coef*w*(u(a|a',o)-u(a|o))
        #     )
        #     eval_and_record(c,f"cfg_BF_cos:wa{w_a}")

        #enhance guidance on action
        for w in cfg_coef+[0,-1]:
        #u = w* u(actions,obs) + (1-w)* u(∅,obs)​​  =   u(∅,obs)​​  +  w (u(actions,obs)-u(∅,obs)​​)
            w_ao=w#w_ao=1+w
            w_o=1-w#w_o=-w
            c = dataclasses.replace(
                config,
                inference_delay=inference_delay,
                execute_horizon=execute_horizon,
                method=CFGMethodConfig(w_1=0.0, w_2=0.0, w_3=w_o,w_4=w_ao),# u = (1-2*w1) u(∅,∅) + w2 u(actions,∅) + w3 u(∅,obs) +w4 u(a',o)
            )
            eval_and_record(c,f"cfg_BF:wa{w}")
        #enhance guidance on obs
        # for w in cfg_coef:
        # #u = w* u(actions,obs) + (1-w)* u(action,∅)​​  =   u(action,∅)​  +  w (u(actions,obs)-u(action,∅))
        #     w_ao=w# w_ao=1+w
        #     w_a=1-w# w_a=-w
        #     c = dataclasses.replace(
        #         config,
        #         inference_delay=inference_delay,
        #         execute_horizon=execute_horizon,
        #         method=CFGMethodConfig(w_1=0.0, w_2=w_a, w_3=0.0,w_4=w_ao),# u = (1-2*w1) u(∅,∅) + w2 u(actions,∅) + w3 u(∅,obs) +w4 u(a',o)
        #     )
        #     eval_and_record(c,f"cfg_BF:wo{w}")

        #independence assumption
        # for w_o in cfg_coef:
        #     w_a=1
        #     w_nn=1-w_o-w_a
        #     c = dataclasses.replace(
        #         config,
        #         inference_delay=inference_delay,
        #         execute_horizon=execute_horizon,
        #         method=CFGMethodConfig(w_1=w_nn, w_2=w_a, w_3=w_o, w_4=0.0),
        #         # u = (1-2*w1) u(∅,∅) + w2 u(actions,∅) + w3 u(∅,obs) +w4 u(a',o)
        #     )
        #     eval_and_record(c,f"cfg_BI:wo{w_o}")

        # for w_a in cfg_coef:
        #     w_o=1
        #     w_nn=1-w_o-w_a
        #     c = dataclasses.replace(
        #         config,
        #         inference_delay=inference_delay,
        #         execute_horizon=execute_horizon,
        #         method=CFGMethodConfig(w_1=w_nn, w_2=w_a, w_3=w_o, w_4=0.0),
        #         # u = (1-2*w1) u(∅,∅) + w2 u(actions,∅) + w3 u(∅,obs) +w4 u(a',o)
        #     )
        #     eval_and_record(c,f"cfg_BI:wa{w_a}")

        for w in cfg_coef:
            c = dataclasses.replace(
                config,
                inference_delay=inference_delay,
                execute_horizon=execute_horizon,
                method=CFG_BI_COS_MethodConfig(w_o=w, w_a=w),
                #u = u(∅,∅) + cos*w_a * [u(actions,∅)-u(∅,∅)] + w_o * [u(∅,obs)-u(∅,∅) ]​​​
            )
            eval_and_record(c,f"cfg_BI_cos:w{w}")

        for w in cfg_coef:
            w_nn=1-w-w
            c = dataclasses.replace(
                config,
                inference_delay=inference_delay,
                execute_horizon=execute_horizon,
                method=CFGMethodConfig(w_1=w_nn, w_2=w, w_3=w, w_4=0.0),
                # u = (1-2*w1) u(∅,∅) + w2 u(actions,∅) + w3 u(∅,obs) +w4 u(a',o)
            )
            eval_and_record(c,f"cfg_BI:w{w}")

        for w in cfg_coef:
            w_nn=1-w-1
            c = dataclasses.replace(
                config,
                inference_delay=inference_delay,
                execute_horizon=execute_horizon,
                method=CFGMethodConfig(w_1=w_nn, w_2=1, w_3=w, w_4=0.0),
                # u = (1-w2-w3) u(∅,∅) + w2 u(actions,∅) + w3 u(∅,obs) +w4 u(a',o)
            )
            eval_and_record(c,f"cfg_BI:wo{w}")

        for w in cfg_coef:
            w_nn=1-w-1
            c = dataclasses.replace(
                config,
                inference_delay=inference_delay,
                execute_horizon=execute_horizon,
                method=CFGMethodConfig(w_1=w_nn, w_2=w, w_3=1, w_4=0.0),
                # u = (1-w2-w3) u(∅,∅) + w2 u(actions,∅) + w3 u(∅,obs) +w4 u(a',o)
            )
            eval_and_record(c,f"cfg_BI:wa{w}")

        #u = 0.5* w_o* u(actions,obs) + 0.5*(1-w_o)* u(action,∅)​​ + 0.5*w_a* u(actions,obs) + 0.5*(1-w_a)* u(∅,obs)​​ 
        #u = (0.5 * w_o + 0.5 * w_a ) * u(actions,obs) + 0.5*(1-w_o)* u(action,∅)​​   + 0.5*(1-w_a)* u(∅,obs)​​ 
        # for w_o in [2,3]:
        #     for w_a in [2,3]:
        #         w_ao = 0.5 * w_o + 0.5 * w_a 
        #         w_an = 0.5*(1-w_o)
        #         w_no = 0.5*(1-w_a)
        #         c = dataclasses.replace(
        #             config,
        #             inference_delay=inference_delay,
        #             execute_horizon=execute_horizon,
        #             method=CFGMethodConfig(w_1=0.5, w_2=w_an, w_3=w_no,w_4=w_ao),# u = (1-2*w1) u(∅,∅) + w2 u(actions,∅) + w3 u(∅,obs) +w4 u(a',o)
        #         )
        #         eval_and_record(c,f"cfg_BF_wo{w_o}_wa{w_a}")
        # naive
        c = dataclasses.replace(
            config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=NaiveMethodConfig()
        )
        eval_and_record(c,"naive_ca",weak_state_dicts=None)

        c = dataclasses.replace(
            config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=NaiveMethodConfig(mask_action=True)
        )
        eval_and_record(c,"naive_un",weak_state_dicts=None)

        # BID
        c = dataclasses.replace(
            config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=BIDMethodConfig()
        )
        eval_and_record(c,"BID_ca",weak_state_dicts=weak_state_dicts)

        c = dataclasses.replace(
            config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=BIDMethodConfig(mask_action=True)
        )
        eval_and_record(c,"BID_un",weak_state_dicts=weak_state_dicts)

        #RTC
        c = dataclasses.replace(
            config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=RealtimeMethodConfig()
        )
        eval_and_record(c,"RTC_ca")

        c = dataclasses.replace(
            config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=RealtimeMethodConfig(mask_action=True)
        )
        eval_and_record(c,"RTC_un",weak_state_dicts=None)


        c = dataclasses.replace(
            config,
            inference_delay=inference_delay,
            execute_horizon=execute_horizon,
            method=RealtimeMethodConfig(prefix_attention_schedule="zeros"),
        )
        eval_and_record(c,"RTC_hard_ca")

        c = dataclasses.replace(
            config,
            inference_delay=inference_delay,
            execute_horizon=execute_horizon,
            method=RealtimeMethodConfig(prefix_attention_schedule="zeros",mask_action=True),
        )
        eval_and_record(c,"RTC_hard_un",weak_state_dicts=None)




    def fmt_secs(s: float) -> str:
        return str(timedelta(seconds=int(s)))

    # --- Build the full task list first (so we can compute ETA) ---
    tasks = []
    inference_delay = 1
    # extra horizons at fixed vel/noise
    for execute_horizon in [2, 4, 6, 8]:
        tasks.append({
            "execute_horizon": execute_horizon,
            "vel_target": 0.0,
            "noise_std": 0.1,
            "label": f"extra_horizon"
        })
    for execute_horizon in [1,3,5,7]:
        # velocity sweeps
        # for vel_target in [0.4, 0.8, 1.2]:
        # # for vel_target in [0.7, 1.3]:
        #     tasks.append({
        #         "execute_horizon": execute_horizon,
        #         "vel_target": vel_target,
        #         "noise_std": 0.1,
        #         "label": f"vel_target={vel_target:.2f}"
        #     })
        # noise sweeps at static target
        for noisestd in [ 0.1, 0.2, 0.3, 0.4]:
        # for noisestd in [0.00, 0.4]:
            tasks.append({
                "execute_horizon": execute_horizon,
                "vel_target": 0.0,
                "noise_std": noisestd,
                "label": f"noise_std={noisestd:.2f}"
            })



    # --- Run with timing / ETA ---
    durations = []
    t_all0 = time.time()
    for idx, t in enumerate(tasks, start=1):
        execute_horizon = t["execute_horizon"]
        vel_target = t["vel_target"]
        test_noise_std = t["noise_std"]
        print(f"\n[inference_delay={inference_delay}] "
            f"[execute_horizon={execute_horizon}] "
            f"[{t['label']}]  -> running...")
        t0 = time.time()
        test_methods(config, levels, env, vel_target, inference_delay, execute_horizon, test_noise_std)
        dt = time.time() - t0
        durations.append(dt)
        avg = sum(durations) / len(durations)
        remaining = avg * (len(tasks) - idx)
        print(f"[{idx}/{len(tasks)}] last={fmt_secs(dt)}  "
            f"avg={fmt_secs(avg)}  ETA={fmt_secs(remaining)}")
        df = pd.DataFrame(results)
        df.to_csv(pathlib.Path(output_dir) / "results.csv", index=False)

    t_all = time.time() - t_all0
    print(f"\nAll tasks done in {fmt_secs(t_all)}. "
        f"Mean per task: {fmt_secs(sum(durations)/len(durations))}")
    
if __name__ == "__main__":
    tyro.cli(main)
