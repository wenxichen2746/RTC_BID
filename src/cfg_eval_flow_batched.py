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
from util.env_wrappers import NoisyActObsHistoryWrapper
import numpy as np


def _ensure_tuple(value) -> tuple:
    """Convert scalars or sequences into tuples for configuration broadcasting."""
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _broadcast_tuple(values: tuple, length: int) -> tuple:
    """Broadcast tuple elements to a target length."""
    if len(values) == length:
        return values
    if len(values) == 1:
        return values * length
    raise ValueError(f"Cannot broadcast length {len(values)} to {length}.")


def _chunk_sequence(sequence, chunk_size: int):
    """Yield fixed-size chunks of an indexable sequence."""
    for idx in range(0, len(sequence), chunk_size):
        yield sequence[idx : idx + chunk_size]


def _split_env_counts(total_envs: int, num_weights: int) -> tuple[int, ...]:
    """Divide total_envs across num_weights variants as evenly as possible."""
    if num_weights <= 0:
        raise ValueError("num_weights must be positive.")
    if total_envs < num_weights:
        raise ValueError(
            f"num_evals ({total_envs}) must be >= number of weight variants ({num_weights})."
        )
    base = total_envs // num_weights
    remainder = total_envs % num_weights
    counts = [base + (1 if i < remainder else 0) for i in range(num_weights)]
    if any(c <= 0 for c in counts):
        raise ValueError("Each weight variant must receive at least one environment.")
    return tuple(int(c) for c in counts)

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
    w_1: float | tuple[float, ...] = 1.0
    w_2: float | tuple[float, ...] = 2.0
    w_3: float | tuple[float, ...] = 2.0
    w_4: float | tuple[float, ...] = 0.0

    def weight_sets(self) -> tuple[tuple[float, float, float, float], ...]:
        w1 = _ensure_tuple(self.w_1)
        w2 = _ensure_tuple(self.w_2)
        w3 = _ensure_tuple(self.w_3)
        w4 = _ensure_tuple(self.w_4)
        target_len = max(len(w1), len(w2), len(w3), len(w4))
        w1 = _broadcast_tuple(w1, target_len)
        w2 = _broadcast_tuple(w2, target_len)
        w3 = _broadcast_tuple(w3, target_len)
        w4 = _broadcast_tuple(w4, target_len)
        return tuple((float(a), float(b), float(c), float(d)) for a, b, c, d in zip(w1, w2, w3, w4))

    def num_weights(self) -> int:
        return len(self.weight_sets())

@dataclasses.dataclass(frozen=True)
class CFGCOS_MethodConfig:
    # weights in u = (1-2*w1) u(∅,∅) + w2 u(actions,∅) + w3 u(∅,obs)
    w_a: float | tuple[float, ...] = 1.0

    def weight_sets(self) -> tuple[float, ...]:
        weights = _ensure_tuple(self.w_a)
        return tuple(float(w) for w in weights)

    def num_weights(self) -> int:
        return len(self.weight_sets())

@dataclasses.dataclass(frozen=True)
class CFG_BI_COS_MethodConfig:
    #u = u(∅,∅) + cos*w_a * [u(actions,∅)-u(∅,∅)] + w_o * [u(∅,obs)-u(∅,∅) ]​​​
    w_a: float | tuple[float, ...] = 1.0
    w_o: float | tuple[float, ...] = 1.0
    weight_schedule: bool | tuple[bool, ...] = False

    def weight_sets(self) -> tuple[tuple[float, float, bool], ...]:
        w_a = _ensure_tuple(self.w_a)
        w_o = _ensure_tuple(self.w_o)
        schedule = _ensure_tuple(self.weight_schedule)
        target_len = max(len(w_a), len(w_o), len(schedule))
        w_a = _broadcast_tuple(w_a, target_len)
        w_o = _broadcast_tuple(w_o, target_len)
        schedule = _broadcast_tuple(schedule, target_len)
        return tuple((float(a), float(b), bool(c)) for a, b, c in zip(w_a, w_o, schedule))

    def num_weights(self) -> int:
        return len(self.weight_sets())

@dataclasses.dataclass(frozen=True)
class EvalConfig:
    step: int = -1
    weak_step: int = 5 #| None = None
    num_evals: int =  4096 #2000
    max_cfg_methods_per_batch: int|None = None
    
    num_flow_steps: int = 5
    

    inference_delay: int = 0
    execute_horizon: int = 1
    method: NaiveMethodConfig | RealtimeMethodConfig | BIDMethodConfig | CFGMethodConfig | CFGCOS_MethodConfig | CFG_BI_COS_MethodConfig = NaiveMethodConfig()

    model: _model.ModelConfig = _model.ModelConfig()

    obs_history_length: int = 1
    act_history_length: int = 8 #4


def eval(
    config: EvalConfig,
    env: kenv.environment.Environment,
    rng: jax.Array,
    level: kenv_state.EnvState,
    policy: _model.FlowPolicyCFG2,
    env_params: kenv_state.EnvParams,
    static_env_params: kenv_state.EnvParams,
    weak_policy: _model.FlowPolicyCFG2 | None = None,
    noise_std: float = 0.0,
):
    if config.num_evals <= 0:
        raise ValueError("EvalConfig.num_evals must be positive.")

    total_envs_config = int(config.num_evals)
    method = config.method

    env_counts: tuple[int, ...] = (total_envs_config,)
    cfg_w1_env = cfg_w2_env = cfg_w3_env = cfg_w4_env = None
    cfg_cos_weight_env = None
    cfg_bi_w_a_env = cfg_bi_w_o_env = None
    cfg_bi_schedule_env = None

    if isinstance(method, CFGMethodConfig):
        weight_sets = method.weight_sets()
        num_weight_variants = len(weight_sets)
        if num_weight_variants == 0:
            raise ValueError("CFGMethodConfig.weight_sets() must contain at least one weight tuple.")
        env_counts = _split_env_counts(total_envs_config, num_weight_variants)
        w1_vals = np.asarray([w[0] for w in weight_sets], dtype=np.float32)
        w2_vals = np.asarray([w[1] for w in weight_sets], dtype=np.float32)
        w3_vals = np.asarray([w[2] for w in weight_sets], dtype=np.float32)
        w4_vals = np.asarray([w[3] for w in weight_sets], dtype=np.float32)
        cfg_w1_env = jnp.asarray(np.repeat(w1_vals, env_counts))
        cfg_w2_env = jnp.asarray(np.repeat(w2_vals, env_counts))
        cfg_w3_env = jnp.asarray(np.repeat(w3_vals, env_counts))
        cfg_w4_env = jnp.asarray(np.repeat(w4_vals, env_counts))
    elif isinstance(method, CFGCOS_MethodConfig):
        weights = method.weight_sets()
        num_weight_variants = len(weights)
        if num_weight_variants == 0:
            raise ValueError("CFGCOS_MethodConfig.weight_sets() must contain at least one weight.")
        env_counts = _split_env_counts(total_envs_config, num_weight_variants)
        w_vals = np.asarray(list(weights), dtype=np.float32)
        cfg_cos_weight_env = jnp.asarray(np.repeat(w_vals, env_counts))
    elif isinstance(method, CFG_BI_COS_MethodConfig):
        weight_sets = method.weight_sets()
        num_weight_variants = len(weight_sets)
        if num_weight_variants == 0:
            raise ValueError("CFG_BI_COS_MethodConfig.weight_sets() must contain at least one weight tuple.")
        env_counts = _split_env_counts(total_envs_config, num_weight_variants)
        w_a_vals = np.asarray([w[0] for w in weight_sets], dtype=np.float32)
        w_o_vals = np.asarray([w[1] for w in weight_sets], dtype=np.float32)
        schedule_vals = np.asarray([w[2] for w in weight_sets], dtype=bool)
        cfg_bi_w_a_env = jnp.asarray(np.repeat(w_a_vals, env_counts))
        cfg_bi_w_o_env = jnp.asarray(np.repeat(w_o_vals, env_counts))
        cfg_bi_schedule_env = jnp.asarray(np.repeat(schedule_vals, env_counts))
    else:
        num_weight_variants = 1

    env_counts_arr = np.asarray(env_counts, dtype=np.int32)
    if np.any(env_counts_arr <= 0):
        raise ValueError("Each weight variant must receive at least one environment.")
    num_weight_variants = int(len(env_counts_arr))
    total_envs = int(env_counts_arr.sum())
    env_offsets = np.concatenate(([0], np.cumsum(env_counts_arr)))
    env_slices = [slice(int(env_offsets[i]), int(env_offsets[i + 1])) for i in range(num_weight_variants)]

    eval_base_env = kenv.make_kinetix_env_from_name(
        "Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params
    )
    eval_env = NoisyActObsHistoryWrapper(
        env,
        obs_history_length=config.obs_history_length,
        act_history_length=config.act_history_length,
        noise_std=noise_std,
    )
    env = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(wrappers.AutoReplayWrapper(eval_env)),
        total_envs,
    )

    render_video = train_expert.make_render_video(renderer_pixels.make_render_pixels(env_params, static_env_params))
    assert config.execute_horizon >= config.inference_delay, f"{config.execute_horizon=} {config.inference_delay=}"

    def execute_chunk(carry, _):
        def step(carry, action):
            rng_, obs_, env_state_ = carry
            rng_, key_ = jax.random.split(rng_)
            next_obs_, next_env_state_, reward, done, info = env.step(key_, env_state_, action, env_params)
            return (rng_, next_obs_, next_env_state_), (done, env_state_, info)

        rng_, obs_, env_state_, action_chunk_, n_ = carry
        rng_, key_ = jax.random.split(rng_)

        cos_hist_this = jnp.full((obs_.shape[0], config.num_flow_steps), jnp.nan)
        action_var_this = jnp.full((obs_.shape[0], config.num_flow_steps), jnp.nan)

        if isinstance(method, NaiveMethodConfig):
            next_action_chunk = policy.action(key_, obs_, config.num_flow_steps, mask_action=method.mask_action)

        elif isinstance(method, RealtimeMethodConfig):
            prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
            next_action_chunk = policy.realtime_action(
                key_,
                obs_,
                config.num_flow_steps,
                action_chunk_,
                config.inference_delay,
                prefix_attention_horizon,
                method.prefix_attention_schedule,
                method.max_guidance_weight,
                mask_action=method.mask_action,
            )

        elif isinstance(method, BIDMethodConfig):
            prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
            next_action_chunk, action_var = policy.bid_action(
                key_,
                obs_,
                config.num_flow_steps,
                action_chunk_,
                config.inference_delay,
                config.execute_horizon,
                prefix_attention_horizon,
                method.n_samples,
                bid_k=method.bid_k,
                bid_weak_policy=weak_policy if method.bid_k is not None else None,
                mask_action=method.mask_action,
            )
            action_var_this = action_var

        elif isinstance(method, CFGMethodConfig):
            next_action_chunk = policy.action_cfg(
                key_,
                obs_,
                config.num_flow_steps,
                w1=cfg_w1_env,
                w2=cfg_w2_env,
                w3=cfg_w3_env,
                w4=cfg_w4_env,
            )

        elif isinstance(method, CFGCOS_MethodConfig):
            next_action_chunk, cos_hist = policy.action_cfg_cos(
                key_,
                obs_,
                config.num_flow_steps,
                w_a=cfg_cos_weight_env,
            )
            cos_hist_this = jnp.transpose(cos_hist, (1, 0))

        elif isinstance(method, CFG_BI_COS_MethodConfig):
            next_action_chunk, cos_hist = policy.action_cfg_BI_cos(
                key_,
                obs_,
                config.num_flow_steps,
                w_o=cfg_bi_w_o_env,
                w_a=cfg_bi_w_a_env,
                weight_schedule=cfg_bi_schedule_env,
            )
            cos_hist_this = jnp.transpose(cos_hist, (1, 0))
        else:
            raise ValueError(f"Unknown method: {method}")

        valid_horizon = max(0, policy.action_chunk_size - config.execute_horizon)

        def _overlap_loss(prev_chunk, new_chunk):
            prev_overlap = prev_chunk[:, :valid_horizon, :]
            new_overlap = new_chunk[:, :valid_horizon, :]
            if valid_horizon == 0:
                return jnp.zeros(prev_chunk.shape[0])
            diff = jnp.linalg.norm(new_overlap - prev_overlap, axis=-1)
            return jnp.mean(diff, axis=-1)

        backward_loss_per_env = _overlap_loss(action_chunk_, next_action_chunk)

        delay_idx = config.inference_delay if config.inference_delay > 0 else 0
        delay_idx = min(delay_idx, action_chunk_.shape[1] - 1)
        prev_boundary_action = action_chunk_[:, delay_idx, :]
        next_boundary_action = next_action_chunk[:, delay_idx, :]
        cross_chunk_distance_per_env = jnp.linalg.norm(next_boundary_action - prev_boundary_action, axis=-1)

        action_chunk_to_execute = jnp.concatenate(
            [
                action_chunk_[:, : config.inference_delay],
                next_action_chunk[:, config.inference_delay : config.execute_horizon],
            ],
            axis=1,
        )
        next_action_chunk = jnp.concatenate(
            [
                next_action_chunk[:, config.execute_horizon :],
                jnp.zeros((obs_.shape[0], config.execute_horizon, policy.action_dim)),
            ],
            axis=1,
        )
        next_n = jnp.concatenate([n_[config.execute_horizon :], jnp.zeros(config.execute_horizon, dtype=jnp.int32)])

        (rng_, next_obs, next_env_state), (dones, env_states, infos) = jax.lax.scan(
            step,
            (rng_, obs_, env_state_),
            action_chunk_to_execute.transpose(1, 0, 2),
        )

        return (
            rng_,
            next_obs,
            next_env_state,
            next_action_chunk,
            next_n,
        ), (dones, env_states, infos, cos_hist_this, action_var_this, backward_loss_per_env, cross_chunk_distance_per_env)

    rng, key = jax.random.split(rng)
    obs, env_state = env.reset_to_level(key, level, env_params)

    rng, key = jax.random.split(rng)
    action_chunk = policy.action(key, obs, config.num_flow_steps)
    n = jnp.ones(action_chunk.shape[1], dtype=jnp.int32)

    scan_length = math.ceil(env_params.max_timesteps / config.execute_horizon)
    _, (dones, env_states, infos, cos_hist_iters, action_var_iters, backward_losses, cross_chunk_distances) = jax.lax.scan(
        execute_chunk,
        (rng, obs, env_state, action_chunk, n),
        None,
        length=scan_length,
    )

    L, H, B = dones.shape
    T = L * H
    assert B == total_envs, f"Inconsistent batch size: {B} vs {total_envs}"

    dones_flat = dones.reshape(T, B)
    env_states = jax.tree.map(lambda x: x.reshape(T, *x.shape[2:]), env_states)

    def _is_array_like(x):
        return hasattr(x, "shape") and hasattr(x, "reshape")

    infos_flat = {}
    for k, v in infos.items():
        if v is None or not _is_array_like(v):
            continue
        if v.shape[0] == L and v.shape[1] == H:
            infos_flat[k] = v.reshape(T, *v.shape[2:])
        elif v.shape[0] == T:
            infos_flat[k] = v

    assert dones_flat.shape[0] >= env_params.max_timesteps, f"{dones_flat.shape=}"

    def _episode_stat(x):
        return jnp.nanmax(x, axis=0)

    num_weights = len(env_slices)

    return_info = {}
    for key_name in ["returned_episode_returns", "returned_episode_lengths", "returned_episode_solved"]:
        if key_name in infos_flat:
            per_batch = _episode_stat(infos_flat[key_name])
            means = [jnp.nanmean(per_batch[s]) for s in env_slices]
            values = jnp.stack(means)
            return_info[key_name] = values

            if key_name == "returned_episode_lengths":
                stds = [jnp.nanstd(per_batch[s]) for s in env_slices]
                return_info["returned_episode_lengths_mean"] = values
                return_info["returned_episode_lengths_std"] = jnp.stack(stds)

            if key_name == "returned_episode_returns":
                stds = [jnp.nanstd(per_batch[s]) for s in env_slices]
                return_info["returned_episode_returns_mean"] = values
                return_info["returned_episode_returns_std"] = jnp.stack(stds)

    if ("returned_episode_lengths" in infos_flat) and ("returned_episode_solved" in infos_flat):
        lens_per_batch = _episode_stat(infos_flat["returned_episode_lengths"])
        solved_mask = _episode_stat(infos_flat["returned_episode_solved"])
        solved_means = []
        solved_stds = []
        for sl in env_slices:
            lens_seg = lens_per_batch[sl]
            mask_seg = solved_mask[sl]
            denom = jnp.maximum(1.0, mask_seg.sum())
            mean_val = (lens_seg * mask_seg).sum() / denom
            diffs = (lens_seg - mean_val) * mask_seg
            var = (diffs * diffs).sum() / denom
            solved_means.append(mean_val)
            solved_stds.append(jnp.sqrt(var))
        return_info["returned_episode_lengths_solved_mean"] = jnp.stack(solved_means)
        return_info["returned_episode_lengths_solved_std"] = jnp.stack(solved_stds)

    if "match" in infos_flat:
        match_vals = infos_flat["match"]
        match_means = [jnp.nanmean(match_vals[:, sl]) for sl in env_slices]
        return_info["match"] = jnp.stack(match_means)

    backward_means = []
    backward_stds = []
    for sl in env_slices:
        seg = backward_losses[:, sl]
        backward_means.append(jnp.nanmean(seg))
        backward_stds.append(jnp.nanstd(seg))
    return_info["backward_overlap_loss"] = jnp.stack(backward_means)
    return_info["backward_overlap_loss_std"] = jnp.stack(backward_stds)

    cross_means = []
    cross_stds = []
    for sl in env_slices:
        seg = cross_chunk_distances[:, sl]
        cross_means.append(jnp.nanmean(seg))
        cross_stds.append(jnp.nanstd(seg))
    return_info["cross_chunk_distance_mean"] = jnp.stack(cross_means)
    return_info["cross_chunk_distance_std"] = jnp.stack(cross_stds)

    cos_means = []
    cos_stds = []
    for sl in env_slices:
        seg = cos_hist_iters[:, sl, :]
        cos_means.append(jnp.nanmean(seg, axis=(0, 1)))
        cos_stds.append(jnp.nanstd(seg, axis=(0, 1)))
    cos_step_mean = jnp.stack(cos_means, axis=0)
    cos_step_std = jnp.stack(cos_stds, axis=0)

    var_means = []
    var_stds = []
    for sl in env_slices:
        seg = action_var_iters[:, sl, :]
        var_means.append(jnp.nanmean(seg, axis=(0, 1)))
        var_stds.append(jnp.nanstd(seg, axis=(0, 1)))
    action_var_mean = jnp.stack(var_means, axis=0)
    action_var_std = jnp.stack(var_stds, axis=0)

    cos_artifacts = {
        "episode_mean": cos_step_mean,
        "episode_std": cos_step_std,
    }

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

    base_env_for_shape = kenv.make_kinetix_env_from_name(
        "Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params
    )
    _env = cfg_train_expert.ActObsHistoryWrapper(
        base_env_for_shape,
        act_history_length=config.act_history_length,
        obs_history_length=config.obs_history_length,
    )
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
        
        if 'hard_lunar_lander' in level_paths[0]:   
            # print(f'training LL, target moving at vel{vel_target}')
            levels = change_polygon_position_and_velocity(levels, pos_x=1,vel_x=vel_target, index=4) #change to vel_y=something here if needed
        elif 'grasp' in level_paths[0]:
            # print(f'training grasp, randomizing target location')
            levels = change_polygon_position_and_velocity(levels, pos_x=1,vel_x=vel_target, index=10)
        elif 'toss_bin' in level_paths[0]:
            # print(f'training grasp, randomizing target location')
            levels = change_polygon_position_and_velocity(levels, pos_x=None,vel_x=vel_target, index=9)
            levels = change_polygon_position_and_velocity(levels, pos_x=None,vel_x=vel_target, index=10)
            levels = change_polygon_position_and_velocity(levels, pos_x=None,vel_x=vel_target, index=11)
        elif 'place_can_easy' in level_paths[0]:
            # print(f'training grasp, randomizing target location')
            levels = change_polygon_position_and_velocity(levels, pos_x=2,vel_x=vel_target, index=9)
            levels = change_polygon_position_and_velocity(levels, pos_x=2.5,vel_x=vel_target, index=10)
        elif 'drone' in level_paths[0]:
            # print(f'training grasp, randomizing target location')
            levels = change_polygon_position_and_velocity(levels, pos_x=1,vel_x=vel_target, index=4)
            levels = change_polygon_position_and_velocity(levels, pos_x=1,vel_x=vel_target, index=7)
        elif 'catapult' in level_paths[0]:
            # print(f'training catapult, randomizing target location')
            levels = change_polygon_position_and_velocity(levels, pos_x=2.73,vel_x=vel_target*0.5, index=7)
            levels = change_polygon_position_and_velocity(levels, pos_x=2.47,vel_x=vel_target*0.5, index=5)
            levels = change_polygon_position_and_velocity(levels, pos_x=2.97,vel_x=vel_target*0.5, index=6)
        else:
            if vel_target!=0.0:
                print(f'skipping moving target for {level_paths[0]}')
                return
            #raise NotImplementedError("*** Level not recognized DR not implemented **")
        if vel_target==0.0:
            env = DR_static_wrapper(env, level_paths, levels=levels)

        def eval_and_record(c, method_names, weak_state_dicts=None, variant_info=None):
            method_labels = [method_names] if isinstance(method_names, str) else list(method_names)
            out, _, artifacts = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
            out = {k: np.asarray(v) for k, v in out.items()}
            cos_art = artifacts.get("cosine") if artifacts is not None else None
            variance_art = artifacts.get("action_variance") if artifacts is not None else None

            num_levels = len(level_paths)
            num_methods = len(method_labels)

            if variant_info is None:
                variant_info = [
                    {"type": method_labels[i], "env_count": int(c.num_evals), "weights": {}}
                    for i in range(len(method_labels))
                ]
            else:
                variant_info = list(variant_info)

            if len(variant_info) != num_methods:
                raise ValueError(f"variant_info length {len(variant_info)} does not match methods {num_methods}")

            for metric, arr in out.items():
                if arr.ndim == 1:
                    arr = arr[:, None]
                if arr.ndim != 2:
                    raise ValueError(f"Unexpected shape {arr.shape} for metric '{metric}'")
                if arr.shape[1] == 1 and num_methods > 1:
                    arr = np.repeat(arr, num_methods, axis=1)
                if arr.shape[1] != num_methods:
                    raise ValueError(f"Metric '{metric}' has shape {arr.shape}, expected second dim {num_methods}")
                out[metric] = arr

            def _maybe_nan(value):
                if isinstance(value, (float, np.floating)):
                    return value if not np.isnan(value) else np.nan
                return value

            for method_idx, label in enumerate(method_labels):
                info = variant_info[method_idx]
                variant_type = info.get("type", "none")
                env_count = int(info.get("env_count", c.num_evals))
                weights_map = info.get("weights", {})
                cfg_values = {
                    "cfg_w1": weights_map.get("w1", np.nan),
                    "cfg_w2": weights_map.get("w2", np.nan),
                    "cfg_w3": weights_map.get("w3", np.nan),
                    "cfg_w4": weights_map.get("w4", np.nan),
                    "cfg_w_a": weights_map.get("w_a", np.nan),
                    "cfg_w_o": weights_map.get("w_o", np.nan),
                    "cfg_weight_schedule": weights_map.get("weight_schedule", np.nan),
                }
                for level_idx in range(num_levels):
                    for metric, arr in out.items():
                        value = arr[level_idx, method_idx]
                        results[metric].append(float(value))
                    results["delay"].append(inference_delay)
                    results["method"].append(label)
                    results["level"].append(level_paths[level_idx])
                    results["execute_horizon"].append(execute_horizon)
                    results["env_vel"].append(vel_target)
                    results["noise_std"].append(test_noise_std)
                    results["cfg_variant_type"].append(variant_type)
                    results["cfg_env_count"].append(env_count)
                    for key, val in cfg_values.items():
                        if key == "cfg_weight_schedule":
                            if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
                                results[key].append(np.nan)
                            else:
                                results[key].append(bool(val))
                        else:
                            results[key].append(_maybe_nan(val))

            if cos_art is not None:
                ep_mean = np.asarray(cos_art["episode_mean"])
                ep_std = np.asarray(cos_art["episode_std"])
                if ep_mean.ndim == 1:
                    ep_mean = ep_mean[None, :]
                    ep_std = ep_std[None, :]
                if ep_mean.shape[0] == 1 and num_methods > 1:
                    ep_mean = np.repeat(ep_mean, num_methods, axis=0)
                    ep_std = np.repeat(ep_std, num_methods, axis=0)
                for method_idx, label in enumerate(method_labels):
                    if method_idx >= ep_mean.shape[0]:
                        continue
                    mean_slice = np.asarray(ep_mean[method_idx])
                    std_slice = np.asarray(ep_std[method_idx])
                    if not np.isfinite(mean_slice).any():
                        continue
                    mean_vals = mean_slice.reshape(-1)
                    std_vals = std_slice.reshape(-1)
                    info = variant_info[method_idx]
                    variant_type = info.get("type", "none")
                    env_count_row = int(info.get("env_count", c.num_evals))
                    weights_map = info.get("weights", {})
                    schedule_val = weights_map.get("weight_schedule", np.nan)
                    if isinstance(schedule_val, (bool, np.bool_)):
                        schedule_record = schedule_val
                    elif isinstance(schedule_val, (float, np.floating)) and np.isnan(schedule_val):
                        schedule_record = np.nan
                    elif schedule_val is None:
                        schedule_record = np.nan
                    else:
                        schedule_record = bool(schedule_val)
                    row = {
                        "method": label,
                        "execute_horizon": execute_horizon,
                        "env_vel": vel_target,
                        "noise_std": test_noise_std,
                        "cfg_variant_type": variant_type,
                        "cfg_env_count": env_count_row,
                        "cfg_w1": _maybe_nan(weights_map.get("w1", np.nan)),
                        "cfg_w2": _maybe_nan(weights_map.get("w2", np.nan)),
                        "cfg_w3": _maybe_nan(weights_map.get("w3", np.nan)),
                        "cfg_w4": _maybe_nan(weights_map.get("w4", np.nan)),
                        "cfg_w_a": _maybe_nan(weights_map.get("w_a", np.nan)),
                        "cfg_w_o": _maybe_nan(weights_map.get("w_o", np.nan)),
                        "cfg_weight_schedule": schedule_record,
                        "cos_overall_mean": float(np.nanmean(ep_mean[method_idx])),
                        "cos_overall_std": float(np.nanstd(ep_mean[method_idx])),
                    }
                    for s, (mean_val, std_val) in enumerate(zip(mean_vals, std_vals)):
                        row[f"cos_mean_s{s}"] = float(np.nanmean(np.asarray(mean_val)))
                        row[f"cos_std_s{s}"] = float(np.nanmean(np.asarray(std_val)))

                    df_cos = pd.DataFrame([row])
                    csv_path = pathlib.Path(output_dir) / "cosine_analysis.csv"
                    header = not csv_path.exists()
                    df_cos.to_csv(csv_path, mode="a", index=False, header=header)

            if variance_art is not None:
                var_mean = np.asarray(variance_art["episode_mean"])
                var_std = np.asarray(variance_art["episode_std"])
                if var_mean.ndim == 1:
                    var_mean = var_mean[None, :]
                    var_std = var_std[None, :]
                if var_mean.shape[0] == 1 and num_methods > 1:
                    var_mean = np.repeat(var_mean, num_methods, axis=0)
                    var_std = np.repeat(var_std, num_methods, axis=0)
                for method_idx, label in enumerate(method_labels):
                    if method_idx >= var_mean.shape[0]:
                        continue
                    mean_slice = np.asarray(var_mean[method_idx])
                    std_slice = np.asarray(var_std[method_idx])
                    if not np.isfinite(mean_slice).any():
                        continue
                    mean_vals = mean_slice.reshape(-1)
                    std_vals = std_slice.reshape(-1)
                    info = variant_info[method_idx]
                    variant_type = info.get("type", "none")
                    env_count_row = int(info.get("env_count", c.num_evals))
                    weights_map = info.get("weights", {})
                    schedule_val = weights_map.get("weight_schedule", np.nan)
                    if isinstance(schedule_val, (bool, np.bool_)):
                        schedule_record = schedule_val
                    elif isinstance(schedule_val, (float, np.floating)) and np.isnan(schedule_val):
                        schedule_record = np.nan
                    elif schedule_val is None:
                        schedule_record = np.nan
                    else:
                        schedule_record = bool(schedule_val)
                    row = {
                        "method": label,
                        "execute_horizon": execute_horizon,
                        "env_vel": vel_target,
                        "noise_std": test_noise_std,
                        "cfg_variant_type": variant_type,
                        "cfg_env_count": env_count_row,
                        "cfg_w1": _maybe_nan(weights_map.get("w1", np.nan)),
                        "cfg_w2": _maybe_nan(weights_map.get("w2", np.nan)),
                        "cfg_w3": _maybe_nan(weights_map.get("w3", np.nan)),
                        "cfg_w4": _maybe_nan(weights_map.get("w4", np.nan)),
                        "cfg_w_a": _maybe_nan(weights_map.get("w_a", np.nan)),
                        "cfg_w_o": _maybe_nan(weights_map.get("w_o", np.nan)),
                        "cfg_weight_schedule": schedule_record,
                        "variance_overall_mean": float(np.nanmean(var_mean[method_idx])),
                        "variance_overall_std": float(np.nanstd(var_mean[method_idx])),
                    }
                    for s, (mean_val, std_val) in enumerate(zip(mean_vals, std_vals)):
                        row[f"variance_mean_s{s}"] = float(np.nanmean(np.asarray(mean_val)))
                        row[f"variance_std_s{s}"] = float(np.nanmean(np.asarray(std_val)))

                    df_var = pd.DataFrame([row])
                    csv_path = pathlib.Path(output_dir) / "action_variance_analysis.csv"
                    header = not csv_path.exists()
                    df_var.to_csv(csv_path, mode="a", index=False, header=header)

        
        def determine_cfg_chunk_size(total_methods: int) -> int:
            override = config.max_cfg_methods_per_batch
            if override is None or override <= 0:
                return total_methods
            return max(1, min(total_methods, override, int(config.num_evals)))

        def run_cfg_method_sweep(weight_specs, method_labels):
            if not weight_specs:
                return
            weight_specs = list(weight_specs)
            method_labels = list(method_labels)
            chunk_size = determine_cfg_chunk_size(len(weight_specs))
            for specs_chunk, labels_chunk in zip(
                _chunk_sequence(weight_specs, chunk_size),
                _chunk_sequence(method_labels, chunk_size),
            ):
                env_counts_chunk = _split_env_counts(int(config.num_evals), len(specs_chunk))
                variant_info_chunk = []
                for spec, label, count in zip(specs_chunk, labels_chunk, env_counts_chunk):
                    variant_info_chunk.append({
                        "type": label,
                        "env_count": int(count),
                        "weights": {
                            "w1": spec[0],
                            "w2": spec[1],
                            "w3": spec[2],
                            "w4": spec[3],
                        },
                    })
                method_cfg = CFGMethodConfig(
                    w_1=tuple(spec[0] for spec in specs_chunk),
                    w_2=tuple(spec[1] for spec in specs_chunk),
                    w_3=tuple(spec[2] for spec in specs_chunk),
                    w_4=tuple(spec[3] for spec in specs_chunk),
                )
                cfg_config = dataclasses.replace(
                    config,
                    inference_delay=inference_delay,
                    execute_horizon=execute_horizon,
                    method=method_cfg,
                )
                eval_and_record(cfg_config, labels_chunk, variant_info=variant_info_chunk)

        def run_cfg_bi_cos_sweep(weight_specs, method_labels):
            if not weight_specs:
                return
            weight_specs = list(weight_specs)
            method_labels = list(method_labels)
            chunk_size = determine_cfg_chunk_size(len(weight_specs))
            for specs_chunk, labels_chunk in zip(
                _chunk_sequence(weight_specs, chunk_size),
                _chunk_sequence(method_labels, chunk_size),
            ):
                env_counts_chunk = _split_env_counts(int(config.num_evals), len(specs_chunk))
                variant_info_chunk = []
                for spec, label, count in zip(specs_chunk, labels_chunk, env_counts_chunk):
                    variant_info_chunk.append({
                        "type": label,
                        "env_count": int(count),
                        "weights": {
                            "w_a": spec[0],
                            "w_o": spec[1],
                            "weight_schedule": spec[2],
                        },
                    })
                method_cfg = CFG_BI_COS_MethodConfig(
                    w_a=tuple(spec[0] for spec in specs_chunk),
                    w_o=tuple(spec[1] for spec in specs_chunk),
                    weight_schedule=tuple(spec[2] for spec in specs_chunk),
                )
                cfg_config = dataclasses.replace(
                    config,
                    inference_delay=inference_delay,
                    execute_horizon=execute_horizon,
                    method=method_cfg,
                )
                eval_and_record(cfg_config, labels_chunk, variant_info=variant_info_chunk)

        cfg_coef = list(np.arange(1, 7.1, 0.5))
        cfg_method_specs = []
        cfg_method_names = []

        for w in cfg_coef + [0, -1, -2, -3]:
            w_ao = w
            w_o = 1 - w
            cfg_method_specs.append((0.0, 0.0, w_o, w_ao))
            cfg_method_names.append(f"cfg_BF:wa{w}")

        for w in cfg_coef:
            w_nn = 1 - w - w
            cfg_method_specs.append((w_nn, w, w, 0.0))
            cfg_method_names.append(f"cfg_BI:w{w}")

            w_nn_alt = 1 - w - 1
            cfg_method_specs.append((w_nn_alt, 1.0, w, 0.0))
            cfg_method_names.append(f"cfg_BI:wo{w}")

            cfg_method_specs.append((w_nn_alt, w, 1.0, 0.0))
            cfg_method_names.append(f"cfg_BI:wa{w}")

        run_cfg_method_sweep(cfg_method_specs, cfg_method_names)

        cfg_bi_cos_specs = []
        cfg_bi_cos_names = []
        for w in cfg_coef:
            cfg_bi_cos_specs.append((w, w, False))
            cfg_bi_cos_names.append(f"cfg_BI_cos:w{w}")
        # for w in cfg_coef:
        #     cfg_bi_cos_specs.append((w, w, True))
        #     cfg_bi_cos_names.append(f"cfg_BI_cos_schedule:w{w}")
        run_cfg_bi_cos_sweep(cfg_bi_cos_specs, cfg_bi_cos_names)


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
        for vel_target in [0.4, 0.8, 1.2]:
        # for vel_target in [0.7, 1.3]:
            tasks.append({
                "execute_horizon": execute_horizon,
                "vel_target": vel_target,
                "noise_std": 0.1,
                "label": f"vel_target={vel_target:.2f}"
            })
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
        test_methods(config, levels, _env, vel_target, inference_delay, execute_horizon, test_noise_std)
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
