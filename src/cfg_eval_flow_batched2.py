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

# --- Dataclasses and the `eval` function remain unchanged ---

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
class EvalConfig:
    step: int = -1
    weak_step: int = 5 #| None = None
    num_evals: int = 2048
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

        cos_hist_this = jnp.full((obs.shape[0], config.num_flow_steps), jnp.nan)

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
            next_action_chunk = policy.bid_action(
                key, obs, config.num_flow_steps, action_chunk,
                config.inference_delay, prefix_attention_horizon,
                config.method.n_samples,
                bid_k=config.method.bid_k,
                bid_weak_policy=weak_policy if config.method.bid_k is not None else None,
                mask_action=config.method.mask_action
            )
        elif isinstance(config.method, CFGMethodConfig):
            next_action_chunk = policy.action_cfg(
                key, obs, config.num_flow_steps,
                w1=config.method.w_1, w2=config.method.w_2, w3=config.method.w_3, w4=config.method.w_4
            )
        elif isinstance(config.method, CFGCOS_MethodConfig):
            next_action_chunk, cos_hist = policy.action_cfg_cos(
                key, obs, config.num_flow_steps, w_a=config.method.w_a
            )
            cos_hist_this = jnp.transpose(cos_hist, (1, 0))
        else:
            raise ValueError(f"Unknown method: {config.method}")

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
        return (rng, next_obs, next_env_state, next_action_chunk, next_n), (dones, env_states, infos, cos_hist_this)

    rng, key = jax.random.split(rng)
    obs, env_state = env.reset_to_level(key, level, env_params)

    rng, key = jax.random.split(rng)
    action_chunk = policy.action(key, obs, config.num_flow_steps)
    n = jnp.ones(action_chunk.shape[1], dtype=jnp.int32)

    scan_length = math.ceil(env_params.max_timesteps / config.execute_horizon)
    _, (dones, env_states, infos, cos_hist_iters) = jax.lax.scan(
        execute_chunk,
        (rng, obs, env_state, action_chunk, n),
        None,
        length=scan_length,
    )

    L, H, B = dones.shape[0], dones.shape[1], dones.shape[2]
    dones_flat = dones.reshape(-1, B)
    env_states = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), env_states)
    assert dones_flat.shape[0] >= env_params.max_timesteps, f"{dones_flat.shape=}"

    done_any_in_chunk = jnp.any(dones, axis=1)
    done_cum_chunks = jnp.cumsum(done_any_in_chunk, axis=0)
    alive_chunk = (done_cum_chunks == 0)[..., None]
    cos_hist_masked = jnp.where(alive_chunk, cos_hist_iters, jnp.nan)
    cos_episode_mean = jnp.nanmean(cos_hist_masked, axis=0)
    cos_episode_std  = jnp.nanstd(cos_hist_masked,  axis=0)

    return_info = {}
    first_done_idx = jnp.argmax(dones_flat, axis=0)
    for key in ["returned_episode_returns", "returned_episode_lengths", "returned_episode_solved"]:
        return_info[key] = infos[key][first_done_idx, jnp.arange(config.num_evals)].mean()

    video = render_video(jax.tree.map(lambda x: x[:, 0], env_states))

    cos_artifacts = {
        "per_chunk":      cos_hist_masked,
        "episode_mean":   cos_episode_mean,
        "episode_std":    cos_episode_std,
    }
    return return_info, video, cos_artifacts

# --- NEW HELPER FUNCTION ---
def get_all_method_configs(base_config, inference_delay, execute_horizon, weak_policy_is_available):
    """
    Generates a list of method configurations to be tested in parallel.
    This logic is extracted from the original `test_methods` function.
    """
    method_names = []
    method_configs = []
    
    def add_method(name, config):
        method_names.append(name)
        method_configs.append(config)

    cfg_coef = list(range(-1, 5))

    # CFG COS
    for w_a in cfg_coef:
        c = dataclasses.replace(
            base_config, inference_delay=inference_delay, execute_horizon=execute_horizon,
            method=CFGCOS_MethodConfig(w_a=w_a)
        )
        add_method(f"cfg_BF_cos:wa{w_a}", c)

    # Naive
    add_method("naive_ca", dataclasses.replace(base_config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=NaiveMethodConfig()))
    add_method("naive_un", dataclasses.replace(base_config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=NaiveMethodConfig(mask_action=True)))

    # RTC
    add_method("RTC_un", dataclasses.replace(base_config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=RealtimeMethodConfig(mask_action=True)))
    add_method("RTC_hard_un", dataclasses.replace(base_config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=RealtimeMethodConfig(prefix_attention_schedule="zeros", mask_action=True)))
    
    # BID (only if a weak policy was loaded)
    if weak_policy_is_available:
        add_method("BID_un", dataclasses.replace(base_config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=BIDMethodConfig(mask_action=True)))

    # CFG BF (Action Guidance)
    for w in cfg_coef + [-1]:
        w_ao = w
        w_o = 1 - w
        c = dataclasses.replace(
            base_config, inference_delay=inference_delay, execute_horizon=execute_horizon,
            method=CFGMethodConfig(w_1=0.0, w_2=0.0, w_3=w_o, w_4=w_ao)
        )
        add_method(f"cfg_BF:wa{w}", c)

    # CFG BF (Observation Guidance)
    for w in cfg_coef:
        w_ao = w
        w_a = 1 - w
        c = dataclasses.replace(
            base_config, inference_delay=inference_delay, execute_horizon=execute_horizon,
            method=CFGMethodConfig(w_1=0.0, w_2=w_a, w_3=0.0, w_4=w_ao)
        )
        add_method(f"cfg_BF:wo{w}", c)

    # CFG BI (Independence Assumption)
    for w_o in cfg_coef:
        w_a = 1
        w_nn = 1 - w_o - w_a
        c = dataclasses.replace(
            base_config, inference_delay=inference_delay, execute_horizon=execute_horizon,
            method=CFGMethodConfig(w_1=w_nn, w_2=w_a, w_3=w_o, w_4=0.0)
        )
        add_method(f"cfg_BI:wo{w_o}", c)

    for w_a in cfg_coef:
        w_o = 1
        w_nn = 1 - w_o - w_a
        c = dataclasses.replace(
            base_config, inference_delay=inference_delay, execute_horizon=execute_horizon,
            method=CFGMethodConfig(w_1=w_nn, w_2=w_a, w_3=w_o, w_4=0.0)
        )
        add_method(f"cfg_BI:wa{w_a}", c)
        
    return method_names, method_configs


# --- REFACTORED main FUNCTION ---
def main(
    run_path: str,
    config: EvalConfig = EvalConfig(),
    level_paths: Sequence[str] = (
        "worlds/l/hard_lunar_lander.json",
    ),
    seed: int = 0,
    output_dir: str | None = "eval_output",
):
    # --- Initial setup remains the same ---
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)

    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)
    _env=cfg_train_expert.ActObsHistoryWrapper(env, act_history_length=4, obs_history_length=1)
    
    state_dicts = []
    weak_state_dicts = []
    for level_path in level_paths:
        level_name = level_path.replace("/", "_").replace(".json", "")
        log_dirs = [p for p in pathlib.Path(run_path).iterdir() if p.is_dir() and p.name.isdigit()]
        if not log_dirs:
            raise FileNotFoundError(f"No checkpoint dirs found under {run_path}")
        log_dirs = sorted(log_dirs, key=lambda p: int(p.name))
        last_dir = log_dirs[-1]
        last_ckpt = last_dir / "policies" / f"{level_name}.pkl"
        if not last_ckpt.exists():
            raise FileNotFoundError(f"Missing {last_ckpt}")
        with last_ckpt.open("rb") as f:
            state_dicts.append(pickle.load(f))

        if config.weak_step is not None:
            weak_dir = pathlib.Path(run_path) / str(config.weak_step)
            weak_ckpt = weak_dir / "policies" / f"{level_name}.pkl"
            if not weak_ckpt.exists():
                raise FileNotFoundError(f"Missing {weak_ckpt}")
            with weak_ckpt.open("rb") as f:
                weak_state_dicts.append(pickle.load(f))

    state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *state_dicts))
    if config.weak_step is not None:
        weak_state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *weak_state_dicts))
    else:
        weak_state_dicts = None

    action_dim = _env.action_space(env_params).shape[0]
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    pspec = jax.sharding.PartitionSpec("x")
    sharding = jax.sharding.NamedSharding(mesh, pspec)

    raw_obs_dim = jax.eval_shape(
        _env.reset_to_level, jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params
    )[0].shape[-1]
    context_act_len = config.act_history_length * action_dim
    context_obs_len = raw_obs_dim - context_act_len
    context_dim = raw_obs_dim
    context_obs_index = (0, context_obs_len)
    context_act_index = (context_obs_len, context_obs_len + context_act_len)

    results = collections.defaultdict(list)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # --- JIT-compiled function for a single method evaluation ---
    @functools.partial(jax.jit, static_argnames=("test_noise_std",))
    def run_eval_for_one_config(
        config: EvalConfig, 
        rng: jax.Array, 
        levels: kenv_state.EnvState, 
        state_dicts, 
        weak_state_dicts, 
        test_noise_std: float
    ):
        # This inner function includes the vmap over levels and shard_map
        @functools.partial(shard_map.shard_map, mesh=mesh, in_specs=(None, pspec, pspec, pspec, pspec, None), out_specs=pspec)
        @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, None))
        def _eval_vmapped(
            config: EvalConfig, rng: jax.Array, level: kenv_state.EnvState, state_dict, weak_state_dict, test_noise_std: float
        ):
            policy = _model.FlowPolicyCFG2(
                context_dim=context_dim, action_dim=action_dim, config=config.model,
                rngs=nnx.Rngs(rng), context_act_index=context_act_index,
                context_obs_index=context_obs_index,
            )
            graphdef, state = nnx.split(policy)
            state.replace_by_pure_dict(state_dict)
            policy = nnx.merge(graphdef, state)

            weak_policy = None
            if weak_state_dict is not None and isinstance(config.method, BIDMethodConfig):
                graphdef, state = nnx.split(policy)
                state.replace_by_pure_dict(weak_state_dict)
                weak_policy = nnx.merge(graphdef, state)

            return eval(config, env, rng, level, policy, env_params, static_env_params, weak_policy, noise_std=test_noise_std)

        # A separate RNG is needed for each level vmapped over.
        rngs_for_levels = jax.random.split(rng, len(level_paths))
        return _eval_vmapped(config, rngs_for_levels, levels, state_dicts, weak_state_dicts, test_noise_std)
    
    def fmt_secs(s: float) -> str:
        return str(timedelta(seconds=int(s)))

    # --- Build the full task list ---
    tasks = []
    inference_delay = 1
    for execute_horizon in [3, 5]:
        tasks.append({"execute_horizon": execute_horizon, "vel_target": 0.0, "noise_std": 0.1, "label": "extra_horizon"})
    for execute_horizon in [1, 8]:
        for vel_target in [0.1, 0.4, 0.7, 1.0, 1.3]:
            tasks.append({"execute_horizon": execute_horizon, "vel_target": vel_target, "noise_std": 0.1, "label": f"vel_target={vel_target:.2f}"})
        for noisestd in [0.00, 0.1, 0.2, 0.4]:
            tasks.append({"execute_horizon": execute_horizon, "vel_target": 0.0, "noise_std": noisestd, "label": f"noise_std={noisestd:.2f}"})

    # --- Main execution loop with asynchronous dispatch ---
    durations = []
    t_all0 = time.time()
    main_rng_key = jax.random.key(seed)

    for idx, t in enumerate(tasks, start=1):
        execute_horizon = t["execute_horizon"]
        vel_target = t["vel_target"]
        test_noise_std = t["noise_std"]
        print(f"\n[inference_delay={inference_delay}] "
              f"[execute_horizon={execute_horizon}] "
              f"[{t['label']}]  -> running...")
        
        t0 = time.time()
        
        # 1. Prepare environment for the current task
        current_levels = copy.deepcopy(levels)
        if 'hard_lunar_lander' in level_paths[0]:
            current_levels = change_polygon_position_and_velocity(current_levels, pos_x=1,vel_x=vel_target, index=4)
        elif 'grasp' in level_paths[0]:
            current_levels = change_polygon_position_and_velocity(current_levels, pos_x=1,vel_x=vel_target, index=10)
        elif 'toss_bin' in level_paths[0]:
            current_levels = change_polygon_position_and_velocity(current_levels, pos_x=None,vel_x=vel_target, index=9)
            current_levels = change_polygon_position_and_velocity(current_levels, pos_x=None,vel_x=vel_target, index=10)
            current_levels = change_polygon_position_and_velocity(current_levels, pos_x=None,vel_x=vel_target, index=11)
        elif 'place_can_easy' in level_paths[0]:
            current_levels = change_polygon_position_and_velocity(current_levels, pos_x=2,vel_x=vel_target, index=9)
            current_levels = change_polygon_position_and_velocity(current_levels, pos_x=2.5,vel_x=vel_target, index=10)
        else:
            raise NotImplementedError("*** Level not recognized DR not implemented **")
        
        current_env = DR_static_wrapper(env,level_paths[0]) if vel_target==0.0 else env

        # 2. Get all method configs for this task
        method_names, method_configs = get_all_method_configs(
            config, inference_delay, execute_horizon, weak_state_dicts is not None
        )
        
        # 3. Launch all evaluations asynchronously
        main_rng_key, method_rng_key = jax.random.split(main_rng_key)
        rngs_for_methods = jax.random.split(method_rng_key, len(method_names))
        
        pending_results = []
        for i in range(len(method_names)):
            # Launch one computation per method. JAX does not block here.
            result_future = run_eval_for_one_config(
                method_configs[i],
                rngs_for_methods[i],
                current_levels,
                state_dicts,
                weak_state_dicts,
                test_noise_std
            )
            pending_results.append(result_future)

        # 4. Wait for all computations to finish and get results from device
        final_results = jax.device_get(pending_results)

        # 5. Process the collected results
        for method_idx, method_name in enumerate(method_names):
            out, _, cos_art = final_results[method_idx]
            
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
                ep_mean = np.array(cos_art["episode_mean"]).squeeze()
                ep_std  = np.array(cos_art["episode_std"]).squeeze()
                if np.isfinite(ep_mean).any():
                    B, S = ep_mean.shape
                    df_cos = pd.DataFrame({
                        "env_idx": np.repeat(np.arange(B), S), "step_idx": np.tile(np.arange(S), B),
                        "cos_mean": ep_mean.reshape(-1), "cos_std":  ep_std.reshape(-1),
                        "method": method_name, "execute_horizon": execute_horizon,
                        "env_vel": vel_target, "noise_std": test_noise_std,
                    })
                    csv_path = pathlib.Path(output_dir) / "cosine_analysis.csv"
                    header = not csv_path.exists()
                    df_cos.to_csv(csv_path, mode="a", index=False, header=header)
        
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