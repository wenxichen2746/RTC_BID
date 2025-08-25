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

        # default: NaN cos-history for methods that don't produce it
        cos_hist_this = jnp.full((obs.shape[0], config.num_flow_steps), jnp.nan)  # (B, S)

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
            # NOTE: your action_cfg_cos returns (x_1, cos_history) with cos_history shape (num_steps, B)
            next_action_chunk, cos_hist = policy.action_cfg_cos(
                key, obs, config.num_flow_steps, w_a=config.method.w_a
            )
            cos_hist_this = jnp.transpose(cos_hist, (1, 0))  # -> (B, S)

        else:
            raise ValueError(f"Unknown method: {config.method}")

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
        return (rng, next_obs, next_env_state, next_action_chunk, next_n), (dones, env_states, infos, cos_hist_this)

    rng, key = jax.random.split(rng)
    obs, env_state = env.reset_to_level(key, level, env_params)

    rng, key = jax.random.split(rng)
    action_chunk = policy.action(key, obs, config.num_flow_steps)  # [B, horizon, action_dim]
    n = jnp.ones(action_chunk.shape[1], dtype=jnp.int32)

    scan_length = math.ceil(env_params.max_timesteps / config.execute_horizon)
    _, (dones, env_states, infos, cos_hist_iters) = jax.lax.scan(
        execute_chunk,
        (rng, obs, env_state, action_chunk, n),
        None,
        length=scan_length,
    )
    # shapes:
    #   dones:           (scan_length, execute_horizon, B)
    #   cos_hist_iters:  (scan_length, B, S)   where S = config.num_flow_steps

    # # Flatten env steps: (T, B)
    # dones = dones.reshape(-1, *dones.shape[2:])             # -> (T, B)
    # env_states = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), env_states)
    # assert dones.shape[0] >= env_params.max_timesteps, f"{dones.shape=}"

    # # Expand cosine histories to per-step by repeating each iteration for execute_horizon steps
    # cos_hist_steps = jnp.repeat(cos_hist_iters, config.execute_horizon, axis=0)  # (T', B, S)
    # cos_hist_steps = cos_hist_steps[:env_params.max_timesteps]                   # (T,  B, S)

    # # Mask out steps at/after first done: alive = (cumsum(dones) == 0)
    # done_csum = jnp.cumsum(dones.astype(jnp.int32), axis=0)  # (T, B)
    # alive_mask = (done_csum == 0)                            # True before first done
    # cos_hist_masked = jnp.where(alive_mask[:, :, None], cos_hist_steps, jnp.nan)  # (T, B, S)

    # # Episode mean per batch over time (ignore NaNs)
    # cos_episode_mean = jnp.nanmean(cos_hist_masked, axis=0)  # (B, S)
    # cos_episode_std  = jnp.nanstd(cos_hist_masked,  axis=0)  # (B, S)

    # shapes from scan:
    #   dones:           (L = scan_length, H = execute_horizon, B)
    #   cos_hist_iters:  (L, B, S)   # S = num_flow_steps
    L, H, B = dones.shape[0], dones.shape[1], dones.shape[2]

    # 1) video still needs env-steps
    dones_flat = dones.reshape(-1, B)  # (T = L*H, B)
    env_states = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), env_states)
    assert dones_flat.shape[0] >= env_params.max_timesteps, f"{dones_flat.shape=}"

    # 2) chunk‑level "done" (any step within the chunk)
    done_any_in_chunk = jnp.any(dones, axis=1)                   # (L, B)

    # 3) alive mask per chunk: true until (and excluding) the first done-chunk
    done_cum_chunks = jnp.cumsum(done_any_in_chunk, axis=0)      # (L, B)
    alive_chunk = (done_cum_chunks == 0)[..., None]              # (L, B, 1)

    # 4) mask cosine history per chunk (avoid any step-vs-chunk broadcasting)
    cos_hist_masked = jnp.where(alive_chunk, cos_hist_iters, jnp.nan)  # (L, B, S)

    # 5) per-episode summaries across chunks (ignore NaNs after done)
    cos_episode_mean = jnp.nanmean(cos_hist_masked, axis=0)      # (B, S)
    cos_episode_std  = jnp.nanstd(cos_hist_masked,  axis=0)      # (B, S)


    # # Summaries you already computed
    # return_info = {}
    # first_done_idx = jnp.argmax(dones, axis=0)
    # for key in ["returned_episode_returns", "returned_episode_lengths", "returned_episode_solved"]:
    #     return_info[key] = infos[key][first_done_idx, jnp.arange(config.num_evals)].mean()
    # Summaries you already computed (env‑step resolution)
    return_info = {}
    first_done_idx = jnp.argmax(dones_flat, axis=0)  # (B,)
    for key in ["returned_episode_returns", "returned_episode_lengths", "returned_episode_solved"]:
        return_info[key] = infos[key][first_done_idx, jnp.arange(config.num_evals)].mean()

    video = render_video(jax.tree.map(lambda x: x[:, 0], env_states))

    # Pack cosine artifacts
    cos_artifacts = {
        "per_chunk":      cos_hist_masked,   # (L, B, S) with NaNs after done-chunk
        "episode_mean":   cos_episode_mean,  # (B, S)
        "episode_std":    cos_episode_std,   # (B, S)
    }
    return return_info, video, cos_artifacts



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
            eval_info, video, cos_artifacts = eval(config, env, rng, level, policy, env_params, static_env_params, weak_policy,noise_std=test_noise_std)
            return eval_info, video, cos_artifacts
        
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
        else:
            raise NotImplementedError("*** Level not recognized DR not implemented **")
        if vel_target==0.0:
            env=DR_static_wrapper(env,level_paths[0])

        def eval_and_record(c,method_name,weak_state_dicts=None):
            # out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
            out, _, cos_art = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))

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
                ep_mean = np.array(cos_art["episode_mean"]).squeeze()     # (B, S)
                ep_std  = np.array(cos_art["episode_std"]).squeeze()      # (B, S)
                if np.isfinite(ep_mean).any():
                    B, S = ep_mean.shape
                    env_idx = np.repeat(np.arange(B), S)
                    step_idx = np.tile(np.arange(S), B)
                    df_cos = pd.DataFrame({
                        "env_idx": env_idx,
                        "step_idx": step_idx,
                        "cos_mean": ep_mean.reshape(-1),
                        "cos_std":  ep_std.reshape(-1),
                        "method": method_name,
                        "execute_horizon": execute_horizon,
                        "env_vel": vel_target,
                        "noise_std": test_noise_std,
                    })
                    csv_path = pathlib.Path(output_dir) / "cosine_analysis.csv"
                    header = not csv_path.exists()
                    df_cos.to_csv(csv_path, mode="a", index=False, header=header)

        cfg_coef=[x for x in range(-1, 5)]
        for w_a in cfg_coef:
            c = dataclasses.replace(
                config,
                inference_delay=inference_delay,
                execute_horizon=execute_horizon,
                method=CFGCOS_MethodConfig(w_a=w_a),#u=u(a|o)+ cos_coef*w*(u(a|a',o)-u(a|o))
            )
            eval_and_record(c,f"cfg_BF_cos:wa{w_a}")

        # naive
        c = dataclasses.replace(
            config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=NaiveMethodConfig()
        )
        eval_and_record(c,"naive_ca",weak_state_dicts=None)

        c = dataclasses.replace(
            config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=NaiveMethodConfig(mask_action=True)
        )
        eval_and_record(c,"naive_un",weak_state_dicts=None)

        #RTC
        # c = dataclasses.replace(
        #     config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=RealtimeMethodConfig()
        # )
        # eval_and_record(c,"RTC_ca")

        c = dataclasses.replace(
            config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=RealtimeMethodConfig(mask_action=True)
        )
        eval_and_record(c,"RTC_un",weak_state_dicts=None)


        # c = dataclasses.replace(
        #     config,
        #     inference_delay=inference_delay,
        #     execute_horizon=execute_horizon,
        #     method=RealtimeMethodConfig(prefix_attention_schedule="zeros"),
        # )
        # eval_and_record(c,"RTC_hard_ca")

        c = dataclasses.replace(
            config,
            inference_delay=inference_delay,
            execute_horizon=execute_horizon,
            method=RealtimeMethodConfig(prefix_attention_schedule="zeros",mask_action=True),
        )
        eval_and_record(c,"RTC_hard_un",weak_state_dicts=None)

        #BID
        # c = dataclasses.replace(
        #     config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=BIDMethodConfig()
        # )
        # eval_and_record(c,"BID_ca",weak_state_dicts=weak_state_dicts)

        c = dataclasses.replace(
            config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=BIDMethodConfig(mask_action=True)
        )
        eval_and_record(c,"BID_un",weak_state_dicts=weak_state_dicts)
        #CFG
        # cfg_coef=[x * 0.5 for x in range(2, 8.1)]
        



        #enhance guidance on action
        for w in cfg_coef+[-1]:
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
        for w in cfg_coef:
        #u = w* u(actions,obs) + (1-w)* u(action,∅)​​  =   u(action,∅)​  +  w (u(actions,obs)-u(action,∅))
            w_ao=w# w_ao=1+w
            w_a=1-w# w_a=-w
            c = dataclasses.replace(
                config,
                inference_delay=inference_delay,
                execute_horizon=execute_horizon,
                method=CFGMethodConfig(w_1=0.0, w_2=w_a, w_3=0.0,w_4=w_ao),# u = (1-2*w1) u(∅,∅) + w2 u(actions,∅) + w3 u(∅,obs) +w4 u(a',o)
            )
            eval_and_record(c,f"cfg_BF:wo{w}")

        #independence assumption
        for w_o in cfg_coef:
            w_a=1
            w_nn=1-w_o-w_a
            c = dataclasses.replace(
                config,
                inference_delay=inference_delay,
                execute_horizon=execute_horizon,
                method=CFGMethodConfig(w_1=w_nn, w_2=w_a, w_3=w_o, w_4=0.0),
                # u = (1-2*w1) u(∅,∅) + w2 u(actions,∅) + w3 u(∅,obs) +w4 u(a',o)
            )
            eval_and_record(c,f"cfg_BI:wo{w_o}")

        for w_a in cfg_coef:
            w_o=1
            w_nn=1-w_o-w_a
            c = dataclasses.replace(
                config,
                inference_delay=inference_delay,
                execute_horizon=execute_horizon,
                method=CFGMethodConfig(w_1=w_nn, w_2=w_a, w_3=w_o, w_4=0.0),
                # u = (1-2*w1) u(∅,∅) + w2 u(actions,∅) + w3 u(∅,obs) +w4 u(a',o)
            )
            eval_and_record(c,f"cfg_BI:wa{w_a}")



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
        #         eval_and_record(c,f"cfg2_wo{w_o}_wa{w_a}")

    # test cases
    # inference_delay=1
    # for execute_horizon in [1,8]:
    #     # execute_horizon=max(1, inference_delay)
    #     # for vel_target in [0.1,0.3,0.5,0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5]:
        
    #     for vel_target in [0.1, 0.4, 0.7, 1.0, 1.3]:
    #     # for vel_target in [0.0, 0.1,0.3,0.5,0.7, 0.9, 1.1, 1.3]:
    #     # for vel_target in [0.0]:
    #         print(f"{inference_delay=} {execute_horizon=} {vel_target=}")
    #         test_noise_std=0.1
    #         test_methods(config,levels,env,vel_target,inference_delay,execute_horizon,test_noise_std)
    #     for noisestd in [0.00, 0.1, 0.2, 0.4]:
    #         print(f"{inference_delay=} {execute_horizon=} {noisestd=}")
    #         vel_target=0.0
    #         test_methods(config,levels,env,vel_target,inference_delay,execute_horizon,noisestd)
    
    # vel_target=0.0
    # test_noise_std=0.1
    # # for execute_horizon in range(max(1, inference_delay), 8 - inference_delay + 1,2):
    # for execute_horizon in [3,5]:
    #     print(f"{inference_delay=} {execute_horizon=} {test_noise_std=}")
    #     test_methods(config,levels,env,vel_target,inference_delay,execute_horizon,test_noise_std)

    def fmt_secs(s: float) -> str:
        return str(timedelta(seconds=int(s)))

    # --- Build the full task list first (so we can compute ETA) ---
    tasks = []
    inference_delay = 1
    # extra horizons at fixed vel/noise
    for execute_horizon in [3, 5]:
        tasks.append({
            "execute_horizon": execute_horizon,
            "vel_target": 0.0,
            "noise_std": 0.1,
            "label": f"extra_horizon"
        })
    for execute_horizon in [1, 8]:
        # velocity sweeps
        for vel_target in [0.1, 0.4, 0.7, 1.0, 1.3]:
            tasks.append({
                "execute_horizon": execute_horizon,
                "vel_target": vel_target,
                "noise_std": 0.1,
                "label": f"vel_target={vel_target:.2f}"
            })
        # noise sweeps at static target
        for noisestd in [0.00, 0.1, 0.2, 0.4]:
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
