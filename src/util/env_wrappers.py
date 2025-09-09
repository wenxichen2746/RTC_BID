
from flax import struct


import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as wrappers


class BatchEnvWrapper(wrappers.GymnaxWrapper):
    """Define our own BatchEnvWrapper (we don't need different levels)"""

    def __init__(self, env, num: int):
        super().__init__(env)
        self.num = num

    def reset(self, rng, params):
        return jax.vmap(self._env.reset, in_axes=(0, None))(jax.random.split(rng, self.num), params)

    def reset_to_level(self, rng, level, params):
        return jax.vmap(self._env.reset_to_level, in_axes=(0, None, None))(
            jax.random.split(rng, self.num), level, params
        )

    def step(self, rng, state, action, params):
        return jax.vmap(self._env.step, in_axes=(0, 0, 0, None))(jax.random.split(rng, self.num), state, action, params)


@struct.dataclass
class DenseRewardState:
    env_state: kenv_state.EnvState
    timestep: int
    action: jax.Array


class DenseRewardWrapper(wrappers.GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step_env(key, state.env_state, action, params)
        dist_penalty = jax.lax.select(reward > 0, 0.0, info["distance"] / 6.0 / params.max_timesteps)
        new_reward = reward - jax.lax.select(done, (params.max_timesteps - state.timestep) * dist_penalty, dist_penalty)

        next_timestep = jax.lax.select(done, 0, state.timestep + 1)
        return obs, DenseRewardState(env_state, next_timestep, action), new_reward, done, info

    def reset(self, rng, params=None):
        obs, env_state = self._env.reset(rng, params)
        return obs, DenseRewardState(env_state, 0, jnp.zeros(self._env.action_space(params).shape[0]))

    def reset_to_level(self, rng, level, params=None):
        obs, env_state = self._env.reset_to_level(rng, level, params)
        return obs, DenseRewardState(env_state, 0, jnp.zeros(self._env.action_space(params).shape[0]))



class NoisyActionWrapper(wrappers.UnderspecifiedEnvWrapper):
    def __init__(self, env,noise_std=0.1):
        super().__init__(env)
        self.noise_std=noise_std

    def step_env(self, key, state, action, params):
        key1, key2 = jax.random.split(key)
        action = action + jax.random.normal(key1, action.shape) * self.noise_std
        return self._env.step_env(key2, state, action, params)

    def reset_to_level(self, rng, level, params):
        return self._env.reset_to_level(rng, level, params)

    def action_space(self, params):
        return self._env.action_space(params)


@struct.dataclass
class StickyActionState:
    env_state: kenv_state.EnvState
    action: jax.Array


class StickyActionWrapper(wrappers.UnderspecifiedEnvWrapper):
    def __init__(self, env, stickiness: float):
        super().__init__(env)
        self.stickiness = stickiness

    def step_env(self, key, state, action, params):
        key1, key2 = jax.random.split(key)
        actual_action = jax.lax.select(jax.random.bernoulli(key1, self.stickiness), state.action, action)
        obs, env_state, reward, done, info = self._env.step_env(key2, state.env_state, actual_action, params)
        return obs, StickyActionState(env_state, actual_action), reward, done, info

    def reset_to_level(self, rng, level, params):
        obs, env_state = self._env.reset_to_level(rng, level, params)
        return obs, StickyActionState(
            env_state,
            jnp.zeros(
                len(self._env.action_space(params).number_of_dims_per_distribution),
                dtype=jnp.int32,
            ),
        )

    def action_space(self, params):
        return self._env.action_space(params)



@struct.dataclass
class ActObsHistoryState:
    env_state: kenv_state.EnvState
    stacked_obs: jax.Array
    stacked_act: jax.Array
    original_obs: jax.Array


class ActObsHistoryWrapper(wrappers.UnderspecifiedEnvWrapper):
    def __init__(self, env, obs_history_length: int, act_history_length: int):
        super().__init__(env)
        self.obs_history_length = obs_history_length
        self.act_history_length = act_history_length

    def step_env(self, key, state: ActObsHistoryState, action, params):
        obs, env_state, reward, done, info = self._env.step_env(key, state.env_state, action, params)
        stacked_obs = jnp.roll(state.stacked_obs, -1, axis=0).at[-1].set(obs)
        stacked_act = jnp.roll(state.stacked_act, -1, axis=0).at[-1].set(action)
        actobs = jnp.concatenate([stacked_obs.flatten(), stacked_act.flatten()])
        return actobs, ActObsHistoryState(env_state, stacked_obs, stacked_act, obs), reward, done, info

    def reset_to_level(self, rng, level, params):
        obs, env_state = self._env.reset_to_level(rng, level, params)
        stacked_obs = jnp.repeat(obs[None], self.obs_history_length, axis=0)
        act_dim = self._env.action_space(params).shape[0]
        stacked_act = jnp.zeros((self.act_history_length, act_dim), dtype=obs.dtype)
        actobs = jnp.concatenate([stacked_obs.flatten(), stacked_act.flatten()])
        return actobs, ActObsHistoryState(env_state, stacked_obs, stacked_act, obs)

    def action_space(self, params):
        return self._env.action_space(params)

    @staticmethod
    def get_original_obs(env_state) -> jax.Array:
        while not isinstance(env_state, ActObsHistoryState):
            env_state = env_state.env_state
        return env_state.original_obs

    @staticmethod
    def get_stacked_obs(env_state) -> jax.Array:
        while not isinstance(env_state, ActObsHistoryState):
            env_state = env_state.env_state
        return env_state.stacked_obs

    @staticmethod
    def get_stacked_act(env_state) -> jax.Array:
        while not isinstance(env_state, ActObsHistoryState):
            env_state = env_state.env_state
        return env_state.stacked_act


# ---------------- Target Attraction Shaping ----------------

@struct.dataclass
class TargetRewardState:
    env_state: kenv_state.EnvState
    target_xy: jax.Array  # shape (2,)


class PreferenceDiversityRewardWrapper(wrappers.GymnaxWrapper):
    """Adds a small, consistent per-seed attraction towards a fixed (x,y) target.

    - Samples target x,y ~ Uniform([low, high]) at reset/reset_to_level using the provided RNG.
    - Stores target in wrapper state so it remains constant across episode resets when wrapped by AutoReplayWrapper.
    - Applies a shaping penalty proportional to distance to target, normalized by norm_factor * params.max_timesteps.
    """

    def __init__(self, env, low: float = 1.0, high: float = 4.0, norm_factor: float = 9.0):
        super().__init__(env)
        self.low = low
        self.high = high
        self.norm_factor = norm_factor

    def step(self, key, state: TargetRewardState, action, params=None):
        # Call underlying Gymnax-style step to preserve wrapper state (e.g., AutoReplay)
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)

        agent_pos = info.get("agent_position", None)
        if agent_pos is not None:
            dist_to_target = jnp.linalg.norm(agent_pos - state.target_xy)
            shaping = dist_to_target / (self.norm_factor * params.max_timesteps)
            shaping = jnp.nan_to_num(shaping, nan=0.0, posinf=0.0, neginf=0.0)
            reward = reward - shaping
            info["target_position"] = state.target_xy
            info["target_distance"] = dist_to_target

        return obs, TargetRewardState(env_state, state.target_xy), reward, done, info

    def reset(self, rng, params=None):
        # Avoid resampling targets on generic resets to keep per-seed targets fixed.
        # Prefer calling reset_to_level in pipelines so the sampled target persists.
        obs, env_state = self._env.reset(rng, params)
        return obs, TargetRewardState(env_state, jnp.array([jnp.nan, jnp.nan]))

    def reset_to_level(self, rng, level, params=None):
        obs, env_state = self._env.reset_to_level(rng, level, params)
        rx, ry = jax.random.split(rng)
        target_xy = jnp.array([
            jax.random.uniform(rx, (), minval=self.low, maxval=self.high),
            jax.random.uniform(ry, (), minval=self.low, maxval=self.high),
        ])
        return obs, TargetRewardState(env_state, target_xy)
