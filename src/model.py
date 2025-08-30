import dataclasses
import functools
from typing import Literal, TypeAlias, Self

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing import Optional,Sequence

@dataclasses.dataclass(frozen=True)
class ModelConfig:
    channel_dim: int = 256
    channel_hidden_dim: int = 512
    token_hidden_dim: int = 64
    num_layers: int = 4
    action_chunk_size: int = 8


def posemb_sincos(pos: jax.Array, embedding_dim: int, min_period: float, max_period: float) -> jax.Array:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


PrefixAttentionSchedule: TypeAlias = Literal["linear", "exp", "ones", "zeros"]


def get_prefix_weights(start: int, end: int, total: int, schedule: PrefixAttentionSchedule) -> jax.Array:
    """With start=2, end=6, total=10, the output will be:
    1  1  4/5 3/5 2/5 1/5 0  0  0  0
           ^              ^
         start           end
    `start` (inclusive) is where the chunk starts being allowed to change. `end` (exclusive) is where the chunk stops
    paying attention to the prefix. if start == 0, then the entire chunk is allowed to change. if end == total, then the
    entire prefix is attended to.

    `end` takes precedence over `start` in the sense that, if `end < start`, then `start` is pushed down to `end`. Thus,
    if `end` is 0, then the entire prefix will always be ignored.
    """
    start = jnp.minimum(start, end)
    if schedule == "ones":
        w = jnp.ones(total)
    elif schedule == "zeros":
        w = (jnp.arange(total) < start).astype(jnp.float32)
    elif schedule == "linear" or schedule == "exp":
        w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * jnp.expm1(w) / (jnp.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return jnp.where(jnp.arange(total) >= end, 0, w)


class MLPMixerBlock(nnx.Module):
    def __init__(
        self, token_dim: int, token_hidden_dim: int, channel_dim: int, channel_hidden_dim: int, *, rngs: nnx.Rngs
    ):
        self.token_mix_in = nnx.Linear(token_dim, token_hidden_dim, use_bias=False, rngs=rngs)
        self.token_mix_out = nnx.Linear(token_hidden_dim, token_dim, use_bias=False, rngs=rngs)
        self.channel_mix_in = nnx.Linear(channel_dim, channel_hidden_dim, use_bias=False, rngs=rngs)
        self.channel_mix_out = nnx.Linear(channel_hidden_dim, channel_dim, use_bias=False, rngs=rngs)
        self.norm_1 = nnx.LayerNorm(channel_dim, use_scale=False, use_bias=False, rngs=rngs)
        self.norm_2 = nnx.LayerNorm(channel_dim, use_scale=False, use_bias=False, rngs=rngs)
        self.adaln_1 = nnx.Linear(channel_dim, 3 * channel_dim, kernel_init=nnx.initializers.zeros_init(), rngs=rngs)
        self.adaln_2 = nnx.Linear(channel_dim, 3 * channel_dim, kernel_init=nnx.initializers.zeros_init(), rngs=rngs)

    def __call__(self, x: jax.Array, adaln_cond: jax.Array) -> jax.Array:
        scale_1, shift_1, gate_1 = jnp.split(self.adaln_1(adaln_cond)[:, None], 3, axis=-1)
        scale_2, shift_2, gate_2 = jnp.split(self.adaln_2(adaln_cond)[:, None], 3, axis=-1)

        # token mix
        residual = x
        x = self.norm_1(x) * (1 + scale_1) + shift_1
        x = x.transpose(0, 2, 1)
        x = self.token_mix_in(x)
        x = nnx.gelu(x)
        x = self.token_mix_out(x)
        x = x.transpose(0, 2, 1)
        x = residual + gate_1 * x

        # channel mix
        residual = x
        x = self.norm_2(x) * (1 + scale_2) + shift_2
        x = self.channel_mix_in(x)
        x = nnx.gelu(x)
        x = self.channel_mix_out(x)
        x = residual + gate_2 * x
        return x


class FlowPolicy(nnx.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        config: ModelConfig,
        rngs: nnx.Rngs,
    ):
        self.channel_dim = config.channel_dim
        self.action_dim = action_dim
        self.action_chunk_size = config.action_chunk_size

        self.in_proj = nnx.Linear(action_dim + obs_dim, config.channel_dim, rngs=rngs)
        self.mlp_stack = [
            MLPMixerBlock(
                config.action_chunk_size,
                config.token_hidden_dim,
                config.channel_dim,
                config.channel_hidden_dim,
                rngs=rngs,
            )
            for _ in range(config.num_layers)
        ]
        self.time_mlp = nnx.Sequential(
            nnx.Linear(config.channel_dim, config.channel_dim, rngs=rngs),
            nnx.swish,
            nnx.Linear(config.channel_dim, config.channel_dim, rngs=rngs),
            nnx.swish,
        )
        self.final_norm = nnx.LayerNorm(config.channel_dim, use_scale=False, use_bias=False, rngs=rngs)
        self.final_adaln = nnx.Linear(
            config.channel_dim, 2 * config.channel_dim, kernel_init=nnx.initializers.zeros_init(), rngs=rngs
        )
        self.out_proj = nnx.Linear(config.channel_dim, action_dim, rngs=rngs)

    def __call__(self, obs: jax.Array, x_t: jax.Array, time: jax.Array) -> jax.Array:
        assert x_t.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), x_t.shape
        time_emb = posemb_sincos(
            jnp.broadcast_to(time, obs.shape[0]), self.channel_dim, min_period=4e-3, max_period=4.0
        )
        time_emb = self.time_mlp(time_emb)
        obs = einops.repeat(obs, "b e -> b c e", c=self.action_chunk_size)
        x = jnp.concatenate([x_t, obs], axis=-1)
        x = self.in_proj(x)
        for mlp in self.mlp_stack:
            x = mlp(x, time_emb)
        assert x.shape == (obs.shape[0], self.action_chunk_size, self.channel_dim), x.shape
        scale, shift = jnp.split(self.final_adaln(time_emb)[:, None], 2, axis=-1)
        x = self.final_norm(x) * (1 + scale) + shift
        x = self.out_proj(x)
        return x

    def action(self, rng: jax.Array, obs: jax.Array, num_steps: int) -> jax.Array:
        dt = 1 / num_steps

        def step(carry, _):
            x_t, time = carry
            v_t = self(obs, x_t, time)
            return (x_t + dt * v_t, time + dt), None

        noise = jax.random.normal(rng, shape=(obs.shape[0], self.action_chunk_size, self.action_dim))
        (x_1, _), _ = jax.lax.scan(step, (noise, 0.0), length=num_steps)
        assert x_1.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), x_1.shape
        return x_1
    
    def loss(self, rng: jax.Array, obs: jax.Array, action: jax.Array):
        assert action.dtype == jnp.float32
        assert action.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), action.shape
        noise_rng, time_rng = jax.random.split(rng, 2)
        time = jax.random.uniform(time_rng, (obs.shape[0],))
        noise = jax.random.normal(noise_rng, shape=action.shape)

        x_t = (1 - time[:, None, None]) * noise + time[:, None, None] * action
        u_t = action - noise
        pred = self(obs, x_t, time)
        return jnp.mean(jnp.square(pred - u_t))

    def bid_action(
        self,
        rng: jax.Array,
        obs: jax.Array,
        num_steps: int,
        prev_action_chunk: jax.Array,  # [batch, horizon, action_dim]
        inference_delay: int,
        prefix_attention_horizon: int,
        n_samples: int,
        # when below two are None, it becomes backwards loss only (i.e., rejection sampling)
        bid_weak_policy: Self | None = None,
        bid_k: int | None = None,
    ) -> jax.Array:
        obs = einops.repeat(obs, "b ... -> (n b) ...", n=n_samples)
        weights = get_prefix_weights(inference_delay, prefix_attention_horizon, self.action_chunk_size, "exp")

        def backward_loss(action_chunks: jax.Array):
            error = jnp.linalg.norm(action_chunks - prev_action_chunk, axis=-1)  # [n, b, h]
            return jnp.sum(error * weights[None, None, :], axis=-1)  # [n, b]

        strong_actions = einops.rearrange(self.action(rng, obs, num_steps), "(n b) h d -> n b h d", n=n_samples)
        loss = backward_loss(strong_actions)  # [n, b]

        if bid_weak_policy is not None or bid_k is not None:
            assert bid_weak_policy is not None and bid_k is not None, (bid_weak_policy, bid_k)
            weak_actions = einops.rearrange(
                bid_weak_policy.action(rng, obs, num_steps), "(n b) h d -> n b h d", n=n_samples
            )
            weak_loss = backward_loss(weak_actions)  # [n, b]
            weak_idxs = jax.lax.top_k(-weak_loss.T, bid_k)[1].T  # [k, b]
            strong_idxs = jax.lax.top_k(-loss.T, bid_k)[1].T  # [k, b]
            a_plus = jnp.take_along_axis(strong_actions, strong_idxs[:, :, None, None], axis=0)  # [k, b, h, d]
            a_minus = jnp.take_along_axis(weak_actions, weak_idxs[:, :, None, None], axis=0)  # [k, b, h, d]
            # compute forward loss for each action in strong_actions
            forward_loss = jnp.sum(
                jnp.linalg.norm(strong_actions[:, None] - a_plus[None, :], axis=-1),  # [n, k, b, h]
                axis=(1, 3),  # [n, b]
            ) - jnp.sum(
                jnp.linalg.norm(strong_actions[:, None] - a_minus[None, :], axis=-1),  # [n, k, b, h]
                axis=(1, 3),  # [n, b]
            )
            loss += forward_loss / n_samples

        best_idxs = jnp.argmin(loss, axis=0)  # [b]
        return jnp.take_along_axis(strong_actions, best_idxs[None, :, None, None], axis=0).squeeze(0)  # [b, h, d]

    def realtime_action(
        self,
        rng: jax.Array,
        obs: jax.Array,
        num_steps: int,
        prev_action_chunk: jax.Array,  # [batch, horizon, action_dim]
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule: PrefixAttentionSchedule,
        max_guidance_weight: float,
    ) -> jax.Array:
        dt = 1 / num_steps

        def step(carry, _):
            x_t, time = carry

            @functools.partial(jax.vmap, in_axes=(0, 0, 0, None))  # over batch
            def pinv_corrected_velocity(obs, x_t, y, t):
                def denoiser(x_t):
                    v_t = self(obs[None], x_t[None], t)[0]
                    return x_t + v_t * (1 - t), v_t

                x_1, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)
                weights = get_prefix_weights(
                    inference_delay, prefix_attention_horizon, self.action_chunk_size, prefix_attention_schedule
                )
                error = (y - x_1) * weights[:, None]
                pinv_correction = vjp_fun(error)[0]
                # constants from paper
                inv_r2 = (t**2 + (1 - t) ** 2) / ((1 - t) ** 2)
                c = jnp.nan_to_num((1 - t) / t, posinf=max_guidance_weight)
                guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)
                return v_t + guidance_weight * pinv_correction

            v_t = pinv_corrected_velocity(obs, x_t, prev_action_chunk, time)
            return (x_t + dt * v_t, time + dt), None

        noise = jax.random.normal(rng, shape=(obs.shape[0], self.action_chunk_size, self.action_dim))
        (x_1, _), _ = jax.lax.scan(step, (noise, 0.0), length=num_steps)
        assert x_1.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), x_1.shape
        return x_1

    def loss(self, rng: jax.Array, obs: jax.Array, action: jax.Array):
        assert action.dtype == jnp.float32
        assert action.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), action.shape
        noise_rng, time_rng = jax.random.split(rng, 2)
        time = jax.random.uniform(time_rng, (obs.shape[0],))
        noise = jax.random.normal(noise_rng, shape=action.shape)

        x_t = (1 - time[:, None, None]) * noise + time[:, None, None] * action
        u_t = action - noise
        pred = self(obs, x_t, time)
        return jnp.mean(jnp.square(pred - u_t))
    






# flow policy with null guidance for classifier free guidance:




class FlowPolicyCFG2(nnx.Module):
    def __init__(
        self,
        *,
        context_dim: int,         # len(stacked_obs.flatten()) + len(stacked_act.flatten())
        action_dim: int,
        config: ModelConfig,
        rngs: nnx.Rngs,
        context_act_index: Sequence[int],  # [a, b) slice within context for actions-history part
        context_obs_index: Sequence[int],  # [c, d) slice within context for obs-history part
    ):
        self.channel_dim = config.channel_dim
        self.action_dim = action_dim
        self.action_chunk_size = config.action_chunk_size

        a, b = context_act_index
        c, d = context_obs_index
        assert 0 <= a <= b <= context_dim
        assert 0 <= c <= d <= context_dim
        self.act_start, self.act_end = int(a), int(b)
        self.obs_start, self.obs_end = int(c), int(d)
        assert (self.act_start < self.act_end) and (self.obs_start < self.obs_end)
        assert (self.act_start, self.act_end) != (self.obs_start, self.obs_end), "slices must not be identical"

        act_slice_dim = self.act_end - self.act_start
        obs_slice_dim = self.obs_end - self.obs_start

        # Learnable null vectors for the two context portions
        self.null_act_embed = nnx.Param(jnp.zeros((act_slice_dim,), dtype=jnp.float32))
        self.null_obs_embed = nnx.Param(jnp.zeros((obs_slice_dim,), dtype=jnp.float32))

        # Same backbone as FlowPolicy
        self.in_proj = nnx.Linear(action_dim + context_dim, config.channel_dim, rngs=rngs)
        self.mlp_stack = [
            MLPMixerBlock(
                config.action_chunk_size,
                config.token_hidden_dim,
                config.channel_dim,
                config.channel_hidden_dim,
                rngs=rngs,
            ) for _ in range(config.num_layers)
        ]
        self.time_mlp = nnx.Sequential(
            nnx.Linear(config.channel_dim, config.channel_dim, rngs=rngs),
            nnx.swish,
            nnx.Linear(config.channel_dim, config.channel_dim, rngs=rngs),
            nnx.swish,
        )
        self.final_norm = nnx.LayerNorm(config.channel_dim, use_scale=False, use_bias=False, rngs=rngs)
        self.final_adaln = nnx.Linear(
            config.channel_dim, 2 * config.channel_dim, kernel_init=nnx.initializers.zeros_init(), rngs=rngs
        )
        self.out_proj = nnx.Linear(config.channel_dim, action_dim, rngs=rngs)

    def _apply_null_masks(
        self,
        context: jax.Array,                  # [B, context_dim]
        use_null_act: jax.Array,             # [B] bool
        use_null_obs: jax.Array,             # [B] bool
    ) -> jax.Array:
        # Split out the two slices
        ctx = context
        act_slice = ctx[:, self.act_start:self.act_end]   # [B, A]
        obs_slice = ctx[:, self.obs_start:self.obs_end]   # [B, O]
        
        # Batch-views of learnable nulls
        null_act = jnp.broadcast_to(self.null_act_embed[None, :], act_slice.shape)
        null_obs = jnp.broadcast_to(self.null_obs_embed[None, :], obs_slice.shape)

        # Per-row replacement where masks are True
        act_slice_eff = jnp.where(use_null_act[:, None], null_act, act_slice)
        obs_slice_eff = jnp.where(use_null_obs[:, None], null_obs, obs_slice)

        # Stitch back the full context in a copy-safe way
        context_eff = ctx.at[:, self.act_start:self.act_end].set(act_slice_eff)
        context_eff = context_eff.at[:, self.obs_start:self.obs_end].set(obs_slice_eff)
        return context_eff

    def __call__(
        self,
        context: jax.Array,                   # [B, context_dim]
        x_t: jax.Array,                       # [B, action_chunk_size, action_dim]
        time: jax.Array,                      # [B]
        use_null_act: Optional[jax.Array] = None,  # [B] bool
        use_null_obs: Optional[jax.Array] = None,  # [B] bool
    ) -> jax.Array:
        B = context.shape[0]
        assert x_t.shape == (B, self.action_chunk_size, self.action_dim), x_t.shape
        if use_null_act is None: use_null_act = jnp.zeros((B,), dtype=bool)
        if use_null_obs is None: use_null_obs = jnp.zeros((B,), dtype=bool)

        # Apply masks to build the effective context
        eff_context = self._apply_null_masks(context, use_null_act, use_null_obs)

        # Time embedding
        time_emb = posemb_sincos(jnp.broadcast_to(time, B), self.channel_dim, min_period=4e-3, max_period=4.0)
        time_emb = self.time_mlp(time_emb)

        # Repeat context across tokens and run backbone
        eff_ctx_tokens = einops.repeat(eff_context, "b e -> b c e", c=self.action_chunk_size)
        x = jnp.concatenate([x_t, eff_ctx_tokens], axis=-1)
        x = self.in_proj(x)
        for mlp in self.mlp_stack:
            x = mlp(x, time_emb)
        assert x.shape == (B, self.action_chunk_size, self.channel_dim), x.shape

        scale, shift = jnp.split(self.final_adaln(time_emb)[:, None], 2, axis=-1)
        x = self.final_norm(x) * (1 + scale) + shift
        x = self.out_proj(x)
        return x

    # Fully-conditional rollout (no nulling)
    # def action(self, rng: jax.Array, context: jax.Array, num_steps: int) -> jax.Array:
    #     dt = 1.0 / num_steps
    #     B = context.shape[0]
    #     def step(carry, _):
    #         x_t, time = carry
    #         v_t = self(context, x_t, time)  # both masks default to False
    #         return (x_t + dt * v_t, time + dt), None
    #     noise = jax.random.normal(rng, shape=(B, self.action_chunk_size, self.action_dim))
    #     (x_1, _), _ = jax.lax.scan(step, (noise, 0.0), length=num_steps)
    #     return x_1
    
    def action(self, rng: jax.Array, context: jax.Array, num_steps: int, mask_action: bool = False) -> jax.Array:
        dt = 1.0 / num_steps
        B = context.shape[0]
        mask_act = jnp.ones((B,), dtype=bool) if mask_action else jnp.zeros((B,), dtype=bool)
        mask_obs = jnp.zeros((B,), dtype=bool)

        def step(carry, _):
            x_t, time = carry
            v_t = self(context, x_t, time, use_null_act=mask_act, use_null_obs=mask_obs)
            return (x_t + dt * v_t, time + dt), None

        noise = jax.random.normal(rng, shape=(B, self.action_chunk_size, self.action_dim))
        (x_1, _), _ = jax.lax.scan(step, (noise, 0.0), length=num_steps)
        return x_1

    # Generalized CFG with two independent nulls
    # u = (1-2*w1) u(∅,∅) + w2 u(actions,∅) + w3 u(∅,obs)
    def action_cfg(self, rng: jax.Array, context: jax.Array, num_steps: int, w1: float, w2: float, w3: float, w4: float=0.0) -> jax.Array:
        dt = 1.0 / num_steps
        B = context.shape[0]
        mask_F = jnp.zeros((B,), dtype=bool)
        mask_T = jnp.ones((B,), dtype=bool)

        def step(carry, _):
            x_t, time = carry
            # compute each term only if its weight is non-zero; otherwise use a zero placeholder
            u_nn = self(context, x_t, time, use_null_act=mask_T, use_null_obs=mask_T) if w1 != 0.0 else jnp.zeros_like(x_t)
            u_an = self(context, x_t, time, use_null_act=mask_F, use_null_obs=mask_T) if w2 != 0.0 else jnp.zeros_like(x_t)
            u_no = self(context, x_t, time, use_null_act=mask_T, use_null_obs=mask_F) if w3 != 0.0 else jnp.zeros_like(x_t)
            u_ao = self(context, x_t, time, use_null_act=mask_F, use_null_obs=mask_F) if w4 != 0.0 else jnp.zeros_like(x_t)  # (actions, obs)

            u = w1 * u_nn + w2 * u_an + w3 * u_no + w4 * u_ao
            return (x_t + dt * u, time + dt), None

        noise = jax.random.normal(rng, shape=(B, self.action_chunk_size, self.action_dim))
        (x_1, _), _ = jax.lax.scan(step, (noise, 0.0), length=num_steps)
        return x_1
    
    # def action_cfg_cos(self, rng: jax.Array, context: jax.Array, num_steps: int, w_a: float) -> jax.Array:
    #     #u=u(a|o)+ cos_coef*w*(u(a|a',o)-u(a|o))
    #     dt = 1.0 / num_steps
    #     B = context.shape[0]
    #     mask_F = jnp.zeros((B,), dtype=bool)
    #     mask_T = jnp.ones((B,), dtype=bool)

    #     def step(carry, _):
    #         x_t, time = carry
    #         u_no = self(context, x_t, time, use_null_act=mask_T, use_null_obs=mask_F)
    #         u_ao = self(context, x_t, time, use_null_act=mask_F, use_null_obs=mask_F)
    #         u_guid = u_ao - u_no
    #         dot = jnp.sum(u_guid * u_no, axis=(1, 2))
    #         norm_g = jnp.linalg.norm(u_guid, axis=(1, 2))
    #         norm_n = jnp.linalg.norm(u_no, axis=(1, 2))
    #         cos_coef = dot / (norm_g * norm_n + 1e-12)
    #         cos_coef = jnp.maximum(cos_coef, 0.0).reshape(B, 1, 1)
    #         u = u_no + w_a * cos_coef * u_guid
    #         return (x_t + dt * u, time + dt), None

    #     noise = jax.random.normal(rng, shape=(B, self.action_chunk_size, self.action_dim))
    #     (x_1, _), _ = jax.lax.scan(step, (noise, 0.0), length=num_steps)
    #     return x_1
    def action_cfg_cos(self, rng: jax.Array, context: jax.Array, num_steps: int, w_a: float):
        """
        Returns:
        x_1:        (B, action_chunk_size, action_dim) final action chunk
        cos_history:(num_steps, B) raw cosine(sim(u_guid, u_no)) per step (unclipped)
        """
        dt = 1.0 / num_steps
        B = context.shape[0]
        mask_F = jnp.zeros((B,), dtype=bool)
        mask_T = jnp.ones((B,), dtype=bool)

        def step(carry, _):
            x_t, time = carry

            # u_no = u(a|o), u_ao = u(a|a',o), guidance = u_ao - u_no
            u_no = self(context, x_t, time, use_null_act=mask_T, use_null_obs=mask_F)
            u_ao = self(context, x_t, time, use_null_act=mask_F, use_null_obs=mask_F)
            u_guid = u_ao - u_no

            # raw cosine similarity (before clipping)
            dot = jnp.sum(u_guid * u_no, axis=(1, 2))
            norm_g = jnp.linalg.norm(u_guid, axis=(1, 2))
            norm_n = jnp.linalg.norm(u_no,   axis=(1, 2))
            cos_coef_raw = dot / (norm_g * norm_n + 1e-12)     # shape (B,)
            cos_coef = cos_coef_raw[:, None, None]
            # clip only for the guidance coefficient used in u
            # cos_coef = jnp.maximum(cos_coef_raw, 0.0).reshape(B, 1, 1)

            # update action
            u = u_no + w_a * cos_coef * u_guid
            return (x_t + dt * u, time + dt), cos_coef_raw  # collect raw cos per step (B,)
        noise = jax.random.normal(rng, shape=(B, self.action_chunk_size, self.action_dim))
        (x_1, _), cos_history = jax.lax.scan(step, (noise, 0.0), length=num_steps)
        # cos_history has shape (num_steps, B)
        return x_1, cos_history
    
    def action_cfg_BI_cos(self, rng: jax.Array, context: jax.Array, num_steps: int, w_o: float =1.0, w_a: float =1.0):
        """
        Returns:
        x_1:        (B, action_chunk_size, action_dim) final action chunk
        cos_history:(num_steps, B) raw cosine(sim(u_guid, u_no)) per step (unclipped)
        """
        #u = u(∅,∅) + cos*w_a * [u(actions,∅)-u(∅,∅)] + w_o * [u(∅,obs)-u(∅,∅) ]​​​
        dt = 1.0 / num_steps
        B = context.shape[0]
        mask_F = jnp.zeros((B,), dtype=bool)
        mask_T = jnp.ones((B,), dtype=bool)

        def step(carry, _):
            x_t, time = carry

            # u_no = u(a|o), u_ao = u(a|a',o), guidance = u_ao - u_no
            u_nn = self(context, x_t, time, use_null_act=mask_T, use_null_obs=mask_T)
            u_no = self(context, x_t, time, use_null_act=mask_T, use_null_obs=mask_F)
            u_an = self(context, x_t, time, use_null_act=mask_F, use_null_obs=mask_T)
            u_guid_o = u_no - u_nn
            u_guid_a = u_an - u_nn

            # raw cosine similarity (before clipping)
            dot = jnp.sum(u_guid_o * u_guid_a, axis=(1, 2))
            norm_o = jnp.linalg.norm(u_guid_o, axis=(1, 2))
            norm_a = jnp.linalg.norm(u_guid_a,   axis=(1, 2))
            cos_coef_raw = dot / (norm_o * norm_a + 1e-12)     # shape (B,)
            cos_coef = cos_coef_raw[:, None, None]  

            # clip only for the guidance coefficient used in u
            # cos_coef = jnp.maximum(cos_coef_raw, 0.0).reshape(B, 1, 1)
            # update action
            u = u_no + w_a * cos_coef * u_guid_a + w_o * u_guid_o
            return (x_t + dt * u, time + dt), cos_coef_raw  # collect raw cos per step (B,)
        noise = jax.random.normal(rng, shape=(B, self.action_chunk_size, self.action_dim))
        (x_1, _), cos_history = jax.lax.scan(step, (noise, 0.0), length=num_steps)
        # cos_history has shape (num_steps, B)
        return x_1, cos_history

    # FM loss with independent drop probabilities for action- and obs-context
    def loss(
        self,
        rng: jax.Array,
        context: jax.Array,                         # [B, context_dim]
        action: jax.Array,                          # [B, C, A]
        p_drop_act: float = 0.3, #0.2
        p_drop_obs: float = 0.3, #0.2
    ):
        assert action.dtype == jnp.float32
        B = context.shape[0]
        assert action.shape == (B, self.action_chunk_size, self.action_dim), action.shape

        noise_rng, time_rng, drop_rng_a, drop_rng_o = jax.random.split(rng, 4)
        time = jax.random.uniform(time_rng, (B,))
        noise = jax.random.normal(noise_rng, shape=action.shape)

        x_t = (1.0 - time[:, None, None]) * noise + time[:, None, None] * action
        u_t = action - noise

        use_null_act = jax.random.bernoulli(drop_rng_a, p=p_drop_act, shape=(B,))
        use_null_obs = jax.random.bernoulli(drop_rng_o, p=p_drop_obs, shape=(B,))

        pred = self(context, x_t, time, use_null_act=use_null_act, use_null_obs=use_null_obs)
        return jnp.mean(jnp.square(pred - u_t))

    
    def bid_action(
        self,
        rng: jax.Array,
        obs: jax.Array,
        num_steps: int,
        prev_action_chunk: jax.Array,  # [batch, horizon, action_dim]
        inference_delay: int,
        prefix_attention_horizon: int,
        n_samples: int,
        bid_weak_policy: Self | None = None,
        bid_k: int | None = None,
        mask_action: bool = False,
    ) -> jax.Array:
        obs = einops.repeat(obs, "b ... -> (n b) ...", n=n_samples)
        weights = get_prefix_weights(inference_delay, prefix_attention_horizon, self.action_chunk_size, "exp")

        def backward_loss(action_chunks: jax.Array):
            error = jnp.linalg.norm(action_chunks - prev_action_chunk, axis=-1)  # [n, b, h]
            return jnp.sum(error * weights[None, None, :], axis=-1)  # [n, b]

        strong_actions = einops.rearrange(
            self.action(rng, obs, num_steps, mask_action=mask_action), "(n b) h d -> n b h d", n=n_samples
        )
        loss = backward_loss(strong_actions)  # [n, b]

        if bid_weak_policy is not None or bid_k is not None:
            assert bid_weak_policy is not None and bid_k is not None, (bid_weak_policy, bid_k)
            weak_actions = einops.rearrange(
                bid_weak_policy.action(rng, obs, num_steps, mask_action=mask_action), "(n b) h d -> n b h d", n=n_samples
            )
            weak_loss = backward_loss(weak_actions)  # [n, b]
            weak_idxs = jax.lax.top_k(-weak_loss.T, bid_k)[1].T  # [k, b]
            strong_idxs = jax.lax.top_k(-loss.T, bid_k)[1].T     # [k, b]
            a_plus = jnp.take_along_axis(strong_actions, strong_idxs[:, :, None, None], axis=0)  # [k, b, h, d]
            a_minus = jnp.take_along_axis(weak_actions,   weak_idxs[:,   :, None, None], axis=0)  # [k, b, h, d]
            forward_loss = jnp.sum(
                jnp.linalg.norm(strong_actions[:, None] - a_plus[None, :], axis=-1), axis=(1, 3)
            ) - jnp.sum(
                jnp.linalg.norm(strong_actions[:, None] - a_minus[None, :], axis=-1), axis=(1, 3)
            )
            loss += forward_loss / n_samples

        best_idxs = jnp.argmin(loss, axis=0)  # [b]
        return jnp.take_along_axis(strong_actions, best_idxs[None, :, None, None], axis=0).squeeze(0)  # [b, h, d]

    def realtime_action(
        self,
        rng: jax.Array,
        obs: jax.Array,
        num_steps: int,
        prev_action_chunk: jax.Array,  # [batch, horizon, action_dim]
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule: PrefixAttentionSchedule,
        max_guidance_weight: float,
        mask_action: bool = False,
    ) -> jax.Array:
        dt = 1 / num_steps
        # B = obs.shape[0]
        # mask_act_B = jnp.ones((B,), dtype=bool) if mask_action else jnp.zeros((B,), dtype=bool)
        # mask_obs_B = jnp.zeros((B,), dtype=bool)

        def step(carry, _):
            x_t, time = carry

            @functools.partial(jax.vmap, in_axes=(0, 0, 0, None))  # over batch
            def pinv_corrected_velocity(o, x_t_i, y_i, t_i):
                def denoiser(x_t_single):
                    m_act_1 = jnp.ones((1,), dtype=bool) if mask_action else jnp.zeros((1,), dtype=bool)
                    m_obs_1 = jnp.zeros((1,), dtype=bool)
                    v_t = self(o[None], x_t_single[None], t_i, use_null_act=m_act_1, use_null_obs=m_obs_1)[0]
                    return x_t_single + v_t * (1 - t_i), v_t

                x_1, vjp_fun, v_t = jax.vjp(denoiser, x_t_i, has_aux=True)
                weights = get_prefix_weights(
                    inference_delay, prefix_attention_horizon, self.action_chunk_size, prefix_attention_schedule
                )
                error = (y_i - x_1) * weights[:, None]
                pinv_correction = vjp_fun(error)[0]
                inv_r2 = (t_i**2 + (1 - t_i) ** 2) / ((1 - t_i) ** 2)
                c = jnp.nan_to_num((1 - t_i) / t_i, posinf=max_guidance_weight)
                guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)
                return v_t + guidance_weight * pinv_correction

            v_t = pinv_corrected_velocity(obs, x_t, prev_action_chunk, time)
            return (x_t + dt * v_t, time + dt), None

        noise = jax.random.normal(rng, shape=(obs.shape[0], self.action_chunk_size, self.action_dim))
        (x_1, _), _ = jax.lax.scan(step, (noise, 0.0), length=num_steps)
        assert x_1.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), x_1.shape
        return x_1