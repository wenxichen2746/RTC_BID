from dataclasses import dataclass, replace
from typing import Sequence
import copy
import jax
import jax.numpy as jnp
import numpy as np
import kinetix.environment.wrappers as wrappers


@dataclass(frozen=True)
class DRRule:
    polygon_indices: tuple[int, ...]
    xy_min: float = 1.0
    xy_max: float = 4.0
    axis: str = "x"

    def __post_init__(self):
        if self.axis not in ("x", "y"):
            raise ValueError(f"axis must be 'x' or 'y', got {self.axis}")
        object.__setattr__(self, "polygon_indices", tuple(int(i) for i in self.polygon_indices))


class RandomizedResetWrapper(wrappers.UnderspecifiedEnvWrapper):
    """
    Randomizes the position of one or more polygons upon every reset.
    If polygon_index is a list, the polygons maintain their relative spacing,
    shifted together along the chosen axis.
    """

    def __init__(self, env, polygon_index=4, xy_min: float = 1.0, xy_max: float = 4.0, move_x_or_y="x"):
        super().__init__(env)
        if isinstance(polygon_index, int):
            self.polygon_indices = [polygon_index]
        else:
            self.polygon_indices = list(polygon_index)
        self.xy_min = xy_min
        self.xy_max = xy_max
        self.x_or_y = move_x_or_y

    def reset_to_level(self, rng, level, params):
        rng_key, random_pos_key = jax.random.split(rng)

        original_positions = level.polygon.position[self.polygon_indices, :]
        ref_pos = original_positions[0]
        rel_offsets = original_positions - ref_pos

        if self.x_or_y == "x":
            new_val = jax.random.uniform(random_pos_key, shape=(), minval=self.xy_min, maxval=self.xy_max)
            new_ref_pos = jnp.array([new_val, ref_pos[1]])
        elif self.x_or_y == "y":
            new_val = jax.random.uniform(random_pos_key, shape=(), minval=self.xy_min, maxval=self.xy_max)
            new_ref_pos = jnp.array([ref_pos[0], new_val])
        else:
            raise ValueError(f"move_x_or_y must be 'x' or 'y', got {self.x_or_y}")

        new_positions_block = new_ref_pos + rel_offsets
        new_positions = level.polygon.position
        for idx, pos in zip(self.polygon_indices, new_positions_block):
            new_positions = new_positions.at[idx].set(pos)

        polygon_with_new_pos = replace(level.polygon, position=new_positions)
        modified_level = replace(level, polygon=polygon_with_new_pos)
        return self._env.reset_to_level(rng_key, modified_level, params)

    def step_env(self, key, state, action, params):
        return self._env.step_env(key, state, action, params)

    def action_space(self, params):
        return self._env.action_space(params)


class MultiLevelRandomizedResetWrapper(wrappers.UnderspecifiedEnvWrapper):
    """Apply deterministic DR rules per level signature, with optional fallback."""

    def __init__(
        self,
        env,
        signature_positions=None,
        signature_velocities=None,
        rules_per_signature: Sequence[Sequence[DRRule]] | None = None,
        default_rules=None,
        match_atol: float = 1e-6,
    ):
        super().__init__(env)
        self._match_atol = match_atol
        self._signature_positions = None
        self._signature_velocities = None
        self._rules_per_signature: list[list[DRRule]] = []
        self._default_rules: list[DRRule] = []
        self._branches: tuple = tuple()
        self._fallback_index: int = 0
        self.update_rules(
            signature_positions,
            signature_velocities,
            rules_per_signature or [],
            default_rules,
        )

    def update_rules(
        self,
        signature_positions,
        signature_velocities,
        rules_per_signature: Sequence[Sequence[DRRule]],
        default_rules=None,
    ):
        self._default_rules = list(default_rules or [])
        self._rules_per_signature = [list(rules) for rules in (rules_per_signature or [])]

        if self._rules_per_signature:
            if signature_positions is None or signature_velocities is None:
                raise ValueError("signature position/velocity data required when providing per-level DR rules.")
            self._signature_positions = jnp.asarray(signature_positions)
            self._signature_velocities = jnp.asarray(signature_velocities)
        else:
            self._signature_positions = None
            self._signature_velocities = None

        self._rebuild_branches()
        return self

    def _rebuild_branches(self):
        self._num_signatures = len(self._rules_per_signature)
        self._fallback_index = self._num_signatures

        branches = [self._make_branch(rules) for rules in self._rules_per_signature]
        branches.append(self._make_branch(self._default_rules))
        self._branches = tuple(branches)

    @staticmethod
    def _make_branch(rules: Sequence[DRRule]):
        def branch(payload):
            level, rng = payload
            modified_level, env_rng = _apply_rules(level, rules, rng)
            return modified_level, env_rng

        return branch

    def _match_signature_index(self, level):
        if self._num_signatures == 0 or self._signature_positions is None:
            return jnp.array(self._fallback_index, dtype=jnp.int32)

        level_pos = level.polygon.position[None, ...]
        level_vel = level.polygon.velocity[None, ...]
        pos_equal = jnp.all(
            jnp.isclose(self._signature_positions, level_pos, atol=self._match_atol),
            axis=tuple(range(1, self._signature_positions.ndim)),
        )
        vel_equal = jnp.all(
            jnp.isclose(self._signature_velocities, level_vel, atol=self._match_atol),
            axis=tuple(range(1, self._signature_velocities.ndim)),
        )
        matches = jnp.logical_and(pos_equal, vel_equal)
        match_indices = jnp.where(
            matches,
            jnp.arange(self._num_signatures, dtype=jnp.int32),
            jnp.array(self._fallback_index, dtype=jnp.int32),
        )
        return jnp.min(match_indices)

    def reset_to_level(self, rng, level, params):
        idx = self._match_signature_index(level)
        modified_level, env_rng = jax.lax.switch(idx, self._branches, (level, rng))
        return self._env.reset_to_level(env_rng, modified_level, params)

    def step_env(self, key, state, action, params):
        return self._env.step_env(key, state, action, params)

    def action_space(self, params):
        return self._env.action_space(params)


def _level_signature(level) -> tuple[bytes, bytes]:
    pos = np.asarray(jax.device_get(level.polygon.position))
    vel = np.asarray(jax.device_get(level.polygon.velocity))
    return (pos.tobytes(), vel.tobytes())


def _apply_rules(level, rules: list[DRRule], rng):
    if not rules:
        return level, rng
    env_rng, random_rng = jax.random.split(rng)
    modified_level = level
    key = random_rng
    for rule in rules:
        key, rule_key = jax.random.split(key)
        modified_level = _apply_rule(modified_level, rule, rule_key)
    return modified_level, env_rng


def _apply_rule(level, rule: DRRule, rng):
    polygon_indices = rule.polygon_indices
    level_mod = copy.deepcopy(level)

    original_positions = level_mod.polygon.position[polygon_indices, :]
    ref_pos = original_positions[0]
    rel_offsets = original_positions - ref_pos

    new_coord = jax.random.uniform(rng, shape=(), minval=rule.xy_min, maxval=rule.xy_max)
    if rule.axis == "x":
        new_ref = jnp.array([new_coord, ref_pos[1]])
    else:
        new_ref = jnp.array([ref_pos[0], new_coord])

    new_positions_block = new_ref + rel_offsets
    new_positions = level_mod.polygon.position
    for idx, pos in zip(polygon_indices, new_positions_block):
        new_positions = new_positions.at[idx].set(pos)

    polygon_with_new_pos = replace(level_mod.polygon, position=new_positions)
    return replace(level_mod, polygon=polygon_with_new_pos)


def _rules_hard_lunar_lander():
    return "env: LL, randomizing target location", [DRRule((4,))]


def _rules_grasp():
    return "env: grasp, randomizing target location", [DRRule((10,), xy_min=1.8)]


def _rules_place_can():
    return "env: place_can, randomizing obstacles&target location", [DRRule((9, 10), xy_min=2.3, xy_max=3.3)]


def _rules_toss_bin():
    return "env: toss_bin, randomizing obstacles&target location", [DRRule((9, 10, 11), xy_min=1.5, xy_max=3.5)]


def _rules_catapult():
    return "env: catapult, randomizing target&supports location", [DRRule((7, 5, 6), xy_min=2.5, xy_max=4.5)]


def _rules_insert_key():
    return "env: insert_key, randomizing target&supports location", [DRRule((4, 10, 11), xy_min=1.0, xy_max=4.0)]


def _rules_drone():
    return "env: drone, randomizing target location", [DRRule((4, 7)), DRRule((8,))]


def _rules_reach_avoid():
    return "env: reach&avoid, randomizing obstacles location", [DRRule((7,), axis="y")]


def _rules_whip():
    return "env: whip, randomizing targetlocation", [DRRule((10,))]


_DR_RULE_BUILDERS = [
    ("place_can_easy", _rules_place_can),
    ("hard_lunar_lander", _rules_hard_lunar_lander),
    ("grasp", _rules_grasp),
    ("place_can", _rules_place_can),
    ("toss_bin", _rules_toss_bin),
    ("catapult", _rules_catapult),
    ("insert_key", _rules_insert_key),
    ("drone", _rules_drone),
    ("reach_avoid", _rules_reach_avoid),
    ("whip", _rules_whip),
]


def _build_rules_for_path(level_path: str):
    for token, builder in _DR_RULE_BUILDERS:
        if token in level_path:
            message, rules = builder()
            return message, list(rules)
    return None, []


def _ensure_sequence(level_paths) -> list[str]:
    if isinstance(level_paths, str):
        return [level_paths]
    if isinstance(level_paths, Sequence):
        return list(level_paths)
    raise TypeError("level_paths must be a string or sequence of strings")


def DR_static_wrapper(env, level_paths, levels=None):
    paths = _ensure_sequence(level_paths)
    if not paths:
        print("no level paths provided for DR_static_wrapper")
        return env

    signature_to_index: dict[tuple[bytes, bytes], int] = {}
    signature_positions: list[np.ndarray] = []
    signature_velocities: list[np.ndarray] = []
    signature_rules: list[list[DRRule]] = []
    default_rules: list[DRRule] = []
    unmatched_paths: list[str] = []

    def configure_for_path(path: str):
        print({path}, "level_path")
        message, rules = _build_rules_for_path(path)
        if message:
            print(message)
        else:
            unmatched_paths.append(path)
        return rules

    if levels is not None:
        batch_size = levels.polygon.position.shape[0]
        if batch_size != len(paths):
            raise ValueError(
                f"Expected levels to have length {len(paths)}, got {batch_size}."
            )
        for idx, path in enumerate(paths):
            rules = configure_for_path(path)
            if not rules:
                continue
            level_slice = jax.tree.map(lambda x: x[idx], levels)
            signature = _level_signature(level_slice)
            if signature in signature_to_index:
                signature_rules[signature_to_index[signature]] = list(rules)
                continue
            signature_to_index[signature] = len(signature_rules)
            signature_rules.append(list(rules))
            signature_positions.append(np.asarray(jax.device_get(level_slice.polygon.position)))
            signature_velocities.append(np.asarray(jax.device_get(level_slice.polygon.velocity)))
    else:
        rules = configure_for_path(paths[0])
        default_rules = list(rules)
        if len(paths) > 1:
            print("DR_static_wrapper received multiple level_paths but no levels; defaulting to first entry.")

    if unmatched_paths:
        for path in unmatched_paths:
            print(f'no env DR implemented for {path}')

    if signature_rules:
        signature_positions_arr = jnp.stack([jnp.asarray(pos) for pos in signature_positions])
        signature_velocities_arr = jnp.stack([jnp.asarray(vel) for vel in signature_velocities])
    else:
        signature_positions_arr = None
        signature_velocities_arr = None

    if not signature_rules and not default_rules:
        return env

    if isinstance(env, MultiLevelRandomizedResetWrapper):
        return env.update_rules(
            signature_positions_arr,
            signature_velocities_arr,
            signature_rules,
            default_rules,
        )

    if not signature_rules and len(default_rules) == 1:
        rule = default_rules[0]
        return RandomizedResetWrapper(
            env,
            polygon_index=rule.polygon_indices,
            xy_min=rule.xy_min,
            xy_max=rule.xy_max,
            move_x_or_y=rule.axis,
        )

    return MultiLevelRandomizedResetWrapper(
        env,
        signature_positions_arr,
        signature_velocities_arr,
        signature_rules,
        default_rules,
    )


def change_polygon_position_and_velocity(
    levels,
    pos_x=None,
    pos_y=None,
    vel_x=None,
    vel_y=None,
    index=4,
    level_paths=None,
    level_match=None,
):
    # levels: pytree of stacked levels (batched)
    batch_size = levels.polygon.position.shape[0]
    if level_paths is not None and len(level_paths) != batch_size:
        raise ValueError(
            f"Expected level_paths to have length {batch_size}, got {len(level_paths)}."
        )

    # Build a mask of which batch items should be updated.
    if level_paths is None or level_match is None:
        update_mask = [True] * batch_size
    else:
        match_tokens = (
            [level_match]
            if isinstance(level_match, str)
            else list(level_match)
        )
        if not match_tokens:
            update_mask = [False] * batch_size
        else:
            update_mask = [
                any(token in level_path for token in match_tokens)
                for level_path in level_paths
            ]

    if not any(update_mask):
        return levels

    new_levels = []

    for batch_idx in range(batch_size):
        level_slice = jax.tree.map(lambda x: x[batch_idx], levels)
        if not update_mask[batch_idx]:
            new_levels.append(level_slice)
            continue

        level_mod = copy.deepcopy(level_slice)

        current_pos = level_mod.polygon.position[index]
        current_vel = level_mod.polygon.velocity[index]

        new_pos = jnp.array([
            pos_x if pos_x is not None else current_pos[0],
            pos_y if pos_y is not None else current_pos[1],
        ])
        new_vel = jnp.array([
            vel_x if vel_x is not None else current_vel[0],
            vel_y if vel_y is not None else current_vel[1],
        ])

        new_positions = level_mod.polygon.position.at[index].set(new_pos)
        new_velocities = level_mod.polygon.velocity.at[index].set(new_vel)
        new_polygon = replace(level_mod.polygon, position=new_positions, velocity=new_velocities)
        level_mod = replace(level_mod, polygon=new_polygon)
        new_levels.append(level_mod)

    return jax.tree.map(lambda *x: jnp.stack(x), *new_levels)
