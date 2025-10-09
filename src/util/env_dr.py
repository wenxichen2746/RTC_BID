
from dataclasses import replace
import kinetix.environment.wrappers as wrappers
from dataclasses import replace
import copy
import jax.numpy as jnp
import jax

class RandomizedResetWrapper(wrappers.UnderspecifiedEnvWrapper):
    """
    Randomizes the position of one or more polygons upon every reset.
    If polygon_index is a list, the polygons maintain their relative spacing,
    shifted together along the chosen axis.
    """
    def __init__(self, env, polygon_index=4, xy_min: float = 1.0, xy_max: float = 4.0, move_x_or_y='x'):
        super().__init__(env)
        # allow int or list
        if isinstance(polygon_index, int):
            self.polygon_indices = [polygon_index]
        else:
            self.polygon_indices = list(polygon_index)
        self.xy_min = xy_min
        self.xy_max = xy_max
        self.x_or_y = move_x_or_y

    def reset_to_level(self, rng, level, params):
        rng_key, random_pos_key = jax.random.split(rng)

        # --- extract original positions ---
        original_positions = level.polygon.position[self.polygon_indices, :]  # shape (N,2)
        ref_pos = original_positions[0]  # first object as reference
        rel_offsets = original_positions - ref_pos  # keep relative distances

        # --- sample new reference position ---
        if self.x_or_y == 'x':
            new_x = jax.random.uniform(random_pos_key, shape=(), minval=self.xy_min, maxval=self.xy_max)
            new_ref_pos = jnp.array([new_x, ref_pos[1]])
        elif self.x_or_y == 'y':
            new_y = jax.random.uniform(random_pos_key, shape=(), minval=self.xy_min, maxval=self.xy_max)
            new_ref_pos = jnp.array([ref_pos[0], new_y])
        else:
            raise ValueError(f"move_x_or_y must be 'x' or 'y', got {self.x_or_y}")

        # --- compute new positions for all selected polygons ---
        new_positions_block = new_ref_pos + rel_offsets  # shape (N,2)

        # --- update level positions ---
        new_positions = level.polygon.position
        for idx, pos in zip(self.polygon_indices, new_positions_block):
            new_positions = new_positions.at[idx].set(pos)

        polygon_with_new_pos = replace(level.polygon, position=new_positions)
        modified_level = replace(level, polygon=polygon_with_new_pos)
        return self._env.reset_to_level(rng_key, modified_level, params)

    def step_env(self, key, state, action, params):
        """Passes the step call to the wrapped environment."""
        return self._env.step_env(key, state, action, params)

    def action_space(self, params):
        """Passes the action_space call to the wrapped environment."""
        return self._env.action_space(params)




def DR_static_wrapper(env,level_path):
    print({level_path},"level_path")
    if 'hard_lunar_lander' in level_path:
        print(f'env: LL, randomizing target location')
        env = RandomizedResetWrapper(env, polygon_index=4)
    elif 'grasp' in level_path:
        print(f'env: grasp, randomizing target location')
        env = RandomizedResetWrapper(env, polygon_index=10,xy_min=1.8)
    elif 'reach_avoid' in level_path:
        print(f'env: reach&avoid, randomizing obstacles location')
        env = RandomizedResetWrapper(env, polygon_index=7,move_x_or_y='y')
    elif 'place_can' in level_path:
        print(f'env: place_can, randomizing obstacles&taget location')
        env = RandomizedResetWrapper(env, polygon_index=[9,10],xy_min=2.3,xy_max=3.3)
    elif 'toss_bin' in level_path:
        print(f'env: toss_bin, randomizing obstacles&taget location')
        env = RandomizedResetWrapper(env, polygon_index=[9,10,11],xy_min=1.5,xy_max=3.5)
    elif 'drone' in level_path:
        print(f'env: drone, randomizing target location')
        env = RandomizedResetWrapper(env, polygon_index=[4,7])
        env = RandomizedResetWrapper(env, polygon_index=[8])
    elif 'catapult' in level_path:
        print(f'env: catapult, randomizing target&supports location')
        env = RandomizedResetWrapper(env, polygon_index=[7,5,6],xy_min=2.5,xy_max=4.5)
    elif 'insert_key' in level_path:
        print(f'env: insert_key, randomizing target&supports location')
        env = RandomizedResetWrapper(env, polygon_index=[4,10,11],xy_min=1,xy_max=4)
    elif 'whip' in level_path:
        print(f'env: whip, randomizing targetlocation')
        env = RandomizedResetWrapper(env, polygon_index=10,xy_min=1,xy_max=4)
    else:
        print(f'no env DR implemented')
        # raise NotImplementedError("*** Level not recognized DR not implemented **")
    return env




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
