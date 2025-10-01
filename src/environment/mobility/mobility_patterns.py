import numpy as np

def _apply_constraints(new_positions, walkers_positions, limits, obstacles, radius):
    """Shared post-processing: snap, clip, avoid obstacles & collisions."""
    x_min, x_max, y_min, y_max = limits

    # Snap to 0.1 grid
    new_positions = np.round(new_positions, 1)

    # Clip within limits
    new_positions[:, 0] = np.clip(new_positions[:, 0], x_min, x_max)
    new_positions[:, 1] = np.clip(new_positions[:, 1], y_min, y_max)

    # Obstacle avoidance
    if obstacles is not None and len(obstacles) > 0:
        diff = new_positions[:, None, :] - obstacles[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        invalid = (dists < radius).any(axis=1)
        new_positions[invalid] = walkers_positions[invalid]

    # Collision avoidance
    _, idx = np.unique(new_positions, axis=0, return_index=True)
    unique_mask = np.zeros(len(walkers_positions), dtype=bool)
    unique_mask[idx] = True
    new_positions[~unique_mask] = walkers_positions[~unique_mask]

    return new_positions


def random_walk(walkers_positions, rng, possible_actions):
    actions = possible_actions[rng.integers(len(possible_actions), size=len(walkers_positions))]
    return walkers_positions + actions


def brownian(walkers_positions, rng, step_size):
    steps = rng.normal(0, step_size, size=(len(walkers_positions), 2))
    return walkers_positions + steps


def levy_flight(walkers_positions, rng, step_size):
    step_lengths = rng.pareto(a=1.5, size=len(walkers_positions)) * step_size
    angles = rng.uniform(0, 2*np.pi, size=len(walkers_positions))
    steps = np.column_stack([step_lengths * np.cos(angles),
                             step_lengths * np.sin(angles)])
    return walkers_positions + steps


def grid_mobility(walkers_positions, mobility_model = 0,
                  possible_actions=np.array([[0, 0], [0, 0.1], [0, -0.1], [0.1, 0], [-0.1, 0], [-0.1, -0.1], [0.1, 0.1]]),
                  limits=np.array([100, 200, 0, 100]),
                  random_numpy_generator=None,
                  obstacles=None,
                  radius=0.1,
                  step_size=0.1):

    rng = random_numpy_generator if random_numpy_generator else np.random.default_rng()

    if mobility_model == 0:
        new_positions = random_walk(walkers_positions, rng, possible_actions)
    elif mobility_model == 1:
        new_positions = brownian(walkers_positions, rng, step_size)
    elif mobility_model == 2:
        new_positions = levy_flight(walkers_positions, rng, step_size)
    else:
        raise ValueError(f"Unknown mobility model: {mobility_model}")

    return _apply_constraints(new_positions, walkers_positions, limits, obstacles, radius)
