import numpy as np

GRID_SIZE = 10
SIGMA = 1.5

ACTIONS = {
    'up':    (0, -1),
    'down':  (0,  1),
    'left':  (-1, 0),
    'right': (1,  0),
    'stay':  (0,  0),
}
ACTION_LIST = list(ACTIONS.keys())


def clip_to_grid(pos):
    """Clamps x and y to [0, GRID_SIZE-1] so agents stay on the grid."""
    x, y = pos
    x = int(np.clip(x, 0, GRID_SIZE - 1))
    y = int(np.clip(y, 0, GRID_SIZE - 1))
    return (x, y)


class Agent:
    """One agent on the grid. Holds its true position and handles movement."""

    def __init__(self, start_pos):
        self.pos = clip_to_grid(start_pos)

    def move(self, action):
        """Applies the action delta and clamps to grid bounds."""
        dx, dy = ACTIONS[action]
        new_x = self.pos[0] + dx
        new_y = self.pos[1] + dy
        self.pos = clip_to_grid((new_x, new_y))

    def random_move(self, rng):
        """Picks a random action uniformly, applies it, and returns the action name."""
        action = rng.choice(ACTION_LIST)
        self.move(action)
        return action


def observe(true_pos, rng, sigma=SIGMA):
    """
    Returns a noisy observation of true_pos.
    Adds independent Gaussian noise to each coordinate, rounds to integer,
    and clamps to the grid.
    """
    x, y = true_pos
    noise_x = rng.normal(0, sigma)
    noise_y = rng.normal(0, sigma)
    obs_x = round(x + noise_x)
    obs_y = round(y + noise_y)
    return clip_to_grid((obs_x, obs_y))


class Environment:
    """Manages the full simulation: both agents, their goals, timesteps, and observations."""

    def __init__(self, seed=None, mode="random"):
        """
        seed: fixed seed for reproducibility; None gives a fresh random run each time.
        mode: "random" picks 4 distinct random cells for starts/goals;
              "classic" puts A at (0,0)->goal(9,9) and B at (9,9)->goal(0,0).
        """
        self.rng = np.random.default_rng(seed)

        if mode == "classic":
            start_a, start_b = (0, 0), (9, 9)
            goal_a,  goal_b  = (9, 9), (0, 0)
        else:
            all_cells = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
            chosen = [all_cells[i] for i in self.rng.choice(len(all_cells), size=4, replace=False)]
            start_a, start_b, goal_a, goal_b = chosen

        self.agent_a = Agent(start_pos=start_a)
        self.agent_b = Agent(start_pos=start_b)
        self.goal_a = goal_a
        self.goal_b = goal_b

        self.timestep = 0

    def get_observation(self, observer, target):
        """
        Returns a noisy observation of target's position as seen by observer.
        observer is unused for now but kept for future distance-based noise models.
        """
        return observe(target.pos, self.rng)

    def step(self):
        """
        Advances by one timestep: both agents move randomly, then each observes the other.
        Returns a dict with positions, observations, and actions for this step.
        """
        action_a = self.agent_a.random_move(self.rng)
        action_b = self.agent_b.random_move(self.rng)

        obs_a = self.get_observation(self.agent_a, self.agent_b)  # A's noisy view of B
        obs_b = self.get_observation(self.agent_b, self.agent_a)  # B's noisy view of A

        self.timestep += 1

        return {
            'timestep': self.timestep,
            'pos_a':    self.agent_a.pos,
            'pos_b':    self.agent_b.pos,
            'obs_a':    obs_a,
            'obs_b':    obs_b,
            'action_a': action_a,
            'action_b': action_b,
        }


if __name__ == '__main__':
    print("=== Environment Test ===\n")

    env = Environment(seed=None)  # random every run

    print(f"Agent A starts at: {env.agent_a.pos}  |  Goal: {env.goal_a}")
    print(f"Agent B starts at: {env.agent_b.pos}  |  Goal: {env.goal_b}")
    print()

    for _ in range(5):
        result = env.step()
        print(f"Timestep {result['timestep']}")
        print(f"  A: {result['pos_a']}  action={result['action_a']}")
        print(f"  B: {result['pos_b']}  action={result['action_b']}")
        print(f"  A observes B at: {result['obs_a']}  (true: {result['pos_b']})")
        print(f"  B observes A at: {result['obs_b']}  (true: {result['pos_a']})")
        print()
