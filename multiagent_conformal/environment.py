import numpy as np

# -------------------------------------------------------
# GRID CONSTANTS
# -------------------------------------------------------
# The grid is 10x10. All positions are (x, y) integers
# where x and y both range from 0 to GRID_SIZE-1.
GRID_SIZE = 10

# Noise level: how much uncertainty is in each observation.
# sigma=1.5 means the noise has a standard deviation of 1.5
# grid cells. So observations are usually within ~1-2 cells
# of the true position, but occasionally further.
SIGMA = 1.5

# -------------------------------------------------------
# ACTION DEFINITIONS
# -------------------------------------------------------
# Each action is a (dx, dy) vector added to current position.
# We define 5 actions: move in 4 directions or stay put.
ACTIONS = {
    'up':    (0, -1),
    'down':  (0,  1),
    'left':  (-1, 0),
    'right': (1,  0),
    'stay':  (0,  0),
}
ACTION_LIST = list(ACTIONS.keys())  # ordered list for indexing


# -------------------------------------------------------
# CLIP FUNCTION
# -------------------------------------------------------
def clip_to_grid(pos):
    """
    Takes a position (x, y) and clips both coordinates
    to stay inside the grid [0, GRID_SIZE-1].

    Why: agents shouldn't walk off the edge of the grid.
    np.clip(value, min, max) constrains a value to [min, max].
    """
    x, y = pos
    x = int(np.clip(x, 0, GRID_SIZE - 1))
    y = int(np.clip(y, 0, GRID_SIZE - 1))
    return (x, y)


# -------------------------------------------------------
# AGENT CLASS
# -------------------------------------------------------
class Agent:
    """
    Represents one agent on the grid.
    Stores its true position and handles movement.
    """

    def __init__(self, start_pos):
        """
        start_pos: (x, y) tuple -- where the agent starts.
        We store position as a tuple of two ints.
        """
        self.pos = clip_to_grid(start_pos)

    def move(self, action):
        """
        Moves the agent one step in the given action direction.

        action: a string key from ACTIONS dict, e.g. 'up'

        We look up the (dx, dy) vector, add it to current
        position, then clip to keep inside the grid.
        """
        dx, dy = ACTIONS[action]
        new_x = self.pos[0] + dx
        new_y = self.pos[1] + dy
        self.pos = clip_to_grid((new_x, new_y))

    def random_move(self, rng):
        """
        Picks a random action and moves.

        rng: a numpy random number generator (passed in so
        we can control randomness with a seed for reproducibility).

        rng.choice(ACTION_LIST) picks one of the 5 actions
        uniformly at random.
        """
        action = rng.choice(ACTION_LIST)
        self.move(action)
        return action


# -------------------------------------------------------
# OBSERVATION FUNCTION
# -------------------------------------------------------
def observe(true_pos, rng, sigma=SIGMA):
    """
    Simulates a noisy observation of true_pos.

    true_pos: (x, y) -- the actual position of the agent
              being observed.
    rng:      numpy random generator for reproducibility.
    sigma:    standard deviation of the Gaussian noise.

    How it works:
    - We draw two independent noise values from N(0, sigma^2),
      one for x and one for y.
    - rng.normal(mean, std) draws from a Gaussian distribution.
      mean=0 means noise is unbiased (doesn't systematically
      shift in any direction). std=sigma controls spread.
    - We add the noise to the true position coordinates.
    - We round to the nearest integer (grid cells are discrete).
    - We clip to keep the observation inside the grid.

    Returns a (x, y) tuple -- the noisy observed position.
    """
    x, y = true_pos
    noise_x = rng.normal(0, sigma)   # draw x noise from N(0, sigma^2)
    noise_y = rng.normal(0, sigma)   # draw y noise from N(0, sigma^2)
    obs_x = round(x + noise_x)       # add noise, round to integer
    obs_y = round(y + noise_y)
    return clip_to_grid((obs_x, obs_y))  # clip to grid boundary


# -------------------------------------------------------
# ENVIRONMENT CLASS
# -------------------------------------------------------
class Environment:
    """
    Manages the full simulation: both agents, timesteps,
    and producing observations.
    """

    def __init__(self, seed=42):
        """
        seed: random seed for reproducibility. Using the same
        seed always produces the same sequence of random numbers,
        so your simulation is repeatable.

        np.random.default_rng(seed) creates a modern numpy
        random generator -- preferred over the older np.random
        functions because it's faster and more statistically sound.
        """
        self.rng = np.random.default_rng(seed)

        # Place agents at opposite corners of the grid
        # so they start far apart and have to navigate toward
        # their goals while being uncertain about each other.
        self.agent_a = Agent(start_pos=(0, 0))
        self.agent_b = Agent(start_pos=(9, 9))

        # Goals: each agent is trying to reach the opposite corner.
        # This creates natural paths that will cross in the middle,
        # making the multi-agent uncertainty interesting.
        self.goal_a = (9, 9)
        self.goal_b = (0, 0)

        # Track how many timesteps have elapsed.
        self.timestep = 0

    def get_observation(self, observer, target):
        """
        Returns a noisy observation of target's true position,
        as seen by observer.

        observer: the Agent doing the observing (not used for
                  position yet, but useful later if you want
                  observation quality to depend on distance).
        target:   the Agent being observed.

        Calls our observe() function with the target's true
        position and the environment's random generator.
        """
        return observe(target.pos, self.rng)

    def step(self):
        """
        Advances the simulation by one timestep.

        Both agents move randomly for now -- we'll replace
        this with the policy-driven movement in later layers.

        Returns a dict with everything that happened this step:
        - true positions of both agents
        - the noisy observation each agent got of the other
        - the current timestep number
        """
        # Both agents move randomly this timestep
        action_a = self.agent_a.random_move(self.rng)
        action_b = self.agent_b.random_move(self.rng)

        # After moving, each agent observes the other's new position
        # Agent A observes B, Agent B observes A
        obs_a = self.get_observation(self.agent_a, self.agent_b)
        obs_b = self.get_observation(self.agent_b, self.agent_a)

        self.timestep += 1

        return {
            'timestep':   self.timestep,
            'pos_a':      self.agent_a.pos,
            'pos_b':      self.agent_b.pos,
            'obs_a':      obs_a,   # what A observed about B
            'obs_b':      obs_b,   # what B observed about A
            'action_a':   action_a,
            'action_b':   action_b,
        }


# -------------------------------------------------------
# QUICK TEST
# -------------------------------------------------------
if __name__ == '__main__':
    print("=== Layer 1: Environment Test ===\n")

    env = Environment(seed=42)

    print(f"Agent A starts at: {env.agent_a.pos}  |  Goal: {env.goal_a}")
    print(f"Agent B starts at: {env.agent_b.pos}  |  Goal: {env.goal_b}")
    print()

    # Run 5 timesteps and print what happens
    for _ in range(5):
        result = env.step()
        print(f"Timestep {result['timestep']}")
        print(f"  A true pos: {result['pos_a']}  |  A's action: {result['action_a']}")
        print(f"  B true pos: {result['pos_b']}  |  B's action: {result['action_b']}")
        print(f"  A observes B at: {result['obs_a']}  (true: {result['pos_b']})")
        print(f"  B observes A at: {result['obs_b']}  (true: {result['pos_a']})")
        print()
