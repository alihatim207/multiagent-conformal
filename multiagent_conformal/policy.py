import numpy as np
import pickle
from environment import Environment, GRID_SIZE, ACTIONS, ACTION_LIST, clip_to_grid
from classifier import idx_to_pos, pos_to_idx
from conformal import ConformalPredictor

# -------------------------------------------------------
# COST FUNCTION
# -------------------------------------------------------
def cost(my_pos, my_next_pos, their_pos, goal):
    """
    Computes the cost of taking an action that moves me
    from my_pos to my_next_pos, given that the other agent
    might be at their_pos.

    The cost has two components:

    1. COLLISION COST: if my next position overlaps with
       the other agent's position, that's very bad.
       We assign a huge penalty to discourage this.

    2. GOAL COST: how far am I from my goal after moving?
       We use Manhattan distance -- sum of absolute differences
       in x and y coordinates. This is the natural distance
       metric on a grid where you can only move in 4 directions.
       Manhattan distance from (x1,y1) to (x2,y2):
           |x1 - x2| + |y1 - y2|

    By combining these two, the agent tries to move toward
    its goal while avoiding positions where the other agent
    might be.

    Args:
        my_pos:      (x, y) -- my current position
        my_next_pos: (x, y) -- my position after taking the action
        their_pos:   (x, y) -- one possible position of other agent
        goal:        (x, y) -- my target destination

    Returns:
        total cost as a float
    """
    # Collision penalty: 1000 if we'd occupy the same cell
    # This is much larger than any possible goal cost
    # (max Manhattan distance on 10x10 grid is 18),
    # so the agent will always avoid collision first.
    COLLISION_PENALTY = 1000
    collision = COLLISION_PENALTY if my_next_pos == their_pos else 0

    # Goal cost: Manhattan distance from next position to goal
    # abs() gives absolute value, we sum over x and y dimensions
    goal_dist = abs(my_next_pos[0] - goal[0]) + abs(my_next_pos[1] - goal[1])

    return collision + goal_dist


# -------------------------------------------------------
# MINIMAX POLICY
# -------------------------------------------------------
def minimax_policy(my_pos, prediction_set, goal):
    """
    Picks the action that minimizes the worst-case cost
    across all positions in the prediction set.

    This is the core decision-making logic. It implements:
        a* = argmin_a [ max_{p in C} cost(a, p) ]

    For each possible action:
        - Compute where I'd end up (my_next_pos)
        - For each possible position p of the other agent
          in the prediction set C:
            - Compute cost(my_pos, my_next_pos, p, goal)
        - Take the MAXIMUM cost across all p
          (worst case: assume the other agent is wherever
           hurts me most)
    Then pick the action with the MINIMUM worst-case cost.

    Why worst case? Because we have a guarantee that the
    true position is in the prediction set. We don't know
    exactly where in the set, so we plan for the worst
    possible location within it.

    Args:
        my_pos:         (x, y) -- my current position
        prediction_set: list of (x, y) -- uncertainty set
                        about other agent's position
        goal:           (x, y) -- my target

    Returns:
        best_action: string -- the chosen action ('up', 'down', etc.)
        action_costs: dict -- worst-case cost for each action (for debugging)
    """
    best_action = None
    best_worst_case_cost = float('inf')  # start with infinity, minimize down
    action_costs = {}

    for action in ACTION_LIST:
        # Compute where I'd end up after taking this action.
        # clip_to_grid ensures I don't walk off the edge.
        dx, dy = ACTIONS[action]
        my_next_pos = clip_to_grid((my_pos[0] + dx, my_pos[1] + dy))

        # If the prediction set is empty (shouldn't happen but just in case),
        # fall back to just optimizing for goal distance.
        if len(prediction_set) == 0:
            worst_case = abs(my_next_pos[0] - goal[0]) + abs(my_next_pos[1] - goal[1])
        else:
            # Compute cost for every possible position of the other agent.
            # max() finds the worst (highest) cost across all positions.
            # This is the "max" part of minimax.
            costs_for_this_action = [
                cost(my_pos, my_next_pos, their_pos, goal)
                for their_pos in prediction_set
            ]
            worst_case = max(costs_for_this_action)

        action_costs[action] = worst_case

        # Update best action if this action has lower worst-case cost.
        # This is the "min" part of minimax.
        if worst_case < best_worst_case_cost:
            best_worst_case_cost = worst_case
            best_action = action

    return best_action, action_costs


# -------------------------------------------------------
# POLICY-DRIVEN AGENT
# -------------------------------------------------------
class PolicyAgent:
    """
    An agent that uses conformal prediction + minimax policy
    to make decisions.

    This replaces the random_move() behavior from Layer 1
    with intelligent, uncertainty-aware movement.

    Each timestep:
    1. Receive a noisy observation of the other agent
    2. Run it through the conformal predictor to get a prediction set
    3. Run minimax policy to pick the best action given that set
    4. Move
    """

    def __init__(self, start_pos, goal, predictor):
        """
        start_pos:  (x, y) -- starting position on grid
        goal:       (x, y) -- target destination
        predictor:  a calibrated ConformalPredictor from Layer 3
        """
        self.pos = clip_to_grid(start_pos)
        self.goal = goal
        self.predictor = predictor

        # Track history for visualization in Layer 5
        self.pos_history = [self.pos]
        self.pred_set_history = []     # prediction sets over time
        self.set_size_history = []     # sizes of prediction sets
        self.action_history = []       # actions taken
        self.coverage_history = []     # whether true pos was in set

    def act(self, observation, true_other_pos=None):
        """
        Takes one step: observe, predict, decide, move.

        Args:
            observation:     (obs_x, obs_y) -- noisy observation of other agent
            true_other_pos:  (x, y) -- the other agent's TRUE position.
                             Only used for tracking coverage (evaluation).
                             The agent does NOT use this for decision making --
                             it only sees the noisy observation.

        Returns:
            action:         the action taken
            prediction_set: the uncertainty set used for decision making
        """
        # Step 1: Build prediction set from noisy observation.
        # This calls the conformal predictor from Layer 3.
        # It returns the set of all positions with probability >= q.
        prediction_set, probs = self.predictor.predict_set(observation)

        # Step 2: Run minimax policy to pick best action.
        action, action_costs = minimax_policy(self.pos, prediction_set, self.goal)

        # Step 3: Move.
        dx, dy = ACTIONS[action]
        self.pos = clip_to_grid((self.pos[0] + dx, self.pos[1] + dy))

        # Step 4: Track everything for visualization and evaluation.
        self.pos_history.append(self.pos)
        self.pred_set_history.append(prediction_set)
        self.set_size_history.append(len(prediction_set))
        self.action_history.append(action)

        # Track coverage: was the true position in the prediction set?
        # This is purely for evaluation -- agent doesn't use true_other_pos
        # in any decision making above.
        if true_other_pos is not None:
            in_set = true_other_pos in prediction_set
            self.coverage_history.append(in_set)

        return action, prediction_set


# -------------------------------------------------------
# FULL SIMULATION WITH POLICY
# -------------------------------------------------------
class PolicyEnvironment:
    """
    Full simulation environment where both agents use
    the conformal prediction + minimax policy.

    Replaces the random movement from Layer 1's Environment class.
    """

    def __init__(self, predictor, seed=42):
        """
        predictor: calibrated ConformalPredictor (shared by both agents)
        seed:      random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)

        # Agent A starts top-left, wants to reach bottom-right
        # Agent B starts bottom-right, wants to reach top-left
        # Their paths cross in the middle -- that's where uncertainty matters most
        self.agent_a = PolicyAgent(
            start_pos=(0, 0),
            goal=(9, 9),
            predictor=predictor
        )
        self.agent_b = PolicyAgent(
            start_pos=(9, 9),
            goal=(0, 0),
            predictor=predictor
        )

        self.timestep = 0

        # Track collisions (both agents at same cell)
        self.collisions = 0

    def get_noisy_observation(self, target_pos):
        """
        Generates a noisy observation of target_pos.
        Uses the same Gaussian noise model as Layer 1.

        sigma=1.5 means noise std dev of 1.5 grid cells.
        """
        from environment import observe
        return observe(target_pos, self.rng)

    def step(self):
        """
        Advances simulation by one timestep.

        Both agents:
        1. Get a noisy observation of the other
        2. Build a prediction set via conformal prediction
        3. Choose action via minimax policy
        4. Move

        Returns a dict with everything that happened.
        """
        # Each agent observes the OTHER agent's position (with noise)
        obs_a = self.get_noisy_observation(self.agent_b.pos)  # A observes B
        obs_b = self.get_noisy_observation(self.agent_a.pos)  # B observes A

        # Each agent acts based on its observation.
        # We pass the TRUE position of the other agent ONLY for coverage tracking.
        # The agent itself only uses the noisy observation for decisions.
        action_a, pred_set_a = self.agent_a.act(
            observation=obs_a,
            true_other_pos=self.agent_b.pos
        )
        action_b, pred_set_b = self.agent_b.act(
            observation=obs_b,
            true_other_pos=self.agent_a.pos
        )

        self.timestep += 1

        # Check for collision (both at same cell after moving)
        if self.agent_a.pos == self.agent_b.pos:
            self.collisions += 1

        return {
            'timestep':   self.timestep,
            'pos_a':      self.agent_a.pos,
            'pos_b':      self.agent_b.pos,
            'obs_a':      obs_a,
            'obs_b':      obs_b,
            'action_a':   action_a,
            'action_b':   action_b,
            'pred_set_a': pred_set_a,
            'pred_set_b': pred_set_b,
            'set_size_a': len(pred_set_a),
            'set_size_b': len(pred_set_b),
        }

    def run(self, n_steps=40):
        """
        Runs the full simulation for n_steps timesteps.

        Returns list of result dicts, one per timestep.
        """
        results = []
        for _ in range(n_steps):
            result = self.step()
            results.append(result)
        return results


# -------------------------------------------------------
# MAIN: RUN SIMULATION AND PRINT RESULTS
# -------------------------------------------------------
if __name__ == '__main__':
    print("=== Layer 4: Minimax Policy ===\n")

    # Load the calibrated predictor saved by Layer 3
    print("Loading calibrated predictor...")
    with open('model.pkl', 'rb') as f:
        saved = pickle.load(f)
    predictor = saved['predictor_90']
    print(f"  Loaded. Threshold q = {predictor.q:.4f}, alpha = {predictor.alpha}\n")

    # Run the simulation
    print("Running 40-step simulation with policy-driven agents...\n")
    env = PolicyEnvironment(predictor, seed=42)
    results = env.run(n_steps=40)

    # Print first 8 timesteps in detail
    print(f"{'Step':>4}  {'Pos A':>8}  {'Pos B':>8}  {'Action A':>8}  {'Action B':>8}  {'Set A':>6}  {'Set B':>6}  {'Collision':>9}")
    print("-" * 75)
    for r in results[:8]:
        collision = "YES" if r['pos_a'] == r['pos_b'] else "no"
        print(f"{r['timestep']:>4}  {str(r['pos_a']):>8}  {str(r['pos_b']):>8}  "
              f"{r['action_a']:>8}  {r['action_b']:>8}  "
              f"{r['set_size_a']:>6}  {r['set_size_b']:>6}  {collision:>9}")

    print("\n  ... (showing first 8 of 40 steps)\n")

    # Summary statistics
    set_sizes_a = [r['set_size_a'] for r in results]
    set_sizes_b = [r['set_size_b'] for r in results]

    print("Summary Statistics:")
    print(f"  Total collisions:         {env.collisions} / {len(results)} timesteps")
    print(f"  Agent A avg set size:     {np.mean(set_sizes_a):.1f} cells")
    print(f"  Agent B avg set size:     {np.mean(set_sizes_b):.1f} cells")

    # Coverage: how often was the true position in the prediction set?
    coverage_a = np.mean(env.agent_a.coverage_history)
    coverage_b = np.mean(env.agent_b.coverage_history)
    print(f"  Agent A empirical coverage: {coverage_a*100:.1f}%  (target: 90%)")
    print(f"  Agent B empirical coverage: {coverage_b*100:.1f}%  (target: 90%)")

    # Final positions
    print(f"\n  Agent A final pos: {env.agent_a.pos}  (goal: {env.agent_a.goal})")
    print(f"  Agent B final pos: {env.agent_b.pos}  (goal: {env.agent_b.goal})")

    # Save simulation results for Layer 5
    saved['sim_results'] = results
    saved['sim_env']     = env
    with open('model.pkl', 'wb') as f:
        pickle.dump(saved, f)
    print("\n  Saved simulation results to model.pkl")
    print("\nLayer 4 complete.")
