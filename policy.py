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
    Cost of moving to my_next_pos assuming the other agent is at their_pos.
    Collision penalty is large enough to always dominate goal distance on this grid.
    """
    COLLISION_PENALTY = 1000
    collision = COLLISION_PENALTY if my_next_pos == their_pos else 0
    goal_dist = abs(my_next_pos[0] - goal[0]) + abs(my_next_pos[1] - goal[1])
    return collision + goal_dist


# -------------------------------------------------------
# MINIMAX POLICY
# -------------------------------------------------------
def minimax_policy(my_pos, prediction_set, goal):
    """
    Picks the action that minimizes worst-case cost over the prediction set.
    If the set is empty, falls back to greedy goal-seeking.
    """
    best_action = None
    best_worst_case_cost = float('inf')
    action_costs = {}

    for action in ACTION_LIST:
        dx, dy = ACTIONS[action]
        my_next_pos = clip_to_grid((my_pos[0] + dx, my_pos[1] + dy))

        if len(prediction_set) == 0:
            worst_case = abs(my_next_pos[0] - goal[0]) + abs(my_next_pos[1] - goal[1])
        else:
            costs_for_this_action = [
                cost(my_pos, my_next_pos, their_pos, goal)
                for their_pos in prediction_set
            ]
            worst_case = max(costs_for_this_action)

        action_costs[action] = worst_case

        if worst_case < best_worst_case_cost:
            best_worst_case_cost = worst_case
            best_action = action

    return best_action, action_costs


# -------------------------------------------------------
# POLICY-DRIVEN AGENT
# -------------------------------------------------------
class PolicyAgent:
    """
    Agent that runs conformal prediction + minimax on every step.
    Keeps a full history of positions, prediction sets, and actions.
    """

    def __init__(self, start_pos, goal, predictor):
        self.pos = clip_to_grid(start_pos)
        self.goal = goal
        self.predictor = predictor

        # Running history for post-hoc analysis
        self.pos_history = [self.pos]
        self.pred_set_history = []
        self.set_size_history = []
        self.action_history = []
        self.coverage_history = []

    def act(self, observation, true_other_pos=None):
        """
        One step: build prediction set from noisy obs, run minimax, move.
        true_other_pos is only used to log empirical coverage after the fact.
        """
        # Build prediction set from noisy observation
        prediction_set, probs = self.predictor.predict_set(observation)

        # Pick action via minimax over the prediction set
        action, action_costs = minimax_policy(self.pos, prediction_set, self.goal)

        # Move
        dx, dy = ACTIONS[action]
        self.pos = clip_to_grid((self.pos[0] + dx, self.pos[1] + dy))

        # Log history
        self.pos_history.append(self.pos)
        self.pred_set_history.append(prediction_set)
        self.set_size_history.append(len(prediction_set))
        self.action_history.append(action)

        if true_other_pos is not None:
            self.coverage_history.append(true_other_pos in prediction_set)

        return action, prediction_set


# -------------------------------------------------------
# BASELINE AGENT (no conformal, no minimax)
# -------------------------------------------------------
class BaselineAgent:
    """
    Greedy agent that ignores the other agent entirely and just moves
    toward its goal using Manhattan distance. No prediction set, no
    uncertainty awareness -- pure naive navigation.
    """

    def __init__(self, start_pos, goal):
        self.pos = clip_to_grid(start_pos)
        self.goal = goal

        self.pos_history  = [self.pos]
        self.action_history = []
        # These stay empty -- no prediction sets -- but kept for API symmetry
        # with PolicyAgent so visualize.py can treat both uniformly.
        self.set_size_history = []
        self.coverage_history = []

    def act(self):
        """
        Picks the action that reduces Manhattan distance to goal the most.
        Ties broken by action order in ACTION_LIST.
        """
        best_action = None
        best_dist = float('inf')

        for action in ACTION_LIST:
            dx, dy = ACTIONS[action]
            next_pos = clip_to_grid((self.pos[0] + dx, self.pos[1] + dy))
            dist = abs(next_pos[0] - self.goal[0]) + abs(next_pos[1] - self.goal[1])
            if dist < best_dist:
                best_dist = dist
                best_action = action

        dx, dy = ACTIONS[best_action]
        self.pos = clip_to_grid((self.pos[0] + dx, self.pos[1] + dy))

        self.pos_history.append(self.pos)
        self.action_history.append(best_action)

        return best_action


# -------------------------------------------------------
# BASELINE ENVIRONMENT
# -------------------------------------------------------
class BaselineEnvironment:
    """
    Simulation where both agents use greedy goal-seeking with no
    awareness of each other. Used as the safety baseline to compare
    against PolicyEnvironment.
    """

    def __init__(self, seed=None, mode="crossing", starts_and_goals=None):
        """
        starts_and_goals: optional (start_a, start_b, goal_a, goal_b) tuple.
            Pass this to lock in the same layout as a PolicyEnvironment run.
        If omitted, layout is sampled from seed/mode like PolicyEnvironment.
        """
        self.rng = np.random.default_rng(seed)

        if starts_and_goals is not None:
            start_a, start_b, goal_a, goal_b = starts_and_goals
        elif mode == "classic":
            start_a, start_b = (0, 0), (9, 9)
            goal_a,  goal_b  = (9, 9), (0, 0)
        else:
            all_cells = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
            chosen = [all_cells[i] for i in self.rng.choice(len(all_cells), size=4, replace=False)]
            start_a, start_b, goal_a, goal_b = chosen

        self.agent_a = BaselineAgent(start_pos=start_a, goal=goal_a)
        self.agent_b = BaselineAgent(start_pos=start_b, goal=goal_b)

        self.timestep  = 0
        self.collisions = 0

    def step(self):
        """Both agents move greedily toward their goals, ignoring each other."""
        action_a = self.agent_a.act()
        action_b = self.agent_b.act()

        self.timestep += 1

        if self.agent_a.pos == self.agent_b.pos:
            self.collisions += 1

        return {
            'timestep': self.timestep,
            'pos_a':    self.agent_a.pos,
            'pos_b':    self.agent_b.pos,
            'action_a': action_a,
            'action_b': action_b,
        }

    def run(self, n_steps=40):
        results = []
        for _ in range(n_steps):
            results.append(self.step())
        return results


# -------------------------------------------------------
# LAYOUT SAMPLING WITH CROSSING-PATH GUARANTEE
# -------------------------------------------------------
def _segments_cross(p1, p2, p3, p4):
    """
    Returns True if line segment p1->p2 crosses p3->p4.
    Uses the standard cross-product / orientation test.
    """
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    def on_segment(p, q, r):
        return (min(p[0],r[0]) <= q[0] <= max(p[0],r[0]) and
                min(p[1],r[1]) <= q[1] <= max(p[1],r[1]))

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Collinear cases
    if d1 == 0 and on_segment(p3, p1, p4): return True
    if d2 == 0 and on_segment(p3, p2, p4): return True
    if d3 == 0 and on_segment(p1, p3, p2): return True
    if d4 == 0 and on_segment(p1, p4, p2): return True

    return False


def sample_layout(seed=None, mode="random"):
    """
    Returns (start_a, start_b, goal_a, goal_b).

    In "crossing" mode (the default for comparison runs), keeps resampling
    until the straight-line path from start_a->goal_a crosses start_b->goal_b,
    guaranteeing the greedy baseline agents are on a collision course.

    mode="classic"  : fixed opposite-corner layout
    mode="random"   : random layout, no crossing check
    mode="crossing" : random layout, retries until paths cross
    """
    rng = np.random.default_rng(seed)

    if mode == "classic":
        return (0, 0), (9, 9), (9, 9), (0, 0)

    all_cells = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]

    for _ in range(1000):
        chosen = [all_cells[i] for i in rng.choice(len(all_cells), size=4, replace=False)]
        start_a, start_b, goal_a, goal_b = chosen

        if mode == "random":
            return tuple(chosen)

        # "crossing" mode: accept only if paths cross
        if _segments_cross(start_a, goal_a, start_b, goal_b):
            return tuple(chosen)

    # Fallback -- should be extremely rare
    return (0, 0), (9, 9), (9, 9), (0, 0)


# -------------------------------------------------------
# FULL SIMULATION WITH POLICY
# -------------------------------------------------------
class PolicyEnvironment:
    """
    Simulation where both agents navigate with conformal prediction + minimax
    instead of moving randomly.
    """

    def __init__(self, predictor, seed=None, mode="crossing", starts_and_goals=None):
        """
        seed=None gives a fresh random layout each run; pass an int to reproduce an episode.
        mode: "random" samples 4 distinct random cells; "classic" uses opposite corners.
        starts_and_goals: optional explicit (start_a, start_b, goal_a, goal_b) -- overrides seed/mode.
            Use sample_layout() to generate and share a layout with BaselineEnvironment.
        """
        self.rng = np.random.default_rng(seed)

        if starts_and_goals is not None:
            start_a, start_b, goal_a, goal_b = starts_and_goals
        elif mode == "classic":
            start_a, start_b = (0, 0), (9, 9)
            goal_a,  goal_b  = (9, 9), (0, 0)
        else:
            all_cells = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
            chosen = [all_cells[i] for i in self.rng.choice(len(all_cells), size=4, replace=False)]
            start_a, start_b, goal_a, goal_b = chosen

        self.agent_a = PolicyAgent(start_pos=start_a, goal=goal_a, predictor=predictor)
        self.agent_b = PolicyAgent(start_pos=start_b, goal=goal_b, predictor=predictor)

        self.timestep = 0
        self.collisions = 0

    def get_noisy_observation(self, target_pos):
        """Gaussian-noisy observation of target_pos using the same model as the base environment."""
        from environment import observe
        return observe(target_pos, self.rng)

    def step(self):
        """Both agents observe each other, pick minimax actions, move. Logs collisions."""
        obs_a = self.get_noisy_observation(self.agent_b.pos)
        obs_b = self.get_noisy_observation(self.agent_a.pos)

        # Pass true positions only for coverage tracking, not for decisions
        action_a, pred_set_a = self.agent_a.act(
            observation=obs_a,
            true_other_pos=self.agent_b.pos
        )
        action_b, pred_set_b = self.agent_b.act(
            observation=obs_b,
            true_other_pos=self.agent_a.pos
        )

        self.timestep += 1

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
        """Runs the simulation for n_steps and returns the result list."""
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
