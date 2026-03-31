import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from classifier import idx_to_pos
from conformal import ConformalPredictor
from policy import PolicyEnvironment

# -------------------------------------------------------
# COLORS
# -------------------------------------------------------
# Agent colors -- bright and distinct
COLOR_A       = '#2196F3'   # blue  for Agent A
COLOR_B       = '#F44336'   # red   for Agent B
COLOR_SET_A   = '#2196F3'   # same blue, transparent for prediction set
COLOR_SET_B   = '#F44336'   # same red, transparent for prediction set
COLOR_GRID    = '#ECEFF1'   # light gray grid background
COLOR_GOAL    = '#4CAF50'   # green for goal markers
COLOR_PATH    = '#B0BEC5'   # gray for path trails
COLOR_BG      = '#FAFAFA'   # off-white figure background


# -------------------------------------------------------
# SETUP FIGURE
# -------------------------------------------------------
def setup_figure():
    """
    Creates the 4-panel figure layout using GridSpec.

    GridSpec lets you define a grid of subplots with custom
    sizes. We use a 2x2 layout:
      - Top left:    Grid world (main visualization)
      - Top right:   Prediction set size over time
      - Bottom left: Empirical coverage over time
      - Bottom right: Distance between agents over time

    figsize=(14, 10) sets the figure to 14 inches wide, 10 tall.
    This is large enough to see detail clearly.
    """
    fig = plt.figure(figsize=(14, 10), facecolor=COLOR_BG)
    fig.suptitle(
        'Multi-Agent Conformal Prediction: Uncertainty-Aware Navigation',
        fontsize=14, fontweight='bold', y=0.98
    )

    # GridSpec(rows, cols) with width and height ratios
    # The grid panel gets more space (ratio 2) than the others (ratio 1)
    gs = GridSpec(
        2, 2,
        figure=fig,
        width_ratios=[2, 1],   # left column is twice as wide
        height_ratios=[1, 1],
        hspace=0.35,
        wspace=0.3
    )

    ax_grid     = fig.add_subplot(gs[:, 0])   # spans both rows, left column
    ax_setsize  = fig.add_subplot(gs[0, 1])   # top right
    ax_coverage = fig.add_subplot(gs[1, 1])   # bottom right

    return fig, ax_grid, ax_setsize, ax_coverage


# -------------------------------------------------------
# DRAW GRID WORLD
# -------------------------------------------------------
def draw_grid(ax, grid_size=10):
    """
    Draws the empty grid background.

    ax: matplotlib axes to draw on
    grid_size: number of cells per side

    We draw thin gray lines to separate cells and color
    the background light gray. The axes are configured
    so that cell (x, y) is centered at (x, y).
    """
    ax.set_facecolor(COLOR_GRID)
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')

    # Draw grid lines -- one per cell boundary
    for i in range(grid_size + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.8, zorder=1)
        ax.axvline(i - 0.5, color='white', linewidth=0.8, zorder=1)

    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.set_title('Grid World', fontsize=11, fontweight='bold')


# -------------------------------------------------------
# MAIN ANIMATION FUNCTION
# -------------------------------------------------------
def animate(predictor, n_steps=50, seed=99, save_path=None):
    """
    Runs the full simulation and renders it as an animation.

    predictor:  calibrated ConformalPredictor from Layer 3
    n_steps:    how many timesteps to animate
    seed:       random seed (different from training/calibration)
    save_path:  if provided, saves animation to file instead of showing
    """
    # -------------------------------------------------------
    # RUN SIMULATION FIRST, COLLECT ALL DATA
    # -------------------------------------------------------
    # We run the full simulation before animating so we have
    # all data available upfront. This makes the animation
    # smoother than computing on the fly.
    print("Running simulation...")
    env = PolicyEnvironment(predictor, seed=seed)
    results = env.run(n_steps=n_steps)
    print(f"  Done. {n_steps} steps, {env.collisions} collisions.")

    # Extract time series data for the line plots
    timesteps    = [r['timestep'] for r in results]
    set_sizes_a  = [r['set_size_a'] for r in results]
    set_sizes_b  = [r['set_size_b'] for r in results]

    # Running coverage: at each timestep, what fraction of steps
    # so far had the true position inside the prediction set?
    # np.cumsum computes cumulative sum: [a, b, c] -> [a, a+b, a+b+c]
    # Dividing by timestep index gives running mean.
    cov_a_running = np.cumsum(env.agent_a.coverage_history) / np.arange(1, n_steps + 1)
    cov_b_running = np.cumsum(env.agent_b.coverage_history) / np.arange(1, n_steps + 1)

    # Distance between agents at each timestep
    distances = [
        abs(r['pos_a'][0] - r['pos_b'][0]) + abs(r['pos_a'][1] - r['pos_b'][1])
        for r in results
    ]

    # -------------------------------------------------------
    # SET UP FIGURE
    # -------------------------------------------------------
    fig, ax_grid, ax_setsize, ax_coverage = setup_figure()
    draw_grid(ax_grid)

    GRID_SIZE = 10

    # Draw goal markers (stars) -- permanent, drawn once
    # zorder controls layering: higher = drawn on top
    ax_grid.plot(*env.agent_a.goal, '*', color=COLOR_GOAL,
                 markersize=18, zorder=5, label='Goals')
    ax_grid.plot(*env.agent_b.goal, '*', color=COLOR_GOAL,
                 markersize=18, zorder=5)

    # Label goals
    ax_grid.annotate('Goal A', env.agent_a.goal,
                     xytext=(env.agent_a.goal[0]-1.5, env.agent_a.goal[1]+0.3),
                     fontsize=7, color=COLOR_A, fontweight='bold')
    ax_grid.annotate('Goal B', env.agent_b.goal,
                     xytext=(env.agent_b.goal[0]+0.2, env.agent_b.goal[1]+0.3),
                     fontsize=7, color=COLOR_B, fontweight='bold')

    # -------------------------------------------------------
    # INITIALIZE ANIMATED ELEMENTS
    # These are objects we'll update each frame.
    # -------------------------------------------------------

    # Prediction set backgrounds -- one patch per grid cell per agent
    # We pre-create all 100 patches for each agent and toggle visibility
    # This is much faster than creating/destroying patches every frame

    # Dict: (x,y) -> Rectangle patch for agent A's prediction set
    set_patches_a = {}
    set_patches_b = {}
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            # Agent A prediction set patches (blue, transparent)
            patch_a = patches.Rectangle(
                (x - 0.45, y - 0.45), 0.9, 0.9,  # position and size
                linewidth=0,
                facecolor=COLOR_SET_A,
                alpha=0.25,
                visible=False,   # hidden by default
                zorder=2
            )
            ax_grid.add_patch(patch_a)
            set_patches_a[(x, y)] = patch_a

            # Agent B prediction set patches (red, transparent)
            patch_b = patches.Rectangle(
                (x - 0.45, y - 0.45), 0.9, 0.9,
                linewidth=0,
                facecolor=COLOR_SET_B,
                alpha=0.25,
                visible=False,
                zorder=2
            )
            ax_grid.add_patch(patch_b)
            set_patches_b[(x, y)] = patch_b

    # Agent position markers (circles)
    agent_a_dot, = ax_grid.plot([], [], 'o',
                                color=COLOR_A, markersize=14,
                                zorder=6, label='Agent A')
    agent_b_dot, = ax_grid.plot([], [], 'o',
                                color=COLOR_B, markersize=14,
                                zorder=6, label='Agent B')

    # Path trails (thin lines showing where agents have been)
    path_a_line, = ax_grid.plot([], [], '-',
                                color=COLOR_A, alpha=0.4,
                                linewidth=1.5, zorder=3)
    path_b_line, = ax_grid.plot([], [], '-',
                                color=COLOR_B, alpha=0.4,
                                linewidth=1.5, zorder=3)

    # Timestep label in top-left corner of grid
    timestep_text = ax_grid.text(
        0.02, 0.97, '', transform=ax_grid.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Legend
    ax_grid.legend(loc='lower right', fontsize=8, framealpha=0.8)

    # -------------------------------------------------------
    # SET UP LINE PLOTS
    # -------------------------------------------------------

    # Panel 2: Prediction set size over time
    ax_setsize.set_title('Prediction Set Size', fontsize=11, fontweight='bold')
    ax_setsize.set_xlabel('Timestep', fontsize=9)
    ax_setsize.set_ylabel('Cells in Set', fontsize=9)
    ax_setsize.set_xlim(0, n_steps)
    ax_setsize.set_ylim(0, 50)
    ax_setsize.axhline(100, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    ax_setsize.set_facecolor(COLOR_BG)

    # Draw the full trajectory lightly in background so you can see
    # where things are heading
    ax_setsize.plot(timesteps, set_sizes_a, color=COLOR_A, alpha=0.15, linewidth=1)
    ax_setsize.plot(timesteps, set_sizes_b, color=COLOR_B, alpha=0.15, linewidth=1)

    # Animated lines that grow frame by frame
    setsize_a_line, = ax_setsize.plot([], [], color=COLOR_A,
                                      linewidth=2, label='Agent A')
    setsize_b_line, = ax_setsize.plot([], [], color=COLOR_B,
                                      linewidth=2, label='Agent B')
    ax_setsize.legend(fontsize=8)

    # Panel 3: Coverage over time
    ax_coverage.set_title('Empirical Coverage', fontsize=11, fontweight='bold')
    ax_coverage.set_xlabel('Timestep', fontsize=9)
    ax_coverage.set_ylabel('Coverage', fontsize=9)
    ax_coverage.set_xlim(0, n_steps)
    ax_coverage.set_ylim(0, 1.05)
    ax_coverage.set_facecolor(COLOR_BG)

    # Target coverage line (90%)
    ax_coverage.axhline(0.9, color='green', linestyle='--',
                        linewidth=1.5, alpha=0.7, label='90% target')

    # Background trajectories
    ax_coverage.plot(range(1, n_steps+1), cov_a_running,
                     color=COLOR_A, alpha=0.15, linewidth=1)
    ax_coverage.plot(range(1, n_steps+1), cov_b_running,
                     color=COLOR_B, alpha=0.15, linewidth=1)

    # Animated coverage lines
    cov_a_line, = ax_coverage.plot([], [], color=COLOR_A,
                                   linewidth=2, label='Agent A')
    cov_b_line, = ax_coverage.plot([], [], color=COLOR_B,
                                   linewidth=2, label='Agent B')
    ax_coverage.legend(fontsize=8, loc='lower right')

    # -------------------------------------------------------
    # ANIMATION UPDATE FUNCTION
    # -------------------------------------------------------
    def update(frame):
        """
        Called once per frame by FuncAnimation.
        Updates all animated elements to reflect timestep `frame`.

        frame: integer from 0 to n_steps-1
        """
        r = results[frame]

        # -- Update prediction set patches --
        # First hide ALL patches from previous frame
        for patch in set_patches_a.values():
            patch.set_visible(False)
        for patch in set_patches_b.values():
            patch.set_visible(False)

        # Show patches for cells in current prediction sets
        for pos in r['pred_set_a']:
            if pos in set_patches_a:
                set_patches_a[pos].set_visible(True)
        for pos in r['pred_set_b']:
            if pos in set_patches_b:
                set_patches_b[pos].set_visible(True)

        # -- Update agent position dots --
        agent_a_dot.set_data([r['pos_a'][0]], [r['pos_a'][1]])
        agent_b_dot.set_data([r['pos_b'][0]], [r['pos_b'][1]])

        # -- Update path trails --
        # Show path from start up to current frame
        # env.agent_a.pos_history[0] is the start position
        # pos_history has n_steps+1 entries (start + one per step)
        path_a_x = [p[0] for p in env.agent_a.pos_history[:frame+2]]
        path_a_y = [p[1] for p in env.agent_a.pos_history[:frame+2]]
        path_b_x = [p[0] for p in env.agent_b.pos_history[:frame+2]]
        path_b_y = [p[1] for p in env.agent_b.pos_history[:frame+2]]
        path_a_line.set_data(path_a_x, path_a_y)
        path_b_line.set_data(path_b_x, path_b_y)

        # -- Update timestep label --
        coverage_a_so_far = np.mean(env.agent_a.coverage_history[:frame+1])
        timestep_text.set_text(
            f'Step {r["timestep"]:02d}/{n_steps}\n'
            f'Set A: {r["set_size_a"]} cells\n'
            f'Set B: {r["set_size_b"]} cells\n'
            f'Coverage A: {coverage_a_so_far*100:.0f}%'
        )

        # -- Update set size line plots --
        # Show data up to current frame only
        xs = timesteps[:frame+1]
        setsize_a_line.set_data(xs, set_sizes_a[:frame+1])
        setsize_b_line.set_data(xs, set_sizes_b[:frame+1])

        # -- Update coverage line plots --
        cov_xs = list(range(1, frame+2))
        cov_a_line.set_data(cov_xs, cov_a_running[:frame+1])
        cov_b_line.set_data(cov_xs, cov_b_running[:frame+1])

        # Return all updated artists (required by blit=True for performance)
        return (list(set_patches_a.values()) +
                list(set_patches_b.values()) +
                [agent_a_dot, agent_b_dot,
                 path_a_line, path_b_line,
                 timestep_text,
                 setsize_a_line, setsize_b_line,
                 cov_a_line, cov_b_line])

    # -------------------------------------------------------
    # CREATE AND RUN ANIMATION
    # -------------------------------------------------------
    # FuncAnimation calls update(frame) for each frame in range(n_steps)
    # interval=400 means 400ms between frames (about 2.5 fps)
    # blit=True only redraws changed elements -- much faster
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_steps,
        interval=400,
        blit=False,   # False for compatibility across backends
        repeat=True
    )

    plt.tight_layout()

    if save_path:
        print(f"Saving animation to {save_path}...")
        # PillowWriter saves as GIF -- no ffmpeg needed
        writer = animation.PillowWriter(fps=3)
        anim.save(save_path, writer=writer)
        print(f"  Saved.")
    else:
        plt.show()

    return anim, env


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == '__main__':
    print("=== Layer 5: Visualization ===\n")

    # Load the calibrated predictor
    print("Loading calibrated predictor...")
    with open('model.pkl', 'rb') as f:
        saved = pickle.load(f)
    predictor = saved['predictor_90']
    print(f"  Loaded. q = {predictor.q:.4f}\n")

    # Run animation and save as GIF
    anim, env = animate(
        predictor,
        n_steps=50,
        seed=99,
        save_path='simulation.gif'
    )

    # Print final summary
    print("\nFinal Summary:")
    print(f"  Collisions:          {env.collisions} / 50 timesteps")
    print(f"  Agent A coverage:    {np.mean(env.agent_a.coverage_history)*100:.1f}%")
    print(f"  Agent B coverage:    {np.mean(env.agent_b.coverage_history)*100:.1f}%")
    print(f"  Avg set size A:      {np.mean(env.agent_a.set_size_history):.1f} cells")
    print(f"  Avg set size B:      {np.mean(env.agent_b.set_size_history):.1f} cells")
    print(f"  Agent A final pos:   {env.agent_a.pos}  (goal: {env.agent_a.goal})")
    print(f"  Agent B final pos:   {env.agent_b.pos}  (goal: {env.agent_b.goal})")
    print("\nLayer 5 complete.")
