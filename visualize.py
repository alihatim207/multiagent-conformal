import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from classifier import idx_to_pos
from conformal import ConformalPredictor
from policy import PolicyEnvironment, BaselineEnvironment, sample_layout

# -------------------------------------------------------
# COLORS
# -------------------------------------------------------
COLOR_A       = '#2196F3'   # blue for Agent A
COLOR_B       = '#F44336'   # red for Agent B
COLOR_GRID    = '#ECEFF1'   # light gray grid background
COLOR_GOAL    = '#4CAF50'   # green goal markers
COLOR_BG      = '#FAFAFA'   # off-white figure background

COLOR_SAFE    = '#4CAF50'   # green -- conformal wins
COLOR_DANGER  = '#F44336'   # red -- baseline worse


# -------------------------------------------------------
# DRAW GRID BACKGROUND
# -------------------------------------------------------
def draw_grid(ax, title, grid_size=10):
    ax.set_facecolor(COLOR_GRID)
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')
    for i in range(grid_size + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.8, zorder=1)
        ax.axvline(i - 0.5, color='white', linewidth=0.8, zorder=1)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_xlabel('x', fontsize=9)
    ax.set_ylabel('y', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')


# -------------------------------------------------------
# DRAW STATIC GOAL MARKERS ON A GRID PANEL
# -------------------------------------------------------
def draw_goals(ax, goal_a, goal_b):
    ax.plot(*goal_a, '*', color=COLOR_GOAL, markersize=16, zorder=5)
    ax.plot(*goal_b, '*', color=COLOR_GOAL, markersize=16, zorder=5)
    ax.annotate('G_A', goal_a,
                xytext=(goal_a[0] - 1.4, goal_a[1] + 0.3),
                fontsize=7, color=COLOR_A, fontweight='bold')
    ax.annotate('G_B', goal_b,
                xytext=(goal_b[0] + 0.2, goal_b[1] + 0.3),
                fontsize=7, color=COLOR_B, fontweight='bold')


# -------------------------------------------------------
# METRICS PANEL
# -------------------------------------------------------
def draw_metrics_panel(ax, base_env, conf_env, n_steps):
    """
    Draws a static side-by-side metrics table comparing baseline vs conformal.
    Called once after both simulations are complete.
    """
    ax.axis('off')
    ax.set_facecolor(COLOR_BG)
    ax.set_title('Safety Metrics Comparison', fontsize=11, fontweight='bold', pad=8)

    # Compute metrics
    base_coll  = base_env.collisions
    conf_coll  = conf_env.collisions

    base_dist_a = _min_dist_to_goal(base_env.agent_a)
    base_dist_b = _min_dist_to_goal(base_env.agent_b)
    conf_dist_a = _min_dist_to_goal(conf_env.agent_a)
    conf_dist_b = _min_dist_to_goal(conf_env.agent_b)

    conf_cov_a = np.mean(conf_env.agent_a.coverage_history) * 100 if conf_env.agent_a.coverage_history else 0
    conf_cov_b = np.mean(conf_env.agent_b.coverage_history) * 100 if conf_env.agent_b.coverage_history else 0

    conf_set_a = np.mean(conf_env.agent_a.set_size_history) if conf_env.agent_a.set_size_history else 0
    conf_set_b = np.mean(conf_env.agent_b.set_size_history) if conf_env.agent_b.set_size_history else 0

    base_inter = _count_close_calls(base_env, threshold=1)
    conf_inter = _count_close_calls(conf_env, threshold=1)

    rows = [
        ('Metric',                   'Baseline',                        'Conformal',                       'Better'),
        ('Collisions',               f'{base_coll}',                    f'{conf_coll}',                    'conf' if conf_coll < base_coll else ('base' if base_coll < conf_coll else 'tie')),
        ('Near-misses (dist≤1)',      f'{base_inter}',                   f'{conf_inter}',                   'conf' if conf_inter < base_inter else ('base' if base_inter < conf_inter else 'tie')),
        ('A final dist to goal',     f'{base_dist_a}',                  f'{conf_dist_a}',                  'conf' if conf_dist_a < base_dist_a else ('base' if base_dist_a < conf_dist_a else 'tie')),
        ('B final dist to goal',     f'{base_dist_b}',                  f'{conf_dist_b}',                  'conf' if conf_dist_b < base_dist_b else ('base' if base_dist_b < conf_dist_b else 'tie')),
        ('Coverage A (target 90%)',  'N/A',                             f'{conf_cov_a:.1f}%',              '--'),
        ('Coverage B (target 90%)',  'N/A',                             f'{conf_cov_b:.1f}%',              '--'),
        ('Avg pred set A (cells)',   'N/A',                             f'{conf_set_a:.1f}',               '--'),
        ('Avg pred set B (cells)',   'N/A',                             f'{conf_set_b:.1f}',               '--'),
    ]

    col_xs  = [0.01, 0.35, 0.60, 0.86]
    col_align = ['left', 'center', 'center', 'center']
    row_height = 1.0 / (len(rows) + 1)

    for i, row in enumerate(rows):
        y = 1.0 - (i + 0.7) * row_height
        is_header = (i == 0)

        for j, (val, x, align) in enumerate(zip(row, col_xs, col_align)):
            weight = 'bold' if is_header else 'normal'
            color  = 'black'

            # Color the "Better" column
            if j == 3 and not is_header:
                if val == 'conf':
                    color = COLOR_SAFE
                    val   = 'Conformal'
                elif val == 'base':
                    color = COLOR_DANGER
                    val   = 'Baseline'
                elif val == 'tie':
                    color = 'gray'
                    val   = 'Tie'

            ax.text(x, y, val,
                    transform=ax.transAxes,
                    fontsize=8.5,
                    ha=align, va='center',
                    fontweight=weight,
                    color=color)

        # Separator line under header
        if is_header:
            line_y = y - row_height * 0.4
            ax.plot([0, 1], [line_y, line_y],
                    color='#B0BEC5', linewidth=0.8,
                    transform=ax.transAxes, zorder=1)

    # Alternating row shading
    for i in range(1, len(rows)):
        if i % 2 == 0:
            y_top = 1.0 - (i + 0.2) * row_height
            rect = patches.FancyBboxPatch(
                (0, y_top), 1, row_height,
                boxstyle='square,pad=0',
                facecolor='#ECEFF1', edgecolor='none',
                transform=ax.transAxes, zorder=0
            )
            ax.add_patch(rect)


def _min_dist_to_goal(agent):
    """Manhattan distance from agent's final position to its goal."""
    return abs(agent.pos[0] - agent.goal[0]) + abs(agent.pos[1] - agent.goal[1])


def _count_close_calls(env, threshold=1):
    """Counts timesteps where agents were within Manhattan distance <= threshold."""
    hist_a = env.agent_a.pos_history[1:]  # skip initial position
    hist_b = env.agent_b.pos_history[1:]
    count = 0
    for pa, pb in zip(hist_a, hist_b):
        if abs(pa[0]-pb[0]) + abs(pa[1]-pb[1]) <= threshold:
            count += 1
    return count


# -------------------------------------------------------
# MAIN COMPARISON ANIMATION
# -------------------------------------------------------
def compare_animate(predictor, n_steps=50, seed=99, save_path=None, mode="crossing"):
    """
    Runs baseline and conformal simulations on the same layout, then
    animates them side by side with a safety metrics panel below.

    predictor:  calibrated ConformalPredictor
    n_steps:    timesteps to animate
    seed:       controls both the layout and the noise sequence
    save_path:  if set, saves as GIF; otherwise shows interactively
    mode:       "random" or "classic"
    """
    # Sample a shared layout so both envs start in the same positions
    layout = sample_layout(seed=seed, mode=mode)
    start_a, start_b, goal_a, goal_b = layout

    print(f"Layout: A {start_a}->{goal_a}  |  B {start_b}->{goal_b}")

    # Run baseline simulation
    print("Running baseline simulation...")
    base_env = BaselineEnvironment(seed=seed, starts_and_goals=layout)
    base_results = base_env.run(n_steps=n_steps)
    print(f"  Done. Collisions: {base_env.collisions}")

    # Run conformal simulation (same seed so noise sequence matches)
    print("Running conformal simulation...")
    conf_env = PolicyEnvironment(predictor, seed=seed, starts_and_goals=layout)
    conf_results = conf_env.run(n_steps=n_steps)
    print(f"  Done. Collisions: {conf_env.collisions}")

    # -------------------------------------------------------
    # FIGURE LAYOUT
    # 2 rows: top has the two grid panels side by side,
    # bottom has the metrics panel spanning the full width.
    # -------------------------------------------------------
    fig = plt.figure(figsize=(16, 11), facecolor=COLOR_BG)
    fig.suptitle(
        f'Safety Comparison: Baseline vs Conformal  |  seed={seed}  |  '
        f'A: {start_a}→{goal_a}   B: {start_b}→{goal_b}',
        fontsize=11, fontweight='bold', y=0.99
    )

    outer = GridSpec(2, 1, figure=fig, height_ratios=[2.2, 1], hspace=0.35)
    top   = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.25)

    ax_base = fig.add_subplot(top[0])
    ax_conf = fig.add_subplot(top[1])
    ax_metrics = fig.add_subplot(outer[1])

    GRID_SIZE = 10

    draw_grid(ax_base, 'Baseline (greedy, no uncertainty)')
    draw_grid(ax_conf, 'Conformal (minimax + prediction sets)')
    draw_goals(ax_base, goal_a, goal_b)
    draw_goals(ax_conf, goal_a, goal_b)

    # -------------------------------------------------------
    # PRE-CREATE PREDICTION SET PATCHES (conformal panel only)
    # -------------------------------------------------------
    set_patches_a = {}
    set_patches_b = {}
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            pa = patches.Rectangle(
                (x - 0.45, y - 0.45), 0.9, 0.9,
                linewidth=0, facecolor=COLOR_A, alpha=0.22,
                visible=False, zorder=2
            )
            ax_conf.add_patch(pa)
            set_patches_a[(x, y)] = pa

            pb = patches.Rectangle(
                (x - 0.45, y - 0.45), 0.9, 0.9,
                linewidth=0, facecolor=COLOR_B, alpha=0.22,
                visible=False, zorder=2
            )
            ax_conf.add_patch(pb)
            set_patches_b[(x, y)] = pb

    # -------------------------------------------------------
    # ANIMATED ELEMENTS -- BASELINE PANEL
    # -------------------------------------------------------
    base_dot_a, = ax_base.plot([], [], 'o', color=COLOR_A, markersize=13, zorder=6, label='Agent A')
    base_dot_b, = ax_base.plot([], [], 'o', color=COLOR_B, markersize=13, zorder=6, label='Agent B')
    base_path_a, = ax_base.plot([], [], '-', color=COLOR_A, alpha=0.4, linewidth=1.5, zorder=3)
    base_path_b, = ax_base.plot([], [], '-', color=COLOR_B, alpha=0.4, linewidth=1.5, zorder=3)
    base_text = ax_base.text(
        0.02, 0.97, '', transform=ax_base.transAxes,
        fontsize=8.5, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
    )
    ax_base.legend(loc='lower right', fontsize=8, framealpha=0.8)

    # -------------------------------------------------------
    # ANIMATED ELEMENTS -- CONFORMAL PANEL
    # -------------------------------------------------------
    conf_dot_a, = ax_conf.plot([], [], 'o', color=COLOR_A, markersize=13, zorder=6, label='Agent A')
    conf_dot_b, = ax_conf.plot([], [], 'o', color=COLOR_B, markersize=13, zorder=6, label='Agent B')
    conf_path_a, = ax_conf.plot([], [], '-', color=COLOR_A, alpha=0.4, linewidth=1.5, zorder=3)
    conf_path_b, = ax_conf.plot([], [], '-', color=COLOR_B, alpha=0.4, linewidth=1.5, zorder=3)
    conf_text = ax_conf.text(
        0.02, 0.97, '', transform=ax_conf.transAxes,
        fontsize=8.5, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
    )
    ax_conf.legend(loc='lower right', fontsize=8, framealpha=0.8)

    # Draw static metrics panel (doesn't change frame to frame)
    draw_metrics_panel(ax_metrics, base_env, conf_env, n_steps)

    # -------------------------------------------------------
    # ANIMATION UPDATE
    # -------------------------------------------------------
    def update(frame):
        rb = base_results[frame]
        rc = conf_results[frame]

        # -- Baseline panel --
        base_dot_a.set_data([rb['pos_a'][0]], [rb['pos_a'][1]])
        base_dot_b.set_data([rb['pos_b'][0]], [rb['pos_b'][1]])

        bpa_x = [p[0] for p in base_env.agent_a.pos_history[:frame+2]]
        bpa_y = [p[1] for p in base_env.agent_a.pos_history[:frame+2]]
        bpb_x = [p[0] for p in base_env.agent_b.pos_history[:frame+2]]
        bpb_y = [p[1] for p in base_env.agent_b.pos_history[:frame+2]]
        base_path_a.set_data(bpa_x, bpa_y)
        base_path_b.set_data(bpb_x, bpb_y)

        base_coll_so_far = sum(
            1 for pa, pb in zip(
                base_env.agent_a.pos_history[1:frame+2],
                base_env.agent_b.pos_history[1:frame+2]
            ) if pa == pb
        )
        base_text.set_text(
            f'Step {rb["timestep"]:02d}/{n_steps}\n'
            f'Collisions: {base_coll_so_far}'
        )

        # -- Conformal panel: update prediction set patches --
        for patch in set_patches_a.values():
            patch.set_visible(False)
        for patch in set_patches_b.values():
            patch.set_visible(False)
        for pos in rc['pred_set_a']:
            if pos in set_patches_a:
                set_patches_a[pos].set_visible(True)
        for pos in rc['pred_set_b']:
            if pos in set_patches_b:
                set_patches_b[pos].set_visible(True)

        conf_dot_a.set_data([rc['pos_a'][0]], [rc['pos_a'][1]])
        conf_dot_b.set_data([rc['pos_b'][0]], [rc['pos_b'][1]])

        cpa_x = [p[0] for p in conf_env.agent_a.pos_history[:frame+2]]
        cpa_y = [p[1] for p in conf_env.agent_a.pos_history[:frame+2]]
        cpb_x = [p[0] for p in conf_env.agent_b.pos_history[:frame+2]]
        cpb_y = [p[1] for p in conf_env.agent_b.pos_history[:frame+2]]
        conf_path_a.set_data(cpa_x, cpa_y)
        conf_path_b.set_data(cpb_x, cpb_y)

        conf_coll_so_far = sum(
            1 for pa, pb in zip(
                conf_env.agent_a.pos_history[1:frame+2],
                conf_env.agent_b.pos_history[1:frame+2]
            ) if pa == pb
        )
        cov_a_so_far = np.mean(conf_env.agent_a.coverage_history[:frame+1]) * 100 if frame >= 0 else 0
        conf_text.set_text(
            f'Step {rc["timestep"]:02d}/{n_steps}\n'
            f'Collisions: {conf_coll_so_far}\n'
            f'Set A: {rc["set_size_a"]} cells\n'
            f'Coverage A: {cov_a_so_far:.0f}%'
        )

        return (
            list(set_patches_a.values()) + list(set_patches_b.values()) +
            [base_dot_a, base_dot_b, base_path_a, base_path_b, base_text,
             conf_dot_a, conf_dot_b, conf_path_a, conf_path_b, conf_text]
        )

    anim = animation.FuncAnimation(
        fig, update, frames=n_steps,
        interval=400, blit=False, repeat=True
    )

    plt.tight_layout()

    if save_path:
        print(f"Saving to {save_path}...")
        writer = animation.PillowWriter(fps=3)
        anim.save(save_path, writer=writer)
        print("  Saved.")
    else:
        plt.show()

    return anim, base_env, conf_env


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == '__main__':
    print("=== Layer 5: Safety Comparison Visualization ===\n")

    print("Loading calibrated predictor...")
    with open('model.pkl', 'rb') as f:
        saved = pickle.load(f)
    predictor = saved['predictor_90']
    print(f"  Loaded. q = {predictor.q:.4f}\n")

    anim, base_env, conf_env = compare_animate(
        predictor,
        n_steps=50,
        seed=99,
        save_path='simulation.gif',
        mode='random'
    )

    print("\nFinal Summary:")
    print(f"  Baseline  -- collisions: {base_env.collisions}")
    print(f"  Conformal -- collisions: {conf_env.collisions}")
    print(f"  Conformal -- coverage A: {np.mean(conf_env.agent_a.coverage_history)*100:.1f}%")
    print(f"  Conformal -- coverage B: {np.mean(conf_env.agent_b.coverage_history)*100:.1f}%")
    print(f"  Conformal -- avg set A:  {np.mean(conf_env.agent_a.set_size_history):.1f} cells")
    print(f"  Conformal -- avg set B:  {np.mean(conf_env.agent_b.set_size_history):.1f} cells")
    print("\nLayer 5 complete.")

