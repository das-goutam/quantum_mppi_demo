"""One look for all figures of the repository."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

INK, MUTED, GRID, WALL = "#0F172A", "#64748B", "#E2E8F0", "#CBD5E1"
BLUE, GREEN, RED, GOLD, VIOLET, CORAL = "#0EA5E9", "#16A34A", "#E11D48", "#F59E0B", "#7C3AED", "#F0725C"


def use_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": WALL, "axes.labelcolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "font.size": 10,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
        "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def draw_corridor(ax, task, t_max=None, now=None):
    """The corridor: walls in grey, obstacles in coral. Time runs to the right."""
    T = task.episode_steps
    t_max = T if t_max is None else t_max
    ax.set_xlim(-0.6, t_max + 0.6)
    ax.set_ylim(-0.6, 7.6)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    for x in range(1, 7):
        ax.axhline(x, color=GRID, lw=0.8, zorder=0.5)
    for y0 in (-0.6, 6.5):
        ax.add_patch(Rectangle((-0.6, y0), t_max + 1.2, 1.1, color=WALL, lw=0, zorder=1))
    for t, blocked in task.obstacles.items():
        if t > t_max:
            continue
        lo, hi = min(blocked) - 0.42, max(blocked) + 0.42
        faded = now is not None and t < now
        ax.add_patch(FancyBboxPatch((t - 0.26, lo), 0.52, hi - lo, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=CORAL, ec="none", alpha=0.3 if faded else 0.9, zorder=2))
    if t_max > T:
        ax.axvline(T, color=MUTED, lw=1, ls=(0, (4, 4)), zorder=1.5)
    ax.set_xticks(range(0, t_max + 1, 3))
    ax.set_yticks(range(0, 8))
    ax.set_xlabel("time step")
    ax.set_ylabel("position")


def path_xy(t0, x0, positions, offset=0.0):
    ts = [t0] + [t0 + i + 1 for i in range(len(positions))]
    xs = [x0] + [p + offset for p in positions]
    return ts, xs


def draw_step(ax, step, show_all=False):
    """Shots of one control step: verified in green, rejected in red, the chosen plan in gold."""
    t0, x0 = step["t"], step["x"]
    n = len(step["paths"])
    for i, (path, ok) in enumerate(zip(step["paths"], step["ok"])):
        ts, xs = path_xy(t0, x0, path, (i - (n - 1) / 2) * 0.05)
        ax.plot(ts, xs, color=GREEN if ok else RED, lw=1.6, alpha=0.75 if ok else 0.6,
                ls="-" if ok else (0, (3, 2)), zorder=4, solid_capstyle="round")
    if step["chosen"]:
        ts, xs = path_xy(t0, x0, step["chosen"])
        ax.plot(ts, xs, color=GOLD, lw=3.4, zorder=5, solid_capstyle="round",
                ls="-" if step["source"] == "sampled" else (0, (2, 1.5)))


def draw_executed(ax, trajectory, upto=None, color=INK, label=None, **kw):
    upto = len(trajectory) - 1 if upto is None else upto
    ax.plot(range(upto + 1), trajectory[:upto + 1], color=color, lw=2.4, zorder=6, solid_capstyle="round",
            label=label, **kw)
    ax.scatter([upto], [trajectory[upto]], s=70, color="white", edgecolors=color, linewidths=2.2, zorder=7)


def bar_plans(ax, values, feasible, ylabel, ylim=None):
    """Bar chart over the 2^N plans, feasible plans highlighted."""
    n = len(values)
    ax.bar(range(n), values, width=0.75, color=[BLUE if f else WALL for f in feasible], zorder=3)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=2)
    ax.set_xlim(-1, n)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xticks([0, n // 4, n // 2, 3 * n // 4, n - 1])
    ax.set_xlabel("plan (codeword as an integer)")
    ax.set_ylabel(ylabel)
    ax.xaxis.grid(False)
