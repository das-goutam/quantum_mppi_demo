"""Render the explainer video of the quantum-sampled MPPI loop on the corridor task.

    python video/make_video.py                  # full render -> video/quantum_mppi_corridor.mp4
    python video/make_video.py --preview        # a few still frames -> video/preview/

Needs ffmpeg on the path. The episode shown is a real run of the code in this repository:
every trajectory in the video is a measured shot of the 12-qubit circuit (ideal simulator).
"""
import argparse
import os
import subprocess
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from episode import BUDGET, record_episode   # noqa: E402

SEED = 5
W, H, FPS = 1920, 1080, 30
OUT = os.path.join(HERE, "quantum_mppi_corridor.mp4")

# ---------------------------------------------------------------- palette
BG, PANEL, GRID = "#0B1020", "#111936", "#232E52"
TEXT, MUTED = "#E8ECF6", "#8C98B8"
CYAN, GREEN, RED, GOLD, VIOLET, CORAL = "#4CC9F0", "#5BE49B", "#FF5C7A", "#FFD166", "#B79CFF", "#F0725C"
WALL = "#1B2444"

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "text.color": TEXT, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": BG,
    "axes.facecolor": BG, "savefig.facecolor": BG, "mathtext.fontset": "dejavusans",
})

STAGES = ["Superpose", "Mark", "Amplify", "Measure", "Verify", "Update"]
# circuit boxes in pixels of the panel: (label, x0, x1, row_lo, row_hi, stage that lights it)
BOXES = [("H", 92, 128, 0, 0, 0), ("ROLLOUT", 142, 232, 0, 2, 1), ("FLIP", 246, 300, 2, 2, 1),
         ("ROLLOUT$^{-1}$", 314, 420, 0, 2, 1), ("REFLECT", 434, 522, 0, 0, 2)]
ROWS = [62, 118, 174]


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def seg(u, a, b):
    """Progress of u through the interval [a, b], eased."""
    return ease((u - a) / (b - a))


def lerp(a, b, w):
    return a + (b - a) * w


# ---------------------------------------------------------------- drawing helpers
def glow_line(ax, xs, ys, color, lw, alpha=1.0, zorder=3, ls="-"):
    if alpha <= 0.01 or len(xs) < 2:
        return
    ax.plot(xs, ys, color=color, lw=lw * 3.2, alpha=0.13 * alpha, solid_capstyle="round", zorder=zorder - 0.1)
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, solid_capstyle="round", solid_joinstyle="round",
            zorder=zorder, ls=ls)


def partial_path(t0, x0, positions, frac, offset=0.0):
    """Points of a path from (t0, x0) through the positions, cut at a fraction of its length."""
    pts = [(t0, x0)] + [(t0 + i + 1, positions[i] + offset) for i in range(len(positions))]
    if len(pts) > 1:
        pts[1] = (pts[1][0], positions[0] + offset * 0.6)
    n = len(positions) * min(max(frac, 0.0), 1.0)
    k = int(np.floor(n))
    out = pts[:k + 1]
    if k < len(positions) and n - k > 1e-6:
        (ta, xa), (tb, xb) = pts[k], pts[k + 1]
        out.append((lerp(ta, tb, n - k), lerp(xa, xb, n - k)))
    return [q[0] for q in out], [q[1] for q in out]


def draw_corridor(ax, task, now=None, window=None, obstacle_alpha=1.0, reveal=1.0, legend=True):
    T = task.episode_steps
    ax.set_xlim(-0.8, T + 5.3)
    ax.set_ylim(-0.75, 7.75)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, labelsize=15, pad=8)
    ax.set_xticks(range(0, T + 1, 3))
    ax.set_yticks(range(0, 8))
    ax.set_xlabel("time step", fontsize=17, labelpad=10)
    ax.set_ylabel("position", fontsize=17, labelpad=10)
    # lattice
    for t in range(0, T + 6):
        ax.plot([t, t], [0.5, 6.5], color=GRID, lw=0.8, alpha=0.55 if t <= T else 0.25, zorder=0.5)
    for x in range(1, 7):
        ax.plot([-0.5, T + 5.1], [x, x], color=GRID, lw=0.8, alpha=0.45, zorder=0.5)
    # walls
    for y0 in (-0.75, 6.5):
        ax.add_patch(Rectangle((-0.8, y0), T + 6.1, 1.25, color=WALL, zorder=1, lw=0))
    ax.text(T + 5.05, 7.12, "wall", color=MUTED, fontsize=13, ha="right", va="center", zorder=2)
    ax.text(T + 5.05, -0.12, "wall", color=MUTED, fontsize=13, ha="right", va="center", zorder=2)
    if legend:
        items = [("measured shot", CYAN), ("verified", GREEN), ("rejected", RED), ("chosen plan", GOLD),
                 ("executed", "white")]
        x = -0.35
        for name, color in items:
            ax.plot([x, x + 0.7], [-0.12, -0.12], color=color, lw=3.2, solid_capstyle="round", zorder=3)
            ax.text(x + 0.95, -0.12, name, color=MUTED, fontsize=13, va="center", zorder=3)
            x += 1.75 + 0.27 * len(name)
    # beyond the episode
    ax.add_patch(Rectangle((T + 0.5, 0.5), 4.8, 6.0, color=BG, alpha=0.55, zorder=1.2, lw=0))
    ax.plot([T, T], [0.5, 6.5], color=MUTED, lw=1.2, ls=(0, (4, 4)), alpha=0.8, zorder=1.3)
    ax.text(T, 6.72, "end", color=MUTED, fontsize=13, ha="center", va="bottom", zorder=2)
    # planning window
    if window is not None and window[1] > 0.01:
        ax.add_patch(Rectangle((window[0], 0.5), 5, 6.0, color=VIOLET, alpha=0.07 * window[1], zorder=1.1, lw=0))
    # obstacles
    for t, blocked in sorted(task.obstacles.items()):
        grow = ease((reveal * 6 - list(sorted(task.obstacles)).index(t) * 0.7))
        if grow <= 0.01:
            continue
        lo, hi = min(blocked) - 0.5, max(blocked) + 0.5
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * grow
        active = 1.0 if now is None else (1.0 if t >= now - 0.3 else 0.35)
        ax.add_patch(FancyBboxPatch((t - 0.27, mid - half + 0.08), 0.54, 2 * half - 0.16,
                                    boxstyle="round,pad=0.0,rounding_size=0.12", fc=CORAL, ec="none",
                                    alpha=0.85 * obstacle_alpha * active, zorder=2))
        ax.add_patch(FancyBboxPatch((t - 0.27, mid - half + 0.08), 0.54, 2 * half - 0.16,
                                    boxstyle="round,pad=0.06,rounding_size=0.16", fc=CORAL, ec="none",
                                    alpha=0.16 * obstacle_alpha * active, zorder=1.9))


def draw_agent(ax, t, x, alpha=1.0):
    for r, a in ((0.42, 0.10), (0.30, 0.18), (0.20, 1.0)):
        ax.add_patch(Circle((t, x), r, color="white", alpha=a * alpha, zorder=9, lw=0))
    ax.add_patch(Circle((t, x), 0.11, color=BG, alpha=alpha, zorder=9.1, lw=0))


def draw_circuit(ax, stage, glow=1.0, classical=0.0):
    ax.set_xlim(0, 672)
    ax.set_ylim(216, 0)
    ax.axis("off")
    for y, label in zip(ROWS, ["plan  ×5", "position  ×2", "flags  ×5"]):
        ax.plot([84, 578], [y, y], color=MUTED, lw=1.6, alpha=0.75, zorder=1)
        ax.text(76, y, label, ha="right", va="center", fontsize=12.5, color=MUTED)
    for label, x0, x1, r0, r1, lit in BOXES:
        on = stage is not None and lit == stage
        done = stage is not None and lit < stage
        fc = VIOLET if on else ("#2A3566" if done else "#1A2347")
        ec = VIOLET if (on or done) else "#3A4780"
        y0, y1 = ROWS[r0] - 21, ROWS[r1] + 21
        if on:
            ax.add_patch(FancyBboxPatch((x0 - 7, y0 - 7), x1 - x0 + 14, y1 - y0 + 14,
                                        boxstyle="round,pad=0.0,rounding_size=11", fc=VIOLET, ec="none",
                                        alpha=0.22 * glow, zorder=2))
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.0,rounding_size=7",
                                    fc=fc, ec=ec, lw=1.6, zorder=3))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, ha="center", va="center", fontsize=12,
                color=BG if on else TEXT, fontweight="bold", zorder=4)
    # meter
    on, done = stage == 3, stage is not None and stage > 3
    x0, x1, y0, y1 = 536, 578, ROWS[0] - 21, ROWS[0] + 21
    if on:
        ax.add_patch(FancyBboxPatch((x0 - 7, y0 - 7), x1 - x0 + 14, y1 - y0 + 14,
                                    boxstyle="round,pad=0.0,rounding_size=11", fc=VIOLET, ec="none",
                                    alpha=0.22 * glow, zorder=2))
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.0,rounding_size=7",
                                fc=VIOLET if on else "#1A2347", ec=VIOLET if (on or done) else "#3A4780", lw=1.6,
                                zorder=3))
    ang = np.linspace(np.pi, 0, 30)
    ink = BG if on else TEXT
    ax.plot(557 + 12 * np.cos(ang), ROWS[0] + 9 - 12 * np.sin(ang), color=ink, lw=1.8, zorder=4)
    ax.plot([557, 567], [ROWS[0] + 9, ROWS[0] - 9], color=ink, lw=1.8, zorder=4)
    # classical part
    on = classical > 0.5
    for dy in (-3, 3):
        ax.plot([578, 596], [ROWS[0] + dy, ROWS[0] + dy], color=GOLD if on else MUTED, lw=1.4, zorder=1)
    ax.add_patch(FancyBboxPatch((596, ROWS[0] - 21), 72, ROWS[2] - ROWS[0] + 42,
                                boxstyle="round,pad=0.0,rounding_size=7", fc=GOLD if on else "#1A2347",
                                ec=GOLD if on else "#3A4780", lw=1.6, zorder=3))
    ax.text(632, ROWS[1], "classical\ncheck\n+ MPPI", ha="center", va="center", fontsize=10.8,
            color=BG if on else TEXT, fontweight="bold", zorder=4, linespacing=1.55)
    ax.text(84, 16, "12-qubit circuit, simulated gate by gate", ha="left", va="center", fontsize=13.5,
            color=MUTED)


def draw_bars(ax, values, feasible, mode, ylim, mean=None, dots=None, dot_ok=None, title=None, note=None):
    n = len(values)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-1, n)
    ax.set_ylim(*ylim)
    ax.tick_params(length=0, labelsize=12.5)
    ax.set_xticks([])
    ax.yaxis.grid(True, color=GRID, lw=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    colors = [CYAN if f else "#4A567F" for f in feasible]
    ax.bar(range(n), values, width=0.74, color=colors, zorder=3, lw=0)
    ax.axhline(0, color=MUTED, lw=1.0, alpha=0.8, zorder=2)
    if mean is not None:
        ax.axhline(mean, color=GOLD, lw=1.4, ls=(0, (5, 4)), zorder=4, alpha=0.9)
        ax.text(-0.5, mean + 0.012 * (ylim[1] - ylim[0]), "mean", color=GOLD, fontsize=12.5, ha="left",
                va="bottom", zorder=5)
    if dots:
        step = (ylim[1] - ylim[0]) * 0.062
        seen = {}
        for z, ok, a in dots:
            k = seen.get(z, 0)
            seen[z] = k + 1
            color = GOLD if ok is None else (GREEN if ok else RED)
            ax.scatter([z], [max(values[z], 0) + step * (k + 0.75)], s=95, color=color, alpha=a, zorder=6,
                       edgecolors=BG, linewidths=1.0)
    ax.set_ylabel(mode, fontsize=14, labelpad=6)
    ax.set_xlabel("the 32 candidate plans (5-bit codewords)", fontsize=13.5, labelpad=8)
    if title:
        ax.set_title(title, fontsize=15, color=TEXT, loc="left", pad=10, fontweight="bold")
    if note:
        ax.text(1.0, 1.045, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=15, color=CYAN,
                fontweight="bold")


def draw_chips(fig, active, x0=0.045, y=0.092, done_upto=None):
    w, gap = 0.0875, 0.008
    for i, name in enumerate(STAGES):
        on = active == i
        done = active is not None and i < active
        fc = VIOLET if on else ("#2A3566" if done else "#141C3A")
        fig.patches.append(FancyBboxPatch((x0 + i * (w + gap), y), w, 0.038, transform=fig.transFigure,
                                          boxstyle="round,pad=0.0,rounding_size=0.008", fc=fc,
                                          ec=VIOLET if (on or done) else "#2B3763", lw=1.2))
        fig.text(x0 + i * (w + gap) + w / 2, y + 0.019, "%d  %s" % (i + 1, name), ha="center", va="center",
                 fontsize=14, color=BG if on else (TEXT if done else MUTED), fontweight="bold")


def draw_stats(fig, rows, x=0.652, y=0.215):
    for i, (label, value, color) in enumerate(rows):
        fig.text(x + i * 0.086, y, label, fontsize=11.5, color=MUTED, va="bottom", fontweight="bold")
        fig.text(x + i * 0.086, y - 0.04, value, fontsize=22, color=color, va="bottom", fontweight="bold")


def base_layout(fig, header_right=None):
    fig.text(0.045, 0.925, "MPPI with a quantum sampler", fontsize=30, fontweight="bold", va="center")
    fig.text(0.045, 0.882, "corridor navigation on 12 simulated qubits", fontsize=18, color=MUTED, va="center")
    if header_right:
        fig.text(0.975, 0.925, header_right, fontsize=20, color=TEXT, ha="right", va="center", fontweight="bold")
    ax = fig.add_axes([0.045, 0.235, 0.565, 0.60])
    axc = fig.add_axes([0.635, 0.655, 0.35, 0.20])
    axb = fig.add_axes([0.682, 0.345, 0.295, 0.235])
    return ax, axc, axb


def caption(fig, title, body, alpha=1.0, color=TEXT):
    fig.text(0.045, 0.108, title, fontsize=24, fontweight="bold", color=color, alpha=alpha, va="center")
    fig.text(0.045, 0.045, body, fontsize=18.5, color=MUTED, alpha=alpha, va="center")


def fade_overlay(fig, alpha):
    if alpha > 0.003:
        fig.patches.append(Rectangle((0, 0), 1, 1, transform=fig.transFigure, color=BG, alpha=min(alpha, 1.0),
                                     zorder=100))


# ---------------------------------------------------------------- planning layer
def draw_planning(ax, s, ghost=0.0, mark=0.0, shots=0.0, verify=0.0, choose=0.0, fade=1.0, weights=0.0):
    """Trajectories of one control step.

    ghost   all 32 plans, faint          mark    feasible plans highlighted among them
    shots   number of measured shots drawn so far (fractional = the next one is being drawn)
    verify  number of shots already checked      choose  highlight of the chosen plan
    """
    t0, x0 = s["t"], s["x"]
    if ghost > 0.01:
        for z, path in enumerate(s["all_paths"]):
            feasible = bool(s["feasible"][z])
            a = ghost * fade * (0.20 + 0.55 * mark if feasible else 0.20 * (1 - 0.7 * mark))
            xs, ys = partial_path(t0, x0, path, 1.0)
            ax.plot(xs, ys, color=CYAN if (feasible and mark > 0.3) else "#7C8AB8", lw=1.3 + 1.0 * mark * feasible,
                    alpha=a, zorder=2.5, solid_capstyle="round")
    n = len(s["shots"])
    wmax = max(s["weight"]) if max(s["weight"]) > 0 else 1.0
    for i in range(n):
        frac = min(max(shots - i, 0.0), 1.0)
        if frac <= 0:
            continue
        ok = s["ok"][i]
        checked = min(max(verify - i, 0.0), 1.0)
        offset = (i - (n - 1) / 2) * 0.05
        xs, ys = partial_path(t0, x0, s["paths"][i], frac, offset)
        if checked <= 0:
            color, alpha, lw = CYAN, 0.85, 2.3
        elif ok:
            color, alpha = GREEN, 0.9
            lw = 2.3 + weights * 2.6 * s["weight"][i] / wmax
        else:
            color, alpha, lw = RED, 0.9 * (1 - ease((verify - i - 1.0) / 1.6)), 2.3
        alpha *= fade * (1 - 0.72 * choose)
        glow_line(ax, xs, ys, color, lw, alpha, zorder=4)
        if checked > 0 and not ok and alpha > 0.05:
            # mark the first violation
            for k, pos in enumerate(s["paths"][i]):
                if not (1 <= pos <= 6) or pos in ax._blocked.get(t0 + k + 1, ()):
                    ax.scatter([t0 + k + 1], [pos + offset], marker="x", s=170, color=RED, lw=3, alpha=alpha,
                               zorder=6)
                    break
    if choose > 0.01 and s["chosen"]:
        xs, ys = partial_path(t0, x0, s["chosen"], 1.0)
        dashed = s["source"] != "sampled"
        glow_line(ax, xs, ys, GOLD, 4.6, choose * fade, zorder=7, ls=(0, (2.2, 1.6)) if dashed else "-")


def executed_path(ax, traj, upto):
    """White path of the moves already executed, up to a fractional time."""
    k = int(np.floor(upto))
    xs, ys = list(range(k + 1)), list(traj[:k + 1])
    if upto - k > 1e-6 and k + 1 < len(traj):
        xs.append(upto)
        ys.append(lerp(traj[k], traj[k + 1], upto - k))
    glow_line(ax, xs, ys, "white", 3.6, 0.95, zorder=8)
    return xs[-1], ys[-1]


# ---------------------------------------------------------------- scenes
class Movie:
    def __init__(self):
        self.task, self.steps, self.completed = record_episode(SEED)
        self.traj = [s["x"] for s in self.steps] + [self.steps[-1]["x"] + self.steps[-1]["move"]]
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        self.scenes = [("title", 3.6), ("task", 7.0), ("walkthrough", 27.0), ("loop", 17 * 1.75), ("outro", 8.5)]

    def new_frame(self):
        self.fig.clear()
        self.fig.patches.clear()
        self.fig.texts.clear()
        return self.fig

    def corridor(self, ax, **kw):
        draw_corridor(ax, self.task, **kw)
        ax._blocked = self.task.obstacles

    # --- scene 1
    def title(self, u, dur):
        fig = self.new_frame()
        a = seg(u, 0.0, 0.25)
        fig.text(0.5, 0.60, "MPPI with a quantum sampler", fontsize=62, fontweight="bold", ha="center", alpha=a)
        fig.text(0.5, 0.49, "A 12-qubit circuit proposes the trajectories.  A classical rollout verifies them.",
                 fontsize=25, color=MUTED, ha="center", alpha=seg(u, 0.15, 0.45))
        fig.text(0.5, 0.36, "corridor navigation   ·   simulated gate by gate with Qiskit Aer", fontsize=20,
                 color=VIOLET, ha="center", alpha=seg(u, 0.3, 0.6), fontweight="bold")
        fade_overlay(fig, 1 - seg(u, 0.0, 0.12) + seg(u, 0.9, 1.0))

    # --- scene 2
    def task_scene(self, u, dur):
        fig = self.new_frame()
        ax, axc, axb = base_layout(fig)
        axc.axis("off"); axb.axis("off")
        self.corridor(ax, reveal=seg(u, 0.18, 0.6), legend=False)
        draw_agent(ax, 0, self.task.x_start, alpha=seg(u, 0.05, 0.2))
        ax.text(0, self.task.x_start - 0.62, "start", fontsize=15, color=TEXT, ha="center", va="center",
                alpha=seg(u, 0.1, 0.25) * (1 - seg(u, 0.5, 0.58)), zorder=10)
        lines = [
            ("The task", TEXT, 26, True),
            ("", TEXT, 10, False),
            ("A point moves up or down by one", MUTED, 19, False),
            ("position at every time step.", MUTED, 19, False),
            ("", TEXT, 10, False),
            ("It must stay off the walls and off", MUTED, 19, False),
            ("the obstacles, which block half of", MUTED, 19, False),
            ("the corridor at fixed times.", MUTED, 19, False),
            ("", TEXT, 10, False),
            ("The cost prefers position 2, exactly", MUTED, 19, False),
            ("where two of the obstacles appear.", MUTED, 19, False),
            ("", TEXT, 10, False),
            ("Every step the controller plans", CYAN, 19, True),
            ("5 moves ahead: 32 candidate plans.", CYAN, 19, True),
        ]
        y = 0.80
        for i, (text, color, size, bold) in enumerate(lines):
            fig.text(0.652, y, text, fontsize=size, color=color, fontweight="bold" if bold else "normal",
                     alpha=seg(u, 0.05 + 0.035 * i, 0.2 + 0.035 * i), va="center")
            y -= (0.058 if size > 22 else 0.040) if size > 12 else 0.022
        if u > 0.55:
            s = self.steps[0]
            draw_planning(ax, s, ghost=seg(u, 0.55, 0.8))
        caption(fig, "Hard constraints make most sampled plans useless",
                "Classical MPPI draws plans at random and throws away the infeasible ones. Here only 5 of the first 32 plans survive.",
                alpha=seg(u, 0.5, 0.7))
        fade_overlay(fig, 1 - seg(u, 0.0, 0.08) + seg(u, 0.94, 1.0))

    # --- scene 3
    def walkthrough(self, u, dur):
        s = self.steps[0]
        t = u * dur
        cuts = np.cumsum([0, 4.0, 5.0, 5.0, 4.5, 4.5, 4.0])     # six stages
        stage = int(np.searchsorted(cuts, t, side="right") - 1)
        stage = min(stage, 5)
        v = (t - cuts[stage]) / (cuts[stage + 1] - cuts[stage])
        fig = self.new_frame()
        ax, axc, axb = base_layout(fig, "control step 1 of %d" % self.task.episode_steps)
        self.corridor(ax, now=0, window=(0, 1.0))
        D, M = 32, s["M"]
        a0 = 1 / np.sqrt(D)
        feas = s["feasible"]
        amp_f, amp_i = np.sin(3 * s["theta"]) / np.sqrt(M), np.cos(3 * s["theta"]) / np.sqrt(D - M)
        mean = (D - 2 * M) * a0 / D
        n = len(s["shots"])
        ylim = (-0.24, 0.50)
        ghost, mark, shots, verify, choose, weights, fade = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
        dots, classical, agent_t = None, 0.0, 0.0
        if stage == 0:
            ghost = seg(v, 0.0, 0.5)
            vals = np.full(D, a0) * seg(v, 0.1, 0.6)
            draw_bars(axb, vals, feas, "amplitude", ylim, title="State of the 5 plan qubits")
            cap = ("1   Superpose", "Five Hadamard gates put all 32 plans into one quantum state, each with the same amplitude.")
        elif stage == 1:
            mark = seg(v, 0.15, 0.6)
            vals = np.where(feas, a0 * (1 - 2 * seg(v, 0.45, 0.85)), a0)
            draw_bars(axb, vals, feas, "amplitude", ylim, title="State of the 5 plan qubits",
                      note="%d feasible plans" % M if v > 0.3 else None)
            cap = ("2   Mark the feasible plans", "The circuit rolls out all plans at once and raises a flag at every violation. Plans with no flag get a sign flip.")
        elif stage == 2:
            mark = 1.0
            w = seg(v, 0.25, 0.75)
            vals = np.where(feas, lerp(-a0, amp_f, w), lerp(a0, amp_i, w))
            pf = float((vals[feas] ** 2).sum())
            draw_bars(axb, vals, feas, "amplitude", ylim, mean=mean if v < 0.9 else None,
                      title="State of the 5 plan qubits", note="P(feasible) = %2.0f %%" % (100 * min(pf, 1)))
            cap = ("3   Amplify", "A reflection about the mean turns the sign flip into a large amplitude. One Grover iteration: 16 %  to  88 %.")
        else:
            mark = 1.0
            ghost = 1 - 0.75 * seg(v, 0.0, 0.3) if stage == 3 else 0.25
            if stage == 3:
                shots = n * seg(v, 0.08, 0.92) if v < 0.92 else n
                shots = n * min(max((v - 0.08) / 0.84, 0), 1)
            else:
                shots = n
            if stage == 4:
                verify = (n + 1.8) * min(max((v - 0.08) / 0.8, 0), 1)
            elif stage == 5:
                verify = n + 3
            if stage == 5:
                weights = seg(v, 0.0, 0.25)
                choose = seg(v, 0.3, 0.55)
                agent_t = seg(v, 0.68, 0.95)
                fade = 1 - seg(v, 0.8, 1.0)
            classical = 1.0 if stage >= 4 else 0.0
            dots = []
            for i, z in enumerate(s["shots"]):
                if shots - i >= 0.85:
                    ok = None if verify - i <= 0 else s["ok"][i]
                    dots.append((z, ok, 1.0))
            vals = s["prob"]
            draw_bars(axb, vals, feas, "probability", (0, 0.30), dots=dots, title="Measurement outcomes",
                      note="%d shots" % min(int(shots + 0.15), n) if stage == 3 else None)
            if stage == 3:
                cap = ("4   Measure", "Each run of the circuit returns one plan. Eight shots give eight candidate trajectories.")
            elif stage == 4:
                cap = ("5   Verify classically", "A classical rollout checks every shot. Plans that hit a wall or an obstacle are discarded, whatever the device did.")
            else:
                cap = ("6   MPPI update", "The verified plans are averaged with the weights exp(-cost / λ). The controller applies the first move and keeps the best plan.")
        draw_planning(ax, s, ghost=ghost, mark=mark, shots=shots, verify=verify, choose=choose, fade=fade,
                      weights=weights)
        at, ax_ = executed_path(ax, self.traj, agent_t)
        draw_agent(ax, at, ax_)
        draw_circuit(axc, min(stage, 4), classical=classical)
        n_ok = sum(s["ok"])
        rows = [("FEASIBLE", "%d of 32" % M, CYAN),
                ("SHOTS", "%d" % min(int(shots + 0.15), n) if stage >= 3 else "–", TEXT),
                ("VERIFIED", "%d" % sum(1 for i in range(n) if s["ok"][i] and verify - i > 0) if stage >= 4 else "–", GREEN),
                ("MPPI UPDATE", "%+.2f" % s["nu_hat"] if stage == 5 and v > 0.2 else "–", GOLD)]
        draw_stats(fig, rows)
        draw_chips(fig, stage)
        caption(fig, "", cap[1])
        fade_overlay(fig, 1 - seg(u, 0.0, 0.02) + 0.0)

    # --- scene 4
    def loop(self, u, dur):
        per = dur / 17
        t = u * dur
        k = min(int(t / per), 16)
        v = (t - k * per) / per
        s = self.steps[k + 1]
        fig = self.new_frame()
        ax, axc, axb = base_layout(fig, "control step %d of %d" % (s["t"] + 1, self.task.episode_steps))
        self.corridor(ax, now=s["t"], window=(s["t"], 1 - seg(v, 0.85, 1.0)))
        n = len(s["shots"])
        feas = s["feasible"]
        amp = seg(v, 0.04, 0.2)
        vals = lerp(np.full(32, 1 / 32), s["prob"], amp)
        shots = n * min(max((v - 0.2) / 0.22, 0), 1)
        verify = (n + 1.6) * min(max((v - 0.44) / 0.2, 0), 1)
        choose = seg(v, 0.64, 0.74)
        move = seg(v, 0.8, 0.98)
        fade = 1 - seg(v, 0.86, 1.0)
        if v < 0.2:
            stage = 2
        elif v < 0.44:
            stage = 3
        elif v < 0.64:
            stage = 4
        else:
            stage = 5
        dots = []
        for i, z in enumerate(s["shots"]):
            if shots - i >= 0.85:
                dots.append((z, None if verify - i <= 0 else s["ok"][i], fade))
        draw_bars(axb, vals, feas, "probability", (0, 0.30), dots=dots, title="Measurement outcomes",
                  note="P(feasible) = %2.0f %%" % (100 * float(vals[feas].sum())))
        draw_planning(ax, s, ghost=0.22 * (1 - seg(v, 0.2, 0.4)), mark=1.0, shots=shots, verify=verify,
                      choose=choose, fade=fade, weights=seg(v, 0.6, 0.66))
        at, ax_ = executed_path(ax, self.traj, s["t"] + move)
        draw_agent(ax, at, ax_)
        draw_circuit(axc, min(stage, 4), classical=1.0 if stage >= 4 else 0.0)
        n_ok = sum(s["ok"])
        rows = [("FEASIBLE", "%d of 32" % s["M"], CYAN),
                ("SHOTS", "%d" % min(int(shots + 0.15), n), TEXT),
                ("VERIFIED", "%d" % sum(1 for i in range(n) if s["ok"][i] and verify - i > 0) if stage >= 4 else "–", GREEN),
                ("MPPI UPDATE", ("%+.2f" % s["nu_hat"] if s["nu_hat"] is not None else "none") if stage == 5 else "–", GOLD)]
        draw_stats(fig, rows)
        draw_chips(fig, stage)
        if n_ok == 0:
            body = "Almost every plan is feasible here, so one Grover iteration overshoots and no shot passes. The controller follows its stored verified plan."
        elif s["p"] > 0.5:
            body = "Most plans are feasible here, so one Grover iteration overshoots. Few shots pass, and the verified ones still decide the move."
        elif n_ok < n:
            body = "Shots that fail the classical check are discarded. They cost budget and never reach the average."
        else:
            body = "Every shot is a feasible plan. The same budget of classical samples would keep about %d of 24." % round(24 * s["p"])
        caption(fig, "", body)
        if k == 16:
            fade_overlay(fig, seg(u, 0.992, 1.0) * 0.0)

    # --- scene 5
    def outro(self, u, dur):
        fig = self.new_frame()
        ax, axc, axb = base_layout(fig, "episode complete")
        axc.axis("off"); axb.axis("off")
        self.corridor(ax, now=None)
        at, ax_ = executed_path(ax, self.traj, self.task.episode_steps)
        draw_agent(ax, at, ax_)
        total = sum(len(s["shots"]) for s in self.steps)
        kept = sum(sum(s["ok"]) for s in self.steps)
        cards = [("%d" % self.task.episode_steps, "control steps, no constraint violated", TEXT),
                 ("%d" % total, "circuit shots, %d passed the classical check" % kept, CYAN),
                 ("13.5 %  →  0.5 %", "episodes with a violation at a tight budget\nclassical sampling  →  quantum, gate error 0.001\n(6 rollouts per step, 200 episodes)", GOLD)]
        y = 0.80
        for i, (big, small, color) in enumerate(cards):
            a = seg(u, 0.04 + 0.08 * i, 0.14 + 0.08 * i)
            fig.text(0.652, y, big, fontsize=40, fontweight="bold", color=color, alpha=a, va="center")
            fig.text(0.652, y - 0.042, small, fontsize=16.5, color=MUTED, alpha=a, va="top", linespacing=1.5)
            y -= 0.17
        caption(fig, "The quantum device proposes.  The classical rollout decides.",
                "Simulation only. Cost is counted in rollout evaluations, and no quantum advantage is claimed.",
                alpha=seg(u, 0.3, 0.4))
        fade_overlay(fig, seg(u, 0.92, 1.0))

    def render(self, name, u, dur):
        {"title": self.title, "task": self.task_scene, "walkthrough": self.walkthrough, "loop": self.loop,
         "outro": self.outro}[name](u, dur)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())


def render_scene(index):
    movie = Movie()
    name, dur = movie.scenes[index]
    frames = int(round(dur * FPS))
    tmp = os.path.join(HERE, "scene_%d.part.mp4" % index)
    final = os.path.join(HERE, "scene_%d.mp4" % index)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", "%dx%d" % (W, H),
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-r", str(FPS), tmp]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(frames):
        proc.stdin.write(movie.render(name, f / frames, dur).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed on scene %d" % index)
    os.replace(tmp, final)
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--scenes", type=int, nargs="*", default=None)
    args = parser.parse_args()
    if args.preview:
        movie = Movie()
        out = os.path.join(HERE, "preview")
        os.makedirs(out, exist_ok=True)
        shots = [("title", 0.6), ("task", 0.9), ("walkthrough", 0.07), ("walkthrough", 0.27), ("walkthrough", 0.45),
                 ("walkthrough", 0.66), ("walkthrough", 0.82), ("walkthrough", 0.93), ("loop", 0.02), ("loop", 0.137),
                 ("loop", 0.905), ("loop", 0.985), ("outro", 0.7)]
        durs = dict(movie.scenes)
        for i, (name, u) in enumerate(shots):
            movie.render(name, u, durs[name])
            movie.fig.savefig(os.path.join(out, "%02d_%s.png" % (i, name)), dpi=100)
        print("preview frames in", out)
        return
    indices = list(range(5)) if args.scenes is None else args.scenes
    with Pool(2) as pool:
        pool.map(render_scene, indices)
    listing = os.path.join(HERE, "scenes.txt")
    with open(listing, "w") as fh:
        for i in range(5):
            fh.write("file 'scene_%d.mp4'\n" % i)
    tmp = OUT.replace(".mp4", ".part.mp4")
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", listing,
                           "-c", "copy", "-movflags", "+faststart", tmp])
    os.replace(tmp, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
