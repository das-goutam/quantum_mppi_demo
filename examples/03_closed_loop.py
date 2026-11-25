"""Step 3. One closed-loop episode, step by step, with quantum shots in the loop.

    python examples/03_closed_loop.py [--budget 24] [--eta 0.0] [--seed 5]

Writes results/closed_loop_episode.json, figures/closed_loop.png and figures/episode_storyboard.png.
"""
import argparse

import _path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qmppi import ClassicalSampler, CorridorTask, QuantumSampler, exact_closed_loop, run_episode
from qmppi import plotting as P

parser = argparse.ArgumentParser()
parser.add_argument("--budget", type=int, default=24, help="rollout evaluations per control step")
parser.add_argument("--eta", type=float, default=0.0, help="gate error of the simulated device, 0 for ideal")
parser.add_argument("--seed", type=int, default=5)
args = parser.parse_args()

task = CorridorTask()
runs = {}
for label, sampler in (("classical", ClassicalSampler()),
                       ("quantum", QuantumSampler(depth=1, eta=args.eta, seed=args.seed, batch=64))):
    print("\n=== %s sampler (%s), budget %d rollouts per step ===" % (label, sampler.name, args.budget))
    runs[label] = run_episode(task, sampler, args.budget, np.random.default_rng(args.seed), verbose=True)
    out = runs[label]
    if out["failed"]:
        print("--> constraint violated at time %d" % out["failed_at"])
    else:
        print("--> episode completed, cost %.1f, %d rollout evaluations" % (out["cost"], out["rollouts"]))

summary = dict(budget=args.budget, eta=args.eta, seed=args.seed, exact_trajectory=exact_closed_loop(task))
for label, out in runs.items():
    summary[label] = dict(trajectory=out["trajectory"], failed=out["failed"], cost=out["cost"],
                          rollouts=out["rollouts"],
                          steps=[{k: s[k] for k in ("t", "x", "feasible_mass", "proposed", "accepted", "nu_hat",
                                                    "nu_exact", "move", "source")} for s in out["steps"]])
print("\nSaved %s" % _path.save_json("closed_loop_episode.json", summary))

# --- figure 1: the trajectories ----------------------------------------------------------------
P.use_style()
T = task.episode_steps
fig, ax = plt.subplots(figsize=(8.6, 3.5))
P.draw_corridor(ax, task)
ax.plot(exact_closed_loop(task), color=P.MUTED, lw=1.4, ls=(0, (2, 2)), label="exact update", zorder=5)
for label, color in (("classical", P.INK), ("quantum", P.BLUE)):
    traj = runs[label]["trajectory"]
    P.draw_executed(ax, traj, color=color, label="%s sampler%s" % (label, ", violation" if runs[label]["failed"] else ""))
ax.set_title("Closed-loop MPPI, %d rollout evaluations per control step" % args.budget)
ax.legend(fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
fig.savefig(_path.figure_path("closed_loop.png"))

# --- figure 2: what the quantum sampler proposed at six control steps --------------------------
quantum = runs["quantum"]
picks = [k for k in (0, 2, 5, 9, 12, 16) if k < len(quantum["steps"])]
fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.0), sharex=True, sharey=True)
for ax, k in zip(axes.ravel(), picks):
    step = quantum["steps"][k]
    P.draw_corridor(ax, task, t_max=T + 4, now=step["t"])
    P.draw_step(ax, step)
    P.draw_executed(ax, quantum["trajectory"], upto=step["t"])
    note = "%d of %d shots verified" % (step["accepted"], step["proposed"])
    if step["accepted"] == 0:
        note += ", stored plan used"
    ax.set_title("control step %d    %s" % (step["t"] + 1, note), fontsize=10)
    if ax not in axes[:, 0]:
        ax.set_ylabel("")
    if ax not in axes[1]:
        ax.set_xlabel("")
handles = [plt.Line2D([], [], color=P.GREEN, lw=2, label="verified shot"),
           plt.Line2D([], [], color=P.RED, lw=2, ls=(0, (3, 2)), label="rejected shot"),
           plt.Line2D([], [], color=P.GOLD, lw=3.4, label="chosen plan"),
           plt.Line2D([], [], color=P.INK, lw=2.4, label="executed")]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout()
fig.savefig(_path.figure_path("episode_storyboard.png"))
print("Saved figures/closed_loop.png and figures/episode_storyboard.png")
