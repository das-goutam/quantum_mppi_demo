"""Step 5. How much gate noise can the sampler take before it stops paying off?

    python examples/05_noise_study.py
    python examples/05_noise_study.py --plot-only   # redraw the figure from results/noise_study.json

For one planning problem, the script measures the cost of a verified sample (in rollout
evaluations) at several gate error rates and compares it with classical rejection sampling.
Writes results/noise_study.json and figures/noise_study.png.
"""
import json
import os
import sys

import _path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qmppi import CorridorTask, PlanningProblem, QuantumSampler
from qmppi import plotting as P

task = CorridorTask()
problem = PlanningProblem(task, x=task.x_start, t=0)
p, M, D = problem.feasible_mass, problem.n_feasible, problem.n_codewords
theta = np.arcsin(np.sqrt(p))
feasible_codewords = np.flatnonzero(problem.feasible_mask)
etas = [0.0, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
shots = 3000

if "--plot-only" in sys.argv:
    out = json.load(open(os.path.join(_path.RESULTS, "noise_study.json")))
else:
    print("Problem: x = %d, t = %d, p = %.4f. Classical rejection needs 1/p = %.2f rollouts per feasible sample." % (
        problem.x, problem.t, p, 1 / p))
    print("\n%d shots per entry. Cost = (2k + 1) / acceptance, in rollouts per verified sample." % shots)
    print("  depth   gate error   acceptance   cost    gain over classical   spread of the verified samples")
    out = dict(p=p, shots=shots, classical_cost=1 / p, etas=etas, depths={})
    for depth in (1, 2):
        rows = dict(acceptance=[], cost=[], gain=[], max_deviation_sigma=[])
        for eta in etas:
            sampler = QuantumSampler(depth=depth, eta=eta, seed=100 + depth)
            outcomes = sampler.shots(problem, shots)
            counts = np.array([(outcomes == z).sum() for z in feasible_codewords])
            K = int(counts.sum())
            acceptance = K / shots
            cost = (2 * depth + 1) / acceptance
            # largest deviation of a count from the uniform expectation, in standard deviations
            sigma = np.sqrt(K / M * (1 - 1 / M))
            deviation = float(np.abs(counts - K / M).max() / sigma)
            rows["acceptance"].append(acceptance)
            rows["cost"].append(cost)
            rows["gain"].append((1 / p) / cost)
            rows["max_deviation_sigma"].append(deviation)
            print("  %d       %-10g   %.3f        %5.2f   %5.2f                 largest deviation %.1f sigma" % (
                depth, eta, acceptance, cost, (1 / p) / cost, deviation))
        out["depths"][str(depth)] = rows
    print("\nA gain above 1 means that the noisy circuit still saves rollout evaluations.")
    print("Saved %s" % _path.save_json("noise_study.json", out))


P.use_style()
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
xs = range(len(etas))
labels = ["0" if e == 0 else "%g" % e for e in etas]
for depth, color in ((1, P.BLUE), (2, P.VIOLET)):
    rows = out["depths"][str(depth)]
    ideal = np.sin((2 * depth + 1) * theta) ** 2
    axes[0].plot(xs, rows["acceptance"], color=color, marker="o", lw=1.8, label="depth %d" % depth)
    axes[0].axhline(ideal, color=color, lw=0.9, ls=(0, (4, 3)))
    axes[1].plot(xs, rows["gain"], color=color, marker="o", lw=1.8, label="depth %d" % depth)
axes[0].axhline(p, color=P.INK, lw=1.2)
axes[0].text(len(etas) - 1, p + 0.02, "classical sampling, p", ha="right", fontsize=8.5)
axes[0].set_title("Shots that pass the classical check")
axes[0].set_ylabel("acceptance")
axes[0].set_ylim(0, 1)
axes[1].axhline(1, color=P.INK, lw=1.2, label="break-even with classical sampling")
axes[1].set_title("Saving in rollout evaluations")
axes[1].set_ylabel("gain over classical sampling")
axes[1].set_ylim(0, 2.2)
for ax in axes:
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_xlabel("gate error of the simulated device")
axes[0].legend(fontsize=8.5, loc="center left")
axes[1].legend(fontsize=8.5, loc="lower left")
fig.tight_layout()
fig.savefig(_path.figure_path("noise_study.png"))
print("Saved figures/noise_study.png")
