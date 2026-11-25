"""Step 4. Many episodes. How often does the loop violate a constraint at a given budget?

    python examples/04_benchmark.py            # 300 episodes, five budgets, about 15 minutes
    python examples/04_benchmark.py --quick    # 60 episodes, three budgets, about 2 minutes
    python examples/04_benchmark.py --plot-only   # redraw the figure from results/benchmark.json

Every sampler gets the same number of rollout evaluations per control step.
Writes results/benchmark.json and figures/benchmark.png.
"""
import argparse
import json
import os
import time

import _path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qmppi import AdaptiveSampler, ClassicalSampler, CorridorTask, QuantumSampler, run_episode
from qmppi import plotting as P

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=300)
parser.add_argument("--budgets", type=int, nargs="+", default=[6, 9, 12, 18, 24])
parser.add_argument("--quick", action="store_true")
parser.add_argument("--plot-only", action="store_true")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
if args.quick:
    args.episodes, args.budgets = 60, [6, 12, 24]

task = CorridorTask()
samplers = [
    ("classical sampling", ClassicalSampler(), P.INK, "-", "o"),
    ("quantum, ideal device", QuantumSampler(depth=1, eta=0.0, seed=10), P.BLUE, "-", "s"),
    ("quantum, gate error 0.001", QuantumSampler(depth=1, eta=1e-3, seed=20), P.BLUE, "--", "s"),
    ("quantum, gate error 0.005", QuantumSampler(depth=1, eta=5e-3, seed=30), P.BLUE, ":", "s"),
    ("adaptive, ideal device", AdaptiveSampler(QuantumSampler(depth=1, eta=0.0, seed=40)), P.GOLD, "-", "^"),
    ("adaptive, gate error 0.001", AdaptiveSampler(QuantumSampler(depth=1, eta=1e-3, seed=50)), P.GOLD, "--", "^"),
]

if args.plot_only:
    stored = json.load(open(os.path.join(_path.RESULTS, "benchmark.json")))
    args.episodes, args.budgets, results = stored["episodes"], stored["budgets"], stored["samplers"]
else:
    results, start = {}, time.time()
    for name, sampler, *_ in samplers:
        rows = dict(failed=[], failed_se=[], steps_without_sample=[], mean_cost=[])
        for budget in args.budgets:
            rng = np.random.default_rng(args.seed)
            episodes = [run_episode(task, sampler, budget, rng) for _ in range(args.episodes)]
            f = float(np.mean([e["failed"] for e in episodes]))
            steps = [s for e in episodes for s in e["steps"]]
            costs = [e["cost"] for e in episodes if not e["failed"]]
            rows["failed"].append(f)
            rows["failed_se"].append(float(np.sqrt(f * (1 - f) / args.episodes)))
            rows["steps_without_sample"].append(float(np.mean([s["accepted"] == 0 for s in steps])))
            rows["mean_cost"].append(float(np.mean(costs)) if costs else None)
        results[name] = rows
        print("finished %-28s (%.0f s)" % (name, time.time() - start), flush=True)

    shots = 0
    for _, sampler, *_ in samplers:
        q = getattr(sampler, "quantum", sampler)
        shots += getattr(q, "shots_simulated", 0)


    def table(key, scale, fmt):
        lines = ["%-28s" % "rollouts per control step" + "".join("%8d" % b for b in args.budgets)]
        for name, rows in results.items():
            lines.append("%-28s" % name + "".join(fmt % (scale * v) for v in rows[key]))
        return "\n".join(lines)


    print("\nEpisodes with a constraint violation, in percent (%d episodes per entry)" % args.episodes)
    print(table("failed", 100, "%8.1f"))
    print("\nControl steps without any verified sample, in percent")
    print(table("steps_without_sample", 100, "%8.1f"))
    print("\nMean cost of the episodes without violation (the exact update gives 10.5)")
    print(table("mean_cost", 1, "%8.2f"))
    print("\n%d circuit shots were simulated gate by gate." % shots)

    path = _path.save_json("benchmark.json", dict(episodes=args.episodes, budgets=args.budgets, seed=args.seed,
                                                  circuit_shots=shots, samplers=results))
    print("Saved %s" % path)


P.use_style()
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
floor = 0.5 / args.episodes
for name, _, color, ls, marker in samplers:
    rows = results[name]
    axes[0].plot(args.budgets, [max(r, floor) for r in rows["failed"]], color=color, ls=ls, marker=marker, ms=5,
                 lw=1.8, label=name)
    axes[1].plot(args.budgets, [100 * r for r in rows["steps_without_sample"]], color=color, ls=ls, marker=marker,
                 ms=5, lw=1.8)
axes[0].axhline(floor, color=P.MUTED, lw=0.8)
axes[0].text(args.budgets[-1], floor * 0.88, "points on this line: no violation observed", fontsize=8,
             color=P.MUTED, va="top", ha="right")
axes[0].set_yscale("log")
axes[0].set_ylim(floor * 0.55, 0.3)
axes[0].set_title("Episodes with a constraint violation")
axes[0].set_ylabel("fraction of episodes")
axes[1].set_title("Control steps without a verified sample")
axes[1].set_ylabel("percent of steps")
for ax in axes:
    ax.set_xscale("log")
    ax.set_xticks(args.budgets)
    ax.set_xticklabels([str(b) for b in args.budgets])
    ax.minorticks_off()
    ax.set_xlabel("rollout evaluations per control step")
fig.legend(loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
fig.savefig(_path.figure_path("benchmark.png"))
print("Saved figures/benchmark.png")
