"""Step 2. One MPPI update from quantum shots, next to the same update from classical samples.

    python examples/02_single_update.py

Writes results/single_update.json, figures/amplification.png and figures/shots.png.
"""
import _path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Statevector

from qmppi import (ClassicalSampler, CorridorTask, PlanningProblem, QuantumSampler, mppi_update, oracle_circuit,
                   preparation_circuit, reflection_circuit, sampler_circuit)
from qmppi import plotting as P

task = CorridorTask()
problem = PlanningProblem(task, x=task.x_start, t=0)
N, M, D = problem.N, problem.n_feasible, problem.n_codewords
p = problem.feasible_mass
theta = np.arcsin(np.sqrt(p))
nu = problem.exact_update()
feasible = problem.feasible_mask
out = dict(x=problem.x, t=problem.t, feasible_plans=M, plans=D, p=p, nu_exact=nu)

print("Problem: x = %d, t = %d, %d feasible plans of %d, p = %.4f" % (problem.x, problem.t, M, D, p))
print("Exact MPPI update (by enumeration): nu = %+.4f" % nu)


def plan_amplitudes(circuit):
    """Amplitudes of the 32 plans. The position and flag registers are back in their initial state."""
    data = Statevector(circuit).data.reshape(-1, D)[problem.x // 2]
    return np.real(data * np.sign(np.real(data).sum()))      # fix the global sign, which has no effect


# --- the three stages of one Grover iteration ------------------------------------------------
stage_1 = preparation_circuit(problem)
stage_2 = stage_1.compose(oracle_circuit(problem))
stage_3 = stage_2.compose(reflection_circuit(problem))
amplitudes = [plan_amplitudes(c) for c in (stage_1, stage_2, stage_3)]
print("\nAmplitude of a plan after each stage of the circuit")
print("  stage                     feasible plan   infeasible plan   P(feasible)")
for name, a in zip(("superposition", "after marking", "after reflection"), amplitudes):
    print("  %-24s  %+.4f         %+.4f           %.4f" % (
        name, a[feasible][0], a[~feasible][0], float((a[feasible] ** 2).sum())))

P.use_style()
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1), sharey=True)
titles = ["1  Superposition: all plans equal", "2  Marking: feasible plans change sign", "3  Reflection: feasible plans amplified"]
for ax, a, title in zip(axes, amplitudes, titles):
    P.bar_plans(ax, a, feasible, "amplitude" if ax is axes[0] else "", ylim=(-0.25, 0.5))
    ax.set_title(title)
    ax.text(0.03, 0.95, "P(feasible) = %2.0f %%" % (100 * (a[feasible] ** 2).sum()), transform=ax.transAxes,
            ha="left", va="top", color=P.BLUE, fontweight="bold")
fig.tight_layout()
fig.savefig(_path.figure_path("amplification.png"))

# --- what the circuit should do --------------------------------------------------------------
print("\nProbability that a shot returns a feasible plan")
print("  depth   theory sin^2((2k+1) theta)   statevector of the circuit")
out["acceptance_theory"], out["acceptance_statevector"] = [], []
for depth in (0, 1, 2):
    state = Statevector(sampler_circuit(problem, depth, measure=False))
    on_z = (np.abs(state.data) ** 2).reshape(-1, D).sum(axis=0)
    theory = float(np.sin((2 * depth + 1) * theta) ** 2)
    out["acceptance_theory"].append(theory)
    out["acceptance_statevector"].append(float(on_z[feasible].sum()))
    print("  %d       %.6f                     %.6f" % (depth, theory, on_z[feasible].sum()))

# --- shots, verification, uniformity ---------------------------------------------------------
shots = 2000
print("\n%d shots at depth 1, each one verified by a classical rollout" % shots)
print("  device              accepted   counts on the %d feasible plans (uniform is expected)" % M)
feasible_codewords = np.flatnonzero(feasible)
out["shots"] = shots
out["devices"] = {}
fig, ax = plt.subplots(figsize=(7.2, 3.2))
for i, (eta, color) in enumerate(((0.0, P.BLUE), (1e-3, P.VIOLET), (5e-3, P.GOLD))):
    label = "ideal device" if eta == 0 else "gate error %g" % eta
    sampler = QuantumSampler(depth=1, eta=eta, seed=1)
    outcomes = sampler.shots(problem, shots)
    counts = np.bincount(outcomes, minlength=D)
    on_feasible = [int(counts[z]) for z in feasible_codewords]
    out["devices"][label] = dict(accepted=sum(on_feasible) / shots, counts_on_feasible=on_feasible)
    print("  %-18s  %.3f      %s" % (label, sum(on_feasible) / shots, on_feasible))
    ax.bar(np.arange(D) + (i - 1) * 0.27, counts / shots, width=0.27, color=color, label=label, zorder=3)
ax.set_xlim(-1, D)
ax.set_xlabel("measured plan (codeword as an integer)")
ax.set_ylabel("fraction of shots")
ax.set_title("Measured plans, %d shots per device. The five tall groups are the feasible plans." % shots)
ax.xaxis.grid(False)
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(_path.figure_path("shots.png"))

# --- the update at a small budget --------------------------------------------------------------
budget, repeats = 12, 500
rng = np.random.default_rng(0)
print("\nMPPI update with a budget of %d rollout evaluations, %d repetitions" % (budget, repeats))
print("  sampler                          no feasible sample   mean |nu_hat - nu|")
samplers = [ClassicalSampler(), QuantumSampler(1, 0.0, seed=2), QuantumSampler(1, 1e-3, seed=3),
            QuantumSampler(1, 5e-3, seed=4)]
out["update"] = dict(budget=budget, repetitions=repeats, samplers={})
for sampler in samplers:
    errors, empty = [], 0
    for _ in range(repeats):
        codewords, used = sampler.propose(problem, budget, rng)
        update = mppi_update(problem, codewords)
        if update is None:
            empty += 1
        else:
            errors.append(abs(update.nu_hat - nu))
    out["update"]["samplers"][sampler.name] = dict(no_feasible_sample=empty / repeats,
                                                   mean_abs_error=float(np.mean(errors)))
    print("  %-32s %5.1f %%              %.3f" % (sampler.name, 100 * empty / repeats, np.mean(errors)))
print("\nA classical sample costs 1 rollout. A depth-1 shot costs 3, so the quantum samplers")
print("use %d shots where the classical sampler uses %d samples." % (budget // 3, budget))
print("\nSaved %s, figures/amplification.png and figures/shots.png" % _path.save_json("single_update.json", out))
