"""Run one closed-loop episode and record everything the video shows."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qmppi import CorridorTask, PlanningProblem, QuantumSampler, mppi_update   # noqa: E402

BUDGET = 24          # rollout evaluations per control step, 8 shots at depth 1


def record_episode(seed, budget=BUDGET, eta=0.0):
    task = CorridorTask()
    sampler = QuantumSampler(depth=1, eta=eta, seed=seed, batch=64)
    rng = np.random.default_rng(seed)
    x, plan, steps = task.x_start, (), []
    for t in range(task.episode_steps):
        problem = PlanningProblem(task, x, t)
        codewords, _ = sampler.propose(problem, budget, rng)
        rolls = [problem.rollout(int(z)) for z in codewords]
        update = mppi_update(problem, codewords)
        if update is not None:
            move, new_plan, source = update.move, update.plan, "sampled"
            best = min((r for r in rolls if r.feasible and r.moves[0] == move), key=lambda r: r.cost)
            chosen = best.positions
        elif plan:
            move, new_plan, source = plan[0], plan[1:], "stored plan"
            chosen = tuple(x + int(s) for s in np.cumsum(plan))
        else:
            move, new_plan, source, chosen = int(rng.choice([-1, 1])), (), "random", ()
        M, D = problem.n_feasible, problem.n_codewords
        theta = np.arcsin(np.sqrt(M / D))
        prob = np.where(problem.feasible_mask, np.sin(3 * theta) ** 2 / max(M, 1),
                        np.cos(3 * theta) ** 2 / max(D - M, 1))
        steps.append(dict(
            t=t, x=x, p=M / D, M=M, theta=theta, feasible=problem.feasible_mask.copy(), prob=prob,
            all_paths=[problem.rollout(z).positions for z in range(D)],
            shots=[int(z) for z in codewords], paths=[r.positions for r in rolls],
            ok=[r.feasible for r in rolls], cost=[r.cost for r in rolls],
            weight=[problem.weight(r.cost) if r.feasible else 0.0 for r in rolls],
            nu_hat=None if update is None else update.nu_hat, nu_exact=problem.exact_update(),
            move=move, source=source, chosen=chosen))
        plan = new_plan
        x += move
        if x not in task.safe_set(t + 1):
            return task, steps, False
    return task, steps, True


if __name__ == "__main__":
    for seed in range(1, 9):
        task, steps, done = record_episode(seed)
        traj = [s["x"] for s in steps] + [steps[-1]["x"] + steps[-1]["move"]]
        acc = [sum(s["ok"]) for s in steps]
        print(seed, done, traj, acc, [s["source"][0] for s in steps])
