"""The MPPI update and the receding-horizon loop. Everything here is classical.

Update. Among the proposed codewords the controller keeps those that pass the classical rollout.
With the weights g = exp(-S / lambda) of the kept trajectories it forms

    nu_hat = sum_j g_j eps_j / sum_j g_j,

where eps_j = +1 or -1 is the first move of trajectory j. This is the hard-constrained MPPI
update for the first move.

Executed move. A move on the lattice is +1 or -1, so the controller executes the sign of nu_hat.
At least one kept trajectory starts with that move, and the cheapest of them is stored as the
current plan. If no proposal passes the verification, the controller follows the rest of the last
plan, and if no plan is left it moves at random.
"""
from dataclasses import dataclass

import numpy as np

from .corridor import PlanningProblem


@dataclass
class Update:
    nu_hat: float         # estimate of the MPPI update
    move: int             # executed move, +1 or -1
    plan: tuple           # remaining moves of the stored verified trajectory
    n_proposed: int
    n_accepted: int


def mppi_update(problem, codewords):
    """Verify the proposals and form the update. Returns None if no proposal is feasible."""
    accepted = [r for r in (problem.rollout(int(z)) for z in codewords) if r.feasible]
    if not accepted:
        return None
    weights = np.array([problem.weight(r.cost) for r in accepted])
    first_moves = np.array([r.moves[0] for r in accepted], dtype=float)
    nu_hat = float((weights * first_moves).sum() / weights.sum())
    cheapest = min(accepted, key=lambda r: r.cost)
    move = 1 if nu_hat > 0 else (-1 if nu_hat < 0 else cheapest.moves[0])
    best = min((r for r in accepted if r.moves[0] == move), key=lambda r: r.cost)
    return Update(nu_hat, move, best.moves[1:], len(codewords), len(accepted))


def run_episode(task, sampler, budget, rng, verbose=False):
    """One closed-loop episode with `budget` rollout evaluations per control step.

    Returns a dictionary with the trajectory, the outcome and per-step records."""
    x, plan = task.x_start, ()
    out = dict(trajectory=[x], failed=False, cost=0.0, rollouts=0, steps=[])
    for t in range(task.episode_steps):
        problem = PlanningProblem(task, x, t)
        codewords, used = sampler.propose(problem, budget, rng)
        update = mppi_update(problem, codewords)
        if update is not None:
            move, plan, source = update.move, update.plan, "sampled"
        elif plan:
            move, plan, source = plan[0], plan[1:], "stored plan"
        else:
            move, plan, source = int(rng.choice([-1, 1])), (), "random"
        exact = problem.exact_update()
        rollouts = [problem.rollout(int(z)) for z in codewords]
        out["steps"].append(dict(
            t=t, x=x, feasible_mass=problem.feasible_mass, proposed=len(codewords),
            accepted=0 if update is None else update.n_accepted,
            nu_hat=None if update is None else update.nu_hat, nu_exact=exact, move=move, source=source,
            paths=[r.positions for r in rollouts], ok=[r.feasible for r in rollouts],
            chosen=tuple(x + int(c) for c in np.cumsum((move,) + tuple(plan)))))
        out["rollouts"] += used
        if verbose:
            s = out["steps"][-1]
            print("t=%2d  x=%d  p=%.3f  proposed=%2d  accepted=%2d  nu_hat=%s  nu_exact=%s  move=%+d  (%s)" % (
                t, x, s["feasible_mass"], s["proposed"], s["accepted"],
                "  n/a " if s["nu_hat"] is None else "%+.3f" % s["nu_hat"],
                "  n/a " if exact is None else "%+.3f" % exact, move, source))
        x += move
        out["trajectory"].append(x)
        if x not in task.safe_set(t + 1):
            out["failed"] = True
            out["failed_at"] = t + 1
            break
        out["cost"] += task.stage_cost(x)
    return out


def exact_closed_loop(task):
    """Closed loop with the exact update in place of a sampled one. Returns the trajectory."""
    x, trajectory = task.x_start, [task.x_start]
    for t in range(task.episode_steps):
        nu = PlanningProblem(task, x, t).exact_update()
        if nu is None:
            break
        x += 1 if nu > 0 else -1
        trajectory.append(x)
    return trajectory
