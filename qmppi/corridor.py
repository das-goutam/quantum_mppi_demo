"""The control task and its classical model.

Plant. A point moves on the positions x in {0, ..., 7}. At every time step it moves by +1 or -1.
The positions 1, ..., 6 are safe, and obstacles block some of them at given times. The positions
0 and 7 are walls.

Codeword. A plan over N steps is a string of N bits z_0, ..., z_{N-1}. Bit z_t = 1 means a move
of +1 at step t, and z_t = 0 means a move of -1. A codeword is stored as an integer whose bit t
is z_t. This is also the integer that Qiskit returns when the sign qubit t is measured into the
classical bit t.

Everything in this file is classical. `PlanningProblem.rollout` is the classical verification
that every quantum sample has to pass before the controller uses it.
"""
from dataclasses import dataclass, field

import numpy as np

CORRIDOR = frozenset(range(1, 7))          # safe positions when no obstacle is present
N_POSITIONS = 8                             # positions 0, ..., 7 (three bits)


def _default_obstacles():
    low, high = frozenset({1, 2, 3}), frozenset({4, 5, 6})
    return {5: low, 6: low, 10: high, 11: high, 15: low, 16: low}


@dataclass
class CorridorTask:
    """The closed-loop task. The default values are the experiment of the README."""
    episode_steps: int = 18                 # T, number of control steps in an episode
    plan_steps: int = 5                     # N, planning horizon and number of sign qubits
    x_start: int = 2
    x_preferred: int = 2                    # the stage cost is cost_weight * |x - x_preferred|
    cost_weight: float = 0.5
    temperature: float = 1.0                # lambda of the path integral weight exp(-S / lambda)
    obstacles: dict = field(default_factory=_default_obstacles)   # time -> blocked positions

    def safe_set(self, t):
        """Safe positions at time t."""
        return CORRIDOR - self.obstacles.get(t, frozenset())

    def stage_cost(self, x):
        return self.cost_weight * abs(x - self.x_preferred)


@dataclass
class Rollout:
    """Result of one classical rollout of a codeword."""
    codeword: int
    feasible: bool
    cost: float
    moves: tuple          # N moves, each +1 or -1
    positions: tuple      # x_1, ..., x_N


class PlanningProblem:
    """The planning problem that the controller sees at time t in state x."""

    def __init__(self, task, x, t):
        self.task, self.x, self.t = task, int(x), int(t)
        self.N = task.plan_steps
        # safe[tau - 1] is the safe set at planning step tau = 1, ..., N
        self.safe = [task.safe_set(t + tau) for tau in range(1, self.N + 1)]
        self._table = None

    @property
    def key(self):
        return (self.x, self.t)

    @property
    def n_codewords(self):
        return 2 ** self.N

    def rollout(self, codeword):
        """Classical rollout of one codeword. Returns feasibility, cost, moves and positions."""
        x, cost, feasible = self.x, 0.0, True
        moves, positions = [], []
        for tau in range(self.N):
            move = 1 if (codeword >> tau) & 1 else -1
            x += move
            moves.append(move); positions.append(x)
            if x not in self.safe[tau]:
                feasible = False
            cost += self.task.stage_cost(x)
        return Rollout(int(codeword), feasible, cost, tuple(moves), tuple(positions))

    def weight(self, cost):
        """Path integral weight of a feasible trajectory."""
        return float(np.exp(-cost / self.task.temperature))

    # ---- exact quantities by enumeration of all 2^N codewords (ground truth only) ----

    def _enumerate(self):
        if self._table is None:
            rolls = [self.rollout(z) for z in range(self.n_codewords)]
            self._table = dict(
                feasible=np.array([r.feasible for r in rolls]),
                cost=np.array([r.cost for r in rolls]),
                first_move=np.array([r.moves[0] for r in rolls], dtype=float),
            )
        return self._table

    @property
    def feasible_mask(self):
        return self._enumerate()["feasible"]

    @property
    def n_feasible(self):
        return int(self.feasible_mask.sum())

    @property
    def feasible_mass(self):
        """p, the probability that a uniformly random codeword is feasible."""
        return self.n_feasible / self.n_codewords

    def exact_update(self):
        """The exact hard-constrained MPPI update nu_0 = E[g eps_0 | feasible] / E[g | feasible].

        Returns None when no codeword is feasible."""
        tab = self._enumerate()
        f = tab["feasible"]
        if not f.any():
            return None
        g = np.exp(-tab["cost"][f] / self.task.temperature)
        return float((g * tab["first_move"][f]).sum() / g.sum())
