"""Checks that the circuit does what the classical model says. Run with `pytest`."""
import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qmppi import (ClassicalSampler, CorridorTask, PlanningProblem, QuantumSampler, mppi_update,
                   rollout_circuit, run_episode, sampler_circuit)

TASK = CorridorTask()
STATES = [(2, 0), (2, 3), (3, 4), (5, 8), (4, 9), (1, 13)]


@pytest.mark.parametrize("x,t", STATES)
def test_rollout_circuit_matches_classical_rollout(x, t):
    """On every basis state the circuit writes the same flags and position as the model."""
    problem = PlanningProblem(TASK, x, t)
    N = problem.N
    circuit = rollout_circuit(problem)
    for codeword in range(2 ** N):
        start = codeword | ((x // 2) << N)
        state = Statevector.from_int(start, 2 ** circuit.num_qubits).evolve(circuit)
        index = int(np.argmax(np.abs(state.data)))
        assert abs(abs(state.data[index]) - 1) < 1e-12          # still a basis state
        assert index & (2 ** N - 1) == codeword                  # sign register unchanged
        roll = problem.rollout(codeword)
        flags = (index >> (N + 2)) & (2 ** N - 1)
        # The register counts modulo 8, so the two walks agree up to the first visit to a wall
        # (position 0 or 7). A wall is always flagged, hence the comparison up to the first flag.
        for step in range(N):
            unsafe = roll.positions[step] not in problem.safe[step]
            assert bool((flags >> step) & 1) == unsafe
            if roll.positions[step] in (0, 7):
                break
        assert (flags == 0) == roll.feasible
        if roll.feasible:
            assert (index >> N) & 3 == roll.positions[-1] // 2


@pytest.mark.parametrize("x,t", STATES)
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_outcome_distribution(x, t, depth):
    """Acceptance sin^2((2k+1) theta), equal probabilities on the feasible plans, clean ancillas."""
    problem = PlanningProblem(TASK, x, t)
    N, M, D = problem.N, problem.n_feasible, problem.n_codewords
    state = Statevector(sampler_circuit(problem, depth, measure=False))
    probabilities = np.abs(state.data) ** 2
    ancilla = np.arange(probabilities.size) >> N
    assert probabilities[ancilla == x // 2].sum() == pytest.approx(1.0, abs=1e-10)
    on_z = probabilities.reshape(-1, D).sum(axis=0)
    theta = np.arcsin(np.sqrt(M / D))
    alpha = np.sin((2 * depth + 1) * theta) ** 2
    feasible = problem.feasible_mask
    assert on_z[feasible].sum() == pytest.approx(alpha, abs=1e-10)
    if M:
        assert np.allclose(on_z[feasible], alpha / M, atol=1e-10)


def test_only_verified_trajectories_are_used():
    problem = PlanningProblem(TASK, 2, 3)
    sampler = QuantumSampler(depth=1, eta=5e-3, seed=7)
    codewords = sampler.shots(problem, 200)
    update = mppi_update(problem, codewords)
    assert update.n_accepted == sum(problem.rollout(int(z)).feasible for z in codewords)
    assert -1 <= update.nu_hat <= 1


def test_estimate_converges_to_exact_update():
    problem = PlanningProblem(TASK, 2, 3)
    sampler = QuantumSampler(depth=1, seed=11)
    update = mppi_update(problem, sampler.shots(problem, 4000))
    assert update.nu_hat == pytest.approx(problem.exact_update(), abs=0.06)


def test_episode_runs():
    rng = np.random.default_rng(0)
    for sampler in (ClassicalSampler(), QuantumSampler(depth=1, seed=3)):
        out = run_episode(TASK, sampler, 24, rng)
        assert len(out["trajectory"]) >= 2
