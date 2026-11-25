"""Samplers. Each one turns a budget of rollout evaluations into proposed codewords.

Cost unit. One rollout evaluation is one evaluation of the model over the planning horizon.
A classical proposal costs 1 (its verification). A shot of the circuit at depth k costs 2k + 1.
Each Grover iteration runs the rollout forward and backward, and the outcome is verified once.

No sampler here decides feasibility. Every proposed codeword is verified by the classical
rollout in `mppi.mppi_update`.
"""
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from .circuits import sampler_circuit

BASIS_GATES = ["cx", "u"]


def gate_noise_model(eta):
    """Depolarizing noise after every gate and symmetric readout flips.

    eta is the depolarizing parameter of a two-qubit gate and the flip probability of a
    measured bit. Single-qubit gates have the parameter eta / 10."""
    model = NoiseModel(basis_gates=BASIS_GATES)
    model.add_all_qubit_quantum_error(depolarizing_error(eta, 2), ["cx"])
    model.add_all_qubit_quantum_error(depolarizing_error(eta / 10, 1), ["u"])
    model.add_all_qubit_readout_error(ReadoutError([[1 - eta, eta], [eta, 1 - eta]]))
    return model


class ClassicalSampler:
    """Uniformly random codewords from a classical generator. This is standard MPPI sampling."""
    name = "classical"

    def propose(self, problem, budget, rng):
        return rng.integers(0, problem.n_codewords, budget), budget


class QuantumSampler:
    """Shots of the amplitude amplification circuit, simulated gate by gate with Qiskit Aer.

    depth   number of Grover iterations k
    eta     gate error of `gate_noise_model`, or 0 for an ideal device
    batch   the simulator is called for at least this many shots at a time. Unused shots of a
            circuit are kept and handed out later, and every shot is used at most once.
    """

    def __init__(self, depth=1, eta=0.0, seed=1234, batch=500):
        self.depth, self.eta, self.batch = depth, eta, batch
        self.name = "quantum, depth %d, %s" % (depth, "ideal" if eta == 0 else "eta = %g" % eta)
        noise = gate_noise_model(eta) if eta > 0 else None
        self.simulator = AerSimulator(method="statevector", noise_model=noise)
        self._seed = seed
        self._circuits, self._pools = {}, {}
        self.shots_simulated = 0

    @property
    def cost_per_shot(self):
        return 2 * self.depth + 1

    def compiled_circuit(self, problem):
        """The circuit of the problem, compiled to CNOT and single-qubit gates."""
        if problem.key not in self._circuits:
            circuit = sampler_circuit(problem, self.depth, measure=True)
            self._circuits[problem.key] = transpile(circuit, basis_gates=BASIS_GATES, optimization_level=1)
        return self._circuits[problem.key]

    def shots(self, problem, n):
        """n measured codewords of the circuit of this problem."""
        pool = self._pools.get(problem.key, np.zeros(0, dtype=np.int64))
        if len(pool) < n:
            self._seed += 1
            n_new = max(n - len(pool), self.batch)
            result = self.simulator.run(self.compiled_circuit(problem), shots=n_new, memory=True,
                                        seed_simulator=self._seed).result()
            # the bit string has classical bit 0, which holds z_0, as its last character
            new = np.array([int(bits, 2) for bits in result.get_memory()], dtype=np.int64)
            pool = np.concatenate([pool, new])
            self.shots_simulated += n_new
        self._pools[problem.key] = pool[n:]
        return pool[:n]

    def propose(self, problem, budget, rng):
        n = budget // self.cost_per_shot
        return self.shots(problem, n), n * self.cost_per_shot


class AdaptiveSampler:
    """Classical pilot samples first, quantum shots only when the pilot finds nothing.

    A fixed depth can overshoot when the feasible plans are common. This rule uses the circuit
    only when feasible plans look rare, and it keeps the accepted samples uniform on the
    feasible set on an ideal device."""

    def __init__(self, quantum, pilot=3):
        self.quantum, self.pilot = quantum, pilot
        self.name = "adaptive (%s)" % quantum.name

    def propose(self, problem, budget, rng):
        n_pilot = min(self.pilot, budget)
        first = rng.integers(0, problem.n_codewords, n_pilot)
        if any(problem.rollout(int(z)).feasible for z in first):
            rest = rng.integers(0, problem.n_codewords, budget - n_pilot)
            return np.concatenate([first, rest]), budget
        shots, used = self.quantum.propose(problem, budget - n_pilot, rng)
        return np.concatenate([first, shots]), n_pilot + used
