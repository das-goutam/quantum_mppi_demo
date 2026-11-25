"""End-to-end MPPI control with samples drawn from a quantum circuit.

The quantum device proposes trajectories by amplitude amplification of the feasible ones.
A classical rollout verifies every proposal, and the MPPI update is a classical weighted average.
"""
from .corridor import CorridorTask, PlanningProblem
from .circuits import (oracle_circuit, preparation_circuit, reflection_circuit, rollout_circuit,
                       sampler_circuit)
from .samplers import ClassicalSampler, QuantumSampler, AdaptiveSampler, gate_noise_model
from .mppi import mppi_update, run_episode, exact_closed_loop

__all__ = [
    "CorridorTask", "PlanningProblem", "rollout_circuit", "sampler_circuit", "preparation_circuit",
    "oracle_circuit", "reflection_circuit",
    "ClassicalSampler", "QuantumSampler", "AdaptiveSampler", "gate_noise_model",
    "mppi_update", "run_episode", "exact_closed_loop",
]
