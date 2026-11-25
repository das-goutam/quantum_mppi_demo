"""Step 1. Build the circuit of one planning problem and look at it.

    python examples/01_show_circuit.py
"""
import os

import _path
from qiskit import transpile

from qmppi import CorridorTask, PlanningProblem, rollout_circuit, sampler_circuit
from qmppi.samplers import BASIS_GATES

task = CorridorTask()
problem = PlanningProblem(task, x=task.x_start, t=0)

print("Planning problem at time t = %d, state x = %d, horizon N = %d" % (problem.t, problem.x, problem.N))
for tau, safe in enumerate(problem.safe, start=1):
    print("  safe positions at planning step %d: %s" % (tau, sorted(safe)))
print("  feasible plans: %d of %d  (p = %.4f)" % (problem.n_feasible, problem.n_codewords, problem.feasible_mass))

rollout = rollout_circuit(problem)
print("\nReversible rollout on %d qubits: %s" % (rollout.num_qubits, dict(rollout.count_ops())))
print(rollout.draw(output="text", fold=120))

circuit = sampler_circuit(problem, depth=1)
compiled = transpile(circuit, basis_gates=BASIS_GATES, optimization_level=1)
print("\nOne shot at depth 1")
print("  logical gates : %s" % dict(circuit.count_ops()))
print("  compiled      : %d CNOT, %d single-qubit gates, depth %d" % (
    compiled.count_ops().get("cx", 0), compiled.count_ops().get("u", 0), compiled.depth()))

_path.save_json("circuit.json", dict(
    x=problem.x, t=problem.t, qubits=circuit.num_qubits, feasible_plans=problem.n_feasible,
    rollout_gates={k: int(v) for k, v in rollout.count_ops().items()},
    logical_gates={k: int(v) for k, v in circuit.count_ops().items()},
    compiled_cnot=int(compiled.count_ops().get("cx", 0)), compiled_single_qubit=int(compiled.count_ops().get("u", 0)),
    compiled_depth=int(compiled.depth())))

try:
    import matplotlib
    matplotlib.use("Agg")
    rollout.draw(output="mpl", fold=-1).savefig(os.path.join(_path.FIGURES, "rollout_circuit.png"), dpi=150, bbox_inches="tight")
    circuit.draw(output="mpl", fold=40).savefig(os.path.join(_path.FIGURES, "sampler_circuit.png"), dpi=130, bbox_inches="tight")
    print("\nSaved results/circuit.json, figures/rollout_circuit.png and figures/sampler_circuit.png")
except Exception as error:      # the picture needs matplotlib and pylatexenc
    print("\nNo picture saved (%s)" % error)
