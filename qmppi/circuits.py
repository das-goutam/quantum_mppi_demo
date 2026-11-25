"""Gate-level circuits.

Registers (2N + 2 qubits)
  z[0..N-1]   sign qubits, one per planning step (z_t = 1 is a move of +1)
  y[0..1]     position register
  f[0..N-1]   violation flags, f[t] = 1 if and only if the position after step t is unsafe

Position register. The position after t steps has the parity of x_0 + t, so its lowest bit is a
classical function of time. The register stores only y = floor(x / 2) in two qubits, and
x = 2 y + c with the known parity c. The register counts modulo 8. The walls 0 and 7 are always
flagged, so a path is flagged before it can wrap around, and "all flags zero" is exactly
feasibility.

One shot of the sampler
  1. Hadamard gates on z prepare all 2^N plans in superposition.
  2. k Grover iterations amplify the feasible plans. One iteration is
       rollout  ->  phase flip if all flags are zero  ->  inverse rollout  ->  reflection.
  3. z is measured. The outcome is a codeword that still has to pass the classical rollout.
"""
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister


def _controlled_increment(qc, control, y):
    """y <- y + 1 (mod 4) if control = 1."""
    qc.ccx(control, y[0], y[1])
    qc.cx(control, y[0])


def _controlled_decrement(qc, control, y):
    """y <- y - 1 (mod 4) if control = 1."""
    qc.cx(control, y[0])
    qc.ccx(control, y[0], y[1])


def _set_flag(qc, y, flag, unsafe_values):
    """flag <- flag XOR [y in unsafe_values] for the two-qubit register y."""
    unsafe = frozenset(unsafe_values)
    if not unsafe:
        return
    if len(unsafe) == 4:
        qc.x(flag)
        return
    # unsafe set decided by one bit of y: one CNOT
    one_bit = {frozenset({2, 3}): (1, 1), frozenset({0, 1}): (1, 0),
               frozenset({1, 3}): (0, 1), frozenset({0, 2}): (0, 0)}
    if unsafe in one_bit:
        bit, value = one_bit[unsafe]
        if value == 0:
            qc.x(y[bit])
        qc.cx(y[bit], flag)
        if value == 0:
            qc.x(y[bit])
        return
    # three unsafe values: invert the flag and correct it for the single safe value
    if len(unsafe) == 3:
        qc.x(flag)
        unsafe = frozenset(range(4)) - unsafe
    # one Toffoli gate per remaining value, with X gates that select the value
    for value in sorted(unsafe):
        zeros = [b for b in range(2) if not (value >> b) & 1]
        for b in zeros:
            qc.x(y[b])
        qc.ccx(y[0], y[1], flag)
        for b in zeros:
            qc.x(y[b])


def rollout_circuit(problem):
    """Reversible rollout |z>|y_0>|0...0>  ->  |z>|y_N>|f_1 ... f_N>."""
    N = problem.N
    z, y, f = QuantumRegister(N, "z"), QuantumRegister(2, "y"), QuantumRegister(N, "f")
    qc = QuantumCircuit(z, y, f, name="rollout")
    for t in range(N):
        parity = (problem.x + t) % 2
        if parity == 0:
            # even position: the move -1 (z_t = 0) decrements y, the move +1 leaves it unchanged
            qc.x(z[t])
            _controlled_decrement(qc, z[t], y)
            qc.x(z[t])
        else:
            # odd position: the move +1 (z_t = 1) increments y, the move -1 leaves it unchanged
            _controlled_increment(qc, z[t], y)
        new_parity = (problem.x + t + 1) % 2
        unsafe = [v for v in range(4) if (2 * v + new_parity) not in problem.safe[t]]
        _set_flag(qc, y, f[t], unsafe)
    return qc


def _phase_flip_if_all_zero(qc, qubits):
    """Multiply the state by -1 when all the given qubits are zero."""
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(list(qubits[:-1]), qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)


def _registers(problem):
    N = problem.N
    return QuantumRegister(N, "z"), QuantumRegister(2, "y"), QuantumRegister(N, "f")


def preparation_circuit(problem):
    """Initial position in the position register and all plans in superposition."""
    z, y, f = _registers(problem)
    qc = QuantumCircuit(z, y, f, name="prepare")
    y0 = problem.x // 2
    for b in range(2):
        if (y0 >> b) & 1:
            qc.x(y[b])
    qc.h(z)
    return qc


def oracle_circuit(problem):
    """Phase oracle: multiplies the amplitude of every feasible plan by -1.

    Rollout, phase flip when all flags are zero, inverse rollout. The position and flag
    registers are returned to their initial values."""
    z, y, f = _registers(problem)
    qc = QuantumCircuit(z, y, f, name="oracle")
    rollout = rollout_circuit(problem)
    qc.compose(rollout, inplace=True)
    _phase_flip_if_all_zero(qc, list(f))
    qc.compose(rollout.inverse(), inplace=True)
    return qc


def reflection_circuit(problem):
    """Reflection about the uniform superposition of all plans (up to a global sign)."""
    z, y, f = _registers(problem)
    qc = QuantumCircuit(z, y, f, name="reflect")
    qc.h(z)
    _phase_flip_if_all_zero(qc, list(z))
    qc.h(z)
    return qc


def sampler_circuit(problem, depth=1, measure=True):
    """The complete circuit of one shot: preparation, `depth` Grover iterations, measurement."""
    z, y, f = _registers(problem)
    qc = QuantumCircuit(z, y, f)
    qc.compose(preparation_circuit(problem), inplace=True)
    for _ in range(depth):
        qc.barrier()
        qc.compose(oracle_circuit(problem), inplace=True)
        qc.compose(reflection_circuit(problem), inplace=True)
    if measure:
        c = ClassicalRegister(problem.N, "c")
        qc.add_register(c)
        qc.barrier()
        qc.measure(z, c)
    return qc
