"""Hardware study of the amplified sampler on an IBM quantum computer.

Two experiments, each one job.

  scan   Planning problems of horizon 2 to 5, compiled circuits from about 30 to about 320
         two-qubit gates. For every circuit the script measures the acceptance probability,
         the gain over classical sampling, the uniformity of the accepted samples and the
         MPPI update, and it fits how fast the device noise grows with the gate count.
         This is the quantitative check of the noise analysis.

  loop   Closed-loop MPPI on the corridor task of the paper (18 steps, horizon 5). Shots of
         the circuit of every planning problem that the controller can reach are drawn on the
         device first. The closed loop then consumes them, every shot at most once per run.
         The samples have the same distribution as samples drawn while the loop runs, since
         the circuit of a planning step depends only on the state and the time.

Circuits. "full" is the circuit of the paper. "lean" is the same algorithm with the flag
gadgets simplified by what the planner already knows: a flag that cannot fire is removed,
and a flag may also mark register values that no safe path can reach when that makes the
gadget cheaper. Before anything is sent, every circuit is checked on an ideal simulator to
give exactly the ideal amplified distribution.

Typical use (from the repository root, after  python hardware/run_ibm.py --save-account)
  python hardware/run_study.py usage                    # processor time used so far
  python hardware/run_study.py scan --fake model        # instant stand-in, to see the output format
  python hardware/run_study.py scan --dry-run           # real device: compile and estimate, submit nothing
  python hardware/run_study.py scan                     # real device, asks before submitting
  python hardware/run_study.py loop --dry-run
  python hardware/run_study.py loop
  python hardware/run_study.py scan --suppress          # optional repeat with error suppression
  python hardware/run_study.py collect <job id>         # collect a job that was submitted earlier
  python hardware/run_study.py report                   # redo the analysis and figures of saved results

Everything is written to results/hardware.
"""
import argparse
import datetime
import glob
import hashlib
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                 # repository root, which holds the package qmppi
sys.path.insert(0, ROOT)

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qpy     # noqa: E402
from qiskit.circuit import ParameterVector                                      # noqa: E402
from qiskit.quantum_info import Statevector                                     # noqa: E402
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # noqa: E402

from qmppi.circuits import sampler_circuit                                      # noqa: E402
from qmppi.corridor import CorridorTask, PlanningProblem                        # noqa: E402
from qmppi.mppi import run_episode                                              # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "hardware")
DEFAULT_BACKEND = "ibm_kingston"
EPS_DEFAULT = 0.0058        # effective error per two-qubit gate seen in the first run (A5, k = 1)
FREE_PLAN_SECONDS = 600

# ----------------------------------------------------------------------------------------
# planning problems of the scan: name, horizon N, start position, blocked positions by time,
# circuits, depths k
# ----------------------------------------------------------------------------------------
SCAN = [
    ("N2M1", 2, 2, {2: (1, 2)},                 ("lean",),        (1,)),
    ("N3M1", 3, 2, {3: (1, 2, 3)},              ("lean", "full"), (1,)),
    ("N3M2", 3, 2, {2: (1, 2)},                 ("lean",),        (1,)),
    ("N4M1", 4, 2, {4: (1, 2, 3, 4)},           ("lean",),        (1, 2)),
    ("N4M2", 4, 2, {3: (1, 2, 3)},              ("lean", "full"), (1,)),
    ("N4M4", 4, 3, {3: (3, 4), 4: (3, 4)},      ("lean",),        (1,)),
    ("C5",   5, 2, {3: (1, 2, 3)},              ("lean", "full"), (1,)),
    ("A5",   5, 2, {3: (3, 4), 4: (3, 4)},      ("lean", "full"), (1,)),
    ("N5M8", 5, 3, {4: (1, 2, 3)},              ("lean",),        (1,)),
    ("N5M1", 5, 2, {4: (1, 2, 3, 4)},           ("lean",),        (1, 2)),
]
SCAN_ROWS_PLAIN, SCAN_ROWS_TWIRL, SCAN_SHOTS_PER_ROW = 4, 4, 1500

LOOP_BUDGETS = (6, 9, 12, 18, 24)      # rollout evaluations per control step
LOOP_EPISODES = 500
LOOP_SHOTS_PER_ROW = 500
LOOP_PILOT = 3


def scan_problem(N, obstacles, x=2):
    task = CorridorTask(episode_steps=N, plan_steps=N, x_start=x,
                        obstacles={int(t): frozenset(v) for t, v in obstacles.items()})
    return PlanningProblem(task, x=x, t=0)


def problem_from_spec(spec):
    """spec = dict(x=..., safe=[[...], ...]) describes one planning problem completely."""
    N = len(spec["safe"])
    task = CorridorTask(episode_steps=N, plan_steps=N, x_start=spec["x"], obstacles={})
    problem = PlanningProblem(task, x=spec["x"], t=0)
    problem.safe = [frozenset(s) for s in spec["safe"]]
    return problem


def spec_of(problem):
    return dict(x=int(problem.x), safe=[sorted(int(v) for v in s) for s in problem.safe])


def spec_key(spec):
    return json.dumps(spec, sort_keys=True)


# ----------------------------------------------------------------------------------------
# lean circuits
# ----------------------------------------------------------------------------------------
ONE_BIT = {frozenset({2, 3}): (1, 1), frozenset({0, 1}): (1, 0),
           frozenset({1, 3}): (0, 1), frozenset({0, 2}): (0, 0)}


def _flag_cost(values):
    values = frozenset(values)
    if len(values) in (0, 4):
        return 0
    if values in ONE_BIT:
        return 1
    return 2 if len(values) == 2 else 6


def flag_sets(problem):
    """For every step, the set of register values that the flag of that step marks.

    A value has to be marked if a path that was safe so far can reach it and it is unsafe.
    Values that no such path can reach may be marked or not, whichever is cheaper."""
    safe_now, out = {problem.x}, []
    for t in range(problem.N):
        new = {x + d for x in safe_now for d in (-1, 1)}
        reach = {x // 2 for x in new}
        need = {x // 2 for x in new if x not in problem.safe[t]}
        free = sorted(set(range(4)) - reach)
        options = [frozenset(need | set(extra)) for r in range(len(free) + 1)
                   for extra in itertools.combinations(free, r)]
        out.append(min(options, key=lambda u: (_flag_cost(u), len(u), sorted(u))))
        safe_now = {x for x in new if x in problem.safe[t]}
    return out


def _set_flag(qc, y, flag, values):
    values = frozenset(values)
    if not values:
        return
    if len(values) == 4:
        qc.x(flag)
    elif values in ONE_BIT:
        bit, value = ONE_BIT[values]
        qc.cx(y[bit], flag)
        if value == 0:
            qc.x(flag)
    elif len(values) == 2:                      # {1, 2} is y0 XOR y1, {0, 3} is its complement
        qc.cx(y[0], flag)
        qc.cx(y[1], flag)
        if values == frozenset({0, 3}):
            qc.x(flag)
    else:
        if len(values) == 3:
            qc.x(flag)
            values = frozenset(range(4)) - values
        (value,) = values
        zeros = [b for b in range(2) if not (value >> b) & 1]
        for b in zeros:
            qc.x(y[b])
        qc.ccx(y[0], y[1], flag)
        for b in zeros:
            qc.x(y[b])


def _lean_rollout(problem, z, y, f, sets):
    qc = QuantumCircuit(z, y, f)
    start_bit0 = (problem.x // 2) & 1
    j = 0
    for t in range(problem.N):
        even = (problem.x + t) % 2 == 0
        if even:                                 # even position: the move -1 (z = 0) decrements y
            qc.x(z[t])
        if t == 0:                               # the register value is known before the first move
            if even:
                qc.cx(z[t], y[0])
                if start_bit0 == 0:
                    qc.cx(z[t], y[1])
            else:
                if start_bit0 == 1:
                    qc.cx(z[t], y[1])
                qc.cx(z[t], y[0])
        elif even:
            qc.cx(z[t], y[0])
            qc.ccx(z[t], y[0], y[1])
        else:                                    # odd position: the move +1 (z = 1) increments y
            qc.ccx(z[t], y[0], y[1])
            qc.cx(z[t], y[0])
        if even:
            qc.x(z[t])
        if sets[t]:
            _set_flag(qc, y, f[j], sets[t])
            j += 1
    return qc


def _mcx_clean(qc, controls, target, clean):
    """Multi-controlled X. `clean` are qubits in state 0 that are returned to 0."""
    controls, clean = list(controls), list(clean)
    c = len(controls)
    if c <= 2 or len(clean) < c - 2:
        qc.mcx(controls, target)
        return
    work = clean[:c - 2]
    ladder = QuantumCircuit(*qc.qregs)
    ladder.rccx(controls[0], controls[1], work[0])
    for i in range(1, c - 2):
        ladder.rccx(controls[i + 1], work[i - 1], work[i])
    qc.compose(ladder, inplace=True)
    qc.ccx(controls[-1], work[-1], target)
    qc.compose(ladder.inverse(), inplace=True)


def _flip_if_all_zero(qc, qubits, clean=()):
    """Multiply the state by -1 when all the given qubits are zero."""
    qubits = list(qubits)
    qc.x(qubits)
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        _mcx_clean(qc, qubits[:-1], qubits[-1], clean)
        qc.h(qubits[-1])
    qc.x(qubits)


def lean_circuit(problem, k):
    sets = flag_sets(problem)
    m = sum(1 for s in sets if s)
    z = QuantumRegister(problem.N, "z")
    y = QuantumRegister(2, "y")
    f = QuantumRegister(max(m, 1), "f")
    qc = QuantumCircuit(z, y, f)
    for b in range(2):
        if ((problem.x // 2) >> b) & 1:
            qc.x(y[b])
    qc.h(z)
    rollout = _lean_rollout(problem, z, y, f, sets)
    for _ in range(k):
        qc.barrier()
        qc.compose(rollout, inplace=True)
        _flip_if_all_zero(qc, f[:m] if m else f[:1])
        qc.compose(rollout.inverse(), inplace=True)
        qc.h(z)
        _flip_if_all_zero(qc, z, clean=list(f))      # the flags are back at zero here
        qc.h(z)
    return qc


def logical_circuit(problem, k, variant):
    """Circuit with measurement of the sign qubits into the classical register c."""
    if variant == "full":
        qc = sampler_circuit(problem, depth=k, measure=False)
    else:
        qc = lean_circuit(problem, k)
    c = ClassicalRegister(problem.N, "c")
    qc.add_register(c)
    qc.barrier()
    qc.measure(qc.qregs[0], c)
    return qc


def ideal_distribution(problem, k):
    """Outcome distribution of an ideal device, and the ideal acceptance probability."""
    feas = problem.feasible_mask
    M, D = int(feas.sum()), problem.n_codewords
    p = M / D
    q = float(np.sin((2 * k + 1) * np.arcsin(np.sqrt(p))) ** 2)
    if M == 0:
        return np.full(D, 1.0 / D), 0.0
    inside = q / M
    outside = (1 - q) / (D - M) if D > M else 0.0
    return np.where(feas, inside, outside), q


def ideal_error(problem, k, variant):
    """Largest deviation of the circuit's exact outcome distribution from the ideal one."""
    qc = logical_circuit(problem, k, variant).remove_final_measurements(inplace=False)
    probs = Statevector(qc).probabilities(list(range(problem.N)))
    return float(np.abs(probs - ideal_distribution(problem, k)[0]).max())


# ----------------------------------------------------------------------------------------
# compilation for the device
# ----------------------------------------------------------------------------------------
def two_qubit_count(circ):
    return sum(1 for inst in circ.data if inst.operation.num_qubits == 2)


def add_mask_parameters(isa, N):
    """Before the measurement of sign qubit t insert  sx, rz(m[t]), sx.

    m[t] = 0 gives an X gate (the measured bit is flipped), m[t] = pi gives a Z gate (no
    effect on the measured bit). One compiled circuit then serves all readout masks."""
    m = ParameterVector("m", N)
    out = isa.copy_empty_like()
    for inst in isa.data:
        if inst.operation.name == "measure":
            q = inst.qubits[0]
            t = isa.find_bit(inst.clbits[0]).index
            out.sx(q)
            out.rz(m[t], q)
            out.sx(q)
        out.append(inst)
    return out, list(m)


def mask_values(circuit, masks, N):
    """Parameter values, one row per mask, columns in the order of circuit.parameters."""
    cols = [p.index for p in circuit.parameters]          # elements of the vector m
    assert sorted(cols) == list(range(N))
    return np.array([[0.0 if (mask >> t) & 1 else math.pi for t in cols] for mask in masks])


def calibration_fidelity(isa, target):
    """Product of (1 - reported error) over all gates and measurements of the circuit."""
    fid, missing = 1.0, 0
    for inst in isa.data:
        name = inst.operation.name
        if name in ("barrier", "delay"):
            continue
        qubits = tuple(isa.find_bit(q).index for q in inst.qubits)
        try:
            err = target[name][qubits].error
        except Exception:
            err = None
        if err is None:
            missing += 1
        else:
            fid *= 1.0 - err
    return fid if missing < len(isa.data) / 2 else None


def compile_circuits(items, backend, tries, cache_tag, quiet=False):
    """items: list of dict(spec, k, variant). Adds the compiled circuit and its figures.

    Several seeds are tried and the version with the fewest two-qubit gates is kept. The
    result is cached on disk for the day, so a dry run and the run that follows compile once."""
    if not {"sx", "rz"} <= set(backend.target.operation_names):
        sys.exit("This script needs a device with sx and rz gates (all IBM devices have them).")
    os.makedirs(os.path.join(OUT_DIR, "cache"), exist_ok=True)
    done = {}
    for n, item in enumerate(items):
        key = (spec_key(item["spec"]), item["k"], item["variant"])
        if key not in done:
            problem = problem_from_spec(item["spec"])
            err = ideal_error(problem, item["k"], item["variant"])
            if err > 1e-9:
                sys.exit(f"Internal check failed: circuit {key} deviates from the ideal distribution by {err:.2e}")
            digest = hashlib.sha1(repr((key, cache_tag, tries)).encode()).hexdigest()[:16]
            path = os.path.join(OUT_DIR, "cache", f"{digest}.qpy")
            best = None
            if os.path.exists(path):
                try:
                    with open(path, "rb") as fh:
                        best = qpy.load(fh)[0]
                except Exception:
                    best = None
            if best is None:
                logical = logical_circuit(problem, item["k"], item["variant"])
                for s in range(tries if item["k"] > 0 else 1):
                    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=s)
                    isa = pm.run(logical)
                    if best is None or two_qubit_count(isa) < two_qubit_count(best):
                        best = isa
                try:
                    with open(path, "wb") as fh:
                        qpy.dump(best, fh)
                except Exception:
                    pass
            try:
                duration = float(best.estimate_duration(backend.target))
            except Exception:
                duration = None
            masked, _ = add_mask_parameters(best, problem.N)
            done[key] = dict(circuit=masked, two_qubit_gates=two_qubit_count(best), depth=best.depth(),
                             duration=duration, cal_fidelity=calibration_fidelity(best, backend.target),
                             qubits=len({best.find_bit(q).index for inst in best.data for q in inst.qubits}))
            if not quiet and sys.stdout.isatty():
                print(f"\r  compiled {len(done)} circuits", end="", flush=True)
        rec = done[key]
        item.update({k: v for k, v in rec.items() if k != "circuit"})
        item["_circuit"] = rec["circuit"]
    if not quiet and sys.stdout.isatty():
        print()


def complementary_masks(N, rows, rng):
    """Readout masks in complementary pairs, so every bit is flipped in exactly half the rows."""
    full = 2 ** N - 1
    pool = [m for m in range(2 ** N) if m < (m ^ full)]
    rng.shuffle(pool)
    masks = []
    for i in range(rows // 2):
        m = int(pool[i % len(pool)])
        masks += [m, m ^ full]
    if rows % 2:
        masks.append(int(rng.integers(0, 2 ** N)))
    return masks


# ----------------------------------------------------------------------------------------
# plans
# ----------------------------------------------------------------------------------------
def scan_plan(shots_per_row, seed):
    rng = np.random.default_rng(seed)
    plan = []
    for name, N, x, obstacles, variants, ks in SCAN:
        problem = scan_problem(N, obstacles, x)
        spec = spec_of(problem)
        plan.append(dict(name=name, spec=spec, k=0, variant="lean", masks=[0], twirl=[0],
                         shots=shots_per_row))
        for variant in variants:
            for k in (ks if variant == "lean" else (1,)):
                masks = [0] * SCAN_ROWS_PLAIN + complementary_masks(N, SCAN_ROWS_TWIRL, rng)
                twirl = [0] * SCAN_ROWS_PLAIN + [1] * SCAN_ROWS_TWIRL
                plan.append(dict(name=name, spec=spec, k=k, variant=variant, masks=masks, twirl=twirl,
                                 shots=shots_per_row))
    return plan


def loop_task():
    return CorridorTask()          # the task of the paper: 18 steps, horizon 5


def reachable_problems(task):
    """All planning problems (x, t) that some sequence of moves through safe positions reaches."""
    states, out = {task.x_start}, []
    for t in range(task.episode_steps):
        for x in sorted(states):
            out.append(PlanningProblem(task, x, t))
        states = {x + d for x in states for d in (-1, 1) if (x + d) in task.safe_set(t + 1)}
    return out


class ModelSampler:
    """Stand-in for the device: ideal outcome with probability r, a uniform outcome otherwise."""
    cost_per_shot = 3

    def __init__(self, r_of_key, seed):
        self.r_of_key, self.rng = r_of_key, np.random.default_rng(seed)
        self.demand = {}

    def propose(self, problem, budget, rng):
        n = budget // self.cost_per_shot
        key = spec_key(spec_of(problem))
        self.demand[key] = self.demand.get(key, 0) + n
        probs, _ = ideal_distribution(problem, 1)
        r = self.r_of_key(key)
        probs = r * probs + (1 - r) / problem.n_codewords
        return self.rng.choice(problem.n_codewords, size=n, p=probs / probs.sum()), n * self.cost_per_shot

    def reset(self):
        self.demand = {}


class PoolEmpty(Exception):
    pass


class PoolSampler:
    """Hands out device shots of the circuit of each planning problem, each shot once per run."""
    cost_per_shot = 3

    def __init__(self, pools):
        self.pools = pools
        self.ptr = {}

    def reset(self):
        self.ptr = {}

    def propose(self, problem, budget, rng):
        n = budget // self.cost_per_shot
        key = spec_key(spec_of(problem))
        start = self.ptr.get(key, 0)
        pool = self.pools.get(key)
        if pool is None or start + n > len(pool):
            raise PoolEmpty(key)
        self.ptr[key] = start + n
        return pool[start:start + n], n * self.cost_per_shot


class ClassicalSampler:
    name = "classical"

    def propose(self, problem, budget, rng):
        return rng.integers(0, problem.n_codewords, budget), budget

    def reset(self):
        pass


class AdaptiveSampler:
    """Classical pilot samples first, device shots only when the pilot finds nothing."""

    def __init__(self, quantum, pilot=LOOP_PILOT):
        self.quantum, self.pilot = quantum, pilot

    def reset(self):
        self.quantum.reset()

    def propose(self, problem, budget, rng):
        n_pilot = min(self.pilot, budget)
        first = rng.integers(0, problem.n_codewords, n_pilot)
        if any(problem.rollout(int(z)).feasible for z in first):
            rest = rng.integers(0, problem.n_codewords, budget - n_pilot)
            return np.concatenate([first, rest]), budget
        shots, used = self.quantum.propose(problem, budget - n_pilot, rng)
        return np.concatenate([first, shots]), n_pilot + used


def closed_loop_runs(task, quantum, episodes, budgets, seed, with_classical=True, keep_episode=False):
    """Runs the closed loop for every sampler and budget. Returns one row per combination."""
    samplers = [("device, k = 1", quantum), ("device, adaptive", AdaptiveSampler(quantum))]
    if with_classical:
        samplers.insert(0, ("classical", ClassicalSampler()))
    rows, example = [], None
    for name, sampler in samplers:
        for B in budgets:
            sampler.reset()
            rng = np.random.default_rng(seed + 1000 * B + len(name))
            fails = costs = done = 0
            src = dict(sampled=0, plan=0, blind=0)
            agree = n_agree = acc = steps = rollouts = 0
            for e in range(episodes):
                try:
                    ep = run_episode(task, sampler, B, rng)
                except PoolEmpty:
                    break
                done += 1
                fails += ep["failed"]
                rollouts += ep["rollouts"]
                if not ep["failed"]:
                    costs += ep["cost"]
                for s in ep["steps"]:
                    steps += 1
                    acc += s["accepted"]
                    src[{"sampled": "sampled", "stored plan": "plan", "random": "blind"}[s["source"]]] += 1
                    if s["nu_exact"] is not None and s["nu_exact"] != 0:
                        n_agree += 1
                        agree += int(s["move"] == (1 if s["nu_exact"] > 0 else -1))
                if keep_episode and example is None and name == "device, k = 1" and B == max(budgets) \
                        and not ep["failed"]:
                    example = dict(B=B, trajectory=ep["trajectory"],
                                   steps=[dict(t=s["t"], x=s["x"], paths=s["paths"], ok=s["ok"]) for s in ep["steps"]])
            if done == 0:
                continue
            fail = fails / done
            rows.append(dict(sampler=name, B=B, episodes=done, fail=fail,
                             fail_se=math.sqrt(max(fail * (1 - fail), 1e-12) / done),
                             mean_cost=(costs / (done - fails) if done > fails else None),
                             sampled=src["sampled"] / steps, plan=src["plan"] / steps, blind=src["blind"] / steps,
                             agree=agree / max(n_agree, 1), mean_accepted=acc / steps,
                             rollouts_per_step=rollouts / steps))
    return rows, example


def loop_plan(shots_per_row, episodes, budgets, eps, seed, gates_of_key=None):
    """Circuits of the closed loop and the number of shots of each. The shot numbers come
    from a trial of the closed loop with the stand-in sampler, with a margin of 30 percent."""
    task = loop_task()
    problems = {}
    for problem in reachable_problems(task):
        problems.setdefault(spec_key(spec_of(problem)), problem)
    gates_of_key = gates_of_key or {}
    model = ModelSampler(lambda key: math.exp(-eps * gates_of_key.get(key, 200)), seed)
    demand = {}
    for B in budgets:
        for sampler in (model, AdaptiveSampler(model)):
            model.reset()
            rng = np.random.default_rng(seed + B)
            for _ in range(episodes):
                run_episode(task, sampler, B, rng)
            for key, n in model.demand.items():
                demand[key] = max(demand.get(key, 0), n)
    rng = np.random.default_rng(seed)
    plan = []
    for key, problem in problems.items():
        rows = max(2, math.ceil(1.3 * demand.get(key, 0) / shots_per_row))
        plan.append(dict(name=f"x{problem.x}t{problem.t}", spec=spec_of(problem), k=1, variant="lean",
                         masks=complementary_masks(problem.N, rows, rng), twirl=[1] * rows,
                         shots=shots_per_row))
    return plan


# ----------------------------------------------------------------------------------------
# running a plan
# ----------------------------------------------------------------------------------------
def total_shots(plan):
    return sum(item["shots"] * len(item["masks"]) for item in plan)


def past_jobs():
    """Job manifests in the results folder: list of (job id, shots, executions)."""
    out = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "job_*.json"))):
        try:
            man = json.load(open(path))
            if "masks" in man["plan"][0] and isinstance(man["plan"][0]["masks"], list):
                shots, runs = total_shots(man["plan"]), sum(len(i["masks"]) for i in man["plan"])
            else:                                             # manifest of run_ibm.py
                shots, runs = sum(i["shots"] for i in man["plan"]), len(man["plan"])
            out.append((man["job_id"], shots, runs, man.get("usage_seconds")))
        except Exception:
            continue
    return out


def job_seconds(job):
    """Processor time of a finished job, or None while IBM has not reported it yet."""
    for read in (lambda: job.metrics()["usage"]["quantum_seconds"], lambda: job.usage()):
        try:
            sec = float(read())
            if sec > 0:
                return sec
        except Exception:
            continue
    return None


def estimate_seconds(plan, backend, service=None):
    """Processor time of the plan. Uses the time that earlier jobs of this folder really took
    when it can be read from the account."""
    rep = getattr(backend, "default_rep_delay", None) or 250e-6
    base = sum(item["shots"] * len(item["masks"]) * (rep + (item.get("duration") or 5e-5)) for item in plan)
    runs = sum(len(item["masks"]) for item in plan)
    per_run, source = 0.1, "default"
    if service is not None:
        ratios = []
        for job_id, shots, n_runs, stored in past_jobs():
            sec = stored
            if sec is None:
                try:
                    sec = job_seconds(service.job(job_id))
                except Exception:
                    sec = None
            if sec:
                ratios.append(max(0.0, (sec - shots * (rep + 4e-5)) / n_runs))
        if ratios:
            per_run, source = float(np.median(ratios)), "earlier jobs"
    return base + runs * per_run + 2.0, source


def account_usage(service):
    try:
        u = service.usage()
        return {k: u[k] for k in u if "usage" in k or "period" in k}
    except Exception:
        return None


def model_counts(plan, eps, seed):
    """Counts of the stand-in device: ideal with probability r = exp(-eps * gates), else uniform."""
    rng = np.random.default_rng(seed)
    out = []
    for item in plan:
        problem = problem_from_spec(item["spec"])
        probs, _ = ideal_distribution(problem, item["k"])
        r = math.exp(-eps * item["two_qubit_gates"])
        probs = r * probs + (1 - r) / problem.n_codewords
        rows = []
        for mask in item["masks"]:
            draw = rng.multinomial(item["shots"], probs / probs.sum())
            rows.append({format(z ^ mask, f"0{problem.N}b"): int(c) for z, c in enumerate(draw) if c})
        out.append(rows)
    return out


def counts_of_result(result, plan):
    out = []
    for pub, item in zip(result, plan):
        data = pub.data.c
        out.append([{k: int(v) for k, v in data.get_counts(i).items()} for i in range(len(item["masks"]))])
    return out


def get_backend(args):
    """Returns (backend, name, service). The stand-in and the simulated devices need no account."""
    if args.fake:
        import qiskit_ibm_runtime.fake_provider as fp
        device = "kingston" if args.fake == "model" else args.fake
        return getattr(fp, "Fake" + device.capitalize())(), ("model" if args.fake == "model" else "fake_" + device), None
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    name = args.backend or DEFAULT_BACKEND
    try:
        backend = service.backend(name)
        if not backend.status().operational:
            raise RuntimeError("not operational")
    except Exception as exc:
        if args.backend:
            sys.exit(f"Device {name} is not available: {exc}")
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=12)
    return backend, backend.name, service


def strip(plan):
    return [{k: v for k, v in item.items() if not k.startswith("_")} for item in plan]


def save_json(payload, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    return path


def execute(kind, plan, backend, backend_name, service, args, extra):
    """Compile, estimate, ask, submit, collect. Returns the saved record or None."""
    today = datetime.date.today().isoformat()
    print(f"Device: {backend_name}")
    compile_circuits(plan, backend, args.tries, cache_tag=(backend_name, today))
    amplified = [i for i in plan if i["k"] > 0]
    gates = [i["two_qubit_gates"] for i in amplified]
    print(f"  {len(plan)} circuits, {sum(len(i['masks']) for i in plan)} executions, {total_shots(plan)} shots, "
          f"two-qubit gates {min(gates)} to {max(gates)}")
    print("  every circuit reproduces the ideal distribution on an ideal simulator")
    if kind == "scan":
        for i in amplified:
            print(f"    {i['name']:<5} {i['variant']:<4} k={i['k']}: {i['two_qubit_gates']:>4} two-qubit gates, "
                  f"depth {i['depth']}")
    manifest = dict(kind=kind, backend=backend_name, submitted=datetime.datetime.now().isoformat(timespec="seconds"),
                    suppress=bool(args.suppress), plan=strip(plan), **extra)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.fake == "model":
        counts = model_counts(plan, args.eps, args.seed)
        manifest.update(counts=counts, note="stand-in model, not device data", model_eps=args.eps)
        return finish(manifest, f"{kind}_model_{stamp}.json")

    seconds, source = estimate_seconds(plan, backend, service)
    if service is not None:
        print(f"  estimated processor time: {seconds:.0f} s (per-execution overhead from {source})")
        usage = account_usage(service) or {}
        left = usage.get("usage_remaining_seconds")
        if left is not None:
            print(f"  processor time left on this account: {left:.0f} s")
            if seconds > left and not args.dry_run:
                sys.exit("The estimate exceeds the time that is left. Reduce --shots, or wait for the 28-day window.")
        else:
            print(f"  the free plan has {FREE_PLAN_SECONDS} s per 28 days")
    if args.dry_run:
        print("Dry run, nothing submitted.")
        return None

    from qiskit_ibm_runtime import SamplerV2
    sampler = SamplerV2(mode=backend)
    if service is not None:
        cap = int(args.cap or max(60, 2.5 * seconds))
        sampler.options.max_execution_time = cap
        manifest["max_execution_time"] = cap
        if args.suppress:
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XpXm"
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.num_randomizations = "auto"
        if not args.yes:
            answer = input(f"Submit to {backend_name}? The job is stopped if it exceeds {cap} s. [y/N] ")
            if answer.strip().lower() not in ("y", "yes"):
                print("Nothing submitted.")
                return None
    pubs = [(item["_circuit"], mask_values(item["_circuit"], item["masks"], len(item["spec"]["safe"])))
            for item in plan]
    job = sampler.run(pubs, shots=plan[0]["shots"])        # the same number of shots for every execution
    if service is not None:
        manifest["job_id"] = job.job_id()
        save_json(manifest, f"job_{job.job_id()}.json")
        print(f"Submitted job {job.job_id()}. Waiting. It is safe to stop this script and collect later with")
        print(f"  python hardware/run_study.py collect {job.job_id()}")
    result = job.result()
    manifest["counts"] = counts_of_result(result, plan)
    if service is not None:
        manifest["usage_seconds"] = job_seconds(job)
        save_json({k: v for k, v in manifest.items() if k != "counts"}, f"job_{job.job_id()}.json")
        if manifest["usage_seconds"] is None:
            print("IBM has not reported the processor time of this job yet. See it later with  python hardware/run_study.py usage")
        else:
            print(f"Processor time used by this job: {manifest['usage_seconds']:.0f} s")
    return finish(manifest, f"{kind}_{backend_name}_{stamp}.json")


def finish(record, filename):
    record["analysis"] = analyse(record)
    path = save_json(record, filename)
    print_report(record)
    figs = make_figures(record, os.path.splitext(path)[0])
    print(f"\nSaved {path}")
    for f in figs:
        print(f"Saved {f}")
    return record


# ----------------------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------------------
def histogram(item, rows, select=None):
    N = len(item["spec"]["safe"])
    hist = np.zeros(2 ** N, dtype=np.int64)
    for j, (mask, counts) in enumerate(zip(item["masks"], rows)):
        if select is not None and item["twirl"][j] != select:
            continue
        for bits, c in counts.items():
            hist[int(bits, 2) ^ mask] += c
    return hist


def uniformity(counts_feasible, rng, draws=20000):
    """Chi-square test of the accepted samples against the uniform distribution (Monte Carlo
    p-value), their total variation distance from uniform, and the typical value of that
    distance for exactly uniform samples."""
    n, M = int(counts_feasible.sum()), len(counts_feasible)
    if n == 0 or M < 2:
        return dict(chi2=None, p_value=None, tv=None, tv_if_uniform=None)
    exp = n / M
    chi2 = float(((counts_feasible - exp) ** 2 / exp).sum())
    tv = float(0.5 * np.abs(counts_feasible / n - 1 / M).sum())
    sim = rng.multinomial(n, np.ones(M) / M, size=draws)
    chi2_sim = ((sim - exp) ** 2 / exp).sum(axis=1)
    tv_sim = 0.5 * np.abs(sim / n - 1 / M).sum(axis=1)
    return dict(chi2=chi2, p_value=float((chi2_sim >= chi2).mean()), tv=tv, tv_if_uniform=float(tv_sim.mean()))


def summarise(problem, k, hist, rng):
    feas = problem.feasible_mask
    M, D = int(feas.sum()), problem.n_codewords
    p = M / D
    _, q = ideal_distribution(problem, k)
    shots = int(hist.sum())
    # the classical check of every measured codeword
    accepted = int(sum(c for z, c in enumerate(hist) if c and problem.rollout(z).feasible))
    acc = accepted / shots if shots else 0.0
    se = math.sqrt(max(acc * (1 - acc), 1e-12) / shots) if shots else None
    row = dict(shots=shots, accepted=accepted, p=p, M=M, accept=acc, accept_se=se, accept_ideal=q)
    if M == 0:
        return row
    cost = 2 * k + 1
    row.update(gain=acc / (cost * p), gain_se=se / (cost * p), gain_ideal=q / (cost * p))
    if k > 0 and abs(q - p) > 0.05:
        row.update(r=(acc - p) / (q - p), r_se=se / abs(q - p), r_needed=2 * k * p / (q - p))
    row.update(uniformity(hist[feas], rng))
    idx = np.flatnonzero(feas)
    w = np.array([problem.weight(problem.rollout(int(z)).cost) for z in idx])
    e0 = np.array([problem.rollout(int(z)).moves[0] for z in idx], dtype=float)
    row["nu_exact"] = float((w * e0).sum() / w.sum())
    if accepted:
        cnt = hist[idx].astype(float)
        row["nu_hat"] = float((cnt * w * e0).sum() / (cnt * w).sum())
        boot = rng.multinomial(accepted, cnt / cnt.sum(), size=2000).astype(float)
        row["nu_se"] = float(((boot * w * e0).sum(axis=1) / np.maximum((boot * w).sum(axis=1), 1e-300)).std())
    return row


def fit_decay(points):
    """Least-squares fit of r = exp(-eps * gates), weighted by the standard errors."""
    pts = [(g, r, max(s, 1e-4)) for g, r, s in points if r is not None]
    if len(pts) < 2:
        return None
    G, R, S = (np.array(v, dtype=float) for v in zip(*pts))

    def loss(eps):
        return float((((R - np.exp(-eps * G)) / S) ** 2).sum())

    grid = np.geomspace(1e-5, 0.1, 4000)
    eps = float(grid[int(np.argmin([loss(e) for e in grid]))])
    h = eps * 1e-3
    curv = (loss(eps + h) - 2 * loss(eps) + loss(eps - h)) / h ** 2
    chi2 = loss(eps)
    dof = len(pts) - 1
    se = math.sqrt(2 / curv) if curv > 0 else None
    # the points scatter more than their counting errors when the noise is not the same for
    # all circuits, the reported error is widened accordingly
    if se is not None and chi2 > dof:
        se *= math.sqrt(chi2 / dof)
    return dict(eps=eps, eps_se=se, chi2=chi2, dof=dof,
                rms=float(np.sqrt(((R - np.exp(-eps * G)) ** 2).mean())))


def analyse(record):
    rng = np.random.default_rng(7)
    rows = []
    for item, counts in zip(record["plan"], record["counts"]):
        problem = problem_from_spec(item["spec"])
        base = dict(name=item["name"], variant=item["variant"], k=item["k"], N=problem.N,
                    two_qubit_gates=item["two_qubit_gates"], depth=item["depth"],
                    cal_fidelity=item.get("cal_fidelity"))
        arms = [("all", None)]
        if 0 in item["twirl"] and 1 in item["twirl"]:
            arms += [("plain", 0), ("twirl", 1)]
        for arm, select in arms:
            hist = histogram(item, counts, select)
            rows.append(dict(base, arm=arm, **summarise(problem, item["k"], hist, rng),
                             histogram=hist.tolist()))
    out = dict(rows=rows)
    pooled = [r for r in rows if r["arm"] == "all" and r["k"] > 0 and r.get("r") is not None]
    out["decay"] = fit_decay([(r["two_qubit_gates"], r["r"], r["r_se"]) for r in pooled])
    if record["kind"] == "loop":
        out["loop"] = analyse_loop(record, rows)
    return out


def analyse_loop(record, rows):
    task = loop_task()
    rng = np.random.default_rng(11)
    pools, r_of = {}, {}
    for item, counts in zip(record["plan"], record["counts"]):
        key = spec_key(item["spec"])
        hist = histogram(item, counts)
        pool = np.repeat(np.arange(len(hist)), hist)
        rng.shuffle(pool)
        pools[key] = pool
    for item, row in zip(record["plan"], [r for r in rows if r["arm"] == "all"]):
        r_of[spec_key(item["spec"])] = min(1.0, max(0.0, row["r"])) if row.get("r") is not None else 0.0
    episodes, budgets = record["episodes"], record["budgets"]
    device, example = closed_loop_runs(task, PoolSampler(pools), episodes, budgets, seed=5, keep_episode=True)
    model, _ = closed_loop_runs(task, ModelSampler(lambda key: r_of.get(key, 0.0), 3), episodes, budgets,
                                seed=6, with_classical=False)
    ideal, _ = closed_loop_runs(task, ModelSampler(lambda key: 1.0, 4), episodes, budgets,
                                seed=8, with_classical=False)
    for r in model:
        r["sampler"] = r["sampler"].replace("device", "model")
    for r in ideal:
        r["sampler"] = r["sampler"].replace("device", "ideal")
    return dict(runs=device + model + ideal, example=example)


# ----------------------------------------------------------------------------------------
# report and figures
# ----------------------------------------------------------------------------------------
def fmt(v, n=3, width=7):
    return f"{'-':>{width}}" if v is None else f"{v:>{width}.{n}f}"


def print_report(record):
    an = record["analysis"]
    rows = an["rows"]
    print(f"\n{record['kind']} on {record['backend']}" + ("  (STAND-IN MODEL, NOT DEVICE DATA)" if record.get("note") else ""))
    if record["kind"] == "scan":
        head = (f"{'name':<6}{'circ':<5}{'k':>2}{'arm':>6}{'2q':>5}{'shots':>7}{'accept':>8}{'ideal':>7}"
                f"{'gain':>7}{'+/-':>6}{'ideal':>6}{'r':>7}{'need':>6}{'unif p':>8}{'TV':>7}{'TV0':>7}"
                f"{'nu_hat':>8}{'+/-':>7}{'nu':>7}")
        print(head)
        print("-" * len(head))
        for r in rows:
            if r["k"] == 0 and r["arm"] != "all":
                continue
            print(f"{r['name']:<6}{r['variant']:<5}{r['k']:>2}{r['arm']:>6}{r['two_qubit_gates']:>5}{r['shots']:>7}"
                  f"{r['accept']:>8.3f}{r['accept_ideal']:>7.3f}{fmt(r.get('gain'), 2)}{fmt(r.get('gain_se'), 2, 6)}"
                  f"{fmt(r.get('gain_ideal'), 2, 6)}{fmt(r.get('r'), 2)}{fmt(r.get('r_needed'), 2, 6)}"
                  f"{fmt(r.get('p_value'), 3, 8)}{fmt(r.get('tv'))}{fmt(r.get('tv_if_uniform'))}"
                  f"{fmt(r.get('nu_hat'), 3, 8)}{fmt(r.get('nu_se'))}{fmt(r.get('nu_exact'))}")
        print("\ngain     accept / ((2k+1) p). Above 1: fewer rollouts per accepted sample than classical sampling")
        print("r        fraction of shots that behave as on an ideal device, from accept = r * ideal + (1 - r) * p")
        print("need     smallest r for which the gain exceeds 1")
        print("unif p   p-value of the uniformity test of the accepted samples (M >= 2 only)")
        print("TV, TV0  distance of the accepted samples from uniform, and its typical value for uniform samples")
        print("nu_hat   MPPI update from the accepted samples with its standard error, nu is the exact update")
    pooled = [r for r in rows if r["arm"] == "all" and r["k"] > 0 and r.get("gain") is not None]
    wins = [r for r in pooled if (r["gain"] - 1) / r["gain_se"] > 3]
    print(f"\nCircuits with gain above 1 by more than 3 standard errors: {len(wins)} of {len(pooled)}"
          + (" (" + ", ".join(f"{r['name']} {r['variant']} k={r['k']}: {r['gain']:.2f}" for r in wins[:12]) + ")"
             if wins else ""))
    tested = [r for r in rows if r["arm"] in ("all",) and r["k"] > 0 and r.get("p_value") is not None]
    if tested:
        low = sum(r["p_value"] < 0.05 for r in tested)
        print(f"Uniformity test at the 5 percent level: rejected for {low} of {len(tested)} circuits "
              f"(about {0.05 * len(tested):.1f} expected by chance for uniform samples)")
    cal = [(r["r"], r["cal_fidelity"]) for r in pooled if r.get("r") and r.get("cal_fidelity")
           and 0 < r["r"] < 1 and 0 < r["cal_fidelity"] < 1]
    if cal:
        ratio = float(np.median([math.log(r) / math.log(f) for r, f in cal]))
        print(f"Measured noise against the device calibration: log r is {ratio:.1f} times the value that the "
              f"reported gate and readout errors give (median over the circuits)")
    d = an.get("decay")
    if d:
        se = f" +/- {d['eps_se']:.5f}" if d.get("eps_se") else ""
        print(f"Noise growth: r = exp(-eps * gates) with eps = {d['eps']:.5f}{se} per two-qubit gate "
              f"(rms deviation of r: {d['rms']:.3f})")
    if record["kind"] == "loop":
        print(f"\nClosed loop, {record['episodes']} episodes per entry, budget B rollouts per control step")
        head = f"{'sampler':<18}{'B':>4}{'episodes':>9}{'fail':>8}{'+/-':>7}{'agree':>7}{'accepted':>9}{'cost':>7}"
        print(head)
        print("-" * len(head))
        for r in an["loop"]["runs"]:
            print(f"{r['sampler']:<18}{r['B']:>4}{r['episodes']:>9}{r['fail']:>8.3f}{r['fail_se']:>7.3f}"
                  f"{r['agree']:>7.3f}{r['mean_accepted']:>9.2f}{fmt(r['mean_cost'], 2)}")
        print("\nfail      fraction of episodes that hit an obstacle or a wall")
        print("agree     fraction of steps whose move equals the move of the exact MPPI update")
        print("accepted  verified samples per control step")
        print("model     the same loop with the stand-in sampler, r of every circuit taken from this job")
        print("ideal     the same loop with a noiseless device")


def make_figures(record, stem):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "STIXGeneral",
                         "mathtext.fontset": "stix", "font.size": 9, "axes.linewidth": 0.6,
                         "legend.frameon": False, "savefig.bbox": "tight"})
    an, files = record["analysis"], []
    pooled = [r for r in an["rows"] if r["arm"] == "all" and r["k"] > 0 and r.get("gain") is not None]
    if pooled:
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7))
        style = {("lean", 1): ("o", "C0", "lean, $k=1$"), ("lean", 2): ("^", "C0", "lean, $k=2$"),
                 ("full", 1): ("s", "C3", "full, $k=1$")}
        seen = set()
        for r in pooled:
            if r.get("r") is None:
                continue
            m, c, lab = style.get((r["variant"], r["k"]), ("d", "C2", f"{r['variant']}, $k={r['k']}$"))
            a.errorbar(r["two_qubit_gates"], max(r["r"], 1e-3), yerr=r["r_se"], fmt=m, color=c, ms=4, lw=0.8,
                       mfc="white" if r["k"] > 1 else c, label=None if lab in seen else lab)
            seen.add(lab)
        d = an.get("decay")
        gmax = max(r["two_qubit_gates"] for r in pooled)
        if d:
            g = np.linspace(0, gmax * 1.05, 100)
            a.plot(g, np.exp(-d["eps"] * g), "k-", lw=0.8, label=r"$e^{-\epsilon G}$, $\epsilon=%.4f$" % d["eps"])
        a.set_yscale("log")
        ticks = [t for t in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)]
        a.set_yticks(ticks)
        a.set_yticklabels([f"{t:g}" for t in ticks])
        a.minorticks_off()
        a.set_xlabel("two-qubit gates $G$ of the compiled circuit")
        a.set_ylabel("ideal fraction $r$")
        a.legend(fontsize=7)
        order = sorted(pooled, key=lambda r: r["two_qubit_gates"])
        xs = np.arange(len(order))
        b.axhline(1.0, color="0.5", lw=0.6, ls="--")
        b.plot(xs, [r["gain_ideal"] for r in order], "_", color="0.4", ms=7, label="ideal device")
        b.errorbar(xs, [r["gain"] for r in order], yerr=[2 * r["gain_se"] for r in order], fmt="o", color="C0",
                   ms=3.5, lw=0.8, label="measured")
        b.set_xticks(xs)
        b.set_xticklabels([f"{r['name']}\n{r['variant']}" + (", k=2" if r["k"] == 2 else "") for r in order],
                          fontsize=5, rotation=90)
        b.set_ylabel("gain over classical sampling")
        b.legend(fontsize=7)
        fig.tight_layout()
        files.append(stem + "_scan.pdf")
        fig.savefig(files[-1])
        plt.close(fig)
    if record["kind"] == "loop" and an.get("loop"):
        task = loop_task()
        runs, example = an["loop"]["runs"], an["loop"]["example"]
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1, 1.5]))
        looks = {"classical": ("k", "o", "-"), "device, k = 1": ("C0", "s", "-"), "device, adaptive": ("C3", "^", "-"),
                 "model, k = 1": ("C0", None, ":"), "model, adaptive": ("C3", None, ":")}
        for name, (c, m, ls) in looks.items():
            sel = [r for r in runs if r["sampler"] == name]
            if sel:
                a.errorbar([r["B"] for r in sel], [r["fail"] for r in sel],
                           yerr=[r["fail_se"] for r in sel] if m else None, color=c, marker=m, ls=ls, ms=3.5,
                           lw=0.9, label=name)
        a.set_xlabel("rollouts per control step $B$")
        a.set_ylabel("failure rate")
        a.legend(fontsize=6.5)
        if example:
            for t in range(task.episode_steps + 1):
                blocked = task.obstacles.get(t, frozenset())
                if blocked:
                    b.add_patch(plt.Rectangle((t - 0.3, min(blocked) - 0.4), 0.6, max(blocked) - min(blocked) + 0.8,
                                              color="C3", alpha=0.35, lw=0))
            for wall in (0, 7):
                b.axhspan(wall - 0.5, wall + 0.5, color="0.85", lw=0)
            for s in example["steps"]:
                for path, ok in zip(s["paths"], s["ok"]):
                    if ok:
                        b.plot(range(s["t"], s["t"] + len(path) + 1), [s["x"]] + list(path), color="C0", lw=0.4,
                               alpha=0.35)
            b.plot(range(len(example["trajectory"])), example["trajectory"], "k-o", ms=2.5, lw=1.1)
            b.set_ylim(-0.5, 7.5)
            b.set_xlabel("time step")
            b.set_ylabel("position")
        fig.tight_layout()
        files.append(stem + "_loop.pdf")
        fig.savefig(files[-1])
        plt.close(fig)
    return files


# ----------------------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------------------
def cmd_scan(args):
    backend, name, service = get_backend(args)
    plan = scan_plan(args.shots or SCAN_SHOTS_PER_ROW, args.seed)
    execute("scan", plan, backend, name, service, args, extra={})


def cmd_loop(args):
    backend, name, service = get_backend(args)
    shots, episodes = args.shots or LOOP_SHOTS_PER_ROW, args.episodes
    print("Sizing the shot pools with a trial of the closed loop ...")
    draft = loop_plan(shots, episodes, LOOP_BUDGETS, args.eps, args.seed)
    compile_circuits(draft, backend, args.tries, cache_tag=(name, datetime.date.today().isoformat()), quiet=True)
    gates = {spec_key(i["spec"]): i["two_qubit_gates"] for i in draft}
    plan = loop_plan(shots, episodes, LOOP_BUDGETS, args.eps, args.seed, gates_of_key=gates)
    execute("loop", plan, backend, name, service, args, extra=dict(episodes=episodes, budgets=list(LOOP_BUDGETS)))


def cmd_collect(args):
    from qiskit_ibm_runtime import QiskitRuntimeService
    path = os.path.join(OUT_DIR, f"job_{args.job_id}.json")
    if not os.path.exists(path):
        sys.exit(f"No record of job {args.job_id} in results/hardware.")
    manifest = json.load(open(path))
    if "kind" not in manifest:
        sys.exit("This job was submitted with run_ibm.py. Collect it with  python hardware/run_ibm.py --job-id " + args.job_id)
    job = QiskitRuntimeService().job(args.job_id)
    print("Job status:", job.status())
    manifest["counts"] = counts_of_result(job.result(), manifest["plan"])
    manifest["usage_seconds"] = job_seconds(job)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    finish(manifest, f"{manifest['kind']}_{manifest['backend']}_{stamp}.json")


def cmd_report(args):
    files = args.files or sorted(glob.glob(os.path.join(OUT_DIR, "scan_*.json")) +
                                 glob.glob(os.path.join(OUT_DIR, "loop_*.json")))
    if not files:
        sys.exit("No saved scan or loop results in results/hardware.")
    for path in files:
        record = json.load(open(path))
        record["analysis"] = analyse(record)
        with open(path, "w") as fh:
            json.dump(record, fh, indent=1)
        print_report(record)
        for f in make_figures(record, os.path.splitext(path)[0]):
            print(f"Saved {f}")


def cmd_usage(args):
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    usage = account_usage(service)
    print("Account usage:", usage if usage else "not reported by this account")
    for job_id, shots, runs, stored in past_jobs():
        sec = stored
        if sec is None:
            try:
                sec = job_seconds(service.job(job_id))
            except Exception:
                sec = None
        print(f"  job {job_id}: {shots} shots in {runs} executions, processor time {sec} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("scan", "loop"):
        p = sub.add_parser(name)
        p.add_argument("--fake", nargs="?", const="kingston",
                       help="'model' for the instant stand-in, or a simulated IBM device (kingston, fez, torino)")
        p.add_argument("--backend", help=f"IBM device, default {DEFAULT_BACKEND}")
        p.add_argument("--dry-run", action="store_true", help="compile and estimate, submit nothing")
        p.add_argument("--suppress", action="store_true",
                       help="dynamical decoupling and gate twirling (real devices only)")
        p.add_argument("--shots", type=int, help="shots per execution")
        p.add_argument("--tries", type=int, default=8, help="compilation seeds to try per circuit")
        p.add_argument("--cap", type=int, help="stop the job after this many seconds of processor time")
        p.add_argument("--eps", type=float, default=EPS_DEFAULT, help="error per two-qubit gate of the stand-in")
        p.add_argument("--yes", action="store_true", help="submit without asking")
        p.add_argument("--seed", type=int, default=2026)
        if name == "loop":
            p.add_argument("--episodes", type=int, default=LOOP_EPISODES)
    p = sub.add_parser("collect")
    p.add_argument("job_id")
    p = sub.add_parser("report")
    p.add_argument("files", nargs="*")
    sub.add_parser("usage")
    args = ap.parse_args()
    dict(scan=cmd_scan, loop=cmd_loop, collect=cmd_collect, report=cmd_report, usage=cmd_usage)[args.command](args)


if __name__ == "__main__":
    main()
