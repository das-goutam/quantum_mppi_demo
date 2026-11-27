"""Run the sampler circuits on an IBM quantum computer and check the accepted samples.

What it does
  For two corridor planning problems (A5 and C5 of the paper) it submits, in one job,
    * the circuit without amplification (k = 0), as the classical baseline on the device,
    * the circuit with one Grover iteration (k = 1),
    * the same circuit with readout twirling (random X gates before the measurement,
      undone on the measured bits),
    * optionally the circuit with two iterations (--k2).
  Every measured bit string is passed through the classical rollout. The script reports the
  acceptance probability, the gain over classical sampling, a test of uniformity of the
  accepted samples over the feasible set, and the estimated MPPI update against the exact one.

Typical use (from the repository folder)
  python hardware/run_ibm.py --save-account          # once, stores the API key on this computer
  python hardware/run_ibm.py --fake --shots 400      # dry run on a simulated IBM device, no account
                                                     # (a few minutes, the simulation is slow)
  python hardware/run_ibm.py --dry-run               # real device: compile and estimate, submit nothing
  python hardware/run_ibm.py                         # real device, least busy
  python hardware/run_ibm.py --backend ibm_kingston --shots 4000
  python hardware/run_ibm.py --job-id <id>           # collect a job that was submitted earlier

Results are written to results/hardware/<backend>_<time>.json.
"""
import argparse
import datetime
import getpass
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qiskit import ClassicalRegister, QuantumCircuit                      # noqa: E402
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # noqa: E402

from qmppi.circuits import sampler_circuit                                # noqa: E402
from qmppi.corridor import CorridorTask, PlanningProblem                  # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "hardware")

INSTANCES = {
    # name: obstacles (time -> blocked positions), start position 2, horizon 5
    "A5": {3: frozenset({3, 4}), 4: frozenset({3, 4})},
    "C5": {3: frozenset({1, 2, 3})},
}


def make_problem(name):
    task = CorridorTask(episode_steps=5, plan_steps=5, x_start=2, obstacles=dict(INSTANCES[name]))
    return PlanningProblem(task, x=2, t=0)


def circuit_with_mask(problem, k, mask):
    """Sampler circuit with k iterations. X gates on the sign qubits in `mask` precede the
    measurement (readout twirling). The measured integer has to be XORed with `mask`."""
    qc = sampler_circuit(problem, depth=k, measure=False)
    z = qc.qregs[0]
    c = ClassicalRegister(problem.N, "c")
    qc.add_register(c)
    qc.barrier()
    for t in range(problem.N):
        if (mask >> t) & 1:
            qc.x(z[t])
    qc.measure(z, c)
    return qc


def build_plan(shots, n_masks, with_k2, seed):
    """List of circuits to run. Each entry: instance, configuration, k, mask, shots.

    The IBM sampler needs the same number of shots for every circuit of a job. Every entry
    therefore has shots / n_masks shots, and the configurations without twirling are entered
    n_masks times."""
    rng = np.random.default_rng(seed)
    per = max(1, shots // n_masks)
    plan = []
    for name in INSTANCES:
        N = make_problem(name).N
        masks = [int(m) for m in rng.choice(np.arange(1, 2 ** N), size=n_masks, replace=False)]
        configs = [("k0", 0, [0] * n_masks), ("k1", 1, [0] * n_masks), ("k1_twirl", 1, masks)]
        if with_k2:
            configs.append(("k2", 2, [0] * n_masks))
        for config, k, mask_list in configs:
            for m in mask_list:
                plan.append(dict(inst=name, config=config, k=k, mask=m, shots=per))
    return plan


def two_qubit_count(circ):
    return sum(1 for inst in circ.data if inst.operation.num_qubits == 2)


def compile_plan(plan, backend, tries):
    """Compile every circuit for the device. Several seeds are tried and the version with the
    fewest two-qubit gates is kept."""
    compiled, cache = [], {}
    for item in plan:
        key = (item["inst"], item["k"], item["mask"])
        if key not in cache:
            logical = circuit_with_mask(make_problem(item["inst"]), item["k"], item["mask"])
            best = None
            for s in range(tries):
                pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=s)
                isa = pm.run(logical)
                if best is None or two_qubit_count(isa) < two_qubit_count(best):
                    best = isa
            cache[key] = best
        best = cache[key]
        item["two_qubit_gates"] = two_qubit_count(best)
        item["depth"] = best.depth()
        compiled.append(best)
    return compiled


def mc_uniformity(counts_feasible, rng, draws=20000):
    """Chi-square statistic of the accepted samples against the uniform distribution, with a
    Monte Carlo p-value, the total variation distance, and its typical value under uniformity."""
    n, M = int(counts_feasible.sum()), len(counts_feasible)
    if n == 0 or M < 2:
        return dict(chi2=None, p_value=None, tv=None, tv_if_uniform=None)
    exp = n / M
    chi2 = float(((counts_feasible - exp) ** 2 / exp).sum())
    tv = float(0.5 * np.abs(counts_feasible / n - 1 / M).sum())
    sim = rng.multinomial(n, np.ones(M) / M, size=draws)
    chi2_sim = ((sim - exp) ** 2 / exp).sum(axis=1)
    tv_sim = 0.5 * np.abs(sim / n - 1 / M).sum(axis=1)
    return dict(chi2=chi2, p_value=float((chi2_sim >= chi2).mean()), tv=tv,
                tv_if_uniform=float(tv_sim.mean()))


def analyse(plan, counts_list, seed=7):
    """Merge the circuits of each configuration, check every outcome classically, and report."""
    rng = np.random.default_rng(seed)
    merged = {}
    for item, counts in zip(plan, counts_list):
        key = (item["inst"], item["config"])
        N = make_problem(item["inst"]).N
        hist = merged.setdefault(key, dict(k=item["k"], hist=np.zeros(2 ** N, dtype=np.int64),
                                           two_qubit_gates=[], depth=[]))
        for bits, c in counts.items():
            hist["hist"][int(bits, 2) ^ item["mask"]] += c
        hist["two_qubit_gates"].append(item["two_qubit_gates"])
        hist["depth"].append(item["depth"])
    rows = []
    for (inst, config), rec in merged.items():
        prob = make_problem(inst)
        feas = prob.feasible_mask
        p = prob.feasible_mass
        k = rec["k"]
        q_ideal = float(np.sin((2 * k + 1) * np.arcsin(np.sqrt(p))) ** 2)
        hist = rec["hist"]
        shots = int(hist.sum())
        # classical check of every measured bit string
        accepted = int(sum(c for z, c in enumerate(hist) if c and prob.rollout(z).feasible))
        assert accepted == int(hist[feas].sum())
        acc = accepted / shots
        se = float(np.sqrt(acc * (1 - acc) / shots))
        uni = mc_uniformity(hist[feas], rng)
        # MPPI update from the accepted samples against the exact one
        idx = np.flatnonzero(feas)
        w = np.array([prob.weight(prob.rollout(int(z)).cost) for z in idx])
        e0 = np.array([prob.rollout(int(z)).moves[0] for z in idx], dtype=float)
        nu_exact = float((w * e0).sum() / w.sum())
        cnt = hist[idx].astype(float)
        nu_hat = float((cnt * w * e0).sum() / (cnt * w).sum()) if accepted else None
        rows.append(dict(
            inst=inst, config=config, k=k, shots=shots, p=p, M=int(feas.sum()),
            two_qubit_gates=int(np.median(rec["two_qubit_gates"])), depth=int(np.median(rec["depth"])),
            accept=acc, accept_se=se, accept_ideal=q_ideal,
            gain=acc / ((2 * k + 1) * p), gain_ideal=q_ideal / ((2 * k + 1) * p),
            fault_free_fit=((acc - p) / (q_ideal - p) if k > 0 and abs(q_ideal - p) > 1e-9 else None),
            accepted=accepted, nu_exact=nu_exact, nu_hat=nu_hat,
            histogram=hist.tolist(), **uni))
    return rows


def print_report(rows, backend_name):
    print(f"\nBackend: {backend_name}")
    head = (f"{'inst':<5}{'config':<10}{'2q gates':>9}{'shots':>7}{'accept':>9}{'+/-':>7}{'ideal':>7}"
            f"{'gain':>6}{'ideal':>6}{'unif. p':>9}{'TV':>8}{'TV0':>7}{'nu_hat':>8}{'nu':>7}")
    print(head)
    print("-" * len(head))
    for r in rows:
        f = lambda v, n=3: "  -  " if v is None else f"{v:.{n}f}"
        print(f"{r['inst']:<5}{r['config']:<10}{r['two_qubit_gates']:>9}{r['shots']:>7}"
              f"{r['accept']:>9.3f}{r['accept_se']:>7.3f}{r['accept_ideal']:>7.3f}"
              f"{r['gain']:>6.2f}{r['gain_ideal']:>6.2f}{f(r['p_value']):>9}"
              f"{f(r['tv']):>8}{f(r['tv_if_uniform']):>7}{f(r['nu_hat']):>8}{r['nu_exact']:>7.3f}")
    print("\naccept   fraction of shots that pass the classical rollout (ideal: sin^2((2k+1) theta))")
    print("gain     accept / ((2k+1) p), values above 1 mean fewer rollouts per sample than classical")
    print("unif. p  p-value of the uniformity test of the accepted samples over the feasible set")
    print("TV, TV0  distance of the accepted samples from uniform, and its typical value when the")
    print("         samples are exactly uniform (sampling noise alone)")
    print("nu_hat   MPPI update estimated from the accepted samples, nu is the exact update")


def save(payload, backend_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_DIR, f"{backend_name}_{stamp}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nSaved {path}")
    return path


def counts_from_result(result):
    return [pub.data.c.get_counts() for pub in result]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save-account", action="store_true", help="store the IBM API key on this computer and exit")
    ap.add_argument("--fake", nargs="?", const="kingston", help="simulate an IBM device locally (kingston, fez, torino)")
    ap.add_argument("--backend", help="name of the IBM device, default: least busy")
    ap.add_argument("--shots", type=int, default=4000, help="shots per configuration")
    ap.add_argument("--masks", type=int, default=8, help="number of random masks for readout twirling")
    ap.add_argument("--k2", action="store_true", help="also run two Grover iterations")
    ap.add_argument("--tries", type=int, default=10, help="compilation seeds to try per circuit")
    ap.add_argument("--dd", action="store_true", help="dynamical decoupling on idle qubits (real devices only)")
    ap.add_argument("--dry-run", action="store_true", help="compile and print the estimate, submit nothing")
    ap.add_argument("--job-id", help="collect the results of a job submitted earlier")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    if args.save_account:
        from qiskit_ibm_runtime import QiskitRuntimeService
        token = getpass.getpass("IBM Quantum API key (input is hidden): ").strip()
        crn = input("Instance CRN (press Enter to let IBM choose): ").strip() or None
        QiskitRuntimeService.save_account(token=token, instance=crn, set_as_default=True, overwrite=True)
        print("Saved. The key is stored in your home folder under .qiskit, not in this repository.")
        return

    from qiskit_ibm_runtime import SamplerV2

    if args.job_id:
        from qiskit_ibm_runtime import QiskitRuntimeService
        manifest_path = os.path.join(OUT_DIR, f"job_{args.job_id}.json")
        manifest = json.load(open(manifest_path))
        job = QiskitRuntimeService().job(args.job_id)
        print("Job status:", job.status())
        rows = analyse(manifest["plan"], counts_from_result(job.result()))
        print_report(rows, manifest["backend"])
        save(dict(manifest, results=rows), manifest["backend"])
        return

    if args.fake:
        import qiskit_ibm_runtime.fake_provider as fp
        backend = getattr(fp, "Fake" + args.fake.capitalize())()
        backend_name = "fake_" + args.fake
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backend = (service.backend(args.backend) if args.backend
                   else service.least_busy(operational=True, simulator=False, min_num_qubits=12))
        backend_name = backend.name
    print("Device:", backend_name)

    plan = build_plan(args.shots, args.masks, args.k2, args.seed)
    print(f"Compiling {len(set((i['inst'], i['k'], i['mask']) for i in plan))} distinct circuits ...")
    compiled = compile_plan(plan, backend, args.tries)
    total_shots = sum(item["shots"] for item in plan)
    for name in INSTANCES:
        for cfg in ("k0", "k1", "k2"):
            g = [i["two_qubit_gates"] for i in plan if i["inst"] == name and i["config"] == cfg]
            if g:
                print(f"  {name} {cfg}: {g[0]} two-qubit gates after compilation")
    print(f"Total shots: {total_shots}. Rough processor time: {total_shots * 0.0004 + 5:.0f} s "
          f"(the free plan has 600 s per 28 days).")
    if args.dry_run:
        return

    sampler = SamplerV2(mode=backend)
    if args.dd and not args.fake:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    job = sampler.run(compiled, shots=plan[0]["shots"])
    manifest = dict(backend=backend_name, submitted=datetime.datetime.now().isoformat(timespec="seconds"),
                    shots=args.shots, masks=args.masks, dd=bool(args.dd), plan=plan)
    if not args.fake:
        os.makedirs(OUT_DIR, exist_ok=True)
        manifest["job_id"] = job.job_id()
        with open(os.path.join(OUT_DIR, f"job_{job.job_id()}.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)
        print(f"Submitted job {job.job_id()}. Waiting in the queue; it is safe to stop this script and")
        print(f"collect later with:  python hardware/run_ibm.py --job-id {job.job_id()}")
    rows = analyse(plan, counts_from_result(job.result()))
    print_report(rows, backend_name)
    save(dict(manifest, results=rows), backend_name)


if __name__ == "__main__":
    main()
