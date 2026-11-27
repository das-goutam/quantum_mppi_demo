"""Figure of the device experiments.

Run from the repository root:  python hardware/make_figure.py
Reads the stored results in results/hardware and writes figures/hardware.pdf and figures/hardware.png.
"""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9,
                            "font.family": "STIXGeneral", "mathtext.fontset": "stix"})
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_study as rs
src = os.path.join(rs.ROOT, "results", "hardware")
out = os.path.join(rs.ROOT, "figures")

J = lambda f: json.load(open(os.path.join(src, f)))
A, B, L = (J("scan_ibm_kingston_20261127_012725.json"), J("scan_ibm_kingston_20261127_013741.json"),
           J("loop_ibm_kingston_20261127_013153.json"))
rows = lambda rec: {(r['name'], r['variant'], r['k']): r for r in rec['analysis']['rows'] if r['arm'] == 'all' and r['k'] > 0}
a, b = rows(A), rows(B)
keys = sorted(a, key=lambda k: a[k]['two_qubit_gates'])
fig, (p1, p2, p3) = plt.subplots(1, 3, figsize=(7.1, 2.45), gridspec_kw=dict(width_ratios=[1.45, 1, 1]))
x = np.arange(len(keys))
p1.axhline(1, color="0.5", lw=0.6, ls="--")
p1.plot(x, [a[k]['gain_ideal'] for k in keys], "_", color="0.35", ms=7, label="ideal device")
for d, fill, lab in ((a, True, "scan A"), (b, False, "scan B")):
    for std, mk in ((False, "o"), (True, "s")):
        sel = [i for i, k in enumerate(keys) if (k[1] == 'full') == std]
        p1.errorbar(x[sel], [d[keys[i]]['gain'] for i in sel], yerr=[2 * d[keys[i]]['gain_se'] for i in sel], fmt=mk,
                    color="tab:blue" if fill else "tab:orange", mfc=("tab:blue" if fill else "white"), ms=3.6, lw=0.7,
                    label=(lab if not std else None))
p1.set_xticks(x)
p1.set_xticklabels([str(a[k]['two_qubit_gates']) for k in keys], rotation=90, fontsize=6.5)
p1.set_xlabel("two-qubit gates of the compiled circuit")
p1.set_ylabel("gain $\\tilde\\Gamma_k$")
p1.legend(fontsize=6.5, frameon=False, loc="upper left")
p1.set_title("(a)", fontsize=9, loc="left")
# (b) accepted distance against the bound of item 1
pts = []
for rec in (A, B, L):
    for item, counts in zip(rec['plan'], rec['counts']):
        if item['k'] == 0: continue
        pr = rs.problem_from_spec(item['spec']); feas = pr.feasible_mask; M = int(feas.sum())
        if M < 2: continue
        h = rs.histogram(item, counts).astype(float); Pt = h / h.sum()
        P, q = rs.ideal_distribution(pr, item['k'])
        tau0 = 0.5 * np.abs(Pt - P).sum(); qt = Pt[feas].sum()
        tau = 0.5 * np.abs(Pt[feas] / qt - 1 / M).sum()
        pts.append((tau0 / max(q, qt), tau, rec['kind']))
for kind, mk, c, lab in (("scan", "o", "tab:blue", "single circuits"), ("loop", "^", "tab:green", "closed-loop circuits")):
    s = [(u, v) for u, v, kd in pts if kd == kind]
    p2.semilogy([u for u, _ in s], [max(v, 8e-4) for _, v in s], mk, color=c, ms=3.2, mfc="none", label=lab)
g = np.linspace(0.15, 1.05, 50)
p2.semilogy(g, g, "k-", lw=0.7)
p2.text(0.22, 0.42, "bound", fontsize=6.5)
p2.set_xlim(0.15, 1.05); p2.set_ylim(6e-4, 1.5)
p2.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
p2.set_xlabel("bound $\\tau_0/\\max\\{q_k,\\tilde q_k\\}$")
p2.set_ylabel("distance of accepted samples\nfrom uniform")
p2.legend(fontsize=6.5, frameon=False, loc="lower right", handletextpad=0.2)
p2.set_title("(b)", fontsize=9, loc="left")
runs = L['analysis']['loop']['runs']
for name, c, mk, ls, lab in (("classical", "k", "o", "-", "classical"), ("device, k = 1", "tab:blue", "s", "-", "device, $k=1$"),
                             ("device, adaptive", "tab:red", "^", "-", "device, adaptive"),
                             ("model, k = 1", "tab:blue", None, ":", "model, $k=1$"), ("model, adaptive", "tab:red", None, ":", "model, adaptive")):
    s = [r for r in runs if r['sampler'] == name]
    p3.errorbar([r['B'] for r in s], [r['fail'] for r in s], yerr=[r['fail_se'] for r in s] if mk else None, color=c, marker=mk,
                ls=ls, ms=3.2, lw=0.9, label=lab)
p3.set_xlabel("rollouts per control step $B$")
p3.set_ylabel("failure rate")
p3.set_xticks([6, 9, 12, 18, 24])
p3.legend(fontsize=6.5, frameon=False)
p3.set_title("(c)", fontsize=9, loc="left")
fig.tight_layout(w_pad=0.8)
fig.savefig(os.path.join(out, "hardware.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(out, "hardware.png"), bbox_inches="tight", dpi=200)
print("points", len(pts), "ratio median %.3f max %.3f" % (np.median([v / u for u, v, _ in pts]), max(v / u for u, v, _ in pts)))
