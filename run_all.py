"""Run every step and store the console output next to the data.

    python run_all.py            # everything, about 20 minutes on two cores
    python run_all.py --quick    # short benchmark, about 5 minutes

Console output goes to results/NN_name.txt, data to results/*.json, pictures to figures/.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
STEPS = [("01_show_circuit.py", []), ("02_single_update.py", []), ("03_closed_loop.py", []),
         ("04_benchmark.py", ["--quick"] if "--quick" in sys.argv else []), ("05_noise_study.py", [])]

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
for script, extra in STEPS:
    start = time.time()
    print("== %s" % script, flush=True)
    done = subprocess.run([sys.executable, os.path.join(ROOT, "examples", script)] + extra, cwd=ROOT,
                          capture_output=True, text=True)
    with open(os.path.join(ROOT, "results", script.replace(".py", ".txt")), "w") as fh:
        fh.write(done.stdout)
    print(done.stdout[-600:] if done.returncode == 0 else done.stderr[-2000:])
    if done.returncode != 0:
        sys.exit("%s failed" % script)
    print("   %.0f s\n" % (time.time() - start), flush=True)
print("Tests")
subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
