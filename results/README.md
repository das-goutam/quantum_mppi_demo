# Results

Everything in this folder is written by the scripts in `examples/`. `python run_all.py` regenerates it.
The random seeds are fixed, so a rerun with the same package versions reproduces the numbers.

| File | Written by | Content |
|---|---|---|
| `01_show_circuit.txt`, `circuit.json` | `01_show_circuit.py` | The planning problem, the rollout circuit as text, gate counts before and after compilation |
| `02_single_update.txt`, `single_update.json` | `02_single_update.py` | Amplitudes after each stage, acceptance against theory, shot counts on the feasible plans, error of one MPPI update |
| `03_closed_loop.txt`, `closed_loop_episode.json` | `03_closed_loop.py` | One episode printed step by step for the classical and the quantum sampler |
| `04_benchmark.txt`, `benchmark.json` | `04_benchmark.py` | Violation rate, steps without a verified sample, and mean cost, for six samplers and five budgets |
| `05_noise_study.txt`, `noise_study.json` | `05_noise_study.py` | Acceptance and saving against the gate error, for depth 1 and depth 2 |

The pictures are in [`../figures`](../figures).

## How to read `benchmark.json`

```
episodes               episodes per entry
budgets                rollout evaluations per control step
samplers.<name>.failed                 fraction of episodes with a constraint violation, one value per budget
samplers.<name>.failed_se              its standard error
samplers.<name>.steps_without_sample   fraction of control steps in which no proposal passed the check
samplers.<name>.mean_cost              mean cost of the episodes without violation
```
