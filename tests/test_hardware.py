"""Checks of the device experiments that need no IBM account."""
import glob
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hardware"))
import run_study as rs  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "hardware")
SCAN_A = os.path.join(RESULTS, "scan_ibm_kingston_20261127_012725.json")
SCAN_B = os.path.join(RESULTS, "scan_ibm_kingston_20261127_013741.json")
LOOP = os.path.join(RESULTS, "loop_ibm_kingston_20261127_013153.json")


def load(path):
    with open(path) as fh:
        return json.load(fh)


@pytest.mark.parametrize("entry", rs.SCAN, ids=[e[0] for e in rs.SCAN])
def test_circuits_give_the_ideal_distribution(entry):
    name, N, x, obstacles, variants, ks = entry
    problem = rs.scan_problem(N, obstacles, x)
    for variant in variants:
        for k in (ks if variant == "lean" else (1,)):
            assert rs.ideal_error(problem, k, variant) < 1e-9


def test_reduced_circuit_for_every_planning_problem_of_the_task():
    for problem in rs.reachable_problems(rs.loop_task()):
        assert rs.ideal_error(problem, 1, "lean") < 1e-9


@pytest.mark.parametrize("path", [SCAN_A, SCAN_B, LOOP], ids=["scan_A", "scan_B", "loop"])
def test_stored_analysis_follows_from_the_raw_counts(path):
    record = load(path)
    stored = [r for r in record["analysis"]["rows"] if r["arm"] == "all"]
    assert len(stored) == len(record["plan"])
    rng = np.random.default_rng(0)
    for item, counts, row in zip(record["plan"], record["counts"], stored):
        problem = rs.problem_from_spec(item["spec"])
        hist = rs.histogram(item, counts)
        # with gate twirling the service runs 1536 shots per execution where 1500 were requested
        assert hist.sum() == row["shots"] >= item["shots"] * len(item["masks"])
        new = rs.summarise(problem, item["k"], hist, rng)
        for key in ("shots", "accepted", "accept", "gain", "r", "tv", "nu_hat"):
            if row.get(key) is not None:
                assert new[key] == pytest.approx(row[key], abs=1e-12)


def test_shot_totals():
    assert rs.total_shots(load(SCAN_A)["plan"]) == 207000
    assert rs.total_shots(load(LOOP)["plan"]) == 105000
    b = load(SCAN_B)
    assert sum(sum(sum(c.values()) for c in rows) for rows in b["counts"]) == 211968


def test_four_circuits_save_rollouts_in_both_scans():
    def winners(path):
        rows = [r for r in load(path)["analysis"]["rows"] if r["arm"] == "all" and r["k"] > 0]
        return {(r["name"], r["variant"], r["k"]) for r in rows if (r["gain"] - 1) / r["gain_se"] > 3}
    both = winners(SCAN_A) & winners(SCAN_B)
    assert both == {("N2M1", "lean", 1), ("N3M1", "lean", 1), ("N3M2", "lean", 1), ("N3M1", "full", 1)}


def test_every_job_has_a_manifest():
    for path in glob.glob(os.path.join(RESULTS, "*_ibm_kingston_*.json")):
        job = load(path)["job_id"]
        assert os.path.exists(os.path.join(RESULTS, f"job_{job}.json"))
