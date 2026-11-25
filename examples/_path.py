"""Lets the examples run from a fresh clone without installing the package."""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
FIGURES = os.path.join(ROOT, "figures")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)


def save_json(name, data):
    path = os.path.join(RESULTS, name)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    return os.path.relpath(path, ROOT)


def figure_path(name):
    return os.path.join(FIGURES, name)
