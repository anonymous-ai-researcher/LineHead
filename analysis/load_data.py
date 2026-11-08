"""Loading the summary of all experiments.

analysis/import_measured.py reads the summary files that the experiment scripts write (m0/results, e3/results,
pilot1/results) and stores their values in results/summary.json, a nested dict with one block per experiment.
analysis/schema.json fixes the structure of that file: every block, every key, and the design constants (models,
formats, arms, seeds). load_merged() returns the schema with the imported values filled in; a cell that has not
been imported is None, so a check or figure that needs it fails loudly instead of using a stale value.
"""
import copy
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent
ROOT = DATA.parent
SCHEMA = DATA / "schema.json"
SUMMARY = ROOT / "results" / "summary.json"
META = ("status", "config_ref")


def leaves(d, path=()):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from leaves(v, path + (k,))
    else:
        yield path


def overlay(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            overlay(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def load_schema():
    return json.load(open(SCHEMA))


def load_measured(path=None):
    p = Path(path) if path else SUMMARY
    return json.load(open(p)) if p.exists() else {}


def load_merged(schema=None, summary=None):
    """The schema with every imported value filled in (cells not yet imported are None)."""
    P = copy.deepcopy(schema if isinstance(schema, dict) else load_schema())
    M = summary if isinstance(summary, dict) else load_measured(summary)
    return overlay(P, M)


def is_measured(x):
    """Every value in the summary comes from the result files (kept for the figure scripts' interface)."""
    return x is not None


def sub_measured(P, block, sub):
    def full(d):
        if isinstance(d, dict):
            return all(full(v) for k, v in d.items() if k not in META)
        return d is not None
    return full(P.get(block, {}).get(sub))


def nan_missing(d):
    """Empty bins or strata are stored as null; the checks treat them as NaN (any comparison fails, no crash)."""
    if isinstance(d, dict):
        return {k: float("nan") if v is None else nan_missing(v) for k, v in d.items()}
    if isinstance(d, list):
        return type(d)(float("nan") if x is None else nan_missing(x) for x in d) if not isinstance(d, str) else d
    return d


def mmax(*args, **kw):
    return max(*args, **kw)


def mmin(*args, **kw):
    return min(*args, **kw)


def mmean(xs):
    xs = list(xs)
    return sum(float(x) for x in xs) / len(xs)


def msum(xs, start=0):
    return sum((float(x) for x in xs), start)
