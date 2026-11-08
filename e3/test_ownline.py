"""Unit test of the position control: position-controlled k sets have no own-line route for any listed
construction (ownline_check.py reads the tokens only), and the checker does flag random orders.
Run: python test_ownline.py"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(*a):
    return subprocess.run([sys.executable, *a], cwd=HERE, capture_output=True, text=True)


def main():
    t = Path(tempfile.mkdtemp())
    env = dict(os.environ)
    env.pop("E2E_TINY", None)
    for ser in ("AL", "NC"):
        for sp in ("kset", "kid"):
            r = subprocess.run([sys.executable, "gen_e3.py", "--task", "dia", "--ser", ser, "--split", sp, "--pairs", "40",
                                "--seed", "3", "--out", str(t / f"dia_{ser}_{sp}.jsonl")], cwd=HERE, env=env, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, "gen_e3.py", "--task", "dia", "--ser", "AL", "--split", "kown", "--pairs", "40",
                        "--seed", "6", "--out", str(t / "dia_AL_kown.jsonl")], cwd=HERE, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    files = sorted(str(p) for p in t.glob("*.jsonl"))
    r = run("ownline_check.py", *files, "--out", str(t / "ok.json"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert all(v["violations"] == 0 and v["positives"] > 0 for v in json.load(open(t / "ok.json")).values())
    # negative control: random orders (the OOD test split) checked as if position-controlled must show violations
    r = subprocess.run([sys.executable, "gen_e3.py", "--task", "dia", "--ser", "AL", "--split", "ood", "--pairs", "300",
                        "--seed", "2", "--out", str(t / "x_AL_kset_random.jsonl")], cwd=HERE, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = run("ownline_check.py", str(t / "x_AL_kset_random.jsonl"), "--out", str(t / "bad.json"))
    assert r.returncode != 0 and json.load(open(t / "bad.json"))["x_AL_kset_random.jsonl"]["violations"] > 0
    print("ownline test ok")


if __name__ == "__main__":
    main()
