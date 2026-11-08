"""Unit tests (no model): E6 demonstrations are resume-invariant; a truncated last line is skipped on resume;
records carry n_gpus; imp_compute counts GPU hours as sec x n_gpus. Run: python test_run_llm_misc.py"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from instances import load, prompts_for
from names import POOL
from run_llm import attach_demos

HERE = Path(__file__).resolve().parent


def test_demos_resume_invariant():
    recs = [r for r in load(HERE / "data/m0_instances.jsonl") if r["pid"] == "D1a"][:6]
    demo = load(HERE / "data/demo_instances.jsonl")
    full = [p for r in recs for p in prompts_for(r, POOL, serializations=("EL", "AL"))]
    a = attach_demos([dict(p) for p in full], demo, POOL, ("EL", "AL"), 3)
    b = attach_demos([dict(p) for p in full[7:]][::-1], demo, POOL, ("EL", "AL"), 3)     # resumed, other order
    pa = {(p["id"], p["ser"], p["tpl"]): p["prompt"] for p in a}
    assert all(pa[(p["id"], p["ser"], p["tpl"])] == p["prompt"] for p in b)


def test_resume_truncated_and_n_gpus():
    out = Path(tempfile.mkdtemp()) / "r.jsonl"
    args = [sys.executable, str(HERE / "run_llm.py"), "--backend", "mock", "--instances", str(HERE / "data/m0_instances.jsonl"),
            "--limit", "2", "--out", str(out), "--name-report", str(out.parent / "nf.json")]
    subprocess.run(args, check=True, capture_output=True)
    lines = out.read_text().splitlines()
    out.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:20])           # truncated last line
    r = subprocess.run(args, check=True, capture_output=True, text=True)
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert "truncated" in r.stdout and len(recs) == len(lines)
    assert all(x["n_gpus"] == 0 for x in recs)


def test_compute_gpu_hours():
    sys.path.insert(0, str(HERE.parent / "analysis"))
    import import_measured as IM
    d = Path(tempfile.mkdtemp())
    rec = {"sec": 3600.0, "n_gpus": 4}
    (d / "e1_Qwen2.5-72B-Instruct.jsonl").write_text(json.dumps(rec) + "\n" + json.dumps(rec) + "\n")
    (d / "e1_Qwen2.5-7B-Instruct.jsonl").write_text(json.dumps({"sec": 3600.0, "n_gpus": 1}) + "\n")
    M = IM.tree()
    IM.imp_compute(M, [d])
    assert M["compute"]["e1_hours"]["Qwen2.5-72B"] == 8.0 and M["compute"]["e1_hours"]["Qwen2.5-7B"] == 1.0


if __name__ == "__main__":
    test_demos_resume_invariant(); test_resume_truncated_and_n_gpus(); test_compute_gpu_hours()
    print("ok: resume-invariant demonstrations; truncated line skipped; n_gpus recorded; GPU hours = sec x n_gpus")
