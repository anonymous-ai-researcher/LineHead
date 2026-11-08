"""Validate the imported summary (results/summary.json) in one call.

  python analysis/validate.py

1. analysis/check_rules.py: the theory constraints, the preregistered predictions and the consistency rules of
   VALIDATION_RULES.md (groups A to F and the rules of every preregistered set).
2. analysis/real_rules.py: computability and common sense (group R). A failure of R1 to R6 or R8 to R11 is a bug in
   a run, a summary script or the import. R7 states what the measurement normally gives; a violation is printed
   as a warning: check the configuration, the seeds and the code, and report it as a finding once they are
   confirmed. Never change a result to remove a warning.
Both steps report every check as passed, failed, warned or skipped (a skipped check is listed with its reason and is
never counted as passed); --verbose adds the counts per rule.
Exit code 1 if any rule of step 1 or any exact rule of step 2 fails.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from load_data import load_merged, nan_missing  # noqa: E402
import real_rules  # noqa: E402

r1 = subprocess.run([sys.executable, str(HERE / "check_rules.py")] + sys.argv[1:])

P = load_merged()
P["mechanism"] = nan_missing(P["mechanism"])
FAIL, WARN, N, COUNT = [], [], [0], {}


def chk(rid, ok, msg=""):
    N[0] += 1
    c = COUNT.setdefault(rid, [0, 0, 0])
    if ok:
        c[0] += 1
        return
    if rid.startswith("R7"):
        c[2] += 1
        WARN.append(f"{rid}: {msg}")
    else:
        c[1] += 1
        FAIL.append(f"{rid}: {msg}")


real_rules.run(chk, P)
if "--verbose" in sys.argv:
    print("rule      passed  failed  warned")
    for rid in sorted(COUNT):
        print(f"{rid:<8}{COUNT[rid][0]:>8}{COUNT[rid][1]:>8}{COUNT[rid][2]:>8}")
print(f"group R: {N[0]} checks, {len(FAIL)} failed, {len(WARN)} warnings, 0 skipped")
for f in FAIL:
    print("FAIL", f)
for w in WARN:
    print("WARN", w)
sys.exit(1 if (r1.returncode or FAIL) else 0)
