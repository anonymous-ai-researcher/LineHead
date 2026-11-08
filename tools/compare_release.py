"""Compare this release with another copy of the experiment code (for example the copy the runs used), file by file.

  python tools/compare_release.py OTHER_ROOT

OTHER_ROOT holds the folders m0/, e3/, pilot1/ and theory_checks/. Every file of these folders in this repository is
put in one category:
  identical      same bytes
  comments       Python: the same syntax tree (ast.dump) once docstrings are removed; shell: the same text once
                 comments and blank lines are removed
  strings        Python: the same syntax tree once docstrings and every string literal are blanked (help texts,
                 messages, default paths)
  code           any other difference
  other          a non-Python, non-shell file whose bytes differ
  new            no counterpart in OTHER_ROOT
Docstrings reach a program only through __doc__; in this repository that is the --help text of the scripts that
pass description=__doc__ to argparse.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLDERS = ("m0", "e3", "pilot1", "theory_checks")


def _tree(src, blank_strings=False):
    t = ast.parse(src)
    for n in ast.walk(t):
        body = getattr(n, "body", None)
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and body \
                and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            n.body = body[1:] or [ast.Pass()]
        if blank_strings and isinstance(n, ast.Constant) and isinstance(n.value, str):
            n.value = ""
        if blank_strings and isinstance(n, ast.JoinedStr):
            n.values = []
    return ast.dump(t)


def _shell(text):
    out = []
    for line in text.splitlines():
        s = line.split("#", 1)[0].rstrip() if not line.lstrip().startswith("#!") else ""
        if s.strip():
            out.append(s)
    return out


def category(a, b):
    if a.read_bytes() == b.read_bytes():
        return "identical"
    if a.suffix == ".py":
        sa, sb = a.read_text(), b.read_text()
        if _tree(sa) == _tree(sb):
            return "comments"
        if _tree(sa, True) == _tree(sb, True):
            return "strings"
        return "code"
    if a.suffix == ".sh":
        return "comments" if _shell(a.read_text()) == _shell(b.read_text()) else "code"
    return "other"


def main():
    other = Path(sys.argv[1]).resolve()
    cats = {}
    for d in FOLDERS:
        for f in sorted((ROOT / d).rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                rel = f.relative_to(ROOT)
                g = other / rel
                cats.setdefault(category(f, g) if g.exists() else "new", []).append(str(rel))
    for k in ("identical", "comments", "strings", "code", "other", "new"):
        files = cats.get(k, [])
        print(f"{k}: {len(files)}")
        if k != "identical":
            for f in files:
                print(f"  {f}")


if __name__ == "__main__":
    main()
