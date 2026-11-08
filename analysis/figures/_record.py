"""Run a figure script while recording every data value it draws and every text it prints.
Usage: python3 figures/_record.py fig3   ->  figures/fig3.values.json  {"numbers": [...], "texts": [...]}
The recorded values can be compared with results/summary.json."""
import json
import runpy
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np                      # noqa: E402
from matplotlib.axes import Axes        # noqa: E402

HERE = Path(__file__).resolve().parent
NUMS, TXT, TXTC = [], [], []


def flat(x):
    try:
        a = np.asarray(x, dtype=float).ravel()
        NUMS.extend(float(v) for v in a if np.isfinite(v))
    except (TypeError, ValueError):
        pass


def wrap(name, pick):
    orig = getattr(Axes, name)

    def f(self, *a, **k):
        try:
            pick(a, k)
        except Exception:
            pass
        return orig(self, *a, **k)
    setattr(Axes, name, f)


wrap("bar", lambda a, k: flat(a[1] if len(a) > 1 else k.get("height")))
wrap("barh", lambda a, k: flat(a[1] if len(a) > 1 else k.get("width")))
wrap("plot", lambda a, k: [flat(x) for x in a if not isinstance(x, str)])
wrap("errorbar", lambda a, k: flat(a[1] if len(a) > 1 else k.get("y")))
wrap("scatter", lambda a, k: (flat(a[0]), flat(a[1])))
wrap("imshow", lambda a, k: flat(a[0]))
wrap("fill_between", lambda a, k: [flat(x) for x in a[1:3]])
wrap("text", lambda a, k: (TXT.append(str(a[2])), TXTC.append([str(a[2]), str(k.get("color"))])))
wrap("annotate", lambda a, k: TXT.append(str(a[0])))

name = sys.argv[1]
sys.argv = [str(HERE / f"{name}.py")]
import os                               # noqa: E402
os.chdir(HERE)
sys.path.insert(0, str(HERE))
runpy.run_path(str(HERE / f"{name}.py"), run_name="__main__")
json.dump({"numbers": NUMS, "texts": TXT, "text_colors": TXTC}, open(HERE / f"{name}.values.json", "w"))
