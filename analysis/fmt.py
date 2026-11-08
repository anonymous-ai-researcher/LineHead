"""Single rounding rule for every number shown in the paper: round half up at the displayed precision,
after clearing binary-float noise at 1e-10 (so 0.8935 -> .894 and a mean of 0.89349999999 -> .894).
Used by the table and figure scripts."""
from decimal import Decimal, ROUND_HALF_UP, ROUND_FLOOR, ROUND_CEILING


def rnd(x, d=3):
    q = Decimal(1).scaleb(-d)
    return float(Decimal(f"{x:.10f}").quantize(q, rounding=ROUND_HALF_UP))


def fstr(x, d=3):
    q = Decimal(1).scaleb(-d)
    return str(Decimal(f"{x:.10f}").quantize(q, rounding=ROUND_HALF_UP))


def ipts(x):
    """Percentage points as an integer, half up."""
    return int(Decimal(f"{100 * x:.10f}").quantize(Decimal(1), rounding=ROUND_HALF_UP))


def lr_round(xs, d=2):
    """Largest-remainder rounding: shares that sum to 1 are shown with d decimals that also sum to 1."""
    from decimal import Decimal
    q = 10 ** d
    raw = [Decimal(f"{x:.10f}") * q for x in xs]
    base = [int(r) for r in raw]
    short = int(round(sum(raw))) - sum(base)
    order = sorted(range(len(xs)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:max(short, 0)]:
        base[i] += 1
    return [b / q for b in base]


def fbound(x, d=3, side="lower", sig=None):
    """A bound stated in the text ("at least", "between a and b"): rounded toward the side that keeps it true
    (floor for a lower bound, ceiling for an upper bound), with d decimals or, if sig is given, sig significant
    digits. 0.9875 as a lower bound is 0.987, not 0.988."""
    v = Decimal(f"{x:.10f}")
    if sig is not None:
        d = sig - 1 - v.adjusted()
    return str(v.quantize(Decimal(1).scaleb(-d), rounding=ROUND_FLOOR if side == "lower" else ROUND_CEILING))
