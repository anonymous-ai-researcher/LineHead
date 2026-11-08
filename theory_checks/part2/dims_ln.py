"""Item (D): dimension bounds for the near-orthogonality event E_eps (Lemma mr:conc and its sharpened form,
layernorm.tex), and a simulation of the matching head with pre-LayerNorm (mirrored coordinates + dilution pair).

Part 1: the smallest d for which each bound guarantees Pr[E_eps] >= 1 - delta, for m names:
  paper      : 2 m^2 exp(-d eps^2/16) <= delta
  chernoff   : m e^{-d a} + m e^{-d b} + m(m-1) e^{-d c} <= delta  (exact Chernoff exponents, Lemma ln:conc2)
  asym       : the same with separate tolerances eps1 (norm below), eps2 (norm above), eps3 (inner products),
               eps1 + max(eps2, eps3) <= 2 eps, which keeps the margin 1/2 - eps of Proposition mr:match
  exact-union: union bound with exact chi^2 tails and the exact tail of <e_a,e_b> (numerical integration)
  Monte Carlo: empirical Pr[E_eps] at the bound's d and at d = 32 (one E3 head)
Part 2: one matching head M_{2,3} phi with three 1-local copy layers before it, every sublayer reading
  LayerNorm(h); compares the exact bit with the truth at every position (as proofs12/m_attention_sim.py).
Usage: python3 -I dims_ln.py
"""
import math

import numpy as np
from scipy import integrate, optimize, stats


def a_(e):  # norm above:  Pr[X/d >= 1+e] <= exp(-d a)
    return (e - math.log1p(e)) / 2


def b_(e):  # norm below
    return (-e - math.log1p(-e)) / 2


def c_(e):  # inner product, one side: Pr[<e_a,e_b> >= e] <= exp(-d c)
    s = math.sqrt(1 + 4 * e * e)
    return (s - 1 - math.log((s + 1) / 2)) / 2


def d_paper(m, eps, delta):
    return 16 / eps ** 2 * math.log(2 * m * m / delta)


def d_from(fail, lo=1, hi=10 ** 6):
    """least integer d with fail(d) <= delta (fail decreasing)."""
    while lo < hi:
        mid = (lo + hi) // 2
        if fail(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def d_chernoff(m, e1, e2, e3, delta):
    A, B, C = a_(e2), b_(e1), c_(e3)
    return d_from(lambda d: m * math.exp(-d * A) + m * math.exp(-d * B) + m * (m - 1) * math.exp(-d * C) <= delta)


def d_asym(m, eps, delta):
    best = None
    for e1 in np.linspace(0.02, 2 * eps - 0.02, 97):
        e23 = 2 * eps - e1
        if e23 >= 1 or e1 >= 1:
            continue
        d = d_chernoff(m, e1, e23, e23, delta)
        if best is None or d < best[0]:
            best = (d, round(float(e1), 3), round(float(e23), 3))
    return best


def inner_tail(d, e):
    """exact Pr[|<e_a,e_b>| >= e] for independent e_a, e_b ~ N(0, I/d): condition on r^2 = |e_b|^2."""
    f = lambda x: 2 * stats.norm.sf(e * d / math.sqrt(x)) * stats.chi2.pdf(x, d)   # x = d |e_b|^2
    # integrate over a finite range that holds the mass of chi^2_d, with breakpoints; quad on (0, inf)
    # missed the narrow peak for d in the hundreds and returned values 30 orders of magnitude too small
    lo, hi = max(0.0, d - 12 * math.sqrt(2 * d)), d + 30 * math.sqrt(2 * d)
    val, _ = integrate.quad(f, lo, hi, limit=500, points=[d * 0.9, d, d * 1.1, d * 1.3])
    return val


def d_exact_union(m, eps, delta):
    def fail(d):
        pn = stats.chi2.sf(d * (1 + eps), d) + stats.chi2.cdf(d * (1 - eps), d)
        return m * pn + m * (m - 1) / 2 * inner_tail(d, eps) <= delta
    return d_from(fail, 8, 20000)


def mc_event(m, d, eps, trials=200, seed=0):
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(trials):
        E = rng.standard_normal((m, d)) / math.sqrt(d)
        G = E @ E.T
        off = G - np.diag(np.diag(G))
        ok += np.all(np.abs(np.diag(G) - 1) < eps) and np.all(np.abs(off) < eps)
    return ok / trials


def part1():
    eps, delta = 0.25, 0.05
    print(f"Part 1: eps = {eps}, delta = {delta}")
    for m in (64, 150, 600):
        dp = d_paper(m, eps, delta)
        dc = d_chernoff(m, eps, eps, eps, delta)
        da = d_asym(m, eps, delta)
        de = d_exact_union(m, eps, delta)
        print(f"  m = {m:4d}: paper 16/eps^2 bound d >= {math.ceil(dp)};  sharpened Chernoff d >= {dc};  "
              f"asymmetric tolerances d >= {da[0]} (eps1={da[1]}, eps2=eps3={da[2]});  exact-tail union bound d >= {de}")
        for d in sorted({dc, de, 32}):
            nt = 20000 if d > 32 else 200
            pm = mc_event(m, d, eps, trials=nt)
            print(f"      Monte Carlo Pr[E_eps] at d = {d}: {pm:.3f} (s.e. {math.sqrt(max(pm * (1 - pm), 1e-12) / nt):.3f}, {nt} trials)")
    grid = np.linspace(1e-4, 1 - 1e-4, 10000)
    print(f"  check on 10^4 grid points of (0,1): c >= a everywhere: {all(c_(e) >= a_(e) for e in grid)}; "
          f"b >= a everywhere: {all(b_(e) >= a_(e) for e in grid)}; a >= eps^2/12 everywhere: "
          f"{all(a_(e) >= e * e / 12 for e in grid)}")
    print(f"  constants: a(1/4) = {a_(0.25):.5f}, b(1/4) = {b_(0.25):.5f}, c(1/4) = {c_(0.25):.5f}, eps^2/16 = {0.25**2/16:.5f}")
    # 32-dimensional head: expected number of pairs violating |<e_a,e_b>| < 1/4, m = 64
    p32 = inner_tail(32, 0.25)
    print(f"  d = 32: Pr[|<e_a,e_b>| >= 1/4] = {p32:.3f}; expected violating pairs among 64 names: {p32 * 64 * 63 / 2:.0f}")


def layernorm(h, eps_ln=1e-5):
    mu = h.mean(axis=-1, keepdims=True)
    var = h.var(axis=-1, keepdims=True)
    return (h - mu) / np.sqrt(var + eps_ln)


def part2(n=400, n_names=150, d=256, s=2, t=3, trials=20, seed=0, Z=None, use_ln=True, eps=1 / 8, calibrate=False):
    """Residual layout: [Z, -Z | bos, -bos | phi, -phi | raw name slot (d) | c1, -c1 | c2, -c2 | c3, -c3].
    Copy layer u reads LN(h_{i-1}) (the raw slot for u = 1, the half-difference of the mirrored slot c^{(u-1)} for
    u >= 2), divides by the nominal scale lam = sqrt(D)/sqrt(2 Z^2) and writes the mirrored slot c^{(u)}.
    The matching head reads LN(h) the same way; scores beta * <q, k> / lam^2 with a BOS sink 3/2."""
    rng = np.random.default_rng(seed)
    beta = (8 / 3) * np.log(4 * (n + 1))
    U = max(s, t)
    D = 2 + 2 + 2 + d + 2 * d * U
    if Z is None:
        Z = 40.0 * math.sqrt(d * 0 + 2 * U + 6)       # dilution: Z^2 >> the varying part of |h|^2 (<= 2U(1+eps)+5)
    lam = math.sqrt(D) / math.sqrt(2 * Z * Z)
    err = 0
    for _ in range(trials):
        is_name = rng.random(n) < 0.4
        is_name[0] = False
        ids = rng.integers(0, n_names, n)
        E = rng.standard_normal((n_names, d)) / math.sqrt(d)
        phi = rng.random(n) < 0.3
        phi[0] = False
        h = np.zeros((n, D))
        h[:, 0], h[:, 1] = Z, -Z
        h[0, 2], h[0, 3] = 1, -1
        h[:, 4], h[:, 5] = phi, -phi.astype(float)
        raw = slice(6, 6 + d)
        h[is_name, raw] = E[ids[is_name]]

        def slot(u):        # (plus, minus) slices of c^{(u)}
            base = 6 + d + 2 * d * (u - 1)
            return slice(base, base + d), slice(base + d, base + 2 * d)

        def scale(hh):      # the fixed constant the weights divide by: nominal, or the median LN scale (control)
            if not calibrate:
                return lam
            c = hh - hh.mean(axis=1, keepdims=True)
            return float(np.median(math.sqrt(D) / np.linalg.norm(c, axis=1)))

        def read(Hn, u, lm):  # value of c^{(u)} (u = 0: raw slot) as seen through LN, divided by the constant
            if u == 0:
                return Hn[:, raw] / lm
            p, q = slot(u)
            return (Hn[:, p] - Hn[:, q]) / 2 / lm

        for u in range(1, U + 1):
            Hn = layernorm(h) if use_ln else h * lam
            src = read(Hn, u - 1, scale(h) if use_ln else lam)
            p, q = slot(u)
            new = np.zeros((n, d))
            new[1:] = src[:-1]                       # 1-local head: position i copies i-1 (position 1 keeps 0)
            h[:, p], h[:, q] = new, -new
        Hn = layernorm(h) if use_ln else h * lam
        lm = scale(h) if use_ln else lam
        cs, ct = read(Hn, s, lm), read(Hn, t, lm)
        const = (Hn[:, 0] - Hn[:, 1]) / 2 / lm / Z  # = 1 up to the LN scale of each position
        bos = (Hn[:, 2] - Hn[:, 3]) / 2 / lm
        ph = (Hn[:, 4] - Hn[:, 5]) / 2 / lm
        Q = np.concatenate([cs, const[:, None], 1.5 * const[:, None]], axis=1) * beta
        K = np.concatenate([ct, (ph * (1 - (np.arange(n) == 0)))[:, None], bos[:, None]], axis=1)
        S = Q @ K.T
        for i in range(1, n):
            sc = S[i, :i]
            w = np.exp(sc - sc.max())
            w /= w.sum()
            o = w[1:].sum()
            bit = max(0.0, 2 * o - 0.5) - max(0.0, 2 * o - 1.5)
            bit = 1.0 if abs(bit - 1) < 1e-9 else (0.0 if abs(bit) < 1e-9 else bit)
            truth = i - s >= 0 and is_name[i - s] and any(
                j - t >= 0 and is_name[j - t] and ids[j - t] == ids[i - s] and phi[j] for j in range(1, i))
            err += (bit not in (0.0, 1.0)) or (bool(bit) != bool(truth))
    return err


def main():
    part1()
    print("Part 2: matching head M_{2,3} after three 1-local copy layers, n = 400, 150 names, 20 sequences")
    for d in (64, 128, 256):
        e_no = part2(d=d, use_ln=False)
        e_ln = part2(d=d, use_ln=True)
        e_ln_small_Z = part2(d=d, use_ln=True, Z=1.0, calibrate=True)
        print(f"  d = {d:4d}: wrong positions without LN {e_no}, with pre-LN and dilution {e_ln}, "
              f"with pre-LN and no dilution (Z = 1, weights calibrated to the median LN scale) {e_ln_small_Z}  (of {20 * 399})", flush=True)


if __name__ == "__main__":
    main()
