"""Numerical check of the matching-head construction of m_realize.tex (M^{>=1}_{s,t} phi, strict causal softmax,
BOS sink, fresh Gaussian names e_a ~ N(0, I/d) per sequence). For random token sequences and random phi bits it
compares the thresholded head output with the exact truth value at every position, for several d, with
beta = (8/3) ln(4(n+1)) and eps = 1/8 as in the proof; it also reports the event E_eps (all |<e_a,e_b>| < eps,
| |e_a|^2 - 1 | < eps) whose probability the proof bounds by 1 - 2 m^2 exp(-d eps^2/16)."""
import numpy as np


def run(n=400, n_names=150, s=2, t=3, d=64, trials=50, seed=0, eps=1/8):
    rng = np.random.default_rng(seed)
    beta = (8 / 3) * np.log(4 * (n + 1))
    err_pos = E_hold = seq_err = 0
    for _ in range(trials):
        is_name = rng.random(n) < 0.4
        is_name[0] = False                                  # position 1 = BOS
        ids = rng.integers(0, n_names, n)
        E = rng.standard_normal((n_names, d)) / np.sqrt(d)
        G = E @ E.T
        off = G - np.diag(np.diag(G))
        ok_E = np.all(np.abs(np.diag(G) - 1) < eps) and np.all(np.abs(off) < eps)
        E_hold += ok_E
        phi = rng.random(n) < 0.3
        phi[0] = False
        def carried(m):                                     # c^{(m)}_i = e_{w_{i-m}} or 0
            C = np.zeros((n, d))
            for i in range(m, n):
                if is_name[i - m]:
                    C[i] = E[ids[i - m]]
            return C
        Cs, Ct = carried(s), carried(t)
        S = beta * (Cs @ Ct.T + phi[None, :].astype(float))  # scores q_i . k_j
        S[:, 0] = 1.5 * beta                                 # BOS sink
        bad = 0
        for i in range(1, n):
            sc = S[i, :i]                                    # strict masking j < i (0-based: j in 0..i-1)
            w = np.exp(sc - sc.max()); w /= w.sum()
            o = w[1:].sum()
            bit = max(0.0, 2 * o - 0.5) - max(0.0, 2 * o - 1.5)      # ReLU(2o-1/2) - ReLU(2o-3/2)
            bit = 1.0 if abs(bit - 1) < 1e-9 else (0.0 if abs(bit) < 1e-9 else bit)
            truth = i - s >= 0 and is_name[i - s] and any(
                j - t >= 0 and is_name[j - t] and ids[j - t] == ids[i - s] and phi[j] for j in range(1, i))
            bad += (bit not in (0.0, 1.0)) or (bool(bit) != bool(truth))
        err_pos += bad
        seq_err += bad > 0
    return dict(d=d, beta=round(beta, 2), P_event=E_hold / trials, seq_with_error=seq_err / trials,
                pos_errors=err_pos)




def e3_scale():
    """E3 scale: d_model = 256 but each head has dimension 32, so a matching head compares names in <= 32 dims."""
    for d in (32, 64, 128, 256):
        r = run(n=800, n_names=64, d=d, trials=20, seed=1)
        r["pos_error_rate"] = r["pos_errors"] / (20 * 799)
        print("E3 scale (n=800, 64 names):", r, flush=True)


if __name__ == "__main__":
    for d in (16, 32, 64, 128, 256, 1024, 4096):
        print(run(d=d), flush=True)
    e3_scale()
