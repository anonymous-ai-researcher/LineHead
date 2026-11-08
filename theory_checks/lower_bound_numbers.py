"""Explicit parameters of the counterexample of Theorem new:thm-lower (theory_new.tex) as functions of (d, r):
R = d(r+1)+4, K = smallest even integer > 4R+4 = 4R+6, Delta = K+2R+1 (degree of s), hub degree K+1,
position of the query s in the hub's list k = K/2+1, and the token length of one piece (AL and NC, word-level
tokens as in gen_e3.py, line breaks included). The number m >= 2I+2 of saturating blocks is not elementary in d
and is not tabulated."""


def line_len(ser, m):            # tokens of one line with m entries, plus its <NL>
    if ser == "AL":
        return (5 + 2 * m - 1 if m else 6) + 1
    return (6 + 5 * m - 1 + 1 if m else 8) + 1


def piece_len(ser, d, r):
    R = d * (r + 1) + 4
    K = 4 * R + 6
    return (line_len(ser, K + 1) + 2 * R * line_len(ser, 1) + line_len(ser, K + 2 * R + 1)
            + K * line_len(ser, 1) + (R + 1) * line_len(ser, 0) + K * line_len(ser, 1) + (2 * R + 2) * line_len(ser, 0))


def params(d, r):
    R = d * (r + 1) + 4
    K = 4 * R + 6
    return dict(R=R, K=K, Delta=K + 2 * R + 1, hub=K + 1, k=K // 2 + 1,
                piece_AL=piece_len("AL", d, r), piece_NC=piece_len("NC", d, r))


if __name__ == "__main__":
    print("d r | R K Delta k_query | piece tokens AL NC")
    for d in (1, 2, 3, 4, 6, 8, 12):
        for r in range(0, 7):
            p = params(d, r)
            print(f"{d} {r} | {p['R']} {p['K']} {p['Delta']} {p['k']} | {p['piece_AL']} {p['piece_NC']}")
    print("round depth L (d = L, r = L-1):")
    for L in (2, 3, 4, 6, 8, 12):
        p = params(L, L - 1)
        print(f"L={L}: R={p['R']} K={p['K']} Delta={p['Delta']} k_query={p['k']} piece AL={p['piece_AL']} NC={p['piece_NC']}")
    assert all(params(d, r)["Delta"] == 6 * d * (r + 1) + 31 for d in range(1, 20) for r in range(0, 20))
    print("check: Delta = 6d(r+1)+31 for all d<20, r<20")
