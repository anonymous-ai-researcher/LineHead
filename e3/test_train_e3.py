"""Unit tests for the hard-since head of train_e3.py (run: python test_train_e3.py)."""
import torch

from train_e3 import Block, Model, alibi_attention, batchify, rightmost_argmax


def test_rightmost_ties():
    s = torch.tensor([[[1.0, 3.0, 3.0, 2.0, 3.0, -1.0]]])
    assert rightmost_argmax(s).item() == 4
    s = torch.zeros(1, 1, 1000)                          # all tied, length 1,000
    assert rightmost_argmax(s).item() == 999
    s[0, 0, 997:] = float("-inf")                        # masked tail
    assert rightmost_argmax(s).item() == 996


def test_block_under_bf16_autocast():
    torch.manual_seed(0)
    T, d = 1000, 64
    blk = Block(d, 8, "local_hardsince").eval()
    with torch.no_grad():                                  # zero q/k weights: every score ties exactly
        blk.qkv.weight[: 2 * d].zero_(); blk.qkv.bias[: 2 * d].zero_()
    x = torch.randn(1, T, d)
    i = torch.arange(T)
    hard_mask = (i[:, None] > i[None, :]) | ((i[:, None] == 0) & (i[None, :] == 0))
    bias = torch.zeros(8, T, T).masked_fill(~((i[:, None] >= i[None, :]))[None], float("-inf"))
    captured = {}
    orig = rightmost_argmax

    def spy(s):
        captured["s_dtype"] = s.dtype
        out = orig(s); captured["idx"] = out
        return out
    import train_e3
    train_e3.rightmost_argmax = spy
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            blk(x, bias, hard_mask)
    finally:
        train_e3.rightmost_argmax = orig
    assert captured["s_dtype"] == torch.float32
    idx = captured["idx"][0]
    assert idx[0].item() == 0 and all(idx[t].item() == t - 1 for t in range(1, T))   # rightmost visible j < t


def test_alibi_fp32_resolution():
    """ALiBi arms: the bias keeps its unit-step resolution at distances > 512 under bf16 autocast."""
    m = Model(10, d=32, layers=1, heads=8, pos="alibi_only")
    T = 700
    bias, _ = m.bias(T, "cpu")
    slope = 2 ** (-8 * 8 / 8)                             # smallest slope, head 8: 2^-8
    b = bias[7, T - 1]                                    # query at T-1, keys at distance T-1-j
    assert bias.dtype == torch.float32
    assert abs((b[T - 1 - 601] - b[T - 1 - 600]).item() + slope) < 1e-6      # delta 600 -> 601: exactly -slope
    assert (b.bfloat16()[T - 1 - 601] == b.bfloat16()[T - 1 - 600]).item(), "bf16 would merge them (test is meaningful)"
    q = torch.zeros(1, 8, T, 4, dtype=torch.bfloat16); k = torch.zeros_like(q)
    v = torch.arange(T, dtype=torch.float32).view(1, 1, T, 1).expand(1, 8, T, 4).bfloat16()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = alibi_attention(q, k, v, bias[None])
    w = torch.softmax(bias[7, T - 1].double(), -1)
    exp = (w * torch.arange(T, dtype=torch.float64)).sum()
    assert abs(out[0, 7, T - 1, 0].double() - exp) / exp < 1e-2      # fp32 weights (output rounded to bf16)


def test_batchify_fresh_names():
    """One fresh vector per distinct name and sequence; repeated names share it within a sequence."""
    exs = [{"tokens": ["<BOS>", "@a", "RED", "@b", "@a"], "label": 1}, {"tokens": ["<BOS>", "@a"], "label": 0}]
    x, nv, isn, ln, y = batchify(exs, {"<BOS>": 2, "RED": 3}, 8, "cpu", torch.Generator().manual_seed(0))
    assert torch.equal(nv[0, 1], nv[0, 4]) and not torch.equal(nv[0, 1], nv[0, 3])
    assert not torch.equal(nv[0, 1], nv[1, 1]) and isn.sum().item() == 4 and ln.tolist() == [5, 2]
    assert x[0, 2].item() == 3 and nv[0, 2].abs().sum().item() == 0


def test_round17_arms():
    """dlocal: 4 global heads, local head 4+j reads exactly offset (j mod D)+1; lastglobal: no global head before the
    last layer, 4 global heads in it."""
    T = 10
    i = torch.arange(T)
    for D, H in ((2, 8), (4, 8), (8, 12)):
        m = Model(10, 32 * H, 3, H, "dlocal", D)
        b, _ = m.bias(T, "cpu")
        for h in range(4):
            assert torch.equal(b[h] == 0, (i[:, None] > i[None, :]) | ((i[:, None] == 0) & (i[None, :] == 0)))
        for h in range(4, H):
            off = (h - 4) % D + 1
            ok = b[h] == 0
            for r in range(T):
                assert ok[r].nonzero().flatten().tolist() == ([r - off] if r >= off else [r])
    m = Model(10, 256, 4, 8, "lastglobal")
    early, last = m.bias(T, "cpu", last=False)[0], m.bias(T, "cpu")[0]
    assert all(int((early[h] == 0).sum()) == T for h in range(8))
    assert [int((last[h] == 0).sum()) for h in range(8)] == [T * (T - 1) // 2 + 1] * 4 + [T] * 4


if __name__ == "__main__":
    test_rightmost_ties(); test_block_under_bf16_autocast(); test_alibi_fp32_resolution(); test_batchify_fresh_names()
    test_round17_arms()
    print("ok: hard-since head selects the rightmost maximum; ALiBi bias in fp32; batchify")
