"""The test's inputs (source lines 137-155: generator seed 0, the three configurations), regenerated
on the CPU, and the quantities the README states about them:

* X = max over (sequence, head, token) of scale * scoreAbs, scoreAbs = sum_e |q_e * k_e| (Model.lean);
* the number of pages T per sequence;
* attnAbs (the softmax-weighted mean of |v|) and |attnRef|;
* the bound tolO(D, P, T, X_s) * attnAbs + tolF per element, with X_s the sequence's own maximum;
* the error of a float32 emulation of the kernel's loop (sequential sums, no fma) against the exact
  attention, as a sanity check that Post's reference (pageOf, tokElem, seqLen) is the reference the
  test computes, and that the bound is not violated by a plain float32 run.

The make_cache logic is copied from the source (lines 100-115), with the CPU as the device, so that
the generator draws the same numbers in the same order. Run: python -I test_data.py
"""
import math
import sys
from decimal import Decimal as Dm

import torch

sys.path.insert(0, ".")
from tol import tolO, tolF  # noqa: E402


def cdiv(a, b):
    return (a + b - 1) // b


def make_cache(seq_lens, num_kv_heads, head_dim, page_size, num_pages, generator):
    seq_lens = [int(n) for n in seq_lens]
    pages_per_seq = [cdiv(n, page_size) for n in seq_lens]
    max_pages = max(pages_per_seq)
    assert sum(pages_per_seq) <= num_pages
    perm = torch.randperm(num_pages, generator=generator)
    block_tables = torch.zeros((len(seq_lens), max_pages), dtype=torch.int32)
    used = 0
    for s, n in enumerate(pages_per_seq):
        block_tables[s, :n] = perm[used:used + n].to(torch.int32)
        used += n
    shape = (num_pages, page_size, num_kv_heads, head_dim)
    k_cache = torch.randn(shape, generator=generator).to(dtype=torch.float16)
    v_cache = torch.randn(shape, generator=generator).to(dtype=torch.float16)
    return k_cache, v_cache, block_tables, torch.tensor(seq_lens, dtype=torch.int32)


def emulate(q, k_cache, v_cache, block_tables, seq_lens, scale32):
    """The kernel's loop in float32, sequentially (one possible order of the reductions)."""
    num_seqs, num_heads, D = q.shape
    P, nkv = k_cache.shape[1], k_cache.shape[2]
    qpk = num_heads // nkv
    out = torch.empty(q.shape, dtype=torch.float32)
    for s in range(num_seqs):
        n = int(seq_lens[s])
        for h in range(num_heads):
            g = h // qpk
            qf = q[s, h].float()
            m = torch.tensor(float("-inf"), dtype=torch.float32)
            l = torch.tensor(0.0, dtype=torch.float32)
            acc = torch.zeros(D, dtype=torch.float32)
            for j in range(cdiv(n, P)):
                page = int(block_tables[s, j])
                rows = torch.arange(P)
                mask = j * P + rows < n
                k = torch.where(mask[:, None], k_cache[page, :, g].float(), torch.zeros(()))
                v = torch.where(mask[:, None], v_cache[page, :, g].float(), torch.zeros(()))
                sc = (k * qf[None, :]).sum(dim=1) * scale32
                sc = torch.where(mask, sc, torch.tensor(float("-inf")))
                m_new = torch.maximum(m, sc.max())
                alpha = torch.exp(m - m_new)
                p = torch.exp(sc - m_new)
                l = l * alpha + p.sum()
                acc = acc * alpha + (p[:, None] * v).sum(dim=0)
                m = m_new
            out[s, h] = acc / l
    return out.to(torch.float16)


def main():
    gen = torch.Generator().manual_seed(0)
    configs = [
        (32, 8, 128, 16, [1, 15, 16, 17, 1000, 4096, 333]),
        (8, 8, 64, 16, [5, 64, 2049]),
        (16, 2, 128, 32, [31, 32, 33, 8191]),
    ]
    worst_ratio = 0.0
    overall_X = 0.0
    overall_attnAbs = 0.0
    overall_bound = 0.0
    for num_heads, num_kv_heads, head_dim, page_size, seq_lens in configs:
        num_pages = sum(cdiv(n, page_size) for n in seq_lens) + 50
        k_cache, v_cache, block_tables, lens = make_cache(seq_lens, num_kv_heads, head_dim, page_size,
                                                          num_pages, gen)
        q = torch.randn((len(seq_lens), num_heads, head_dim), generator=gen).to(dtype=torch.float16)
        scale = head_dim ** -0.5
        scale32 = torch.tensor(scale, dtype=torch.float32)
        group = num_heads // num_kv_heads
        emu = emulate(q, k_cache, v_cache, block_tables, lens, scale32)
        cfg_X = 0.0
        sum_X = 0.0
        cnt_X = 0
        cfg_attnAbs = 0.0
        cfg_bound = 0.0
        cfg_err = 0.0
        for s, n in enumerate(seq_lens):
            t = torch.arange(n)
            pages = block_tables[s, t // page_size].long()
            k = k_cache[pages, t % page_size].double().repeat_interleave(group, dim=1)  # [n, H, D]
            v = v_cache[pages, t % page_size].double().repeat_interleave(group, dim=1)
            qs = q[s].double()  # [H, D]
            score = torch.einsum("hd,nhd->hn", qs, k)
            scoreAbs = torch.einsum("hd,nhd->hn", qs.abs(), k.abs())
            Xs = scale * scoreAbs  # [H, n]
            cfg_X = max(cfg_X, Xs.max().item())
            sum_X += Xs.sum().item()
            cnt_X += Xs.numel()
            w = torch.softmax(scale * score, dim=-1)  # [H, n]
            ref = torch.einsum("hn,nhd->hd", w, v)
            attnAbs = torch.einsum("hn,nhd->hd", w, v.abs())
            cfg_attnAbs = max(cfg_attnAbs, attnAbs.max().item())
            T = cdiv(n, page_size)
            err = (emu[s].double() - ref).abs()
            cfg_err = max(cfg_err, err.max().item())
            for h in range(num_heads):
                Xh = Dm(repr(Xs[h].max().item()))
                tO = float(tolO(head_dim, page_size, T, Xh))
                tF = float(tolF(head_dim, page_size, T, Xh))
                bound = tO * attnAbs[h] + tF
                cfg_bound = max(cfg_bound, bound.max().item())
                worst_ratio = max(worst_ratio, (err[h] / bound).max().item())
        print(f"heads {num_heads}/{num_kv_heads} head_dim {head_dim} page {page_size} seq_lens {seq_lens}:")
        print(f"  max scale*scoreAbs {cfg_X:.4f}, mean {sum_X / cnt_X:.4f}, max pages {cdiv(max(seq_lens), page_size)}")
        print(f"  max attnAbs {cfg_attnAbs:.4f}, max bound tolO*attnAbs+tolF {cfg_bound:.4e}")
        print(f"  float32 emulation: max abs error vs exact {cfg_err:.3e}")
        overall_X = max(overall_X, cfg_X)
        overall_attnAbs = max(overall_attnAbs, cfg_attnAbs)
        overall_bound = max(overall_bound, cfg_bound)
    print(f"overall: max scale*scoreAbs {overall_X:.4f}, max attnAbs {overall_attnAbs:.4f}, "
          f"max bound {overall_bound:.4e}")
    print(f"largest (emulation error / bound) over every element: {worst_ratio:.4f}")


if __name__ == "__main__":
    main()
