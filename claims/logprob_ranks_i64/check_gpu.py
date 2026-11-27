"""GPU check of the claim `Claims.LogprobRanks`.

Runs the kernel's own host, `compute_token_ranks` of `source/logprob_ranks.py` (imported by
path), on the setup's launches (cut down where `SETUP.md`'s size needs more than 4 GB) and on
launches at the edges of `Inst.ok` and `Pre`, and direct launches of the module's kernel with the
host's own arguments (other `BLOCK_SIZE`s, strides, an output that is a slice of a larger buffer
filled with a sentinel), and checks `Post` exactly as it is stated:

* element `r` of `token_ranks` is `rank d r`: the number of `j < vocab_size` with
  `x_r <= logits[r, j]` (ordered: false when either is NaN), `x_r = logits[r, t_r]`, plus
  `padLanes = cdiv(vocab_size, BLOCK_SIZE) * BLOCK_SIZE - vocab_size` when `x_r <= -inf` (that is,
  `x_r = -inf`); computed here on the CPU from the host's inputs in `float64` (exact for the three
  dtypes), independently of the module's `reference`;
* the frame: the inputs are unchanged, and in a guarded launch every element outside the first
  `batch_size` keeps the sentinel.

Every value is an integer, so the check is exact (no tolerance). Negative tests: plausible wrong
outputs (the strict count `>`, no padding term, the padding term always, a NaN entry counted, a
0-based rank, the rank among the first block only) that `Post` must reject.

Usage: `CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import pathlib

import torch
import triton

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "source" / "logprob_ranks.py"
spec = importlib.util.spec_from_file_location("logprob_ranks_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEVICE = triton.runtime.driver.active.get_active_torch_device()
SENTINEL = -1234567890123
HOST_BLOCK = 8192


def post_rank(logits, tok, block=HOST_BLOCK, variant="post"):
    """`rank d r` for every row (Post), from the host's inputs; `variant` gives the wrong outputs
    of the negative tests instead. Rows are taken in chunks to keep the `float64` copy small."""
    B, V = logits.shape
    pad = (V + block - 1) // block * block - V
    out = []
    step = max(1, (2 ** 24) // max(V, 1))
    for r0 in range(0, B, step):
        lg = logits[r0:r0 + step].double()  # exact: every float32/bfloat16/float16 is a float64
        t = tok[r0:r0 + step].long()
        x = lg.gather(1, t[:, None])
        if variant == "strict":
            cnt = (lg > x).sum(1)
        elif variant == "nan_counts":
            cnt = ((lg >= x) | lg.isnan()).sum(1)
        elif variant == "first_block":
            cnt = (lg[:, :block] >= x).sum(1)
        else:
            cnt = (lg >= x).sum(1)
        xneg = (x[:, 0] == float("-inf")).long()
        if variant == "no_pad":
            r = cnt
        elif variant == "always_pad":
            r = cnt + pad
        else:
            r = cnt + pad * xneg
        if variant == "zero_based":
            r = r - 1
        out.append(r.cpu())
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.int64)


def same_bits(a, b):
    """Bit-for-bit equality of two float tensors (NaN payloads included; `torch.equal` is false on
    NaN)."""
    it = {2: torch.int16, 4: torch.int32}[a.element_size()]
    return a.shape == b.shape and torch.equal(a.contiguous().view(it), b.contiguous().view(it))


results = []
VARIANTS = ["strict", "no_pad", "always_pad", "nan_counts", "zero_based", "first_block"]
neg = {v: [0, 0] for v in VARIANTS}  # launches on which the wrong output is rejected, tried


def negatives(logits, tok, exp, block):
    for v in VARIANTS:
        w = post_rank(logits, tok, block, v)
        neg[v][0] += int(not torch.equal(w, exp))
        neg[v][1] += 1


def check(name, logits, tok):
    """The host's launch, then a guarded direct launch with the host's arguments."""
    before_l, before_t = logits.cpu(), tok.cpu()
    got = mod.compute_token_ranks(logits, tok)
    torch.cuda.synchronize()
    exp = post_rank(logits, tok)
    exact = torch.equal(got.cpu(), exp)
    kept = same_bits(logits.cpu(), before_l) and torch.equal(tok.cpu(), before_t)
    frame = guarded(logits, tok, HOST_BLOCK, exp)
    ok = exact and kept and frame
    results.append(ok)
    negatives(logits, tok, exp, HOST_BLOCK)
    print(f"{'ok ' if ok else 'BAD'} {name}: {tuple(logits.shape)} stride {logits.stride(0)} "
          f"{str(logits.dtype)[6:]}: ranks exact {exact}, frame {frame and kept}", flush=True)
    torch.cuda.empty_cache()


def guarded(logits, tok, block, exp, guard=64):
    """`_ranks_kernel` launched as the host launches it (source lines 75-84), `BLOCK_SIZE = block`,
    the output a slice of a sentinel-filled buffer; checks the ranks and the guards."""
    B, V = logits.shape
    buf = torch.full((guard + B + guard,), SENTINEL, dtype=torch.int64, device=logits.device)
    out = buf[guard:guard + B]
    mod._ranks_kernel[(B,)](out, logits, logits.stride(0), tok, V, BLOCK_SIZE=block,
                            num_warps=4, num_stages=3)
    torch.cuda.synchronize()
    fb = buf.cpu()
    return (bool((fb[:guard] == SENTINEL).all()) and bool((fb[guard + B:] == SENTINEL).all())
            and torch.equal(fb[guard:guard + B], exp))


def check_block(name, logits, tok, block):
    """A direct launch with another `BLOCK_SIZE` (Inst.ok: any power of two up to 2^20)."""
    exp = post_rank(logits, tok, block)
    ok = guarded(logits, tok, block, exp)
    results.append(ok)
    negatives(logits, tok, exp, block)
    print(f"{'ok ' if ok else 'BAD'} {name}: {tuple(logits.shape)} BLOCK_SIZE {block} "
          f"{str(logits.dtype)[6:]}: ranks exact and frame {ok}", flush=True)
    torch.cuda.empty_cache()


gen = torch.Generator(device=DEVICE).manual_seed(0)


def randn(B, V, dtype, stride=None):
    if stride is None or stride == V:
        return torch.randn(B, V, generator=gen, device=DEVICE).to(dtype)
    full = torch.randn(B, stride, generator=gen, device=DEVICE).to(dtype)
    return full[:, :V]


def toks(B, V):
    return torch.randint(0, V, (B,), generator=gen, device=DEVICE, dtype=torch.int64)


# The setup's launches (SETUP.md): the host's test, then the code base's.
for name, B, V, dt, S in [
    ("test", 1, 32000, torch.float32, None), ("test", 8, 128256, torch.float32, None),
    ("test", 64, 151936, torch.bfloat16, None), ("test", 256, 32000, torch.float16, None),
    ("test", 1024, 8192, torch.float32, None), ("test", 3, 262144, torch.bfloat16, None),
    ("sampler", 1024, 128256, torch.float32, None), ("sampler", 256, 151936, torch.bfloat16, None),
    ("sampler", 256, 32000, torch.float16, None), ("sampler slice", 64, 50257, torch.float32, 50304),
    ("spec", 2560, 128256, torch.float32, None), ("prompt", 1024, 151936, torch.bfloat16, None),
    ("prompt", 37, 262144, torch.float32, None), ("empty batch", 0, 32000, torch.float32, None),
]:
    check(name, randn(B, V, dt, S), toks(B, V))
# vLLM's int64-offset regression test (16448 x 131072 float32, 8.6 GB) cut to 4100 rows (2.1 GB):
# the byte offset of the last rows passes 2^31.
check("vllm int64 offset (cut to 4100 rows)", randn(4100, 131072, torch.float32), toks(4100, 131072))

# Edges of Inst.ok and Pre.
# Ties and repeated values (bfloat16 logits on a coarse grid), the token at 0 and at V - 1.
lg = (torch.randint(-4, 5, (300, 70000), generator=gen, device=DEVICE).float() / 2).to(torch.bfloat16)
t = toks(300, 70000); t[0] = 0; t[1] = 69999
check("edge ties", lg, t)
# Special values: a NaN, -inf or +inf token logit; NaN, +-inf, -0/+0 entries; vocab sizes with
# and without padding lanes (V a multiple of 8192, and not).
for V in [8192, 8193, 32000, 16384 * 3 + 1]:
    for dt in [torch.float32, torch.bfloat16, torch.float16]:
        lg = randn(64, V, dt)
        lg[:, ::97] = float("nan")
        lg[:, 1::89] = float("-inf")
        lg[:, 2::83] = float("inf")
        lg[:, 3::79] = 0.0
        lg[:, 4::73] = -0.0
        t = toks(64, V)
        t[0:8] = 0      # NaN
        t[8:16] = 1     # -inf: the padding lanes count
        t[16:24] = 2    # +inf
        t[24:32] = 3    # +0 (ties with -0)
        t[32:40] = 4    # -0
        lg[40:48] = float("-inf")  # whole rows of -inf
        check(f"edge special values V {V}", lg, t)
# V = 1 (vocab_size and logits_stride are then compile-time constants 1), V = 2.
check("edge V 1", randn(500, 1, torch.float32), toks(500, 1))
lg = randn(500, 1, torch.float16); lg[::3] = float("-inf")
check("edge V 1 -inf", lg, toks(500, 1))
check("edge V 2", randn(500, 2, torch.bfloat16), toks(500, 2))
# Strides: a stride of 0 (one row broadcast, `expand`), overlapping rows (stride < V), a large
# stride.
base = randn(1, 50000, torch.float32)
check("edge stride 0", base.expand(777, 50000), toks(777, 50000))
flat = randn(1, 4000 * 9000 + 30000, torch.float32).reshape(-1)
check("edge overlapping rows", flat.as_strided((4000, 30000), (9000, 1)), toks(4000, 30000))
check("edge stride 2^22", randn(40, 2 ** 22, torch.bfloat16)[:, :1000], toks(40, 1000))
# Large vocabularies and many rows.
check("edge V 2^20", randn(16, 2 ** 20, torch.float32), toks(16, 2 ** 20))
check("edge 65536 rows", randn(65536, 1000, torch.float16), toks(65536, 1000))
# Other BLOCK_SIZEs (direct launches), with -inf tokens so that the padding term varies.
for blk in [1, 2, 16, 1024, 2 ** 16, 2 ** 17]:  # 2^20 lanes: compiles too long (SETUP.md)
    V = 50001
    lg = randn(32, V, torch.float32)
    lg[:, 7::101] = float("nan")
    t = toks(32, V)
    lg[0:4, :] = float("-inf")
    lg[torch.arange(4, 8, device=DEVICE), t[4:8]] = float("-inf")
    check_block("edge BLOCK_SIZE", lg, t, blk)

n_bad = results.count(False)
for v in VARIANTS:
    print(f"negative test {v}: rejected on {neg[v][0]} of {neg[v][1]} launches")
    if neg[v][0] == 0:
        n_bad += 1  # a wrong output that Post never rejects: the test is too weak
print(f"{len(results)} launches: every rank exact (error 0, tolerance 0) and every frame clause held"
      if n_bad == 0 else f"{n_bad} of {len(results)} launches FAILED")
sys.exit(1 if n_bad else 0)
