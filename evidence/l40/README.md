# Evidence on the NVIDIA L40 (sm_89)

The measurements behind `TVA.Device.L40` (`lean/TVA/Device/L40.lean`), the device facts of an NVIDIA L40
running Triton 3.8.0's code. Every script of `evidence/` was run on an L40, with the same software as the
H200's runs: Triton 3.8.0, torch 2.11.0+cu130, Python 3.13 (driver 580.105.08, CUDA 13.0).

`run.sh` ran them (one job per GPU at a time, ten L40s); each `<folder>/<script>.out` here is the standard
output of `evidence/<folder>/<script>.py` on the L40. The scripts are the H200's, with three exceptions:

* **Shapes that do not fit.** The L40 has 101376 bytes of shared memory per block, the H200 232448. A block
  shape that needs more is rejected by Triton (`OutOfResources`), which stops the H200's scripts.
  `oor_wrap.py` wraps the body of each `for (M, N, nw…) in …:` loop in `try` / `except OutOfResources`,
  printing a row "rejected: out of resources on this device" instead; on a GPU where every shape fits it
  changes nothing. It was applied to `num-dot16-model/run.py` and `num-dot-chunk16/run.py` (the changes are
  `<folder>/run.py.patch`, indentation only plus the handler). `num-dot/ieee_ratio.py` has its own copy
  here (`num-dot/ieee_ratio.py`), which skips `K = 1024`. `num-dot/run.py` already reports a rejected shape.
* **Scripts that need sm_90.** `num-dot-chunk/run.py` asserts that the dot runs on `wgmma`, which sm_89 does
  not have (`num-dot-chunk/run.out`: the assertion). Its role on the L40, the order and width of the fp8
  instructions, is `num-dot-chunk8-order/run.py` (new). `num-dot-chunk-loss/run.py` runs the fp8 dots on
  `mma.sync` instead and reports every check.
* **New scripts.** `num-cvt8/double.py`, `dump.py`, `CvtCheck.lean`: the two roundings of `float32 → fp8`;
  `num-div-f64/run.py`: `float64` division against the CPU's IEEE division (the H200's script needs a
  Lean driver v3 no longer builds); `num-dot-chunk8-order/run.py`; `num-cvt8/example.py`, one input that tells the two apart.

## Results, fact by fact

| fact (`Facts` field) | evidence here | result on the L40 | statement |
|---|---|---|---|
| `expF32` … `rsqrtF32` | `num-math-f32/run.out` | every `float32` input; the output is the H200's, bit for bit (same `ex2.approx.ftz.f32`, `lg2.approx.ftz.f32`, `sqrt.approx.ftz.f32`, `rsqrt.approx.ftz.f32`) | H200's |
| `expF32NegInf`, `exp2F32NegInf`, `expF32Overflow`, `divF32InfDivisor` | `num-special-f32/run.out` | identical to the H200's | H200's |
| `divFull` | `num-div-f32/run.out` | 30757794871 pairs, 0 outside; largest ratio 0.65 | H200's |
| `divF64` | `num-div-f64/run.out` | 452986128 pairs, 0 differ from IEEE division (`div.rn.f64`) | H200's |
| `cvtSatF16E4m3` … `cvtSatBF16E5m2` | `num-cvt8/run.out`, `specials.out` | every `float16` / `bfloat16` input: 0 differ from one saturating rounding | H200's |
| **`cvtSatF32E4m3`, `cvtSatF32E5m2`** | `num-cvt8/double.out`, `cvtcheck.out` | **two roundings**: `cvt.rz.f16.f32` then `cvt.rn.satfinite.{e4m3,e5m2}x2.f16x2`. Every `float32` input equals `cvtSat .rn (RZ16 x)` (0 differences); one rounding differs on 1032066 (`e4m3`) / 1146756 (`e5m2`) inputs. The library's own definition checked against the device on 1409006 inputs: 0 differences | **L40's own** (`L40.CvtSatF32Two`) |
| `atomicAddF32` | `num-atomic-add/run.out`, `sem_scope.out` | every sem and scope; 0 outside, ratio 0.50 | H200's |
| `dotF16`, `dotBF16` | `num-dot/run.out`, `num-dot16-model/run.out` | 0 outside; ratio 0.25 (`num-dot`), 0.17 (`num-dot16-model`, 49.8 million elements) | H200's |
| `dotF32Ieee` | `num-dot/run.out`, `num-dot/ieee_ratio.out` | bit for bit the sequential `fma`; largest ratio `(K-1)/(2(K+1))`, as on the H200 | H200's |
| `dotI8` | `num-dot-int8/run.out` | exact in range | H200's |
| `dotF16Chunk`, `dotBF16Chunk` | `num-dot-chunk16/run.out`, `num-dot-chunk16-order/run.out` | chunks of 64 in increasing order (the H200's rule); 0 outside; largest ratio 0.80 (H200: 0.50) | H200's (less slack, see below) |
| `dotE4m3ChunkLoss` … `dotE5m2E4m3ChunkLoss` | `num-dot-chunk-loss/run.out`, `num-dot-chunk8-order/run.out` | native `mma.sync…m16n8k32…e4m3/e5m2`; chunks of 32 in increasing order; 0 outside; largest ratio 0.52 | H200's |
| `dotScaledE4m3`, `dotScaledE4m3E2m1` | `num-dot-scaled/run.out` | the same `bfloat16` emulation; 0 outside, ratio 0.22 | H200's |
| `sumDepth` | `reduce-tree/run.out` | every shape: the tree is the library's; 0 failures | H200's |

**What differs from the H200.**

* `float32 → fp8` rounds twice (above). For `x` just above `2^-10` in `float8e4nv` the L40 gives `0`, not
  the smallest subnormal. So the H200's `CvtSatF32E4m3`, `CvtSatF32E5m2` and the round-to-nearest facts
  derived from them are false on the L40, and a proof that uses them does not check against the L40.
* The 16-bit `mma.sync…m16n8k16` keeps one bit less of the accumulator (`num-dot/run.out` part 1: `s ≤ 24`,
  H200 `25`), and the H200's bit-level model of it does not match (`num-dot16-model/run.out` part A). The
  chunk facts still hold with their constants: the arm `2·A_j + 2^-22·Mj` covers any truncating window (a
  product below the window loses at most its own magnitude), and the measured ratio is 0.80. The slack of
  about 2 the library keeps elsewhere is 1.25 here.
* `tl.dot` uses only `mma.sync` (no `wgmma`), and the fp8 instructions are native (`m16n8k32`, a 13-bit
  window like sm_90's `wgmma`), not converted to `float16`. One fp8 instruction is not one window
  (`num-dot-chunk8-order/run.out`: a small term in a big term's chunk survives in 241 of 2009 cases),
  which the fp8 chunk facts do not need.
* Shapes whose shared memory exceeds 101376 bytes are rejected; Triton raises `OutOfResources` before the
  kernel runs (for example `tl.dot` blocks of `64 x 64` with `K = 1024`).

## The swap test

With `Device` set to `Device.L40.Facts` (`lean/TVA/Device.lean`, three lines), the library and `TVA.Proof`
build, and 59 of the 60 claims' oracle proofs check. The one that does not is `kv_cache_write_bf16`: its fp8
configuration promises the single saturating rounding of `key / k_scale`, which the L40 does not give.
`num-cvt8/example.py` runs the claim's operation on one input (`key = 1.0625`, `k_scale = 1 - 2^-12`): the
L40 stores `1.0`, the claim says `1.125` (`num-cvt8/example.out`). So the claim is false on the L40, and the
proof fails where it should.
