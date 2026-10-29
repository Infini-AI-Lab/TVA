# What an autotuned call runs around the launch a claim is about

**Question.** A kernel decorated with `@triton.autotune` is called like any kernel, but the call may
run the kernel many times on the caller's own buffers (benchmarking each config), and runs host
hooks (`reset_to_zero`, `restore_value`, `pre_hook`, `post_hook`, a config's `pre_hook`) that write
those buffers. Which runs happen when, and what memory does the final launch start from? Library
issue L21 (round 8): how a claim's `init` relates to such a call.

**Sources** (Triton 3.8.0, `primitives/triton-3.8.0-autotuner.py`; the copy installed in `.venv` has the same
sha256 as the pinned one).
- `Autotuner.run`, lines 217-282: the tuning key is the values of the `key` arguments plus the dtype
  of every tensor argument (222-227); a key not in the in-process cache is benchmarked (238-245),
  unless pruning leaves one config (232-235) or there is one config at all (267-268); then the
  chosen config's `pre_hook` runs (273-275) and the kernel is launched once with that config
  (276-280).
- `Autotuner._bench`, lines 133-173: each benchmark run is `config.pre_hook`, the autotuner's
  `pre_hook`, one launch on the real arguments, the autotuner's `post_hook` (150-166), repeated by
  the benchmarker (`triton.testing.do_bench`, warm-up 25 ms and 100 ms of repetitions by default,
  so the number of runs depends on the kernel's speed). Between runs `do_bench` also zeroes a
  separate 256 MB buffer to flush the L2 (`primitives/triton-3.8.0-nvidia-driver.py` lines 399-409).
- the default hooks, lines 57-82: `reset_to_zero` zeroes the named arguments before each benchmark
  run and once more after benchmarking (`reset_only`, line 245); `restore_value` copies the named
  arguments before each benchmark run and copies them back after it.
- `check_disk_cache`, lines 175-215: with `TRITON_CACHE_AUTOTUNING=1` (`knobs.autotuning.cache`,
  `primitives/triton-3.8.0-knobs.py` line 383) the timings are stored in Triton's cache folder and a
  later process with the same key reads them and does not benchmark; configs with a `pre_hook` are
  never cached (177).

**Method.** `run.py` defines a kernel that adds 1 in place to every element of an `int32` buffer
`x` (65536 elements), with configs `BLOCK` = 128, 256, 512, so after a call each element holds the
number of launches that ran on it. Each case runs in its own process on GPU 0.

- A1: three configs, no hooks; the first call and a second call.
- A2: `restore_value=['x_ptr']` with `x = 7`; `reset_to_zero=['x_ptr']` with `x = 7`, two calls.
- A3: an autotuner `pre_hook` that fills `x` with 100 and `post_hook` that fills it with -5, and a
  config `pre_hook` that fills it with 1000 (two configs); two calls, counting the hooks' calls.
- A4: a second call with the same key; calls with new keys (`n` = 32768, 65535).
- A5: one config; `prune_configs_by` leaving one config; leaving two.
- A6: `TRITON_CACHE_AUTOTUNING=1` in two processes sharing one new cache folder.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/autotune-launch/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`output.txt`; benchmark counts vary from run to run).

| case | call | `x` before | `x` after | launches on `x` |
|---|---|---|---|---|
| A1 | first | 0 | 3884 | 3884 (benchmark, then one) |
| A1 | second, same key | 0 | 1 | 1 |
| A2 `restore_value` | first | 7 | 8 | the final one counts; the benchmark's are undone |
| A2 `reset_to_zero` | first | 7 | 1 | the final launch starts from 0, not 7 |
| A2 `reset_to_zero` | second | 7 | 8 | 1; no reset on a cached key |
| A3 | first | 0 | 1001 | config `pre_hook` sets 1000, then one launch |
| A3 | second | 0 | 1001 | the same: the config `pre_hook` runs on every call |
| A4 | same key / new key | 0 | 1 / about 4000-4900 | a new key benchmarks again |
| A5 | one config / pruned to one / pruned to two | 0 | 1 / 1 / 3199 | |
| A6 | first process / second process | 0 | 4149 / 1 | the disk cache skips the benchmark |

A3's hook counts on the first call: the autotuner's `pre_hook` and `post_hook` 2601 times each (once
per benchmark run), the config `pre_hook` 1198 + 1404 times, then the `reset_only` call, then the
chosen config's `pre_hook` once before the final launch. On the second call: only the config
`pre_hook`, once.

**Conclusions.**

1. An autotuned call is host code that may run the kernel many times on the real arguments and
   runs hooks that write them; only its last step is one launch of the kernel with the chosen
   config. A claim is about that one launch, from the memory it starts with: after benchmark runs
   (A1: thousands of `+1`s), after `reset_to_zero` (A2: zeros, not the caller's values) and after a
   config `pre_hook` (A3: 1000), that memory is not the caller's.
2. Whether a call benchmarks depends on the process's in-memory cache (A1, A4), the number of
   configs left after pruning (A5), and the disk cache (A6); a `restore_value` argument is copied
   back after each benchmark run (A2), so with it the final launch starts from the caller's values
   of those arguments only.
3. What the host code does around the launch is the host's: a statement about the caller's memory
   before the call must account for it (benchmark runs, hooks); the kernel's claim covers the final
   launch, whose `init` is the memory after the last hook.
