# Host code that changes what a launch runs, or whether it runs

Library evidence (not a kernel's), for library issues L91, L104 and L105. Round 20.

**Question.** Which host code, with no knob, hook, launch option or statement of the kernel's
module, changes what a launch of a supported kernel computes, or whether its programs run?
- L91: a call made while a CUDA graph capture is in progress, on the capturing stream and on
  another stream: round 17's account said every such call runs no program, and lr-17b found that
  a call on another stream ran at once;
- L104: a name of the kernel's module rebound from another module, an attribute of Triton's module
  changed, a `JITFunction`'s source replaced;
- L105: the kernel's `JITFunction` attribute `debug`, and `JITFunction.preload`.

**Sources** (pinned in `primitives/`). `triton-3.8.0-jit.py:727` (`JITFunction.run` takes
`kwargs.get("debug", self.debug)`), `:561-569` (`_unsafe_update_src`), `:823-875` (`preload`: it
compiles the kernel with the options its data names and stores the result under the key its data
names). The front end reads a helper or a `tl` function by name when it compiles the kernel, from
the function's globals and Triton's modules as they are then.

**Method** ([run.py](run.py), output [out.txt](out.txt)). Each case writes the kernel module (every
line in the supported subset) to a temporary file, imports it from the script (another module, as a
host program would), and runs in a process of its own with an empty Triton cache:
- G0–G3: `o = x + 1` (`add1`, compiled before the capture) called inside `torch.cuda.graph(g)`, G0 on
  the capturing stream, G1–G3 under `with torch.cuda.stream(s2)` in the capture modes `global`,
  `thread_local` and `relaxed` (the capture also records a `torch` operation on its own stream);
  `o` is read after the capture, before any replay, and again after it is reset to `-7` and the
  graph replayed;
- M0–M4: `k` stores `double(x)` (a helper, `x * 2`), `expk` stores `tl.exp(x)`: M0 the control; M1
  `kmod.double = kmod.triple`; M2 `kmod.k.fn.__globals__["double"] = kmod.triple`; M3 `tl.exp =
  tl.log`; M4 `kmod.k._unsafe_update_src(<a def of k that stores 5·x>)`;
- D0, D1: an `int32` `a + b` that overflows in lane 0 (`2^31 − 1 + 1`), D1 after
  `kmod.add_i32.debug = True`, with no launch keyword, decorator argument, environment variable or
  knob;
- Pgen, P1, P0: a first process launches `tl.math.exp2(x)` with `enable_reflect_ftz=False` and
  records its specialisation data under the key of a plain launch (a `jit_cache_hook` that lets the
  compile go on, in that process only); P1, a second process with an empty cache and no hook, calls
  `k.preload(<the data>)` and launches `k[(1,)](x, y)` with no option, on `x = -130`; P0, the
  control, launches plainly.

**Triton version.** 3.8.0 (torch 2.11.0+cu130, Python 3.13.9).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`out.txt`).

| case | result |
|---|---|
| G0, the capturing stream | returned without an error; `o` kept `-7` after the capture; the replay wrote `x + 1 = 2` |
| G1–G3, another stream (`global`, `thread_local`, `relaxed`) | returned without an error; `o = 2` after the capture, before any replay (the programs ran at once); after the reset, the replay left `-7` |
| M0, control | `2·x` |
| M1 `kmod.double = kmod.triple`, M2 the same through `fn.__globals__` | `3·x` |
| M3 `tl.exp = tl.log` | `log x` |
| M4 `_unsafe_update_src` | `5·x` |
| D0, plain | wrapped: `-2147483648` |
| D1, `kmod.add_i32.debug = True` | device-side assert, `int32 overflow detected for operation add` |
| P0, a plain launch | `exp2(-130) = 0` (Triton's default `enable_reflect_ftz=True`, the model) |
| P1, after `preload` | a plain launch stored `2^-130` (`7.3468e-40`): the code compiled with `enable_reflect_ftz=False` |

**Conclusions.**
1. A call on the capturing stream is captured: it runs no program, and each replay runs it. A call
   on another stream during a capture is not captured: it runs at once, and no replay runs it.
   `Compiler.Faithful` scope 9 states both and covers neither (library issue L91; the user's
   decision of 2026-10-28 keeps the exclusion).
2. Code outside the kernel's file can change what the front end compiles, with every line of the
   kernel's module in the subset (M1–M4). `Compiler.Faithful` scope 10 states the limit (Triton and
   the kernel's module as installed and written), since round 21 an assumption of the scope's part
   (B), which nobody checks (library issue L104; the user's decision, 2026-10-29: "So basically, we
   want to verify the kernel itself, rather than triton installation or other code programmed by
   users.").
3. `kernel.debug = True` makes a launch's `debug` true (D1): `Launch.debug` is the call's `debug`
   or else the attribute; and `preload` makes a launch with no option run code compiled with other
   options (P1): scope 4 covers no launch of a kernel `preload` was called on (library issue L105);
   since round 21 a `kernel.debug` set from outside the kernel's source and a `preload` are
   assumed not to happen (scope 4's part (B)), and nobody checks them.
