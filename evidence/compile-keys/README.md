# Compile-time values that Triton's keys do not tell apart

Library evidence (not a kernel's), for library issues L98 and L103. Round 20.

**Question.** Triton 3.8.0 finds a launch's compiled code by a key of its compile-time values (its
`tl.constexpr` arguments, among others), and names each copy of a helper by its compile-time
arguments. Which different values share a key or a name, and does a launch or call with such a
value then run the code compiled for another value? lr-17a and lr-19a found cases (`C = 1.0` after
`C = 1`; `MODE = "1"` after `MODE = 1` on one cache folder; a NaN after another NaN); this folder
checks the whole family the sources give.

**Sources** (pinned in `primitives/`).
- `triton-3.8.0-jit.py:592-614` (`compute_cache_key`): in the process, the kernel's cache is looked
  up through a dictionary of the specialisation, which equates values Python's `==` equates, and
  keyed by the text `str` writes for it, where every NaN is `nan`.
- `triton-3.8.0-compiler.py:71-76` (`ASTSource.hash`): on disk, the key is the `str` of each value,
  joined with `-`. A launch that misses the process's cache compiles, and the compile finds what the
  cache folder holds under that key, so the disk key counts within one process too.
- `triton-3.8.0-code-generator.py:32-41` (`mangle_fn`) and `triton-3.8.0-core.py:196-201`: a helper
  copy's name is built from the `repr` of each compile-time argument, with `'` written `_sq_` and `[`,
  `]` written `_`; a second call whose name is already generated uses the first copy.

**Method** ([run.py](run.py), output [out.txt](out.txt)). Every launch sequence runs in a process
of its own, with an empty `TRITON_CACHE_DIR` of its own ("fresh") or one shared by two processes one
after the other; each value is also launched alone, fresh, as the control. The cases (the script's
docstring gives each kernel):
- K1, one process: `C` one value then another among `1`, `1.0`, `True`, `0`, `0.0`, `-0.0`, `False`;
- K2, one process: `C = float('nan')` then `-float('nan')`, both orders;
- K3, two processes on one cache folder: `MODE` `1`/`"1"`, `None`/`"None"`, `2.5`/`"2.5"`,
  `True`/`"True"`, `float('nan')`/`"nan"`; and `1` then `"1"` in one process;
- K4, two processes on one cache folder: `(A, B) = ("a-b", "c")` then `("a", "b-c")`;
- K5, one kernel calling a helper twice in one compilation: `float('nan')` and `-float('nan')`,
  `"x[y"` and `"x_y"`, both orders; the controls `±inf`, `1`/`1.0`/`True`, `"a"`/`"x_y"`.

**Triton version.** 3.8.0 (torch 2.11.0+cu130, Python 3.13.9).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`out.txt`: `launches that ran code compiled for other values than their own: 23`).

| case | pairs | the later launch or call |
|---|---|---|
| K1 | `1`/`1.0`, `1`/`True`, `0`/`0.0`, `0`/`False` (both orders), `-0.0` then `0` | ran the earlier value's code in all 8: `C = 1.0` after `C = 1` stored `16777217` (the integer product) and `isinstance(C, float)` false |
| K2 | `nan`/`-nan`, both orders | stored the earlier NaN's pattern (`0x7fc00000` for `-nan` after `nan`) |
| K3 | the 7 pairs | ran the earlier value's code: `"1"` after `1` gave `10`, not the string's `20`; also `"1"` after `1` in one process |
| K4 | `("a-b", "c")` then `("a", "b-c")` | stored `31`, not `13` |
| K5 | the NaNs and the strings, both orders | the second call ran the first call's copy (4 of 4); the controls `±inf`, `1`/`1.0`/`True` and `"a"`/`"x_y"` gave each call its own value |

**Conclusions.**
1. A launch whose compile-time values share Triton's key with those of an earlier launch or warm-up
   of the kernel, in the process or through the same cache folder, runs the code compiled for the
   earlier values. Since round 21 (the user's decisions, 2026-10-29: "To make the verifier easy, we
   do not verify triton cache." and "we assume the kernel is new."), TVA does not verify Triton's
   caches: `Compiler.Faithful`'s scope 11 assumes the kernel is new, and nobody checks it; this
   result is the record of why the assumption is needed
   ([rules/compiler-faithful.md](../../docs/rules/compiler-faithful.md), "Scope" 11; SEMANTICS
   "Launches"). Round 19's exclusion of a NaN `tl.constexpr` argument is a case of it.
2. Two calls of one helper in one compilation whose compile-time arguments give the same
   `mangle_fn` name are outside the supported subset (SEMANTICS "Compile-time values Triton keys
   apart"; [exclusions.md](../../docs/exclusions.md)).
