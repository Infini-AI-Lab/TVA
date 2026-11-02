---
name: tva-model-map
description: Write the model (the kernel transcribed into TVA's tl.* vocabulary in Lean), kernel.json's line classes and model-map.json. Use at stages 2-4 of the prover's path, or when the gate's map check fails or a reviewer files MODEL_MISMATCH or MAP_INCOMPLETE.
---

# The model, the classes and the map

The model is the kernel's source rewritten line by line as a Lean `Kernel`; a reviewer reads the two side
by side, so keep the source's names, order and line structure. What every construct means is in
`~/fpv/tva/docs/SEMANTICS.md`; `docs/TRANSCRIPTION.md` §2 is a table of source → model patterns and §3 the
pitfalls. Read §2 and §3 before you start; come back to them for each unfamiliar line.

## Procedure

1. `tva-kit classify`, check, `tva-kit classify --write`. Kernel lines = the launched `@triton.jit`
   function, every JIT function it calls, and the imports of `triton` and `triton.language as tl`.
2. Open the closest example's Lean file (CLAUDE.md lists them) and copy its top: `import TVA`, the module
   docstring (rewrite it for your kernel), `namespace <your namespace>`, `open TVA`.
3. Write the kernel definition: `def <kernel_name> (<pointer args> : Ptr .f32) (<int args> : I32)
   (<CONSTEXPR> : Nat) : Kernel where body ctx := do …`. A `tl.constexpr` argument is a `Nat` (or the
   Lean type of its value). Then one model line per source line, with a `--` comment where the choice
   needs a reason (which `+`, which conversion).
4. For each source line: `tva-kit examples '<regex of the construct>'` shows how certified kernels wrote
   it (and their notes say why); `tva-kit lookup tl.<op>` shows the library's definition and arguments;
   `tva-kit doc <construct>` the SEMANTICS section.
5. The file builds after each edit; fix errors as they come.
6. `tva-kit map-skeleton`, then fill each `TODO` entry (below); `tva-kit map-check` until regular.

## The points that decide most models (PROVER_GUIDE §1)

- **Float `+`/`-`**: `tl.add`/`tl.sub` only when no product feeds the operation; `tl.mul_add a b c` for
  `a * b + c`; `tl.addc`/`tl.subc` when a product may feed it or you cannot tell. Same for reductions:
  `tl.sum` vs `tl.sumc`. Say which and why in a comment. This is the most common `MODEL_MISMATCH`.
- **Every implicit conversion and broadcast is explicit**: `tl.cvt` for `.to(...)` and promotions,
  `tl.itof` from integers, `tl.cvt8`/`tl.cvtFrom8` for fp8, `.col`/`.row`/`bc` for broadcasts. A value
  stored into a narrower pointer is converted first.
- **Python numbers** become the constant Triton builds (SEMANTICS.md "Constants"): in an arithmetic
  operation, `FloatV.litF32 …` (or `litF16`, `litBF16`) of the number in the operation's dtype; a name
  assigned a number holds a scalar tensor of its own type.
- **Masks**: keep `mask` and `other` exactly; a masked load without `other` leaves the disabled lanes
  unspecified.
- **Index arithmetic** is `int32` and wraps: `pid * (IntV.ofNat B : I32)`; `Inst.ok` must keep it from
  wrapping. `offs.ltS n` for a signed `<`.
- **Loops**: `range(lo, hi, step)` is `tl.forRange` with the carried variables as a tuple state;
  `tl.static_range(N)` is the body written `N` times; `while` is `tl.whileLoop`.
- **Float operations are `Prog`s** (`let y ← tl.add x z`); integer and index operations are pure (`let`).
- **Hints** (`tl.multiple_of`, `tl.max_contiguous`) are model lines where the value is defined.
- **Names**: declare everything in your namespace; no `instance`, `notation`, `export`; no name that also
  names a library constant; no binder called `tl`. `tl.debug_barrier()` is a no-op (a comment).
- An operation missing from the library is never defined in your file: it is a library request.

## kernel.json

You set only `title`, `source.ranges` (kernel lines), `source.host` (host lines) and `source.unused`
(names of functions nothing runs). Every function and every module-level code line must be in one class;
an unused function's header lines (decorators and `def` line) go in the host lines.

## model-map.json entries

`{"entries": [{"src": [44], "model": "let block_start := pid * (IntV.ofNat BLOCK_SIZE : I32)", "note":
"…"}, …]}`. Each entry covers source lines (`[44]`, `["30-36"]`, `[47, 48]`) with either

- `"model"`: text that occurs in your Lean file (whitespace may differ), or a list of such texts; or
- `"drop"`: a reason (`import`, `decl`, `comment`, `no-op:print`, `no-op:hint`, `config`, `dead`, … —
  TRANSCRIPTION.md §4), with a `note` giving the condition that makes the drop hold.

The `note` says what a reviewer needs to see the correspondence: why this operation, which conversion,
which condition of `Inst.ok`. Copy the style of the example's map (`tva-kit examples` prints notes).
