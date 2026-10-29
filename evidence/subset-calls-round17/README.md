# The calls of the in-scope tutorial kernels, and where TVA gives each a meaning

**Question.** The supported subset (`docs/SEMANTICS.md`, "The supported subset"; library issue L88)
admits calls of the `tl.*` functions and the methods that SEMANTICS.md and `TVA.Lang` give meaning to.
lr-15a found one the in-scope kernels call without a stated meaning, `tl.cdiv`, and asked for the list
to be checked once more for any other. Does every call, method and `tl` attribute of the in-scope
kernels now have a meaning in SEMANTICS.md or `TVA.Lang`? Round 17.

**Method.** `run.py` parses the pinned tutorials (`tutorials/`) and scans the 23 in-scope functions of
`tutorials/inventory.json` (all but tutorial 05's `_layer_norm_bwd_dx_fused`, tutorial 08's
`grouped_matmul_kernel`, tutorial 10's `block_scaled_matmul_kernel_cdna4` and tutorial 11, which
`docs/exclusions.md` excludes): their parameters (annotations and defaults) and bodies, listing every
call (of a `tl.*` function, of a method, of a builtin, of a helper of the same file, with the
keywords given) and every attribute read that is not a call; the decorators, which run on the host,
are listed apart. Each name is matched with the SEMANTICS.md section or `TVA.Lang` definition that
gives its meaning, from a table written by reading both (`MEANING` in the script), and the scan prints
the names the table lacks. No GPU.

Run: `python3 -I evidence/subset-calls-round17/run.py` from TVA's root.

**Triton version.** 3.8.0 (the pinned tutorials, `tutorials/manifest.json`).
**Date.** 2026-10-28.

**Result** (`output.txt`). 23 functions. Every name has a meaning: 37 `tl` functions and attributes (dtypes and `tl.constexpr` included)
(`tl.cdiv` among them, 23 uses in 8 functions, stated in SEMANTICS "`tl.cdiv`" this round;
`tl.num_programs` and `tl.zeros` by their `TVA.Lang.Ops` definitions), 11 methods and attributes of
values (`.to`, `.reshape`, `.permute`, `.split`, `.trans`, `.T`, a descriptor's `.load` and
`.store`, `.dtype`, `.element_ty`, `.shape`), the builtins `float`, `isinstance`, `min` and `range`,
and six helpers of the same files. The keywords given are `mask` and `other` (`tl.load`), `mask`
(`tl.store`), `axis` (`tl.max`, `tl.sum`, `tl.program_id`), `dtype` (`tl.zeros`), `shape`, `strides`
and `block_shape` (`tl.make_tensor_descriptor`), the scheduling keywords `num_stages`,
`warp_specialize`, `flatten` and `disallow_acc_multi_buffer` (`tl.range`), and a helper's own
parameters; each is stated where its function is. The decorators are `triton.jit` (23) and
`triton.autotune` (8).

**Conclusion.** With `tl.cdiv` stated (evidence/cdiv-round17), no call, method or `tl` attribute of
the in-scope kernels lacks a meaning in SEMANTICS.md or `TVA.Lang`.
