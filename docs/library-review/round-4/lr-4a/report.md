# Library review, round 4, by lr-4a

## Verdicts

- **L7** ACCEPT — `foreign_refs` (`tools/tvr.py`) now flags a component `fpv` followed by any
  component other than `tva` (after `.` and `..` are resolved), starts a home path after any
  punctuation, reads `${HOME…}` and `$(…)` as `~`, and decodes escapes before the tests; protocol §2
  says the scan is lexical and that the guarantee for results is the import and lake checks.
  `tvr audit` prints "independent" with this report's work files present. My 12 probes
  (`work/l7_probes.py`, `.out.txt`): a path into a sibling written plain, with the `fpv` component
  double- or single-quoted, with empty quotes inside it, or quoted whole, is flagged, and TVA's own
  path passes. Brace expansion (`fpv/{tva,<sibling>}`), globs (`fpv/*/…`, `fpv/[o]…`),
  `cd ~/fpv && cat <sibling>/…` and a shell variable holding `~/fpv` pass: they are expansions the
  shell computes at run time, inside the stated lexical limit (a wording item below asks §2 to name
  them).
- **L12** ACCEPT — `transcription_rules` ends a rule at its first unindented non-blank line, and
  reports an unindented line that continues a rule's paragraph as a page error. On a copy of the
  tree (`work/gate_rerun.py`, `.out.txt`), case I (the closing paragraph of §3) changes the `map`
  and `model` digests of `vec_add_f32`, whose map does not name `[atomic]`, and my case S (an
  unindented line right after the first line of `[mask]`) fails the gate's `map` check with that
  page error.
- **L17** ACCEPT — `evidence/pdl` reran (`work/rerun/pdl.out.txt`): in part D, `y` is wrong in 20
  of 20 runs only when the claimed kernel calls `gdc_launch_dependents()` and the next launch has
  `launch_pdl=True`, and in 0 of 20 for each of the three other variants; parts A–C give the
  recorded results (C: 19 of 20 runs stale). `Faithful`'s scope point 1, the split
  `[no-op:gdc]`, TRANSCRIPTION §9 and exclusions.md state that case, and the wider "runs alone"
  condition, consistently.
- **L18** ACCEPT — `evidence/launch-options` reran with output identical to `out.txt` (D3, D4, F1,
  S1, A2, A3). Points 2 and 3 cover every knob class of `knobs.py:320-513` and the hooks that `run`
  and `_do_compile` call (`jit.py:726-781`, `:877-882`, read in the pinned file); the docstrings of
  `Launch.debug`, `Launch.fusion` and `DotCfg.precision` agree with them.
- **L19** ACCEPT — Cases A1, D1, E1 and B reran with the recorded output. `_pack_args`
  (`jit.py:702-717`) accepts every `CUDAOptions` field as a keyword, and point 4 excludes all of them
  except the four `Launch` records and the four that do not change the meaning; `warmup` cannot be
  passed through `kernel[grid](…)` (`jit.py:374` fixes it to `False`).
- **L20** ACCEPT — On a copy, cases L–O reran as recorded (L, N and O fail `namespace`, M fails
  `hygiene`), and the kernel-reviewer prompt, REVIEWER_GUIDE §1 and protocol §2 now put the meaning
  of every name on the kernel reviewer, who reads the whole Lean, as the user decided. The gate
  remains a partial backstop: a local binder named `tl` passes it (new issue below), which the
  reviewer's full read is meant to catch.
- **L23** ACCEPT — `evidence/literals` reran with output identical to `out.txt` (9364 literals, four
  dtypes, 0 differences from `FloatV.lit…` and from own type then one rounding for `other=`), and
  the source agrees (`semantic.py:118-158`, `:175-206`, `:601-623`, `:1638-1652`; `core.py`
  `where`). `work/literal_uses.py` runs the uses the evidence does not, with `1e-7` on `bfloat16`: a
  stored value, `tl.maximum` and `/` give `0x33d7` (own type `float32`, then converted), and
  `tl.where` and `+` give `+0` (built in `bfloat16`), as `[literal]` says. Two text defects remain
  (integer `tl.full`, the module text of `Lang/Scalar.lean`; wording items below).
- **L24** ACCEPT — `evidence/itof-rtz` reran with output identical to `out.txt` (0 of `2^32` differ
  from the new rule in both formats; two `tt.fp_to_fp` in the TTIR). `cast` (`semantic.py:811-845`)
  sends a downcast with a mode other than `rtne` to `create_fp_to_fp`, which no fold touches, and
  rejects `fp_downcast_rounding` on int-to-float casts; `FloatV.cvtWith .rz` overflows to the
  largest finite value, as `cvt.rz` does. The docstring of `tl.ItofRel` is stale (wording item
  below).
- **L25** REOPEN — soundness. The fix says that when a condition is not pure, "its operations are
  model lines, as [pure] says" ([pure]: in Python's order, as if each stood alone). But the front
  end does not evaluate every operation of a condition: `and` and `or` stop at a compile-time
  operand that decides them (`visit_BoolOp`, `code-generator.py:1481-1500`), and an `elif` after a
  taken compile-time `if` is never visited (`visit_If`, `:967-996`, visits only the active block).
  So the rule puts into the model operations that never run. `work/short_circuit.py` (4 programs
  each): `if FLAG and tl.atomic_add(c, 1) is not None:` with `FLAG = False` (the author's own C4),
  `if FLAG or tl.atomic_add(c, 1) is None:` with `FLAG = True`, and `if FLAG: … elif
  tl.atomic_add(c, 1) is not None:` with `FLAG = True` each run 0 atomics and have none in the TTIR,
  while the control `if tl.atomic_add(c, 1) is not None and FLAG:` runs 4. The README's "the rule
  asks for a pure condition anyway" does not hold: the rule covers impure conditions by transcribing
  them. Fix: say which operations of a non-pure condition are model lines, namely those the front
  end evaluates: left to right, stopping an `and` at a compile-time false operand and an `or` at a
  compile-time true one, only the taken side of a compile-time conditional expression, and an
  `elif` condition only when the branches before it are not taken; or require instead that no
  operation of a non-pure condition follows a compile-time operand of `and`, `or` or a conditional
  expression, and that it is not the condition of an `elif` after a taken branch. [pure]'s
  paragraph "When an argument is not pure" needs the same fix (new issue below).
- **L26** ACCEPT — `evidence/bcast-mul` reran with output identical to `out.txt`.
  `CombineBroadcastMulReducePattern` (`Combine.cpp:112-198`) needs a rank-3 reduce on axis 1, a
  combine region of one `addf` of at most 32 bits, and `N ≥ 16` and `M ≥ 16` from the two
  broadcasts, with any `K`; the second disjunct of `tl.sum_bcast_mul`
  (`d.bytes ≤ 4 ∧ 16 ≤ M ∧ 16 ≤ N`, `P.dot` with `ieee`) covers exactly that, and the first covers
  the plain reduction with fusable additions of the IEEE products. The pass does not check that the
  `addf` adds the region's two arguments, so a `tl.reduce` whose combine returns `a + a` or `b + b`
  is rewritten too (`work/bcast_reduce_aa.py`: 10, the sum of products, at `M = N = 16`; 4 and 16
  at `M = N = 8`). No rule transcribes such a `tl.reduce`, so no claim reaches it, and the rule's
  "returns the sum of its two arguments" is right; the definition's docstring is looser (wording
  item below).
- **L27** ACCEPT — On a copy, cases J (a sentence of SEMANTICS.md: `map` and `model` change), K (the
  docstring of `TVA.tl.add`: `model` and `assumption:faithful` change) and K2 (whitespace only:
  nothing changes) reran as recorded; `rule_parts` folds SEMANTICS.md into the general text, and the
  claim reader hashes the normalised docstring of every `TVA` constant in a cone. The rule pages
  under `docs/rules/` are still outside every digest (new issue below).

## New issues

- **NEW** soundness — `[pure]`'s paragraph "When an argument is not pure" (TRANSCRIPTION §10)
  transcribes every operation of a non-pure argument "in Python's order (left to right), as if each
  stood alone", but `and`/`or` with a compile-time operand that decides them, and a conditional
  expression with a compile-time condition, evaluate only part of the expression:
  `tl.static_print(FLAG and tl.atomic_add(c, 1))` and
  `tl.device_print("v", tl.atomic_add(c, 1) if FLAG else 0)`, each with `FLAG = False`, run 0
  atomics in 4 programs (`work/short_circuit.py`, K4 and K5). So the rule adds operations that do not
  run, the same defect as the L25 reopen. Fix: as for L25, transcribe only what the front end
  evaluates (left to right, stopping `and`/`or` at a compile-time operand that decides them, only
  the taken side of a compile-time conditional expression), or require that no operation of the
  argument follows such an operand.
- **NEW** soundness — The list of library device facts, `docs/rules/num.md` "Facts the library
  states", says which instances of the `Num.*` definitions, with which arguments, a kernel may cite
  (a field with other constants is `RULE_MISCITED`, "What a kernel reviewer checks"), but it is
  outside every digest: `rule_parts` (`tools/tvr.py`) folds only TRANSCRIPTION.md, exclusions.md and
  SEMANTICS.md, and the claim reader hashes Lean docstrings. So withdrawing a fact from the table, or
  changing its arguments or constants there, re-opens no `assumption:<f>` item. Today each stated
  fact is tied to a Lean constant (`DivFull` carries its constants in its definition), but L4 is to
  add facts with constants. Fix: state every library fact as a named Lean constant whose definition
  carries its arguments and constants, as `DivFull` does, and have the table only name them; or fold
  the table into the digest of every assumption item whose statement mentions a `TVA.Rules.Num`
  constant.
- **NEW** gap — The gate's `namespace` and `hygiene` checks see declared constants and source
  keywords, not local binders, so a binder named `tl` changes what a model line means.
  `work/lean/LocalTl.lean`: with a kernel `structure Ops` (field `add`) and
  `let tl : Ops := ⟨fun x y => TVA.tl.addc x y⟩` before `let output ← tl.add x y`, the line is
  `addc`, by `rfl`. `work/gate_q.py` puts such a binder before that line of `vec_add_f32`'s model on
  a copy of the tree, and the gate passes (with the field set to `TVA.tl.add`, so that the kernel's
  proof still builds; no check looks at the binder). Protocol §8 ("whatever the source syntax")
  reads as if the gate covered it. REVIEWER_GUIDE §1's step of printing the model with full names
  would show `Ops.add`, but its list (`def`, `notation`, `open`, `namespace`, …) does not name local
  binders. Fix: have `hygiene` forbid a binder named `tl` in a kernel's Lean, or check, as L20's
  to-do proposed, that each `tl.*` application in the elaborated model is a `TVA.tl.*` constant;
  and add local binders and field access to the reviewer's list.
- **NEW** gap — L4's list of device facts (`tl.math.exp2`, `tl.math.log2`, `/`, `tl.dot` with
  `float16`/`bfloat16` operands) leaves out operations of the tutorials TVA pins, whose kernels can
  therefore cite no accuracy fact: `tl.exp` on `float32` (`tutorials/02-fused-softmax.py:103`,
  `MathAcc .exp`), `tl.sqrt` on `float32`, which is `sqrt.approx.ftz.f32`
  (`tutorials/05-layer-norm.py:81`, `MathAcc .sqrt`), a `float32` → `float8e4nv` conversion
  (`tutorials/09-persistent-matmul.py:151`, `CvtAcc`), and `tl.dot_scaled`
  (`tutorials/10-block-scaled-matmul.py:230-234`, `DotScaledAcc`). Fix: add them to L4's list, with
  their lowering on sm_90: `tl.exp` is a `mul.f32` by `log2 e`, then `ex2.approx.f32` without `.ftz`
  (subnormal results kept), while `tl.exp2` is libdevice's `ex2.approx.ftz.f32`, which flushes
  results below `2^-126` (`work/ptx_ops.out.txt`, `work/num_questions.out.txt`).
- **NEW** wording — `DivFull`'s docstring and `docs/rules/num.md` ("within 2 ulp"; "Division": "a
  maximum error of 2 ulp") quote the PTX ISA, but on the device the error of `/` on `float32`
  reaches 2.11 ulp of the quotient's binade, near the top of a binade
  (`work/div_full_binades.out.txt`). The card itself, `2^-22·|x/y| + 2^-148` for `|x/y| ≤ 2^126`,
  holds with margin: no pair outside it and no non-finite result, and at most 0.654 of the bound,
  over every divisor mantissa at six exponents (subnormal, `2^-126`, `1`, `2^125`, `2^126`,
  `2^127`, 16 dividends each, `work/div_full_exh.py`) and about `2·10^9` adversarial pairs
  (`work/div_full.py`, `work/div_full_binades.py`). Fix: describe the card as the relative bound it
  is rather than as "2 ulp", and cite TVA's measurement when L4 adds one.
- **NEW** wording — The module text of `lean/TVA/Lang/Scalar.lean` ("Python number constants
  (library issue L23)") lists `tl.maximum` among the operations that build the constant directly in
  the operation's dtype. TRANSCRIPTION `[literal]` puts `tl.maximum` under "a tensor of its own
  type", rightly: `tl.maximum(x, 1e-7)` on `bfloat16` gives `0x33d7`, not `+0`
  (`work/literal_uses.py`). Fix: drop `tl.maximum` from that list.
- **NEW** wording — `[literal]` says that in an integer dtype the constant is "`c` itself", but
  `tl.full` does not check the range and wraps: `tl.full([1], 300, tl.int8)` is 44, `256` in `uint8`
  is 0, `2^31` in `int32` is `-2^31` (`work/literal_int.py`). A negative `c` in an unsigned dtype or
  a `float` `c` is rejected on the host, and the operators reject an out-of-range `c`. Lean's
  `OfNat` on `IntV` wraps as well, so a model that writes the literal agrees. Fix: say "`c` modulo
  `2^w` (`IntV.ofInt c`)".
- **NEW** wording — The docstring of `tl.ItofRel` still says that MLIR folds "any narrowing of an
  integer-to-float result" into one conversion and that "the model writes each such narrowed value
  as one `tl.itof`"; since L24 that holds only for the default rounding (`[itof-narrow]`,
  SEMANTICS.md). Fix: add the restriction and the `rtz` case to the docstring, which is hashed since
  L27.
- **NEW** wording — The docstring of `tl.sum_bcast_mul` ("a `tl.reduce` whose combine function is
  one addition") and SEMANTICS.md ("a `tl.reduce` with one addition") admit a combine that returns
  `a + a`, which the pass also rewrites into a dot at `M, N ≥ 16`, but which is not the sum of
  products at smaller sizes (`work/bcast_reduce_aa.py`: 4 or 16 where the sum is 10).
  `[sum-bcast-mul]` says "whose `f` returns the sum of its two arguments", which is right. Fix: use
  the rule's words in both places, and say that another single-addition combine (`a + a`, `b + b`)
  is rewritten too and has no rule.
- **NEW** wording — `lean/TVA/Rules/Num.lean`'s module text ("The reviewer judges each instance
  against the device and Triton 3.8.0's lowering, with the evidence it asks for") and `DotAcc`'s
  docstring ("the reviewer judges them with its input precision …") still give the device check to
  "the reviewer", which L11 moved to library review, and kernel reviewers read these docstrings.
  Fix: say "library review".
- **NEW** wording — Protocol §2, "The text scan is lexical", names computed paths (`os.path.join`,
  f-strings, `Path.home()`, shell variables) but not shell expansions, which pass too: brace
  expansion `fpv/{tva,<sibling>}`, globs `fpv/*/…` and `fpv/[o]…`, and `cd ~/fpv` followed by a
  relative path (`work/l7_probes.py`). Fix: name them in that paragraph, or flag `fpv/` followed by
  `{`, `*` or `[`.

## Re-ran

- The batch's device evidence, Triton 3.8.0 on the H200 NVL (GPU 0), 2026-10-27, outputs in
  `work/rerun/`: `config-cond`, `literals`, `bcast-mul`, `launch-options` and `itof-rtz` give output
  identical to their `out.txt`; `pdl` too, except the count of stale elements in part C (a race:
  29931520 against 28327936, 19 of 20 runs both times).
- The earlier rounds' device evidence, the same way: `argmax`, `int-reductions`, `permute`,
  `env-knobs` and `pure-operands` identical; `launch-outcomes` identical but for timings;
  `intra-program` the same results with other race counts (for example 41.6% against 41.4% stale
  in the reversed read-back); `hints-dry-run` the same (the misaligned run with the hint faults, the
  one without completes), but for rounding errors of the runs; `debug-calls` the same, plus two
  lines `tl.static_print` printed at compile time.
- `evidence/gate-digests` on a copy of the tree (`work/gate_rerun.py`, `.out.txt`): cases A and I–P
  as recorded, plus my cases Q (a local binder `tl`; it did not build there, because the binder also
  captured `tl.store`) and S (an unindented continuation of `[mask]`: a page error). `work/gate_q.py`
  is case Q made to build: the gate passes.
- `tvr audit` on the tree: "independent".
- Tests added, in `work/`, with their outputs (`*.out.txt`):
  - `ptx_ops.py`: the PTX of `+`, `-`, `*`, `tl.fma`, `/`, `tl.div_rn`, `tl.sqrt_rn`, `tl.sqrt`,
    `tl.exp`, `tl.exp2` and the conversions, per dtype: the exact operations are `add`/`sub`/`mul`,
    `fma.rn`, `div.rn.f32`, `sqrt.rn.f32` and `cvt.rn`, none with `.ftz`; `/` is `div.full.f32`
    (`float32`, after `cvt.f32` from `float16`/`bfloat16`) and `div.rn.f64`;
  - `exact_ops.py`: "Exact operations need no rule" (rules/num.md) and "no flushing of subnormals"
    (SEMANTICS.md), 0 differences everywhere: `tl.sqrt_rn` on every `float32` (`2^32`);
    `float32` → `float16` and → `bfloat16` on every `float32`; `float16` and `bfloat16` `+` and `*`
    on every pair of patterns (`2^32` each); `float32` `+`, `*` and `tl.div_rn` on `2^28` random
    pairs, a third subnormal; `tl.fma` in the four standard formats, 20000 triples each against the
    exact result (cancellation and subnormal results included); `float64` `+`, `*`, `/` on `2^24`
    pairs against numpy (so `DivRn P .binary64`); `float64` → `float32`, `float16`, `bfloat16`, and
    `float16` ↔ `bfloat16` on every pattern;
  - `div_full.py`, `div_full_binades.py`, `div_full_exh.py`: `DivFull` holds (wording item above);
  - `num_questions.py`: `tl.exp`/`tl.exp2` at `±∞`, NaN and in the subnormal range (`exp(-∞) = +0`,
    `MathAt`'s example); `float32` `tl.atomic_add` flushes subnormal operands and results (so
    `abs ≥ 2^-125`, as rules/num.md says); a scalar `tl.atomic_add(p, ±0.0)` with `relaxed` or
    `acquire` has no atomic instruction and leaves `-0.0` in memory, with `release`/`acq_rel` it is
    an atomic and leaves `+0` (a `-0.0` literal is `+0`, `[literal]`);
  - `subnormal_ops.py`: `tl.maximum`, `tl.minimum`, `tl.clamp`, `tl.abs`, unary `-`, `<`, `==`,
    `tl.where`, `tl.max`, `tl.min`, `tl.sum` and float `tl.atomic_max`/`tl.atomic_min` keep
    `float32` subnormals (`max.f32`, `min.f32`, `setp` without `.ftz`);
  - `literal_uses.py`, `literal_int.py` (L23), `short_circuit.py` (L25), `bcast_reduce_aa.py` (L26),
    `l7_probes.py` (L7), `lean/LocalTl.lean` (L20; `lake env lean` accepts it, `rfl` included).

## Not checked

- `evidence/independence-audit`'s 79 probes (I ran my 12 against `foreign_refs` instead), and
  `evidence/gate-digests` cases B–H of round 3.
- The excluded launch options other than `ir_override`, `allowed_dot_input_precisions`,
  `default_dot_input_precision` and `enable_reflect_ftz` were read in the source, not run, as by the
  author; "does not change the meaning" for `num_warps`, `num_stages`, `maxnreg` and
  `launch_cooperative_grid` rests on the evidence's 9 variants.
- fp8 constants in operations (read in the source only, as by the author), and the conversions into
  fp8 (`CvtAcc` is not a stated fact).
- No accuracy constants (`MathAcc`, `DotAcc`, `CvtAcc`, `AtomicAddAcc`, `DotScaledAcc`) are stated
  yet (L4), so none was checked.
