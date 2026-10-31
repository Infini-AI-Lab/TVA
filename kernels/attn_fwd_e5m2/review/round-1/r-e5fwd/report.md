# Review of attn_fwd_e5m2, round 1, by r-e5fwd

## Verdict

accept-partly
The model, the model map, the conditions and the twelve [rule] assumptions are right, and I accept
them. The model follows the source line by line by SEMANTICS.md, including line 98's transposed
load of v through the [HEAD_DIM, Z·H·N_CTX] descriptor. That load gives vEff, under which every head
without vIn outputs ±0, and the spec states this honestly. The conditions cover every test and
benchmark launch, and the cap Xh holds on the host's data. I do not accept the specification. On the
test's and the benchmark's data, Post accepts an all-zero o on 96.8-100% of the elements of the
heads that do get their v. At the benchmark's D = 128 the tolerance is about 10^4 for values of
magnitude at most 4, about three orders of magnitude above a bound of the same facts' form (F1).
Part of this is the library's DotE5m2 and part is slack in the kernel's bound. The write-up should
put the vEff defect up front and stop calling the bound meaningful on the test's data (F2).

## Findings

- **SPEC_TOO_WEAK** `decl:Kernels.AttnFwdE5m2.tolO` — tolO accepts o = 0 on 96.8-100% of elements on the test's and the benchmark's data; at the benchmark (D = 128) it is about 10^4 for |v| <= 4, about 2000x a bound the same facts give
- **DOC** `doc:kernels/attn_fwd_e5m2/README.md` — say up front that only the first ceil(Z·H/D) heads get their v; drop "meaningful on the test's data"; give the benchmark's D = 128 tolerance

## Accepted

- `item:model`
- `item:map`
- `item:conditions`
- `item:assumption:faithful`
- `item:assumption:dot`
- `item:assumption:dot_total`
- `item:assumption:cvt`
- `item:assumption:cvt_total`
- `item:assumption:exp2`
- `item:assumption:exp2_ninf`
- `item:assumption:exp2_total`
- `item:assumption:log2`
- `item:assumption:log2_total`
- `item:assumption:div_full`
- `item:assumption:div_total`

## Checked

- claim.md in full: the conditions, specification, assumptions, model, model map and gate.
- The source: the kernel lines (20-21, 47-110, 168-173, 176-245) line by line against SEMANTICS.md and the model map. That covers every entry and every drop: the fusions (mul_add at lines 76 and 106, subc at 81, IEEE sub at 78 and 84, IEEE add at 241), the constants, the hints (59 checked, 70 dropped), the dead compile-time branches (66-67, 88-93, 99-100, 200-202), isinstance in _maybe_make_tensor_desc, and Z = 1 or H = 1 as constexprs with the same model.
- kernel.json's classing of the lines. The claimed launch runs only _attn_fwd, _attn_fwd_inner and _maybe_make_tensor_desc. A scan of the source's AST found that the module-level names it reads are only the imports of triton and triton.language as tl and the two helpers' defs. Nothing rebinds them, and the source has no walrus and no global.
- The host: the wrapper (505-580), the test (631-695) and the benchmark (698-780). Every shape meets Inst.ok. Both descriptor kinds (warp_specialize) are covered by descArg and descArgV.
- Every Lean file of the kernel: Model, Facts, Inner, Tile, Program, Lane, Real and the root in full; Row, RowMax, RowAcc and Final in part; Step's declarations. The kernel has no instance, notation, attribute, export or renaming, and its only set_options are a linter and maxHeartbeats.
- work/print_names.lean (output in print_names.out): the model, launch, Inst.ok, Pre, Post, HeadOk, Assumptions and Spec with full names. Every operation is a TVA definition or a kernel helper I read. main, consistent and satisfiable use only propext, Classical.choice and Quot.sound.
- The vIn / vEff reading: the v descriptor is [HEAD_DIM, Z·H·N_CTX] with strides [N_CTX, 1], loaded at [0, bh·N·D + 64t]. I checked it against the model, SEMANTICS.md's tensor descriptors and Facts.lean's v lanes, and counted the heads with vIn on the host's shapes.
- Each [rule] assumption against docs/rules/num.md and compiler-faithful.md for the operation the model uses. I also checked that the launches meet points 4, 5, 6, 8 and 9 of part (A) of Compiler.Faithful's scope.
- The tolerances evaluated with Model.lean's formulas on test-like and benchmark-like data (work/tol.py, tol.out), and the cap Xh against the host's data (work/cap.py, cap.out).
- The README against the Lean: the Specification, Conditions, Assumptions, Proof and Narrowings sections.

## Not checked

- The proof bodies of the numeric lemma files (RowMax, RowAcc, Step, Final): I read them in part and rely on Lean having checked them.
- Part (B) of Compiler.Faithful's scope (the environment), which is assumed, and whether the library's facts and constants are true, which is library review's job.
- Whether the magnitude bound I use in F1 is provable as stated. It is an estimate of the form the facts give, offered as a direction, not as a proved bound.
- The tolerance on the benchmark's data at N above 1024, beyond the v = 1 case. tol.py samples rows at N = 1024 only for the benchmark.
