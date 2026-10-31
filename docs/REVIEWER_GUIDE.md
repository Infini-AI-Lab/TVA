# Kernel reviewer guide

You review one kernel's claim. The process is [REVIEW_PROTOCOL.md](REVIEW_PROTOCOL.md) (§9:
filings, finding kinds, acceptance), and your prompt is
[prompts/kernel-reviewer.md](prompts/kernel-reviewer.md). Start from the kernel's `claim.md`
(`kernels/<kernel>/`).

The prover claims, with a Lean proof:

```lean
theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K
```

*If the device behaves as the assumptions say, then on every instance `Inst.ok` covers and every
input `Pre` allows, every launch that runs ends normally with outputs satisfying `Post`.* Lean
guarantees the implication, and `consistent` that the assumptions can hold together. Lean cannot
tell whether the model is the kernel, whether `Post` is what the kernel is for, whether `Inst.ok`
covers how it is used, or whether a `[kernel]` assumption is about what the write-up says. You
can. You do not change the kernel; you raise findings.

## Why it matters, and how to approach it

A certificate tells its readers that the kernel does what `Post` says wherever it is used. If the
model differs from the source in one line, or `Post` is weaker than it looks, or `Inst.ok` leaves out
the sizes the host really uses, the certificate is wrong, and Lean's check will not reveal it.
Nobody downstream re-reads the claim. So do not review by ticking off a list: ask **how this claim
could be wrong**, and look for the place where it is. The questions below are where claims usually
fail; they are not the limit of what you may find.

## Your scope

**You trust the TVA library**, which library reviewers check separately:
- what every `tl.*` operation, launch and outcome means, and every other fact about Triton
  [SEMANTICS.md](SEMANTICS.md) states (how the front end evaluates the source, the constants it
  builds, the shapes its compiler rewrites);
- `Compiler.Faithful` ([rules/compiler-faithful.md](rules/compiler-faithful.md));
- the device facts ([rules/num.md](rules/num.md));
- the lemmas.

You need not know how Triton compiles anything, and you do not read compiler sources or PTX,
consult other projects, or run the kernel. A launch that Triton reports to the host with an
error before it runs (resources, unsupported sizes) has no outcome, so whether the kernel
compiles is not part of the claim. If you doubt the library, file a `LIBRARY_REPORT` and carry on
as if the library were right. It does not block the kernel.

[TRANSCRIPTION.md](TRANSCRIPTION.md) is guidance: worked patterns, the pitfalls that most often
make a model differ from its source, and common reasons to drop a line. It is a help, not a
standard: a model that departs from a pattern is fine if it means what the source does, and a
model that follows the patterns is wrong if it does not.

## 1. The model and the map (`MODEL_MISMATCH`, `MAP_INCOMPLETE`, `UNSUPPORTED`)

The model is the prover's transcription of the source. Read the two side by side (claim.md §6
shows the map), and ask of each line: on every covered instance, does this model line do what
this source line does, by SEMANTICS.md? The usual slips are in TRANSCRIPTION.md §3: an IEEE `+`
that a product feeds, a missing conversion or broadcast, a constant built in the wrong dtype, a
different `mask` or `other`, a loop bound or carried variable, a hint left out or misplaced.

**The supported subset** (library issue L88). TVA gives meaning only to the forms of Python that
SEMANTICS.md lists in "The supported subset"; every other form is excluded
([exclusions.md](exclusions.md)). Check every line the launch runs, on the device or at compile time
(the kernel, its helpers, the decorators and the module-level names they read), against that list:
its statements, names, expressions, calls, decorators and parameters. A kernel that uses a form
outside it (a `break`, a `with`, a bare annotation or an annotation other than `tl.constexpr`, a
`@triton.heuristics`, a `@triton.autotune`, a `@triton.constexpr_function` or another Python function called at compile
time, a module-level value other than Triton's modules and the module's `@triton.jit` functions, a
`for` loop's variable used after the loop, an assignment in a runtime `if` to a name holding `None`
or a dtype, a helper whose visited `return`s give values of different kinds, or compile-time values
that are not the same value of one Python type, element by element for a tuple (`1` and `1.0`,
`(1, 2)` and `(1.0, 2)`, `0.0` and `-0.0`, two NaNs), or that can fall off its end while another
`return` gives a value, two calls of one helper whose compile-time arguments differ but name the
same copy, such as two NaNs, an operator computed in an fp8 dtype or a unary `-` on an fp8 tensor,
…) has no claim: file `UNSUPPORTED`, whatever the model says.

**The ranges.** claim.md §2 lists every function the source defines (every `def` and `lambda`)
as the manifest classes it: in the kernel lines, in the host lines (`source.host`), or unused
(`source.unused`). The gate checks only that each is in one class (and no `@triton.jit` function
in the host lines); which functions the launch runs is yours to judge. Check that the ranges hold
every function the launch runs, on the device or at compile time, as the kernel or as anything it
calls: a JIT function however it is made (`@triton.jit`, a decorator under another name, a call
`triton.jit(f)`), a `triton.constexpr_function`, and any plain Python function the front end calls
at compile time (one reached through a module, such as the kernel's own module, or called from a
constexpr function); the last two are outside the supported subset (above). A host function must
run only on the host, and the body of an unused one must be run by nothing: their lines are mapped
by nobody. An unused function's `def` statement still runs at import (below). A function the launch
runs that the source does not define (imported from another file), other than Triton's own, is
mapped by nobody either (`MAP_INCOMPLETE` or `UNSUPPORTED`).

**The code outside every function.** claim.md §2 also lists the code lines that no function spans
(imports, module-level assignments, a class's own lines, a `__main__` block), and the header of
each unused function (its decorators and `def` lines), as the manifest classes them: in the kernel
lines or in the host lines (the gate checks only that each is in one, library issue L79). All of
them run when Python imports the module: a `def` evaluates its decorators and its parameters'
defaults then (and Python 3.13 its annotations), even for a function nothing calls, so
`def unused(f=(double := triple))` rebinds the JIT function `double` the kernel calls. The front end
resolves a name of the kernel to a module-level value at compile time, and the value is part of the
compiled kernel. The module-level names a launch may read are the imports of `triton` and of
`triton.language as tl`, and the module's `@triton.jit` functions, each bound only by its own
`import` or `def` (SEMANTICS.md, "The supported subset"). A kernel that reads any other module-level
value (a `tl.constexpr(…)` global, a dtype such as `OUT_T = tl.float16`), or a name another statement
rebinds, is `UNSUPPORTED` (L88). Check that the import that binds `tl` and every JIT function the
launch reads lie in the kernel lines and are mapped there (an `import … as tl` of another module
would give `tl.load` another meaning), and that the host lines bind nothing the launch reads: in
particular, that the header of each unused function (its decorators, its defaults, its
annotations) binds or changes no name the kernel lines read. The gate checks only that every such
line is classed, not what it binds (protocol §7). A name the launch reads from the host lines, or
from another file, is mapped by nobody (`MAP_INCOMPLETE`).

**Drops.** A dropped line is a claim that the line changes nothing. Check the reason against
SEMANTICS.md, in particular "How the front end evaluates the source": an operation inside a
condition, a print or an assertion runs as far as the front end evaluates it, and a branch the
configuration does not take is never visited. Does the note's condition hold on every covered
instance? A line dropped that does something is `MAP_INCOMPLETE`.

**Names: read the kernel's whole Lean.** A model line can read as a library operation and mean
something else, without any change to its text. Read every Lean file of the kernel
(`kernels/<kernel>/lean/`) in full, not only what claim.md quotes, and judge what each name in the
model means: every `def`, `notation`, `open`, `namespace`, `instance`, `attribute`, `export` and
`set_option`, every binder (`fun`, `let`, a parameter with a default value, a pattern) and every
field access. The routes found so far: a kernel definition named like a library operation
(`tl.add` in the kernel's namespace, `Launch.mk`), a name that shadows a root name (`min`), a
look-alike (`tℓ.add`), a local notation, an instance of any class (`HDiv Nat Nat Nat` changes the
grid formula), a local binder `tl` whose field `add` is something else, `open … renaming`, `export`.
Print the model with full names (`set_option pp.fullNames true`, `#print`): every operation should
be `TVA.*`, or a helper of the kernel whose definition you have read. A model line that means
something else is a `MODEL_MISMATCH` (user, 2026-10-27, library issue L20). The gate's `hygiene`
and `namespace` checks catch the common routes mechanically; they are a backstop, not a
substitute.

**The launch**: the grid formula, the `program_id` axes, every block shape in `blocks`
(`Launch.Valid` checks only the listed ones), `num_ctas`, `debug` and `enable_fp_fusion`.

**The kernel and its launches** must stay within part (A) of `Compiler.Faithful`'s scope
([rules/compiler-faithful.md](rules/compiler-faithful.md), "Scope"; the same list is the page's
checklist, "How a bundle cites it"), point by point (the points keep their numbers):
- 4. only the launch options `Launch` records, and `num_warps`, `num_stages`, `maxnreg` and
  `launch_cooperative_grid`, wherever the launch's own code gives them (the launch,
  `kernel.warmup(…)`, `triton.compile`'s options, a `triton.Config`), and no `@triton.heuristics`;
  the decorator's `debug` is `Launch.debug`, which must be `false`;
- 5. when the kernel builds descriptors (`tl.make_tensor_descriptor`), an allocator
  (`triton.set_allocator`) whose buffer holds none of the claim's buffers and that nothing else
  writes during the launch;
- 6. for a `CompiledKernel` (`kernel.warmup(…)`, `triton.compile`), the kernel's own source (an
  `ASTSource`, not an IR file) and arguments in its specialisation: the warm-up's dtypes, each
  pointer 16-byte aligned exactly when the warm-up's was, each integer in the warm-up's class, each
  host `TensorDescriptor` with the warm-up's dtype and block shape;
- 8. no host `TensorDescriptor` with `round_f32_to_tf32=True`;
- 9. no CUDA graph capture: the launch is not made while a CUDA stream capture is in progress
  (`torch.cuda.graph`, `triton.testing.do_bench_cudagraph`), on the capturing stream or any other,
  and is not a replay of a captured graph.

The launch is a direct launch of the `@triton.jit` function: `@triton.autotune` is outside the
supported subset, and a source that still has it is `UNSUPPORTED` (library issue L116). For a
kernel a tutorial autotunes, the user and the coordinator chose one config when they set up the
kernel's folder, removed the decorator and pass the config's values at the launch
([REVIEW_PROTOCOL.md](REVIEW_PROTOCOL.md) §3); judge the kernel as given in `source/`, not the
choice of config ("Which launch"). A launch that leaves
part (A) is `CONDITION_WRONG` (the claim then covers nothing it runs) or a narrowing the write-up
must state.

**The environment** is part (B) of the scope, and you do not check it: TVA verifies the kernel,
not the Triton installation or other code users write (the user's decisions, 2026-10-29: "To make
the verifier easy, we do not verify triton cache.", "we assume the kernel is new." and "So
basically, we want to verify the kernel itself, rather than triton installation or other code
programmed by users."). Points 1 (nothing else writes the claim's buffers), 2 (Triton's knobs and
environment variables), 3 (hooks), the environment's part of 4 (no `kernel.debug` set from outside,
no `preload`), 7 (the H200 and an x86-64 host with CPython 3.13 without `-O`/`-OO` or
`PYTHONOPTIMIZE`, in the default floating-point environment), 10 (Triton as installed, and no other code changing Triton or the
kernel's module) and 11 (the kernel is new: the code the launch runs is the code Triton compiles
for that launch's own argument values, compile-time and runtime, and settings; code a cache returns
counts when it is that code) are assumptions of
the rule, and you do not read host code for them. If the claim's own launches or the kernel's
source visibly break one (the source sets a knob, rebinds a Triton name, or launches a second
kernel that writes the claim's buffers), that is still a finding.

**A construct the library gives no meaning to**, or whose meaning is open (marked with a library
issue, such as a runtime `if`, L28), is `UNSUPPORTED`. A model that relies on a fact about Triton
SEMANTICS.md does not state is `UNSUPPORTED` too: the library must state the fact first.

## 2. The conditions (`CONDITION_TOO_NARROW`, `CONDITION_WRONG`)

A claim is only as useful as the instances it covers. Do `Inst.ok` and `Pre` cover the kernel's
real use: the host code's sizes, strides, dtypes, alignment and configs, and realistic input values
(NaN, infinities, large values, unless the kernel is not meant for them)? Is every narrowing stated
in the write-up with its reason, such as a bound that `int32` indexing needs? Are the conditions
satisfiable, and are they what the write-up says? A runtime argument's value is part of the
instance, and the prover's conditions constrain it, for example `n ≤ x.len` for a length
`n_elements` of the buffer `x` (the user's decision, 2026-10-29, library issue L117: "for the
viriable n_elements, its value should be merged into the assumptions of the prover. e.g., x.len >=
n_elements, etc."): point 11 assumes that the launch runs the code Triton compiles for those values,
so judge the conditions on them as on any other.

## 3. The specification (`SPEC_TOO_WEAK`, `SPEC_WRONG`)

`Post` is what the certificate promises. Does it say what the kernel is for, each output
determined or bounded? Is a tolerance meaningful at the **largest** sizes the conditions allow?
Evaluate it there: a bound that grows with `n` may say nothing at `n = 2^30`. Would it reject
plausible wrong kernels, such as a rounding to a lower precision or a dropped term? Does it say
that memory outside the outputs is unchanged?

## 4. The assumptions (`RULE_MISCITED`, `ASSUMPTION_*`, `EVIDENCE_REQUEST`)

An assumption is where a claim can hide what it did not prove.
- **`[rule]`**: a library rule or device fact. Is it the library's fact for the operation the model
  uses: the same function, format, configuration and target? Read what the rule says on its page.
  Whether it is true is the library's matter. The gate ensures that the field applies a fact the
  library states to the platform, by its name, as [rules/num.md](rules/num.md)'s "Facts the
  library states" lists it (`Exp2F32`, `DivFull`, `DivRn P .binary64`, a `…Total` rule), or
  `Compiler.Faithful` as `Faithful P (model d) (launch d) (K d)` for one bound `d` (the model of
  the instance launched), never a shape such as `MathAcc` or `DotAcc` with constants of its own,
  and that it adds no constant of the kernel's own (protocol §5, library issue L62). What you
  judge is whether it is the right listed fact, and whether the binders and hypotheses around it
  (`∀ d, d.ok → …`) say what the write-up says.
- **`[kernel]`**: a fact about the kernel's own values. It mentions neither `K` nor `P` (the gate
  checks it, and its `shape` check refuses an `Assumptions` that is not a `structure`, library issue
  L106, or has any parameter but `(K : Claim.Behaviour Inst) (P : Platform)`, library issue L94;
  check, reading the structure, that it is one and has exactly those two), so it is a closed
  statement, and Lean proves it through `consistent`: it
  cannot be false as stated. What you
  judge is what it states. Is it needed, and clear? Do the kernel constants it constrains (an error
  constant, a bound) mean what the write-up says, so that the statement says what the write-up
  claims? Ask for an argument (`EVIDENCE_REQUEST`) when that is not evident.

## 5. The write-up (`DOC`)

Does `README.md` say in words what the Lean says: the specification, the conditions with every
narrowing, the assumptions, the proof's outline?

## Before you accept

- You have looked for the way the claim could be wrong, and every finding you raised is settled.
- `#print axioms main` shows only `propext`, `Classical.choice` and `Quot.sound` (the gate checks
  it too).
