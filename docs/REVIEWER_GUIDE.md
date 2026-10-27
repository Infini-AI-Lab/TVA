# Kernel reviewer guide

You review one kernel's claim. The process is [REVIEW_PROTOCOL.md](REVIEW_PROTOCOL.md) (§9:
filings, finding kinds, acceptance), and your prompt is
[prompts/kernel-reviewer.md](prompts/kernel-reviewer.md). This page says what to check. Start
from the kernel's `claim.md` (`kernels/<kernel>/`).

The prover claims, with a Lean proof:

```lean
theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K
```

*If the device behaves as the assumptions say, then on every instance `Inst.ok` covers and every
input `Pre` allows, every launch that runs ends normally with outputs satisfying `Post`.* Lean
guarantees the implication, and `consistent` that the assumptions can hold together. You decide
whether the prover's choices are right. You do not change the kernel; you raise findings.

## Your scope

**You trust the TVA library**, which library reviewers check separately:
- the meaning of every `tl.*` operation, of launches and of outcomes ([SEMANTICS.md](SEMANTICS.md));
- the transcription rules ([TRANSCRIPTION.md](TRANSCRIPTION.md));
- `Compiler.Faithful` ([rules/compiler-faithful.md](rules/compiler-faithful.md));
- the device facts ([rules/num.md](rules/num.md));
- the lemmas.

You need not know how Triton compiles anything, and you do not read compiler sources or PTX,
consult other projects, or run the kernel. A launch that Triton reports to the host with an
error before it runs (resources, unsupported sizes) has no outcome, so whether the kernel
compiles is not part of the claim.

If you doubt the library, file a `LIBRARY_REPORT` and carry on as if the library were right. It
does not block the kernel.

## 1. The model and the map (`MODEL_MISMATCH`, `MAP_INCOMPLETE`, `UNSUPPORTED`)

Read the source and the model side by side; claim.md §6 shows the model map.
- **Each mapped line.** Is the model text what the transcription rules give for that construct,
  and does the entry name every rule it applies (its `rules`, shown in brackets in claim.md §6,
  with the rules' text below the map)? A rule left out of an entry is `MAP_INCOMPLETE`: a later
  change to that rule would not re-open the kernel. The usual slips: a missing conversion or
  broadcast, an IEEE `+` that a product feeds, a different `mask` or `other`, a loop bound or
  carried variable, a hint left out, an integer `tl.max` without its widening to `int32`.
- **Each drop.** Is it one of the drop reasons (each a rule of TRANSCRIPTION.md §10), and does
  its note hold on every covered instance? For `no-op:print`, `compile-check` and `dead`, is every
  operand of a pure type (the rule `[pure]`: no aggregate class)?
- **Names.** Print the model with full names (`set_option pp.fullNames true`, `#print`): every
  operation is `TVA.*`. The gate also checks that the kernel declares nothing in `TVA` and no
  instance of a library class.
- **The launch** (rules §9): the grid formula, the `program_id` axes, every block shape in
  `blocks`, `num_ctas`, `debug`, `enable_fp_fusion`, and the autotune configs.
- **A construct the rules do not cover** is `UNSUPPORTED`. So is one whose rule is open (marked
  `open: L<n>`).

## 2. The conditions (`CONDITION_TOO_NARROW`, `CONDITION_WRONG`)

- Do `Inst.ok` and `Pre` cover the kernel's real use? Check the host code's sizes, strides,
  dtypes, alignment and configs, and realistic input values: NaN, infinities and large values,
  unless the kernel is not meant for them.
- Is every narrowing stated in the write-up with its reason, such as a bound that `int32`
  indexing needs?
- Are the conditions satisfiable, and are they what the write-up says?

## 3. The specification (`SPEC_TOO_WEAK`, `SPEC_WRONG`)

- Does `Post` say what the kernel is for: each output determined, or bounded?
- Is a tolerance meaningful at the **largest** sizes the conditions allow? Evaluate it there.
  Does it reject plausible wrong kernels, such as a rounding to a lower precision or a dropped
  term?
- Does it say that memory outside the outputs is unchanged?

## 4. The assumptions (`RULE_MISCITED`, `ASSUMPTION_*`, `EVIDENCE_REQUEST`)

- **`[rule]`**: a library rule or device fact. Check that it is the library's fact for the
  operation the model uses: the same function, format, configuration and target. Read what the
  rule says on its page. Whether it is true is the library's matter. The gate ensures the
  statement adds no constant of the kernel's own.
- **`[kernel]`**: a fact about the kernel's own values that the prover did not prove. Is it true,
  needed and clear? Ask for a proof or an argument (`EVIDENCE_REQUEST`) unless it is evident.

## 5. The write-up (`DOC`)

Does `README.md` say in words what the Lean says: the specification, the conditions with every
narrowing, the assumptions, the proof's outline?

## Before you accept

- Every finding you raised is settled.
- `#print axioms main` shows only `propext`, `Classical.choice` and `Quot.sound` (the gate checks
  it too).
