---
name: tva-claim
description: Write the claim's conditions and specification (Inst, Inst.ok, launch, Pre, Post, Spec) and the Assumptions structure for a TVA kernel. Use at stage 5, or when a reviewer files CONDITION_*, SPEC_*, RULE_MISCITED or ASSUMPTION_* findings, or the gate's shape/kinds/usage checks fail.
---

# Conditions, specification, assumptions (PROVER_GUIDE §2–§3)

Copy the shape from the closest example (`tva-kit lookup <ExampleNamespace>.Inst.ok` etc.), then adapt.

## The parts, in the order they appear in the file

```lean
structure Inst where            -- everything the host chooses: sizes, block sizes, scalars, buffers
  n : Nat
  BLOCK_SIZE : Nat
  x : Buf .f32                  -- a tensor argument: `Buf d` (base address, length in elements)
  ...
structure Inst.ok (d : Inst) : Prop where   -- the instances covered, one named field per condition
  block_pow2 : ∃ e, d.BLOCK_SIZE = 2 ^ e
  block_le : d.BLOCK_SIZE ≤ 2 ^ 20
  n_lt : d.n < 2 ^ 31           -- int32 index arithmetic must not wrap
  x_len : d.n ≤ d.x.len         -- buffers hold what the kernel reads/writes
  out_x : Disjoint (d.out.elems {j | j < d.n}) (d.x.elems {j | j < d.n})  -- written buffers disjoint
def model (d : Inst) : Kernel := my_kernel (d.x.ptr 0) ... (IntV.ofNat d.n) d.BLOCK_SIZE
def launch (d : Inst) : Launch := Launch.grid1 ((d.n + d.BLOCK_SIZE - 1) / d.BLOCK_SIZE) [[d.BLOCK_SIZE]] (fusion := true)
def Pre (d : Inst) (init : Mem) : Prop := d.x.Valid init ∧ ...        -- buffers valid; input values if needed
def Post (d : Inst) (init fin : Mem) : Prop := (∀ j < d.n, …) ∧ ∀ z, z ∉ d.out.elems {j | j < d.n} → fin z = init z
def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K
structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: … -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
```

## Conditions (`Inst.ok`, `Pre`)

- Cover the kernel's real use: the host code's sizes, dtypes, configs (SETUP.md or the source's host
  function). Every narrowing (a bound you need for the proof) has a docstring and is stated in README's
  Conditions/Narrowings with its reason.
- Typical fields: block sizes powers of two and `≤ 2^20` (Triton's limits), sizes that keep `int32`
  arithmetic from wrapping (`n + BLOCK ≤ 2^31` or `n < 2^31`), buffer lengths (`n ≤ x.len`), disjointness
  of what is written from everything else read or written.
- `launch` must be valid on covered instances (`Launch.Valid`: block a power of two ≤ 2^20, fewer than
  2^31 programs); `Launch.valid_grid1` proves it for a 1-D grid.
- Say in README's Conditions that the launches stay within part (A) of `Compiler.Faithful`'s scope
  (`docs/rules/compiler-faithful.md`, "Scope"; copy the example README's paragraph and check each point
  against your kernel's launch).

## Specification (`Post`)

- Say what each output element is: exactly (`FloatV.NaNEq (FloatV.add a b) c`: equal up to NaN payload),
  or within a tolerance that is explicit and meaningful at the **largest** covered sizes (evaluate it
  there; reviewers will).
- Say that every byte outside the outputs is unchanged.
- Approximate operations (`tl.exp`, `/`, `tl.dot`, fp8 conversions, float atomics) give tolerances
  through the library's accuracy facts; never invent constants.

## Assumptions

- `faithful` exactly as above, always.
- A `[rule]` field for each device fact an approximate operation needs, by its name as
  `docs/rules/num.md` "Facts the library states" lists it, applied to `P`: e.g. `exp : Rules.Num.ExpF32 P`,
  `dot : Rules.Num.DotF16 P`, `div : Rules.Num.DivFull P`, plus the matching `…Total` rule when the proof
  needs a result to exist (`tva-kit doc "Facts the library states"`). Never a shape with constants
  (`MathAcc …`, `DotAcc …`).
- A `[kernel]` field only for a closed fact about your own constants (no `K`, no `P`); prefer a lemma.
- Every field must be used by `main` (gate `usage`); each has a docstring starting `[rule]` or `[kernel]`.
- `consistent` is proved with `Platform.any` (no device facts) or `Platform.sm90ref` (which satisfies
  every fact of `TVA.Rules.Num.Sm90`, `DivFull`, `DivRn`, the `…Total` rules): see skill tva-proof.
