# Rule `Compiler.Faithful`

Lean: `TVA.Rules.Compiler.Faithful` (`lean/TVA/Rules/Compiler.lean`).

```lean
def Faithful (P : Platform) (k : Kernel) (L : Launch) (K : Mem → Outcome → Prop) : Prop :=
  L.debug = false → L.Valid → ∀ init, WellBehaved (L.env P) k L init → ∀ o, K init o →
    match o with
    | .ok fin => Outcomes (L.env P) k L init (.ok fin)
    | .fault => False
    | .hang => Outcomes (L.env P) k L init .hang
```

## What it says

Take the compiled kernel `K`, launched as `L` from global memory `init`, without `debug` and
with a valid launch (`Launch.Valid`: CUDA's grid limits, `num_ctas = 1`, and block shapes
Triton accepts). If the model `k`'s launch is well behaved from `init` (no undefined behaviour
and no race between programs on any interleaving), then:

- a normal end of the device run leaves a memory that some interleaving of the model also
  leaves;
- the device run does not fault;
- the device run hangs only if some interleaving of the model runs forever.

## What it trusts

- The Triton compiler (3.8.0) and its documentation: the device runs the Triton program as
  `TVA.Sem` reads the documentation. User, 2026-10-26: "we trust triton compiler and
  documentations".
- The device's meaning of approximate operations, which is the platform `P` the claim
  quantifies over. Accuracy facts about `P` are separate rules (`Rules.Num.*`), not part of this
  one.

## What it does not cover

- **The model's correspondence to the source.** The model is the prover's transcription, and
  the reviewer checks it line by line (`MODEL_MISMATCH`). The rule is about the model.
- **Launches that are not well behaved, or not valid.** For those the rule claims nothing. A
  proof must show the launch is well behaved, for example with `TVA.independent`, and valid
  (`meets_of_faithful` asks for `(launch d).Valid`).
- **Compile and launch limits the library does not state.** A launch that fails is a
  `fault`, so the reviewer accepts the rule only for instances that compile and launch. The
  limits include:
  - shared memory and registers;
  - dtypes the architecture lacks (`float8e4nv` before sm_89);
  - `tl.dot` minimum sizes;
  - a descriptor block whose last dimension is under 16 bytes;
  - a gather or scatter of fewer than 8 rows.

  `Launch.Valid` checks only the block shapes listed in `blocks`.
- **The host.** Arguments, grid and allocation are the instance `d`: `launch d`, and the buffers
  in `Pre`.
- **The [exclusions](../exclusions.md).**

## How a bundle cites it

```lean
structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful` -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
```

The reviewer checks four things:

- `model d` is the kernel's source, with every implicit conversion, broadcast and mask;
- `launch d` is the host's launch: the grid formula, `enable_fp_fusion`, `num_ctas`, and the
  source's block shapes in `blocks`;
- `d.ok` covers the instances the claim is about, and each of them compiles and launches;
- the bundle proves `Claim.Consistent Assumptions` (`exists_total_faithful` gives the behaviour
  for this rule).

## Consistency

The rule holds of some total behaviour for every model and launch:
`TVA.exists_total_faithful` (`lean/TVA/Lemmas/Consistency.lean`). The behaviour follows the
model where the launch is well behaved and faults elsewhere. It is total because a
well-behaved launch always ends normally or runs forever (`WellBehaved.outcome`): no state of
the semantics is stuck (`Step.progress`).
