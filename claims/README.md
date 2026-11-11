# Claims

Each folder `claims/<kernel>/` holds one fixed claim about one Triton kernel. A claim says that
every launch of the kernel it covers ends normally (no fault, no race, no hang) and leaves memory
satisfying a specification. It holds no proof: a prover proves `theorem main : claim.Holds`
separately (see [../docs/PROVING.md](../docs/PROVING.md)).

## What a folder holds

| File | What |
|---|---|
| `Claim.lean` | the claim: the model, the instance type `Inst`, the conditions `Inst.ok`, the launch, `Pre`, `Post`, packed as `def claim : TVA.Claim`; then `claim_satisfiable` and the `setup_*_ok` theorems |
| `README.md` | the claim in prose: the kernel, what is claimed, the conditions, the launches covered, the device facts the tolerances rest on, and what is not covered |
| `SETUP.md` | the kernel's source and configuration, and the launches the source file makes, which the claim must cover |
| `source/` | the kernel's source files, unchanged, so that a reader can compare the model with them line by line |
| `check_gpu.py` | a GPU test: runs the kernel on an H200 and checks `Post` on the launches of `SETUP.md` and on edges of `Inst.ok` |
| `claim.json` | metadata: `kernel`, `title`, `namespace` (the Lean namespace of `claim`), `lean` (`Claim.lean`), `source` and its `source_sha256`, and the line ranges of the kernel and of the host code in the source |

Every `Claim.lean` imports `TVA`, the trusted part of the library (`attn_bwd_pre_f16` also imports
`TVA.Proof.Float.Tree`, a proved module, for a lemma about its tolerance). Its module docstring
says which source lines the model transcribes, what is claimed and under which conditions; each
model line has a comment saying how it reads its source line.

## Claims are fixed

A claim is written once and not changed afterwards: a prover proves the claim as it stands. The
checker pins `Claim.lean` by its sha256 and refuses a proof whose claim file differs. A claim that
turns out wrong (too strong for the kernel, or covering a launch the kernel does not handle) is
not weakened to fit a proof: it stays unproved.

## What `Claim.lean` proves itself

- **`claim_satisfiable : claim.Satisfiable`**: some instance and some initial memory satisfy the
  conditions (`Inst.ok` and `Pre`). Without it, a claim covering nothing (`ok := fun _ => False`)
  would hold of every kernel. Its device-side counterpart is `Device.ref_ok` in the library: some
  platform satisfies all the device facts, so assuming them is not a contradiction.
- **`setup_<name>_ok : setup_<name>.ok`**, one per launch listed in `SETUP.md`: that launch is a
  covered instance. A launch the claim does not cover is named in its `README.md` with the reason
  (for example `softmax_f32` leaves out the tutorial's test shape, `1823 × 781`, because `781` is
  not a multiple of `128`).

## Checking a claim

From `lean/`, after `lake build`:

```bash
lake env lean ../claims/<kernel>/Claim.lean
```

It elaborates the claim and checks `claim_satisfiable` and every `setup_*_ok`. To read a claim,
start with its `README.md`, then compare the model in `Claim.lean` with `source/`, using
[../docs/SEMANTICS.md](../docs/SEMANTICS.md) for what each `tl.*` operation means, and check the
source against [../docs/EXCLUSIONS.md](../docs/EXCLUSIONS.md): a claim's model uses no excluded
feature, and its launches are in no excluded situation.

## The GPU test

`check_gpu.py` runs the kernel's own host code (and direct launches of the kernel) with Triton 3.8.0
on one H200, and checks `Post` as it is stated: values within the claim's tolerance (or bit for bit
where `Post` is exact), and the frame `Post` states: the inputs unchanged and, where the script
places an output inside a larger sentinel-filled buffer, nothing written outside it. It is evidence that the claim is not false on the launches
tried; it proves nothing.

Run it on a machine with an H200, Python 3.13, `triton==3.8.0` and PyTorch:

```bash
cd claims/<kernel>
CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
```

Each script prints one line per case and fails (a non-zero exit) when a case fails. Each case
keeps its device memory under 4 GB, so a test can share a GPU with other work; sizes in `SETUP.md`
that would need more are cut down in the test, and the script says so.
