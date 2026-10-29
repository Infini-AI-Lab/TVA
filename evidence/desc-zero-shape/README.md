# Tensor descriptors with a zero shape entry

**Question.** `tl.make_tensor_descriptor` takes its shape at run time, and a shape entry may be `0`.
Before round 11 the library's `TensorDesc.Valid` set no lower bound, so in the model such a
descriptor makes every lane out of bounds: a load gives the padding, a store or a reduction writes
nothing. What does the device do with an access through it, and with a descriptor that has one but
is never accessed? Library issue L5 (round 11; reopened by lr-8a, whose
`docs/library-review/round-8/lr-8a/work/desc_edges.py` found the fault for load and store).

**Method.** `run.py` launches one program per case, each in its own process (a fault ends the
process's CUDA context), with an allocator for the descriptor scratch. The input is a 32 x 32
`float32` matrix, the block 32 x 32 (or 1 x 32 for gather and scatter, 32 for rank 1):

- Z1: a load through an in-kernel descriptor of shape [32, 32] (control); Z4: shape [1, 32].
- Z2, Z3: a load, shape [0, 32] and [32, 0]; Z11: shape [0, 32] at offsets [32, 0].
- Z5, Z6: a store, shape [0, 32] and [32, 0]; Z7: an `atomic_add` (TMA reduction), shape [0, 32].
- Z8, Z9: a gather and a scatter of 8 rows, shape [0, 32]; Z14, Z15: the same at shape [32, 32];
  Z16: a gather through a host-built descriptor of shape [32, 32].
- Z10: a load through a rank-1 descriptor of shape [0].
- Z12: a descriptor of shape [0, 32] built, but accessed only in a runtime branch no program takes.
- Z13: a host-built `TensorDescriptor` of shape [0, 32].

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/desc-zero-shape/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`output.txt`).

| case | access | shape | result |
|---|---|---|---|
| Z1 | load | [32, 32] | as the model |
| Z4 | load | [1, 32] | as the model (row 0, then the padding) |
| Z2, Z3, Z11 | load | [0, 32], [32, 0] | fault: "an illegal instruction was encountered" |
| Z10 | load, rank 1 | [0] | fault: "an illegal instruction was encountered" |
| Z5, Z6 | store | [0, 32], [32, 0] | fault: "an illegal instruction was encountered" |
| Z7 | `atomic_add` | [0, 32] | fault: "an illegal instruction was encountered" |
| Z8, Z9, Z14, Z15, Z16 | gather, scatter | any | rejected on the host: ptxas refuses `.tile::gather4` (needs sm_100) and `.tile::scatter4` (not on sm_90a) |
| Z12 | none (built only) | [0, 32] | as the model |
| Z13 | host-built | [0, 32] | rejected on the host: `AssertionError: shape must be positive` |

**Conclusions**, used by `docs/SEMANTICS.md` ("Tensor descriptors") and `TensorDesc.Valid`:

1. An access (load, store, reduction) through an in-kernel descriptor with a zero shape entry faults
   on the H200, whatever the offsets and the rank. So such an access is undefined behaviour in the
   model: `TensorDesc.Valid` asks `0 < shape[k]` for every dimension. With that condition the
   model's accesses are those Z1 and Z4 show the device performs.
2. Building such a descriptor without accessing it is harmless (Z12): `tl.make_tensor_descriptor`
   stays pure and accepts a zero entry; only an access requires `Valid`.
3. A host-built descriptor with a zero entry never reaches the device (Z13): the launch has no
   outcome.
4. On the H200, `desc.gather` and `desc.scatter` do not compile, through in-kernel and host-built
   descriptors alike (Z8, Z9, Z14-Z16): a launch that uses one is rejected on the host and has no
   outcome.
