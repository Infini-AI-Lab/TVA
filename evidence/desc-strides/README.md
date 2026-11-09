# Host descriptors with a zero or short leading stride (library issue L154 (4))

**Question.** `TensorDesc.Valid` (`lean/TVA/Lang/Desc.lean`) asks of each leading stride only that it be a
multiple of 16 bytes, as Triton 3.8.0's host `TensorDescriptor` does, so it admits a stride of `0` (every row
the same memory) and one shorter than the row (rows that overlap). The library's meaning of an access is
`base + Σ_k (offs_k + i_k)·stride_k` for each in-bounds lane and the padding for the others
(`TensorDesc.elemAt`, `tl.desc_load`, `tl.desc_atomic`). Does the H200 compute that for such strides?

**Method** (`run.py`). Host `TensorDescriptor`s over a buffer just long enough for the formula's last element:
- Z1: `desc.load` of every block, at every block offset that tiles the shape and one block past each edge
  (padding), of a 2-D descriptor of shape `[24, 40]`, blocks `[8, 16]` and `[16, 32]`, for `float32` (strides
  `[0, 1]`, `[4, 1]`, `[8, 1]`, `[16, 1]`, and the control `[40, 1]`), `float16` (`[0, 1]`, `[8, 1]`, `[16, 1]`,
  `[32, 1]`, `[40, 1]`) and `int8` (`[0, 1]`, `[16, 1]`, `[32, 1]`, `[64, 1]`, `[48, 1]`; the `[8, 16]` block only
  where its last dimension has 16 bytes), against the formula with zero padding.
- Z2: one load of a 3-D `float32` descriptor of shape `[4, 6, 32]`, block `[4, 8, 32]`, strides `[0, 32, 1]` and
  `[32, 8, 1]`.
- Z3: `desc.atomic_add` of a block of ones through strides `[0, 1]` (8 lanes per element) and `[8, 1]` (up to 4):
  against one addition per lane to its element.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/desc-strides/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130). **GPU.** NVIDIA H200 NVL (sm_90). **Date.** 2026-11-08.

**Result** (`output.txt`). Every launch ran. Z1: 0 lanes of 6656 differ from the formula in each of the 15
cases. Z2: 0 of 1024 in both. Z3: 0 elements differ from one addition per lane in both.

**Conclusion.** On the H200 a host descriptor with a zero leading stride or one shorter than its row loads, and
reduces into, the elements the library's formula names, lane by lane, with the padding outside the shape, as
`TensorDesc.Valid` and `tl.desc_load`/`tl.desc_atomic` state. (A store through such a descriptor writes some
element from two lanes, a conflict within the program, which no well-behaved model has.)
