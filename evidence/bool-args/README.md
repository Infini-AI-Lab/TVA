# `torch.bool` tensor arguments (library issue L158)

Library evidence (not a kernel's), for library issue L158. Triton 3.8.0 (TVA's `.venv`, torch
2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-11-08. Script: [run.py](run.py); output:
[out.txt](out.txt) (seconds).

**Question.** docs/SEMANTICS.md did not say what a `torch.bool` tensor argument is, or what `tl.load`
and `tl.store` do with its elements; the library stated it only in the Lean docstrings of
`DType.bytes` and `DType.ofBits`. In Triton 3.8.0's source, a `torch.bool` tensor's signature type is
`*u1` (`triton/_utils.py`, `type_canonicalisation_dict`: `"bool": "u1"`), the language type
`pointer<int1>`, and `semantic.load`/`semantic.store` treat a `pointer<int1>` as a `pointer<int8>`
(`primitives/triton-3.8.0-semantic.py:1036-1040`, `1236-1240`): a load reads an `int8` and casts it to
`int1`, which is `x != 0` (`semantic.py:874-877, 1061-1062`); a store casts the value to `int8`
(`semantic.py:1243`), not to `int1`. Does the device do that?

**Method** (`run.py`). A `torch.bool` tensor whose bytes are `0, 1, 2, 127, 128, 255, 0, 1` (written
through a `uint8` view of the same storage) is passed to a kernel that prints the argument's type at
compile time, loads it (unmasked; and with `mask = j < 3`, `other = 5`) and stores the loaded values
converted to `int32`; a second kernel stores comparisons (`int1`) into a `torch.bool` output; a third
stores `int32` values `0, 1, 2, 255, 256, 257, -1, 512` into a `torch.bool` output. Output bytes are
read through a `uint8` view.

**Result** (`out.txt`).
- The argument is `pointer<int1>` (`!tt.ptr<i1>` in the TTIR), its element type `int1`, and its
  loads are byte loads (`ld.global.b8`).
- Loaded: `0, 1, 1, 1, 1, 1, 0, 1`: every non-zero byte is `true`. Masked-off lanes with `other = 5`
  hold `true`.
- Comparisons stored: bytes `0` and `1` only.
- `int32` values stored: bytes `0, 1, 2, 255, 0, 1, 255, 0`: the low byte of each value, so `256`
  and `512` store `0` (`false`) and `2` stores `2` (which a load reads as `true`).

**Conclusions.** A `torch.bool` argument is `Ptr .i1`, one byte per element. A load gives `byte ≠ 0`
(`DType.ofBits .i1`), and a store of an `int1` value gives `0` or `1` (`DType.toBits .i1`), as the
library's `Mem.read` and `Mem.write` of `.i1` state. A store of another dtype to it is the
conversion to `int8` and a byte store, not a test `≠ 0`; docs/SEMANTICS.md ("Parameters and
arguments") states both.
