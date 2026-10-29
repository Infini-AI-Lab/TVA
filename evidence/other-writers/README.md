# Writes by others next to a launch's bytes, while it runs

**Question.** `Rules.Compiler.Faithful` asked that a claimed launch "run alone": nothing else writes
global memory while it runs. Library issue L49 (round 8) proposes the weaker condition that nothing
else writes the bytes the launch is given (`init`'s buffers) from its start to its end. Is a write
to *other* bytes harmless even when it is as close as possible: the same 32-byte sector, the same
128-byte line, the same 4-byte word, written concurrently from another stream?

**Method.** `run.py` runs a claimed kernel (16 programs) that does `y[2i] += 1` 4000 times on its
own elements (a load and a store each time, so a disturbed byte would show in the final count), and
at the same time, on a second stream, another kernel (16 programs) that stores 120000 times to the
elements in between, `y[2i + 1]`. Both read `%globaltimer` at their start and end (program 0), so
the overlap in time is measured. 20 trials alone and 20 with the other writer, each case in its own
process on GPU 0.

- W1: `int32` elements: every 32-byte sector and 128-byte line holds claimed and other elements.
- W2: `uint8` elements: the claimed and the other bytes alternate inside each 4-byte word.
- W3: control: the other kernel writes the claimed launch's own elements `y[2i]`.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/other-writers/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`output.txt`).

| case | alone: correct | with the other writer: correct | overlapped | time covered |
|---|---|---|---|---|
| W1 `int32` neighbours | 20 of 20 | 20 of 20 (0 elements wrong) | 19 of 20 | 97 % |
| W2 `uint8` neighbours in one word | 20 of 20 | 20 of 20 (0 bytes wrong) | 19 of 20 | 79 % |
| W3 control: the same elements | 20 of 20 | 0 of 20 (up to 4096 of 4096 wrong) | 19 of 20 | 97 % |

"Time covered" is the median fraction of the claimed launch's run (program 0) during which the
other kernel was running; the trial without overlap is the first (the other kernel's compile).

**Conclusions.**

1. Concurrent writes by another launch to bytes the claimed launch is not given, even in the same
   sector, line or 4-byte word, did not change the claimed launch's bytes or its result (W1, W2):
   the GPU's stores are byte-granular. The condition a claim needs is about the bytes it is given.
2. A concurrent writer to the claimed launch's own bytes does change its result (W3), so the
   condition cannot be dropped: from the launch's start to its end, nothing else may write the
   bytes `init` gives it.
