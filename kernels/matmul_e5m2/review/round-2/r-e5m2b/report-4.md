# Review of matmul_e5m2, round 2, by r-e5m2b

## Verdict

accept

Library round 25 re-opened assumption:dot, dot_total, cvt and cvt_total, and spec, at gate 5e4b5331abc9b970. The kernel's files did not change. Nothing these items rely on changed in meaning, so I accept all items again.

## Findings

None.

## Accepted

All items.

## Checked

- `liftMag` moved to `Rules.Num` with the same formula, `if q = 0 then 0 else max |q| (minNormal8 d)`. `minNormal8 .f8e5` is still `2^-14`; the new cases are `.f16` and `.bf16` only. So `liftV`, `errAt` and `tol` in the spec mean what they meant before.
- `DotE5m2Chunk`, `DotF8Chunk`, `DotChunkAcc` and the chunk constants `rhoChunk8`, `sigmaChunk8` and `alphaChunk8` are unchanged. The fact the kernel cites for `dot` is therefore the same.
- `CvtF32E5m2`, `CvtTotal` and `DotTotal` are unchanged. The facts cited for `cvt`, `cvt_total` and `dot_total` are therefore the same.
- In num.md, the listing rows and reviewer-check rows for the fp8 chunk fact, the e5m2 conversion and the totality facts are unchanged. The diff touches the 16-bit text and rows, the new `DotLiftAcc` row, and the mma.sync text in the chunk section.
- In work file `Round25`, `lake env lean` prints these definitions and checks `minNormal8 .f8e5 = 1/2^14`, `liftMag .f8e5 (1/2^16) = 1/2^14` and `liftMag .f8e5 1 = 1`. `main`, `consistent` and `satisfiable` depend only on `propext`, `Classical.choice` and `Quot.sound`.

## Not checked

- I did not re-check whether the 16-bit facts (`DotF16`, `DotBF16`, `DotLiftAcc`) are true on the device. This kernel does not cite them.
