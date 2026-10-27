# Prover's notes on vec_add_f32, round 1

## Changes

- README.md, Narrowings: the bullet on `n + BLOCK_SIZE ≤ 2^31` now gives each excluded range its own reason: `n > 2^31`, where the last program's `int32` product `pid * BLOCK_SIZE` wraps (out-of-bounds accesses before the buffers, and from `n > 2^32` on, other programs' elements); `n = 2^31`, where nothing wraps but `n_elements` is passed as `int64`, which the model's `I32` does not cover; and `2^31 − BLOCK_SIZE < n < 2^31`, excluded by `Block.Fits` (library issue L8). This answers F1.
- README.md, Conditions: the `n_fits` row names the same three reasons; the overlap row says a partial overlap "can make programs race" (the reviewer's remark (2)).
- README.md, Narrowings, Aliasing: the conditions are stated on whole buffers (`Buf.bytes`), so they also exclude an overlap lying only in elements from `n` on, which would be harmless.
- No Lean, map or manifest change, so the recorded gate stays current.

## Responses

- **F1** CLARIFIED — the README now says the wrap starts at `n > 2^31` (out-of-bounds accesses) and gives `n = 2^31` its own reason (an `int64` `n_elements`, another model).

## Library requests

None.

## Gate

PASS, digest `b7289baba9114347`
