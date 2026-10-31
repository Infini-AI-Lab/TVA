# Prover's notes on attn_bwd_pre_f16, round 1

## Changes

- README.md, Narrowings, bullet 'The grid': the sentence that joined two launches' maxima now attributes each to its own launch (the largest Z·H, 192, to the test with N_CTX = 4096; the largest grid, 128 · 128 programs, to the benchmark with Z·H = 128). Wording only: no Lean file, map, manifest or other text changed, and no item digest changed.

## Responses

- **F1** CLARIFIED — the 'The grid' sentence now gives the test's largest Z·H and the benchmark's largest grid separately.
- **F2** CLARIFIED — the same fix as F1.

## Library requests

None.

## Gate

PASS, digest `74a1e051af7b1c1b`
