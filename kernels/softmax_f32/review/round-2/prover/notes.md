# Prover's notes on softmax_f32, round 2

## Changes

- README, Conditions, 'Covered in particular': the `n_rows` limit is attributed to `in_off`, stated as the last row starting below element `2^31`, with the examples 131072 × 16384 and 169467 × 12672 (F7, F8).
- Err.lean: the docstring of `lane_facts` gives the bound the theorem proves (F7). No statement or proof changed.

## Responses

- **F7** CLARIFIED — README names in_off as the n_rows limit, and lane_facts' docstring states its relative bound.
- **F8** CLARIFIED — README: covered while the last row starts below element 2^31 (131072 rows of 16384 columns), not by element count.

## Library requests

None.

## Gate

PASS, digest `b822ec44212cc840`
