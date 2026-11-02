---
name: tva-filing
description: Record a TVA review with tva-kit filing (findings with valid kinds and targets, acceptances, releases, re-raises, Checked/Not checked) so that filing.json and report.md are regular. Use when you write up a review and whenever tva-kit filing lint or the service reports a problem.
---

# Writing the filing

Never edit `filing.json` or `report.md` by hand: `tva-kit filing …` keeps the two consistent (the report's
verdict, its Findings bullets in the filing's order, its Accepted section), which is what `tvr lint` and
`tvr file` check.

```
tva-kit filing init                                   # once per round (or per refiling)
tva-kit targets                                       # the kinds, and the targets that exist
tva-kit filing add --kind SPEC_TOO_WEAK --target item:spec \
    --summary "Post bounds out[j] only for j < n/2" \
    --detail "Post's first conjunct quantifies j < d.n / 2 (VecAdd.lean:112), but the kernel writes all n elements (source line 54). Quantify over j < d.n." \
    --evidence "n = 10: elements 5..9 are unconstrained by Post."
tva-kit filing accept item:map item:conditions item:assumption:faithful   # or: accept all
tva-kit filing release F2 --note "Settled: Post now quantifies every j < n (Lean line 112)."
tva-kit filing reraise F3 --note "Still open: the note claims no product feeds line 52, but line 50's value is a product."
tva-kit filing checked "Compared all 12 map entries with source lines 23-54 against SEMANTICS.md."
tva-kit filing checked "Evaluated the tolerance at n = 2^31 - 1: 2.4e-7 relative."
tva-kit filing not-checked "The tolerance's derivation in rules/num.md (trusted library)."   # if any
tva-kit filing show
tva-kit filing lint                              # until: regular — then stop with your final message
```

## Kinds and targets

| Kind | Blocking | Targets |
|---|---|---|
| `MODEL_MISMATCH` | yes | `item:model`, `map:<line>`, `src:<line>`, `decl:<name>` |
| `MAP_INCOMPLETE` | yes | `item:map`, `map:<line>`, `src:<line>` |
| `UNSUPPORTED` | yes | `src:<line>`, `map:<line>` |
| `CONDITION_TOO_NARROW`, `CONDITION_WRONG` | yes | `item:conditions`, `decl:<name>` |
| `SPEC_TOO_WEAK`, `SPEC_WRONG` | yes | `item:spec`, `decl:<name>` |
| `RULE_MISCITED` | yes | `item:assumption:<field>` of a `[rule]` |
| `ASSUMPTION_FALSE`, `ASSUMPTION_TOO_STRONG`, `ASSUMPTION_UNCLEAR`, `EVIDENCE_REQUEST` | yes | `item:assumption:<field>` of a `[kernel]` |
| `LIBRARY_REPORT` | no | `decl:`, `lib:<path>`, `src:` |
| `DOC`, `QUESTION` | no | any `item:`, `decl:`, `src:`, `map:`, `lib:`, `doc:` |

`src:N`/`map:N` must be a line in the kernel or host lines; `decl:NAME` a declaration in an item's cone
(`tva-kit targets` lists them); `--affects item:X …` says which items a `decl:` finding holds.

## Rules of the record

- You cannot accept an item while a blocking finding of yours holds it (tva-kit refuses and says so).
- The verdict follows: `accept` = all items accepted, no new blocking finding, no re-raise; `reject` = no
  item accepted; otherwise `accept-partly`.
- At least one Checked bullet. Not checked: bullets, or nothing (it then says `Nothing.`).
- The summary is one line ≤ 200 characters; a blocking finding has a `--detail`.
- Cite only TVA and Mathlib (paths under `~/fpv/tva`, never other projects).
