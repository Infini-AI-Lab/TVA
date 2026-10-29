# The gate on a copy of the tree: digests, the map check, hygiene, replay and names

Library evidence (not a kernel's), for library issues L12, L13 (round 3), L12, L20, L27 (round 4),
L20, L32 (round 5) and L36 (round 6). It tests TVA's gate (`tools/tvr.py check`), not the device:
Lean 4.35.0 (with `leanchecker` from the same toolchain), Python 3.13, 2026-10-28, every case rerun
with the round-7 tool (round 7: the results are those of round 6, word for word but for the wording
of the `hygiene` check, which now also covers the library's code, and of a `replay` failure; the
round-7 gate's own cases are in [evidence/gate-round7](../gate-round7/README.md)); no GPU and no
Triton run. Script: [run.py](run.py); output:
[out.txt](out.txt).

**Question (round 6, L36).**
1. Digests are complete by construction: an item's digest holds the whole text of every library
   file its cone reaches (the Lean modules, the fact pages SEMANTICS.md and exclusions.md when it
   reaches the library, the rule pages of the `TVA.Rules` modules it reaches). Does any change to
   such a file re-open exactly the items that reach it, whitespace and comments included, while a
   change to a library file no item reaches, to the kernel's proof, or to TRANSCRIPTION.md (now
   guidance, in no digest) re-opens nothing?
2. The map check checks coverage and model text only. Does a map with no `rules`, with unknown
   `rules`, or with a free-text drop reason pass, and does a map with an uncovered line, an entry
   with neither model text nor a drop, or an entry without `src` fail?
3. Do the checks kept from earlier rounds still stop what they stopped: meta code that adds an
   unchecked theorem (`hygiene`, and `replay` behind it, L13), and model text that means another
   operation (`namespace`, `hygiene`, L20)?

**Method.** The script copies the TVA tree to a fresh temporary folder (the Lean build is copied
and the Lean packages are linked; nothing else is shared) and works only there. The real tree is
read and not written, except for `out.txt`, and `tvr check` never runs on it. In the copy it
brings `kernels/vec_add_f32` up to date with the library (`Launch.valid_grid1` for
`launch_valid`, L17; disjoint output and inputs with the proof of `hRW`, L15) and leaves its map as
it is, with no `rules`. Then it runs `python3 -I tools/tvr.py check vec_add_f32` there, changes one
thing, runs the gate again, compares the item digests in `gate.json` with case A's, and restores the
file.

| Case | Change in the copy |
|---|---|
| A | none (the updated kernel); prints the library files in each item's digest |
| B | a pitfall of TRANSCRIPTION.md (guidance) |
| C | TRANSCRIPTION.md replaced by one line |
| D | a sentence of exclusions.md (a fact page) |
| J | a sentence of SEMANTICS.md (a fact page: what `tl.maximum` does with bf16) |
| E | a map entry names `rules: ["no-such-rule"]` |
| E2 | the drop reason `decl` of the first entry written as free text |
| E3 | the first entry has neither model text nor a drop reason |
| E4 | the entry of source line 52 removed (the line is uncovered) |
| E5 | an entry without `src` added |
| F | the kernel's Lean ends with a `run_cmd` that adds `extraFalse : False := True.intro` through `addDeclCore … (doCheck := false)` |
| G | F, with the copy's tool not running the hygiene list (`bad = []`), so the file is built |
| H | everything restored |
| K | the docstring of `TVA.tl.add`, which the model uses (`lean/TVA/Lang/Ops.lean`) |
| K2 | the same docstring, whitespace only |
| K3 | a comment at the end of `lean/TVA/Lang/Ops.lean` |
| K4 | a comment at the end of `lean/TVA/Lemmas/Softmax.lean`, a module no item reaches |
| K5 | a comment in the kernel's proof (`main`) |
| Y | a bullet of the module docstring of `TVA.Claim` |
| Z | the docstring of the field `TVA.Launch.blocks`, which no term projects |
| AA | a heading of `docs/rules/compiler-faithful.md`, the rule page of `TVA.Rules.Compiler` |
| AB | a heading of `docs/rules/num.md`, the rule page of `TVA.Rules.Num` (the kernel cites no `Num` fact) |
| L | the kernel declares `def tl.add … := TVA.tl.addc x y` in its namespace |
| M | the kernel declares `local notation "tl.add" => TVA.tl.addc` |
| N | the kernel declares `instance : HAdd F32 F32 F32 := ⟨FloatV.sub⟩` |
| O | the kernel declares `theorem lane_tile`, a name of `TVA.Block`, which its `section Proof` opens |
| Q | the kernel declares `def Launch.mk …` with `debug := !debug` |
| R | the kernel declares `instance instHDivNatK : HDiv Nat Nat Nat := ⟨fun a b => a / b + 1⟩` |
| S | the kernel declares `def tℓ.add … := TVA.tl.addc x y` (`ℓ` is U+2113) |
| T | `structure Ops`, `def ops : Ops := ⟨fun x y => TVA.tl.add x y⟩`, and `let tl : Ops := ops` before the model line `let output ← tl.add x y` |
| U | `structure Ops` and `def lookBinder … (tℓ : Ops := ⟨TVA.tl.addc⟩) … := tℓ.add x y` |
| V | the kernel's `open Block` becomes `open TVA` with `Block` on the next line, and it declares `theorem lane_tile` |
| W | the kernel declares `def min (a b : Nat) : Nat := a + b` |
| X | the kernel declares `set_option pp.notation false in theorem ppProbe : True := trivial` (a control) |
| AD | everything restored |

**Result** (`out.txt`).

| Case | Gate | Failing check | `map` | `model` | `spec` | `conditions` | `assumption:faithful` |
|---|---|---|---|---|---|---|---|
| A | ok | none (`hygiene`, `build`, `replay`, `namespace`, `map` pass) | | | | | |
| B | ok | none | unchanged | unchanged | unchanged | unchanged | unchanged |
| C | ok | none | unchanged | unchanged | unchanged | unchanged | unchanged |
| D | ok | none | changed | changed | changed | changed | changed |
| J | ok | none | changed | changed | changed | changed | changed |
| E | ok | none | changed (the map file) | unchanged | unchanged | unchanged | unchanged |
| E2 | ok | none | changed (the map file) | unchanged | unchanged | unchanged | unchanged |
| E3 | fails | `map`: "entry 0 (src [29]): an entry has model text or a drop reason, one of them" | changed | unchanged | unchanged | unchanged | unchanged |
| E4 | fails | `map`: "uncovered source lines [52]" | changed | unchanged | unchanged | unchanged | unchanged |
| E5 | fails | `map`: "entry 10: `src` is a non-empty list of line numbers and ranges" | changed | unchanged | unchanged | unchanged | unchanged |
| F | fails | `hygiene` (`addDeclCore`, `doCheck`, `setEnv`, `run_cmd`); `build`, `replay`, `shape` not run | not computed | not computed | not computed | not computed | not computed |
| G | fails | `replay`: "leanchecker found a problem in VecAdd … while replaying declaration 'Kernels.VecAdd.extraFalse'" | unchanged | unchanged | unchanged | unchanged | unchanged |
| H | ok | none | unchanged | unchanged | unchanged | unchanged | unchanged |
| K | ok | none | changed | changed | changed | unchanged | changed |
| K2 | ok | none | changed | changed | changed | unchanged | changed |
| K3 | ok | none | changed | changed | changed | unchanged | changed |
| K4 | ok | none | unchanged | unchanged | unchanged | unchanged | unchanged |
| K5 | ok | none | unchanged | unchanged | unchanged | unchanged | unchanged |
| Y | ok | none | unchanged | unchanged | changed | unchanged | changed |
| Z | ok | none | changed | changed | unchanged | unchanged | changed |
| AA | ok | none | unchanged | unchanged | unchanged | unchanged | changed |
| AB | ok | none | unchanged | unchanged | unchanged | unchanged | unchanged |
| L | fails | `namespace`: "Kernels.VecAdd.tl.add (a name with a component `tl` …)" | unchanged | unchanged | unchanged | unchanged | unchanged |
| M | fails | `hygiene` (`notation`, `local notation`); `build`, `replay`, `shape` not run | not computed | not computed | not computed | not computed | not computed |
| N | fails | `namespace`: "Kernels.VecAdd.instHAddF32_vecAdd (an instance; a kernel declares none)" | unchanged | unchanged | unchanged | unchanged | unchanged |
| O | fails | `namespace`: "Kernels.VecAdd.lane_tile (written `lane_tile`, it can be read as `TVA.Block.lane_tile`)" | unchanged | unchanged | unchanged | unchanged | unchanged |
| Q | fails | `namespace`: "Kernels.VecAdd.Launch.mk (written `Launch.mk`, it can be read as `TVA.Launch.mk`)" | unchanged | unchanged | unchanged | unchanged | unchanged |
| R | fails | `namespace`: "Kernels.VecAdd.instHDivNatK (an instance; a kernel declares none)" | unchanged | unchanged | unchanged | unchanged | unchanged |
| S | fails | `namespace`: "Kernels.VecAdd.tℓ.add (a name with a character other than ASCII letters, …)" | unchanged | unchanged | unchanged | unchanged | unchanged |
| T | fails | `namespace`: "Kernels.VecAdd.add_kernel (binds `tl`, … not the library's `tl`)" | unchanged | changed (the model changed) | unchanged | unchanged | changed |
| U | fails | `namespace`: "Kernels.VecAdd.lookBinder (binds `tℓ`, … not the library's `tl`)" | unchanged | unchanged | unchanged | unchanged | unchanged |
| V | fails | `namespace`: "Kernels.VecAdd.lane_tile (written `lane_tile`, it can be read as `TVA.Block.lane_tile`)" | unchanged | unchanged | unchanged | unchanged | unchanged |
| W | fails | `namespace`: "Kernels.VecAdd.min (written `min`, it can be read as `Min.min`, which `min` names at the root)" | unchanged | unchanged | unchanged | unchanged | unchanged |
| X | ok | none (`hygiene` passes) | unchanged | unchanged | unchanged | unchanged | unchanged |
| AD | ok | none | unchanged | unchanged | unchanged | unchanged | unchanged |

In F and M the build does not run, so `gate.json` holds no digest of the claim's items; `out.txt`
prints "CHANGED" for them because they are absent. In case A, `out.txt` lists the library files in
each item's digest: the `model` and `map` items reach eleven files (SEMANTICS.md, exclusions.md
and nine Lean modules, `Lang/Ops.lean` and `Sem/Launch.lean` among them), `spec` reaches
`Claim.lean` and `Lang/Ops.lean` (through `FloatV.NaNEq`) but not `Sem/Launch.lean`, `conditions`
reaches four Lean modules, and `assumption:faithful` reaches sixteen: the model's eleven,
`Claim.lean`, `Math/Bytes.lean`, `Sem/Step.lean`, `Rules/Compiler.lean` and
`rules/compiler-faithful.md`.

**Conclusions.**

1. The kernel, once updated (two of round 3's three updates; it needs no `rules` now), passes the
   round-6 gate, `replay` and `namespace` included (A, H, AD).
2. L36, digests: a change to a library file re-opens exactly the items whose cones reach it,
   whatever the change: a docstring (K), whitespace (K2, which re-opened nothing before round 6) or
   a comment (K3) in `Lang/Ops.lean` re-opens `map`, `model`, `spec` and `assumption:faithful`, and
   not `conditions`, which does not reach the file; a module docstring (Y), an unprojected field's
   docstring (Z) and a rule page (AA) re-open the items that reach their files. A fact page
   re-opens every item (D, J). A library module no item reaches (K4), a rule page whose module no
   item reaches (AB), and a comment in the kernel's proof (K5) re-open nothing.
3. L36, guidance: TRANSCRIPTION.md is in no digest; changing a sentence of it, or replacing it
   whole, re-opens nothing (B, C; before round 6 a rule's text re-opened the maps that named it).
4. L36, the map check: a map without `rules` (A), with `rules` nothing defines (E) or with a
   free-text drop reason (E2) passes, and only the `map` digest moves, since the map file changed;
   an entry with neither model text nor a drop (E3), an uncovered line (E4) and an entry without
   `src` (E5) fail.
5. L13: the hygiene list stops F before anything is built; with the list switched off (G), the file
   builds and `replay` rejects the unchecked declaration.
6. L20: every route of rounds 3–5 still fails the gate, with the same reason as in round 5: a kernel
   name with a component `tl` (L), a notation (M), any instance (N, R), a name a library namespace the
   kernel opens also has (O, V), a definition the kernel writes whatever its last component (Q), a
   name outside plain ASCII (S), a binder named `tl` or a look-alike of it (T, U), a name that
   shadows a root alias (W). A printing option passes `hygiene` (X).
