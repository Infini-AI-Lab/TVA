# Library review, round 16, by lr-16a

## Verdicts

- **L7** ACCEPT — Every git the tool runs goes through `_git` (all 12 calls in `tools/tvr.py`, each `-C <repo>`
  first where it has one), which runs it in the empty folder of `_git_home` with `GIT_CEILING_DIRECTORIES` at the
  folder above the repository it is given, or at `_git_home` (above `cwd`) for the rest; protocol §2 says exactly
  that. evidence/tool-round16 Part G re-ran identical (network). End to end, in `work/gitcfg.py`, the release
  check's packages step (`make_packages`) run from inside a repository whose `.git/config` sends github.com to
  `example.invalid` and sets `http.proxy`/`https.proxy` to a closed port, with the caller's folder, `TMPDIR` (so
  git's home) and the folders git writes all inside it, made all nine packages from upstream, Mathlib's pin
  accepted as the tag's commit and every file its blob, while a plain git in the same folder was redirected and
  refused (`work/gitcfg.out`). TVA's own configuration is read only by `ls-files`, which decides only which of
  TVA's files are copied, as §2 says.
- **L62** ACCEPT — `faithful_problem` accepts `Faithful` only with the arguments `P`, `<ns>.model x`,
  `<ns>.launch x`, `K x` for one bound `x`, and `rule_fact_problem` a head only as `stated_facts` lists it; Parts F
  and L re-ran identical (F1 refused by `kinds` alone; `model e`, `launch e`, `K e`, the chosen instance,
  `model ⟨0⟩`, a written launch and an eta-expanded `K d` refused; the helpers, the shapes, `DotF8` and
  `DivRn P .binary32` refused; a missing page, section or table fails closed and the audit names num.md). The 32
  listed facts are exactly the set the round-13 rule admitted (the 24 `Platform → Prop` facts of
  `TVA.Rules.Num.Sm90`, which I matched against the file, `DivFull`, the six `…Total` rules, `DivRn P .binary64`),
  so no kernel item changes; the six names listed alone are totality statements (`∀ x, ∃ r, …`), true for every
  argument. The forms the check still admits beside the standard one (no `d.ok`, extra hypotheses,
  `∀ d, d.ok → ∀ e, Faithful … e`) give `main` nothing more: `K` occurs only in `Faithful` and `Spec` reads only
  covered instances, so a proof from them gives one from the standard form (take `K` empty outside the covered
  instances). The guides, the prompts and LIBRARY_REVIEW.md check 5 say what the gate does. Two defects next to it
  are under New issues: the reader finds `K` and `P` by position (here it fails closed: a shifted `K` refuses
  every `Faithful`), and every table of the section is read, not only "its table".
- **L79** REOPEN — Every code line no function spans must now lie in the kernel or the host lines (Part M re-ran
  identical; vec-add's manifest fails on its imports, `DEVICE`, prints and benchmark run, as the notes say). But a
  function named in `source.unused` is exempt from classing, while its `def` statement runs when Python executes
  the module: its decorators, defaults and annotations are evaluated and can bind a module-level name the launch
  reads. In `work/probes_src.py` the kernel lines bind `SCALE = tl.constexpr(2)`, or define the `@triton.jit`
  helper `double` (`x * 2`) that the kernel calls, and the unused `_unused_helper(x=(SCALE := tl.constexpr(3)))`,
  or `_unused_helper(f=(double := triple))` with the `@triton.jit` function `triple` (`x * 3`) also unused,
  rebinds the name at import. `map_problems` passes both with every code line of the kernel lines mapped
  (`work/probes.out` Part U), and on the H200 NVL with Triton 3.8.0 both kernels store `3·x`
  (`work/unused_gpu.out`). The `double` case uses only module-level names the supported subset admits (the
  module's `@triton.jit` functions). So protocol §7's "covers … every module-level value it reads" does not hold,
  and REVIEWER_GUIDE §1, PROVER_GUIDE §1 and claim.md §2 treat an unused function as one nothing runs. Fix (cheap
  and complete, as L79 asked): class the header of every `source.unused` function (its decorators and its `def`
  lines up to the body) as module code that must lie in the kernel or the host lines; say in §7 and the guides
  that a `def` statement evaluates its decorators, defaults and annotations at import; and in SEMANTICS' supported
  subset (L88) admit a module-level name only as its own `def` or `import` binds it.
- **L81** REOPEN — lr-13a's route is closed where the folder's events name it as `dir` (Part R re-ran
  identical), and `tvr start`/`tvr lint` refuse an unassigned reviewer. But `kernel_folder_problems` counts a
  folder as recorded by any event that names one of its files, and `ledger_problems` freezes only folders an
  event names as `dir`. (1) vec-add's own ledger has `file`/`response` events written before `dir` existed (2, 5,
  7): their folders are not frozen (`review/round-1/r-vecadd/work/` holds four files no event records, and nothing
  names them), and until event 11 its `round-2/r-vecadd/` was such a folder. In a scratch copy at events 1-9 with
  a passing gate made by hand at the digests r-vecadd accepted, r-vecadd files `filing-2.json` with a blocking
  `SPEC_TOO_WEAK`; with that filing's `file` and `raise` lines removed, `kernel_record_problems`,
  `unrecorded_current` and the records step name nothing and the certificate is granted beside the filing
  (`work/probes.out` Part R, L1). (2) On a folder the current tool froze, a coordinator's `tvr note … --files` of a
  filing and its report accounts for both, so removing the filing's two lines again grants the certificate (N).
  Fix (cheap): freeze the folder of every `file` and `response` event's path whether or not the line names `dir`
  (vec-add's four round-1 work files then show, and need recording once, or its ledger restarts with the
  re-proof), and let only a role's own `file`/`response` events account for a role folder and its files, never a
  `note`.
- **L82** ACCEPT — LIBRARY_REVIEW.md ("Rounds" 4), protocol §2 (`records`), §3, §10 and `ledger_digests` say the
  lines are trusted as the coordinator keeps them and that an edit shows only against a copy kept elsewhere,
  compared by hand; no live text says they rest on the git history. §3, the comment at `ESCALATE_AFTER` and
  `filing_events` state the re-derivation with today's defaults, and Part C shows it (vec-add's ledger fails at
  its `DOC` finding once `DOC` blocks by default). §8 and §9 say what a gate in the tree rests on, which my Part R
  confirms (a hand-made `gate.json` grants the certificate). `_PLACEHOLDER` reads a capitals-only component as a
  name (Part C), and the only placeholder components among the tree's citations are `round-N` (checked). On the
  coordinator's question: on the merged tree Part C's "the tree's citations" line reads "every cited file
  exists", the one line of the re-run that differs from `out.txt`; the README's claim holds, but `out.txt`
  records an earlier text of the protocol and should be regenerated.

## New issues

- **NEW** soundness — The claim reader finds `K` and `P` by position: `mentions 0`/`mentions 1` and the
  `platform` and `K` flags read `xs[0]` and `xs[1]` of the field projection's binders
  (`tools/claim_reader.lean.in:395-431`), and nothing requires `Assumptions` to have only those two parameters.
  With `structure Assumptions [Inhabited Unit] [Inhabited Bool] (K : Claim.Behaviour Inst) (P : Platform)`, both
  instances are synthesised, the reader's `shape` examples elaborate, and the flags read the instances: a field
  `[kernel] spec : Claim.Meets Inst.ok Pre Post K` (the specification itself, `main := h.spec`) and a field
  `[kernel] div : ∀ x y r : FloatV .binary32, P.div .binary32 x y r → r = x` (a device fact false on the H200,
  satisfied in `consistent` by a platform built for it) pass hygiene, replay, shape, axioms, usage, namespace and
  `kinds`, while the control with the plain structure fails `kinds` on both (`work/probes.out` Part K). Protocol
  §5 ("a `[kernel]` assumption cannot be about the device or the compiler") and REVIEWER_GUIDE §4 ("It mentions
  neither `K` nor `P` … it cannot be false as stated") tell the reviewer the gate guarantees what it does not.
  Fix (cheap and complete): have the reader fail unless `Assumptions` has exactly the two parameters
  `K : Claim.Behaviour Inst` and `P : Platform` (`numParams = 2`), and say so in §8's `shape`.
- **NEW** gap — `stated_facts` (`tools/tvr.py:1893-1926`) takes every table row of the section "Facts the
  library states", up to its next heading, not the first column of "its table" as protocol §5 and the library
  author's prompt say. A second table in that section, even one headed as shapes a field never cites, makes its
  first-column names citable alone, with any arguments: with `MathAcc` and `DotAcc` in such a table, 34 facts are
  listed and lr-9a's field `MathAcc P .exp .binary32 Set.univ (2^-24) (2^-150)` passes (`work/probes.out` Part
  L). The page has one table today. Fix: read only the first table of the section (stop at its first non-table
  line), and say so in §5 and the author's prompt.
- **NEW** gap — The release check's `pages` step compares the pages the sandbox regenerates with the live tree's
  (`page_mismatch(ROOT, copy)`, `tools/tvr.py:4040`), not with the copy that was checked, since the sandbox
  overwrites the copy's pages. A page fixed in the tree during the run passes although the checked copy shipped
  it stale, and a page regenerated in the tree during the run fails: the coordinator's release check of the
  merged tree named vec-add's `claim.md`, which its prover's `tvr check` rewrote in the tree while the check ran.
  Protocol §2 says each page "must be the one regenerated from the shipped files". Fix: read the copy's generated
  pages (and gate times) before the sandbox runs, and compare with those.

## Re-ran

- evidence/tool-round16/run.py in full (Parts F, L, M, R, C and G, with the network), on the merged tree:
  identical to its `out.txt` except line 139 (L82 above).
- `work/probes.py`: Part K (two kernels compiled with `lean -o` into a scratch folder, the tree's claim reader with
  all four `example`s, the gate's hygiene patterns, `leanchecker` and the Lean-side checks), Part U (the map check
  on `work/probes_src.py`), Part R (a scratch copy of the tree with vec-add's ledger), Part L (a scratch num.md)
  and Part C (the tree's citations, fact page and release lint list); output `work/probes.out`.
- `work/unused_gpu.py` on GPU 0 (Triton 3.8.0, NVIDIA H200 NVL, 2026-10-28): both Part U kernels store `3·x`.
- `work/gitcfg.py` (network): the packages step inside a hostile repository, as under L7.
- The coordinator's release check of the merged tree (read, not re-run): 8 of 10 steps pass; vec_add_f32's gate
  (its proof does not build; its shipped gate is stale) and the pages fail, as expected. I did not run my own;
  the L7 route was tested at its packages step instead.
- `tvr audit` on the tree with my work files: `independent`.

## Not checked

- The device truth of the 24 `Sm90` facts, `DivFull` and `DivRn P .binary64` (the device area, round 12): I
  checked only which names the list holds and how the gate reads them.
- A full `tvr check` of Part K's kernel, which needs `lake build`: the reader, the hygiene patterns, `leanchecker`
  and the Lean-side checks were run on the compiled module; `map` and `doc` do not depend on it.
- vec_add_f32's gate and pages (being re-proved), and the full release check from inside a hostile repository.
