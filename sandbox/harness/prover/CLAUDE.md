# Prover's operating manual: TVA kernel `{{kernel}}`

Your task (the instruction you were given) is TVA's prover prompt plus the sandbox's rules: prove the
Triton kernel `{{kernel}}` in Lean 4 with the TVA library, then get the proof **certified** by {{reviewers}}
through `tva-review`. The task succeeds only when `tva-review status` says
`certificate: GRANTED`. This file tells you how to get there step by step; the instruction and TVA's
documents say what each part must be.

**Time.** You have {{time_limit}} in all, from your first start. Your own sense of the time passed is
unreliable (an earlier prover believed 5 hours had passed after 15 minutes, and nearly gave up its claim):
never estimate it; run `tva-kit clock`, which prints the time used, the time left and when to submit by.
Every message of the harness ends with the clock too. Record each step with `tva-kit clock "what is done,
what is next"`: it adds the line to `~/scratch/progress.md` with the time used, and says how long the
step took since the last note. A round of review takes 20 to 60 minutes, and findings need a second round:
submit by three quarters of the time at the latest. A proof that is not finished when time runs out
earns nothing, so narrow early rather than late (below, "When you are stuck"). Most lost time goes into
proving statements that turn out wrong: design and audit them (stages 6 and 7) before the long proofs.

## Where things are

- TVA: `~/fpv/tva` (also `/home/tva/fpv/tva`). Your kernel's folder: `~/fpv/tva/kernels/{{kernel}}/`
  (`source/`, {{setup}}`kernel.json`, `model-map.json`, `README.md`, `lean/`). The Lean module is
  `{{module}}` (file `lean/{{module_path}}.lean`), the namespace `{{namespace}}`.
- Your notes and trials: `~/scratch` (keep `~/scratch/progress.md` up to date: it is shown to you
  again after a compaction).
- **Worked examples** (certified kernels: read their `lean/`, `model-map.json`, `kernel.json`,
  `README.md`): {{examples}}, the closest first. Copy their structure; most of your files have the same
  shape. `tva-kit outline KERNEL` shows how a kernel's proof is designed, file by file; every certified
  kernel in `kernels/` can serve as an example (the skill **tva-design** lists their designs).
- Helpers: `tva-kit` (run `tva-kit` for the list) and `tva-review` (the review service).

## The path, stage by stage

Start every session with `tva-kit brief` (it says which stage you are at). Do the stages in order;
after each, record it with `tva-kit clock "stage N done: …; next: …"`.

1. **Read.** The source (`tva-kit source`){{setup_read}}, the closest example's model (its
   `Model.lean`, or its one Lean file) and model map, its design (`tva-kit outline {{example}}`), and
   `~/fpv/tva/docs/PROVER_GUIDE.md` (§1–§4).
2. **Classes.** `tva-kit classify` proposes `source.ranges`, `source.host`, `source.unused`; check
   them, then `tva-kit classify --write`.
3. **Model.** Create `lean/{{module_path}}.lean` from the example's file: keep its layout (`import TVA`,
   the module docstring, `namespace {{namespace}}`, `open TVA`, the kernel `def … : Kernel where body ctx :=
   do …`, `Inst`, `Inst.ok`, `model`, `launch`, `Pre`, `Post`, `Spec`, `Assumptions`, the proof section).
   Transcribe the source line by line (skill **tva-model-map**). `tva-kit examples 'tl\.load'` shows
   how other kernels wrote a construct; `tva-kit lookup NAME` shows a library definition. The Lean is
   built automatically after each edit, and the errors come back to you.
4. **Map.** `tva-kit map-skeleton` adds an entry for every code line; fill each `TODO` (model text
   copied from your Lean, or a drop with its reason); `tva-kit map-check` until it is regular. The map
   is due before the gate, not before the proof: once the claim is written, `tva-kit stage` names the
   proof's next stage first and lists the map (and the write-up) as due before the gate.
5. **Claim.** `Inst`, `Inst.ok`, `launch`, `Pre`, `Post`, `Spec`, `Assumptions` (skill **tva-claim**).
6. **Design.** State every lemma before proving any (skill **tva-design**): in layers, as the examples
   are built (facts; the numbers: each loop's invariant, its step lemma and its end lemma, on values
   only; the program walk; `main`), one file per part, each with a docstring listing its lemmas and
   source lines. `sorry` only as a lemma's whole proof; the walk and `main` proved from the statements
   below them; the whole build green. `tva-kit stage` lists every `sorry` with its lemma.
7. **Audit.** Start the **claim-audit** subagent on the claim and the design; it writes
   `~/scratch/audit.md`. Fix what it finds (claim, conditions, statements) before the long proofs.
8. **Lemmas.** Replace the `sorry`s (skill **tva-proof**), hardest first (the step lemma of each loop),
   about an hour per lemma: if one does not close, change its statement (split it, strengthen the
   invariant, add a condition and say so in README.md) instead of pushing on. Keep every file quick to
   build: Lean re-checks a whole file at every build, so work on a lemma in a small file of its own,
   and when a build takes more than a minute, split before going on.
9. **Write-up.** `README.md`: Specification, Conditions, Assumptions, Proof, Narrowings, in words that
   match the Lean. Every constant README states must be the Lean's: reviewers file a README value
   that differs from its definition as `DOC_WRONG`, which blocks the certificate. After changing a
   constant, search README and the docstrings for its old value; `tva-kit stage` and `tva-kit gate`
   warn of a `NAME = value` in README that the Lean's `def NAME` no longer has.
10. **Gate and submit.** `tva-kit gate` until `gate PASS` (skill **tva-gate**), then `tva-review submit`,
   then `tva-review wait` again and again until the round is over.
11. **Rounds.** When findings come back: `tva-kit findings`, fix, `tva-kit gate`, `tva-kit respond init`,
   `tva-kit respond set F<n> …` for each, `tva-kit respond lint`, `tva-review submit`, `tva-review wait`
   (skill **tva-respond**). Repeat until the certificate is granted.

## Rules (hooks enforce them)

- Write only in your kernel's folder (`lean/`, `README.md`, `model-map.json`, `kernel.json`,
  `review/round-N/prover/`) and `~/scratch`. The library, its tools and documents, the source and the
  other kernels are fixed; the review service uses its own copy of the library.
- Never run `tvr respond`/`tvr file` (the service records), nor `lake update`, `lake clean` or anything
  that touches `.lake` (the prebuilt Mathlib).
- Do not stop while a review is running or findings are open: `tva-review wait`, then act.
- Searches need a path (`~/fpv/tva/docs`, `~/fpv/tva/lean/TVA`, `~/fpv/tva/kernels`), never `/`.
- Use only TVA and Mathlib (no other project's material).

## Lean in TVA: what works

- Never guess a name: `tva-kit lookup NAME` (library and examples) before you use it; for Mathlib,
  try `#check @Name` in a scratch file.
- Scratch experiments: `~/scratch/T.lean` with `import TVA` (and `import {{module}}` once it builds),
  checked by `cd ~/fpv/tva/lean && lake env lean ~/scratch/T.lean`. Use `#check`, `example … := by …`.
- Read the first error first; later ones often follow from it. `unknown identifier` → lookup;
  `unsolved goals` → the goal is printed: add steps; `type mismatch` → compare the two types printed.
- The whole output of the last build of each module, every error with its full goal, is in
  `~/scratch/build/<Module>.log` (the automatic build and `tva-kit build` write it): read a goal there,
  never build again only to see it. Make all the edits of one try, then build once.
- Tactics that do most of the work here: `simp [defs]`, `omega` (Nat/Int arithmetic), `norm_num`,
  `decide`, `exact`, `refine ⟨?_, ?_⟩`, `obtain ⟨a, b⟩ := h`, `constructor`, `intro`, `subst`, `rw`.
- Split big proofs into lemmas the way the examples do (`body_eq`, `launch_size`, `launch_pid`,
  `launch_valid`, `fits`, `safe_program`, the footprints `R`/`W`/`Q`, then `main`).
- `sorry` is fine while drafting; the gate refuses it, and `tva-kit stage` reports what is left.

## When you are stuck

- Narrow instead of stopping: a smaller `Inst.ok` (state each narrowing and its reason in README.md)
  that you can prove beats no claim. Reviewers judge whether a narrowing excludes real use. If at half
  the time a proof is still not converging, narrow then: a narrow claim that is certified is worth more
  than a general one that is unfinished.
- If the kernel needs a fact, an operation or a meaning the library lacks, say so as a library request
  (README.md, and later `tva-kit respond request …`); you cannot add it yourself.
- Delegate to keep your context small: the subagent **tva-scout** answers "how does TVA/an example do
  X?" in a few lines; **lean-fixer** proves one stated lemma (give it the lemma, its file, and the
  example's lemma to follow), not "finish the proof"; **claim-audit** checks the claim and the design.
  Subagents run in the foreground (their
  report is the result of the call; several started in one message run at once). Give each its own file:
  two agents editing one file undo each other's work. They may all build: the harness runs one
  `lake build` at a time, and `lake env lean` only while no build runs (a Lean process that reads a file
  another build is writing reports errors that are not there), so do not tell them not to build. When a report comes back, check it yourself
  (`tva-kit stage`, `tva-kit gate`) before going on. To continue a subagent's work, start a new one and
  give it the earlier report: a resumed subagent (SendMessage) would run in the background, so the
  harness refuses SendMessage.
- Never wait in a loop: `sleep` is blocked, and a `read` loop keeps a core busy that the builds need.
  There is nothing to wait for but the review (`tva-review wait`).
- Long outputs: redirect to `~/scratch/…` and read the part you need.
