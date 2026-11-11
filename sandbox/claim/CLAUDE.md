# Working in this container

**The task is in [/instruction.md](/instruction.md): read it first.** Before you stop, kill or clean anything,
read [/process_safety.md](/process_safety.md): a broad `kill`/`pkill` can end your own session.

The task: prove `theorem main : claim.Holds`, in the claim's namespace, in `/app/claim/Proof.lean` (lemmas may go
in modules under `/app/claim/Proof/`). The claim is fixed; Lean checks the proof, and nobody else reads it.

## Paths

| Path | What |
|---|---|
| `/app/claim/` | the fixed claim (`Claim.lean`, `claim.json`, `README.md`, `SETUP.md`, `source/`: never change them) and your proof (`Proof.lean`, `Proof/`) |
| `/app/examples/vec_add_f32/` | a worked example: another fixed claim with its complete proof (`Proof.lean`). Read it; it is not built here (its modules have the same names as yours) |
| `/app/lean/` | the TVA library with Mathlib, built: `TVA/Claim.lean` (what `claim.Holds` means), the `tl.*` semantics (`TVA/Triton`, `TVA/Launch`), and the proof kit `TVA/Proof` |
| `/app/docs/` | TVA's docs: `PROVING.md` (how proofs go, which kit module to use), `SEMANTICS.md`, `LIBRARY.md` (a map of the library), `DESIGN.md` |
| `/tmp/` | your scratch files |

## Building and checking

- Run Lean from `/app/lean`: `lake build Proof` (everything `Proof.lean` imports), `lake build Proof.Lanes` (one
  module, for `/app/claim/Proof/Lanes.lean`), `lake env lean /tmp/T.lean` (a scratch file; `import TVA.Proof` or
  `import Claim`). `Claim` is built already. Proof files import `Claim` and `TVA.Proof`, the proof kit (`import TVA`, which `Claim.lean`
  uses, is only the trusted part: the semantics and what a claim means).
- `/usr/local/bin/tva-check check /app/claim` runs the grader's five steps on the whole workspace (`Claim.lean`,
  `Proof.lean`, every file under `Proof/`; it never checks one file alone) and prints each as PASS or FAIL,
  reusing your build. It re-checks everything with `leanchecker`, so it takes minutes: run it when you think the
  proof is done, not after every edit.
- `/usr/local/bin/tva-check grade /app/claim` is the grader's procedure exactly: your proof's build outputs
  deleted, a fresh build, the five steps within the grader's time limit, then `reward 1` or `reward 0`. Run it
  last, with `run_in_background`, when no subagent is building.
- Put `timeout N` in front of every build (`timeout 1200 lake build Proof`). The Bash tool stops a command after
  10 minutes unless you pass its `timeout` (at most 30 minutes); run longer builds with `run_in_background`.
- `lake` runs one build at a time: a build may first wait while another agent's build runs.
- Look names up before you use them: `grep -rn "theorem NAME" /app/lean/TVA`, or `#check @NAME` in `/tmp/T.lean`.

## Subagents

Subagents run in the background (the harness sets it): the call returns at once, and each subagent's report
arrives as a message when it ends. Give each subagent one lemma, with its exact statement, in its own file under
`/app/claim/Proof/`, and do not edit that file while it works. The `lean-fixer` subagent iterates on one file's
build errors. Never wait with `sleep` or a polling loop: end your turn when only waiting is left.

## Planning the proof

- **Keep a plan in `/tmp/plan.md`**: the lemmas of your proof, one line each (name, what it states, status:
  stated, proved, or stuck and why). Update it when a status changes. Your context is summarized now and then;
  after each summary the harness shows you the plan again, so it is how you know what is done and what is next.
- **Decide what is true before you walk the program.** Write down first each program's footprints (what it
  reads, `R i`, and writes, `W i`), what each output's value satisfies and, for each loop, the invariant after
  `k` iterations (with its error bound), and check the bound at the largest covered instance. Then state the
  lemmas and `main`, and only then step through the kernel body. A wrong invariant found after the walk costs
  hours.
- **One walk for every mode.** When the kernel branches on a flag (variable length, a mask, a window, a dtype),
  prove the body once and split on the flag where the code differs, not by copying the whole walk per mode.
- **The shape of the proof** is the example's: one program at a time (`Safe`, walked line by line with one rule
  per `tl.*` operation), disjoint footprints, then `correct_of_safe` (its docstring shows the skeleton of `main`).
- **Look in the library first** (the list in `/app/lean/TVA/Proof.lean`, `/app/docs/LIBRARY.md`, `grep -rn` in
  `/app/lean/TVA/Proof`). Among its modules:
  - `TVA.Proof.Rules.*`: the rule of each operation (`Memory`: loads and stores; `Arith`, `Convert`, `Reduce`,
    `ArgReduce`, `Dot`, `Loop`);
  - `TVA.Proof.Buf`, `TVA.Proof.Mem`: buffer elements, footprints, reads after writes;
  - `TVA.Proof.Pattern`: 1-D blocks (`pid*B + tl.arange(0, B)`), 2-D strided tiles, block pointers, descriptors;
  - `TVA.Proof.Int`: integer values at any width, offsets and masks of lanes;
  - `TVA.Proof.Float`: float constants, rounding, error bounds, sums along a tree;
  - `TVA.Proof.Dot`: `acc += tl.dot(a, b)` loops and chunked accumulation, with the device's dot facts;
  - `TVA.Proof.Shared`: programs that meet only through atomics on shared cells (`correct_of_safeA` in place
    of `correct_of_safe`);
  - `TVA.Proof.Scan`, `TVA.Proof.Random`: scans, sorts, histograms; `tl.rand`;
  - `TVA.Proof.Softmax`: online softmax;
  - `TVA.Proof.Recurrence`: a state carried across trips with a decay.

  A lemma you write is yours to keep proving.

## Time

The session has a time limit ([/instruction.md](/instruction.md) gives it); `/usr/local/bin/tva-check time` shows
the time used and left. The harness reminds you as the end nears. Use the time on the proof's structure first: a
`main` that builds from stated lemmas tells you early what is left to prove.

## Stopping

The harness does not let you end the session while `/usr/local/bin/tva-check check /app/claim` fails: it runs the
check when you try, and shows you what fails. It lets you stop while a subagent is still running (its report
starts a new turn), and after eight tries in a row with the proof files unchanged (the task then fails). When the
check passes, you are done.
