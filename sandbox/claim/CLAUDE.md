# Working in this container

**The task is in [/instruction.md](/instruction.md): read it first.** Before you stop, kill or clean anything,
read [/process_safety.md](/process_safety.md): a broad `kill`/`pkill` can end your own session.

## Paths

| Path | What |
|---|---|
| `/app/claim/` | the fixed claim (`Claim.lean`, `claim.json`: never change them) and your proof (`Proof.lean`, `Proof/`) |
| `/app/examples/vec_add_f32/` | a worked example: another fixed claim with its complete proof. Read it; it is not built here (its modules have the same names as yours) |
| `/app/lean/` | the TVA library (`TVA/`: `Lang`, `Sem`, `Rules`, `Lemmas`, `Math`) with Mathlib, already built |
| `/app/docs/` | `SEMANTICS.md` (what each `tl.*` operation means), `PROVER_GUIDE.md` §1–4 (models, specs, proofs), `LIBRARY.md`, `TRANSCRIPTION.md`, `rules/` |
| `/tmp/` | your scratch files |

`PROVER_GUIDE.md` also describes TVA's review workflow (`tvr`, `tva-kit`, gates, reviewers, `kernels/`): none of
it is here. The only tool is `/usr/local/bin/tva-claim`.

## Building and checking

- Run Lean from `/app/lean`: `lake build Proof` (everything `Proof.lean` imports), `lake build Proof.Lanes` (one
  module, for `/app/claim/Proof/Lanes.lean`), `lake env lean /tmp/T.lean` (a scratch file; `import TVA` or
  `import Claim`). `Claim` is built already.
- `/usr/local/bin/tva-claim check /app/claim` runs the grader's five steps on the whole workspace
  (`Claim.lean`, `Proof.lean`, every file under `Proof/`; it never checks one file alone) and prints each as PASS or FAIL, reusing your build. It
  re-checks everything with `leanchecker`, so it takes minutes: run it when you think the proof is done, not
  after every edit.
- `/usr/local/bin/tva-claim grade /app/claim` is the grader's procedure exactly: your proof's build outputs deleted, a fresh build of
  `Proof.lean` and `Proof/` only, the five steps within the grader's time limit, then `reward 1` or `reward 0`.
  Run it last, with `run_in_background`, when no subagent is building.
- Put `timeout N` in front of every build (`timeout 1200 lake build Proof`). The Bash tool stops a command
  after 10 minutes unless you pass its `timeout` (at most 30 minutes); run longer builds with
  `run_in_background`.
- `lake` runs one build at a time: a build may first wait while another agent's build runs.
- Look names up before you use them: `grep -rn "theorem NAME" /app/lean/TVA`, or `#check @NAME` in
  `/tmp/T.lean`.

## Subagents

Subagents run in the background (the harness sets it): the call returns at once, and each subagent's report
arrives as a message when it ends. Give each subagent one lemma, with its exact statement, in its own file under
`/app/claim/Proof/`, and do not edit that file while it works. The `lean-fixer` subagent iterates on one file's
build errors. Never wait with `sleep` or a polling loop: end your turn when only waiting is left.

## Planning the proof

- **Keep a plan in `/tmp/plan.md`**: the lemmas of your proof, one line each (name, what it states, status:
  stated, proved, or stuck and why). Update it when a status changes. Your context is summarized now and then;
  after each summary the harness shows you the plan again, so it is how you know what is done and what is next.
- **Decide what is true before you walk the program.** Write down first what each output's value satisfies and,
  for each loop, the invariant after `k` iterations (with its error bound), and check the bound at the largest
  covered instance. Then state the lemmas and `main`, and only then step through the kernel body. A wrong
  invariant found after the walk costs hours.
- **One walk for every mode.** When the kernel branches on a flag (variable length, a mask, a window, a dtype),
  prove the body once and split on the flag where the code differs, not by copying the whole walk per mode.
- **Look in the library first** (`/app/docs/LIBRARY.md`, `grep -rn` in `/app/lean/TVA/Lemmas`): it has loop and
  error lemmas for common patterns. Among them:
  - integer values at any width (`IntW`);
  - 2-D masked tiles: offsets, footprints, load/store, disjointness (`Tile2`);
  - float constants and exact conversions (`FloatFacts`);
  - dot-product accumulation (`DotLoop`);
  - online softmax (`OnlineSoftmax`);
  - a state carried across trips with a decay (`Recurrence`).

  A lemma you write is yours to keep proving.

## Time

The session has a time limit ([/instruction.md](/instruction.md) gives it); `/usr/local/bin/tva-claim time` shows the time used
and left. The harness reminds you as the end nears. Use the time on the proof's structure first: a `main` that
builds from stated lemmas tells you early what is left to prove.

## Stopping

The harness does not let you end the session while `/usr/local/bin/tva-claim check /app/claim` fails: it runs the check when you try, and
shows you what fails. It lets you stop while a subagent is still running (its report starts a new turn), and
after eight tries in a row with the proof files unchanged (the task then fails). When the check passes, you are
done.
