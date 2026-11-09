#!/usr/bin/env python3
"""Generate the Harbor task in which an agent proves one of TVA's fixed claims (claims/<kernel>/).

  sandbox/make_claim_task.py KERNEL [--out DIR] [--tva DIR] [--oracle] [--agent-timeout S] [--cpus N]
               [--memory-mb N] [--context-window N]

writes OUT/tva-claim-KERNEL/:

  task.toml, instruction.md                     Harbor's prompt: it points to /instruction.md
  environment/Dockerfile, environment/claim/,   the prover's image: the claim base image, this claim,
  environment/instruction.md                    and the task at /instruction.md
  tests/Dockerfile, tests/claim/, tests/test.sh the verifier's image: the same, with its own copy of the claim
  solution/                                     with --oracle: a proof oracle.py derives from TVA's certified one

Both images start from tva-claim-base:<commit> (claim/build_base.sh), whose /app holds the library (lean/),
the docs, one worked example (examples/vec_add_f32: a claim with its proof) and the tool `tva-claim`. The
claim goes to /app/claim; the agent writes only the proof there (`Proof.lean`, `Proof/`), and Harbor hands
only those files to the verifier, a machine check (`tva-claim verify`: the build, `main : Statement`, the
standard axioms only, the gate's hygiene list, `leanchecker`) whose reward is 1 or 0.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = "vec_add_f32"  # the claim base image's worked example: not a task
CLAUDE_CODE = """RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm \\
    && rm -rf /var/lib/apt/lists/* \\
    && npm install -g @anthropic-ai/claude-code@2.1.316 && npm cache clean --force"""


def write(path, text, mode=0o644):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, mode)


def claim_copy(src, dst):
    """The claim folder without its GPU check (the containers have no GPU), with Claim.lean's sha256 in
    claim.json."""
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "check_gpu.py", "Proof.lean", "Proof"))
    cj = json.load(open(os.path.join(dst, "claim.json")))
    cj.pop("check", None)
    cj["claim_sha256"] = hashlib.sha256(open(os.path.join(dst, "Claim.lean"), "rb").read()).hexdigest()
    write(os.path.join(dst, "claim.json"), json.dumps(cj, indent=2) + "\n")
    return cj


def hm(s):
    return f"{s // 3600} h" + (f" {s % 3600 // 60} min" if s % 3600 else "")


def what_you_prove(cd):
    """The claim README's `## What you prove` section (its body), which the instruction shows as is."""
    text = open(os.path.join(cd, "README.md")).read()
    start = text.find("\n## What you prove\n")
    if start < 0:
        sys.exit(f"{cd}/README.md has no `## What you prove` section")
    body = text[start + len("\n## What you prove\n"):]
    end = body.find("\n## ")
    return (body if end < 0 else body[:end]).strip() + "\n"


CHECK = "/usr/local/bin/tva-claim check /app/claim"
GRADE = "/usr/local/bin/tva-claim grade /app/claim"
TIME = "/usr/local/bin/tva-claim time"


def instruction(k, cj, wyp, a_timeout, v_timeout=3600):
    ns = cj["namespace"]
    return f"""# Prove a fixed claim about the Triton kernel `{k}`

{cj['title']}.

## What to do

1. Read `/app/claim/Claim.lean` and the section **Your claim** below.
2. Write a Lean 4 proof of `theorem {ns}.main : {ns}.Statement` in `/app/claim/Proof.lean` (and, if you
   want, more modules under `/app/claim/Proof/`).
3. Run `{CHECK}` until it ends with `PASS: Statement is proved`.
4. Run `{GRADE}` (in the background: it rebuilds your proof from scratch) and see `reward 1`.

You have {hm(a_timeout)}. Grading is all or nothing: the reward is 1 if the check passes and 0 otherwise.
You may not change the claim. [/app/CLAUDE.md](/app/CLAUDE.md) explains how to work in this container (paths,
builds, subagents); Claude Code loads it for you (you start in `/app`).
**Read [/process_safety.md](/process_safety.md) before you stop or kill any process**:
a broad `kill` or `pkill` can end your own session, and the task then fails.

## The commands

Run them exactly like this, from any directory:

```
{CHECK}   # the grader's five steps, reusing your build (takes minutes)
{GRADE}   # the grader's procedure, from a fresh build of your proof
{TIME}               # the time used and left
```

`check` and `grade` take the workspace `/app/claim` as their one argument and check all of it: the fixed
`Claim.lean` and your proof, `Proof.lean` and every file under `Proof/`. They never check one file alone. To
build while you work, run `lake` from `/app/lean`: `timeout 1200 lake build Proof`, or one module with
`lake build Proof.Lanes` (/app/CLAUDE.md).

You work as the user `prover`. Everything in this container is read-only to you except your proof
(`/app/claim/Proof.lean` and `/app/claim/Proof/`), the library's build folder (your builds add to it) and
`/tmp`: the claim, this file, `/app/CLAUDE.md`, the library, the docs, the example and the tool cannot be changed.

## What the theorem says

`Claim.lean` defines, for this kernel:

| Name | What it is |
|---|---|
| `model`, `launch` | the kernel transcribed into TVA's Lean semantics, and the grid it is launched on |
| `Inst`, `Inst.ok` | an instance (sizes, strides, constexprs, buffers) and the instances the claim covers |
| `Pre d init` | what the initial memory `init` must satisfy (valid buffers, and any condition on the input values) |
| `Post d init fin` | what the final memory `fin` must satisfy: the outputs, and that every other byte is unchanged |
| `Assumptions K P` | the facts you may use: how the device `K` relates to `model` (`faithful`), and facts about the platform `P` (rounding, `exp`, dot products, ...) |
| `Statement` | `∀ K P, Assumptions K P → Spec K`, with `Spec K := Claim.Meets Inst.ok Pre Post K` |

Unfolded, `Statement` is:

```lean
∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

`K d init o` says that running the compiled kernel on instance `d` from memory `init` can end with outcome
`o`: a normal end `.ok fin`, a `.fault`, or a `.hang`. You know nothing about `K` except `Assumptions`. Its
field `faithful` (`Rules.Compiler.Faithful`) says that every outcome of `K` is an outcome of the model's
launch, provided the launch is valid, has `debug = false`, and the model's launch is well behaved from `init`
(no out-of-bounds access, no race). So you prove the claim about the model:

- `TVA.meets_of_faithful` reduces `Spec K` to: each covered launch is valid with `debug = false`, and from
  every `init` with `Pre`, the model's launch is well behaved and every one of its outcomes is `.ok fin` with
  `Post d init fin`.
- `TVA.meets_independent` reduces it further, for programs that write disjoint memory, to one proof per
  program (`Safe`, with read and write footprints), the footprints' disjointness, and how the programs'
  results give `Post`. Programs that share cells through atomics use `TVA.shared` instead.
- `PROVER_GUIDE.md` §4 (`/app/docs/`) gives the rule for each Triton operation (`safe_load_block`,
  `safe_dot_add`, `safe_forRange`, ...), and `/app/examples/vec_add_f32/Proof.lean` is a complete proof of
  another claim in this form.

`Claim.lean` also proves `consistent`, `satisfiable` and the `setup_*_ok` theorems (the claim is not
vacuous, and it covers the source's real launches). They are done: `main` does not need them.

## Your claim

This is the section "What you prove" of `/app/claim/README.md`. The rest of that file explains the
conditions, how each tolerance follows from the library's facts, and what the claim leaves out. In it,
"line N" is a line of `/app/claim/Claim.lean`, and "source line N" a line of the kernel's source file in
`/app/claim/source/`.

{wyp}
## Files to read

1. `/app/claim/Claim.lean`: the claim itself.
2. `/app/claim/README.md`: what the claim says, and how each tolerance is derived (`## Tolerances`).
3. `/app/claim/SETUP.md` and `/app/claim/source/`: the kernel's source and the launches the claim covers;
   `/app/claim/model-map.json`: which model line stands for which source line. `README.md` mentions a GPU
   check (`check_gpu.py`): it was run when the claim was prepared, and it is not in this container.
4. `/app/examples/vec_add_f32/`: a worked example, `Claim.lean` and its complete `Proof.lean`.
5. `/app/docs/PROVER_GUIDE.md` §1–4 and `/app/docs/SEMANTICS.md` (what each `tl.*` operation means).

## The proof file

```lean
-- /app/claim/Proof.lean
import Claim

namespace {ns}

open TVA   -- the opens of Claim.lean do not carry over: open what you use

theorem main : Statement := by
  ...

end {ns}
```

Your proof may use several files: `/app/claim/Proof/Lanes.lean` is the module `Proof.Lanes`, imported with
`import Proof.Lanes`. Proof files may import only `TVA`, `Mathlib`, `Claim` and `Proof` modules. A module that
`Proof.lean` does not import (directly or through another module) is not built, but it is still checked for
the forbidden constructs below.

## Grading

Only `/app/claim/Proof.lean` and `/app/claim/Proof/` are graded. The grader starts a fresh container from the
same base image (no agent tools), with its own copy of the claim, copies those files into it, and runs
`{CHECK}`, the same command you can run. Nothing else of yours reaches it: not your files elsewhere (`/tmp`,
scratch modules), not your build state. So a `PASS` of `{CHECK}` here, with every module your proof needs
under `/app/claim/Proof/`, is a pass there. The grader's whole check must finish within {hm(v_timeout)}: a
proof whose build takes much longer (heavy `decide` or `simp`, very large `maxHeartbeats`) scores 0. The
reward is 1 when every step passes, and 0 otherwise:

1. `Claim.lean` is the claim, unchanged.
2. The proof files pass TVA's hygiene list. It is matched on the **whole text, comments and strings included**,
   so do not write these words even in a comment:
   - proof escapes: `sorry`, `admit`, `native_decide`, `ofReduceBool`, `ofReduceNat`, `axiom`;
   - kernel or compiler bypasses: `implemented_by`, `extern`, `unsafe`, `set_option debug.*`, `skipKernelTC`,
     `addDecl`, `modifyEnv`, `setEnv`, the word `Environment`;
   - meta code: `#eval`, `#exit`, `run_cmd`, `run_tac`, `elab`, `macro`, `syntax`, `initialize`, `simproc`,
     `delab`, ..., and meta types (`IO`, `CoreM`, `MetaM`, `TacticM`, `Lean.Elab`, `Lean.Meta`, ...);
   - attributes that install handlers (`@[tactic ...]`, `@[norm_num ...]`, `@[aesop ...]`, `@[init]`, ...);
   - `notation`, `infix`, `prefix`, `postfix`, `export`, `renaming`, `unif_hint`;
   - `local` or `scoped` instances, attributes or notation, `@[instance]`, `@[default_instance]`, `@[coe]`, and
     attribute erasure (`attribute [-simp] ...`);
   - `set_option` other than `maxHeartbeats`, `maxRecDepth`, `synthInstance.maxHeartbeats`,
     `synthInstance.maxSize`, `autoImplicit`, `relaxedAutoImplicit`, `linter.*`, `pp.*`, `trace.*`;
   - a string literal holding `{{`, and a line break other than a line feed.

   They import only `TVA`, `Mathlib`, `Claim` and `Proof` modules. `{CHECK}` names the file and line of each
   match.
3. `lake build Claim Proof` succeeds.
4. `{ns}.main` has type `{ns}.Statement`, and it depends on no axioms other than Lean's `propext`,
   `Classical.choice` and `Quot.sound`.
5. `leanchecker` re-checks every declaration of `Claim` and `Proof`.

## Time

You have {hm(a_timeout)} from the start of the session; then the session ends, whatever state the proof is
in. A proof that does not pass `{CHECK}` scores 0, however close it is. So plan: get `main` building early,
as a skeleton that states the lemmas it needs, and fill the lemmas in afterwards (a placeholder proof is fine
while you work, but none may be left at the end). `{TIME}` shows the time used and left; the harness also
tells you at the start, at half time, at 75% and 90%, and with 30 and 10 minutes left.

**Check yourself exactly as the grader will**, before you finish:

```
{GRADE}
```

It does what the grader does, here: it deletes your proof's build outputs (the grader has none), builds
`Proof.lean` and `Proof/` from scratch, runs the five steps within the grader's {hm(v_timeout)}, and ends with
`reward 1` or `reward 0`. It takes as long as a fresh build of your proof, often longer than the Bash tool's
10-minute default: run it with the Bash tool's `run_in_background` and read its output when it ends. While you
work, `{CHECK}` runs the same five steps faster, reusing your build. Run `{GRADE}` when no subagent is
building (it waits for their builds, then deletes the outputs they share).

The harness does not let you end the session while `{CHECK}` fails; when it passes, you are done.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kernel")
    ap.add_argument("--tva", default=os.path.dirname(HERE))
    ap.add_argument("--out", default="tasks")
    ap.add_argument("--oracle", action="store_true",
                    help="add a solution derived from TVA's certified proof (only where the claim's definitions "
                         "are the certified bundle's)")
    ap.add_argument("--agent-timeout", type=int, default=8 * 3600)
    ap.add_argument("--cpus", type=int, default=16)
    ap.add_argument("--memory-mb", type=int, default=65536)
    ap.add_argument("--context-window", type=int, default=128000)
    a = ap.parse_args()
    k, tva = a.kernel, os.path.realpath(a.tva)
    if k == EXAMPLE:
        sys.exit(f"{EXAMPLE} is the worked example every claim task shows (with its proof): not a task")
    if os.path.realpath(a.out) == tva or os.path.realpath(a.out).startswith(tva + os.sep):
        sys.exit(f"--out {a.out} is inside TVA's tree: choose a folder outside it")
    cd = os.path.join(tva, "claims", k)
    rev = subprocess.run(["git", "-C", tva, "log", "-1", "--format=%h", "--", ".", ":(exclude)sandbox",
                          ":(exclude)claims"], capture_output=True, text=True, check=True).stdout.strip()
    base = f"tva-claim-base:{rev}"
    out = os.path.join(a.out, f"tva-claim-{k}")
    if os.path.exists(out):
        shutil.rmtree(out)

    cj = claim_copy(cd, os.path.join(out, "environment", "claim"))
    claim_copy(cd, os.path.join(out, "tests", "claim"))
    src = os.path.join(cd, cj["source"]["file"])
    if hashlib.sha256(open(src, "rb").read()).hexdigest() != cj["source"]["sha256"]:
        sys.exit(f"{src} does not match claim.json's sha256")

    # the task, in the prover's container at /instruction.md; Harbor's prompt points to it
    write(os.path.join(out, "environment", "instruction.md"), instruction(k, cj, what_you_prove(cd), a.agent_timeout))
    write(os.path.join(out, "instruction.md"), f"""Prove a fixed claim about the Triton kernel `{k}` in Lean 4.

The task is in `/instruction.md`: read it first, all of it, and follow it. Read it again whenever you are unsure
what is asked. The task succeeds when `/usr/local/bin/tva-claim check /app/claim` passes.
""")
    write(os.path.join(out, "task.toml"), f'''schema_version = "1.4"

[task]
name = "tva/claim-{k}"
description = "Prove TVA's fixed claim about the Triton kernel {k} in Lean 4; a machine check grades the proof."
keywords = ["lean4", "triton", "formal-verification"]

[metadata]
kernel = "{k}"
tva_commit = "{rev}"
claim_sha256 = "{cj['claim_sha256']}"

[agent]
timeout_sec = {a.agent_timeout}
user = "prover"

[verifier]
timeout_sec = 3600
environment_mode = "separate"
network_mode = "no-network"

[environment]
build_timeout_sec = 3600
cpus = {a.cpus}
memory_mb = {a.memory_mb}
# the agent runs in /app, where Claude Code loads its harness (/app/CLAUDE.md, /app/.claude)
workdir = "/app"

# only the proof goes to the verifier, whose image has its own copy of the claim
[[artifacts]]
source = "/app/claim/Proof.lean"

[[artifacts]]
source = "/app/claim/Proof"
''')
    write(os.path.join(out, "environment", "Dockerfile"), f"""# The prover's image: the claim base image, Claude Code, and the claim in /app/claim.
FROM {base}
{CLAUDE_CODE}
ENV CLAUDE_CODE_MAX_CONTEXT_TOKENS={a.context_window} CLAUDE_CODE_AUTO_COMPACT_WINDOW={a.context_window} \
    TVA_CLAIM_AGENT_TIMEOUT_SEC={a.agent_timeout} TVA_CLAIM_VERIFIER_TIMEOUT_SEC=3600
COPY claim/ /app/claim/
COPY instruction.md /instruction.md
# the claim and the task are root's and read-only to the prover (user `prover`); it may only add files to
# /app/claim (sticky: its Proof.lean and Proof/, never replacing or removing the claim's)
RUN tva-claim prepare \
    && chown -R root:root /app/claim /instruction.md && chmod -R a+rX,go-w /app/claim && chmod 444 /instruction.md \
    && chmod 1777 /app/claim
# the agent starts in /app, where Claude Code loads its harness (/app/CLAUDE.md, /app/.claude)
WORKDIR /app
""")
    write(os.path.join(out, "tests", "Dockerfile"), f"""# The verifier's image: the claim base image and its own copy of the claim; the proof arrives as artifacts.
FROM {base}
COPY claim/ /app/claim/
RUN tva-claim prepare
COPY test.sh /tests/test.sh
""")
    write(os.path.join(out, "tests", "test.sh"), "#!/bin/bash\n# the machine check: reward 1 when the proof "
          "passes, 0 otherwise (/logs/verifier/reward.json)\nexec tva-claim verify\n", 0o755)

    if a.oracle:
        S = os.path.join(out, "solution")
        subprocess.run([sys.executable, "-I", os.path.join(HERE, "claim", "oracle.py"), k, S, "--tva", tva],
                       check=True)
        write(os.path.join(S, "solve.sh"), """#!/bin/bash
# The oracle: TVA's certified proof of the kernel, moved into the claim's namespace (claim/oracle.py).
set -euo pipefail
cp /solution/Proof.lean /app/claim/Proof.lean
if [ -d /solution/Proof ]; then cp -r /solution/Proof /app/claim/; fi
/usr/local/bin/tva-claim check /app/claim
""", 0o755)
    print(f"wrote {out} (images from {base}; " + ("with the oracle)" if a.oracle else "no oracle)"))


if __name__ == "__main__":
    main()
