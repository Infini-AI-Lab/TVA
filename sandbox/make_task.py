#!/usr/bin/env python3
"""Write the Harbor task in which an agent proves one of TVA's fixed claims (claims/<kernel>/).

  sandbox/make_task.py KERNEL... [--out DIR] [--image IMAGE] [--oracle] [--agent-timeout S] [--model M]
               [--budget USD] [--context-window N] [--cpus N] [--memory-mb N]

writes, for each KERNEL, OUT/tva3-claim-KERNEL/:

  task.toml, instruction.md                      Harbor's task; its prompt points to /instruction.md
  environment/Dockerfile, environment/claim/,    the prover's image: the claim image, Claude Code, this claim
  environment/instruction.md                     in /app/claim, and the task at /instruction.md
  environment/docker-compose.yaml,               the prover's network: sealed, its one way out a relay to the
  environment/meter_relay.py                     credit meter
  tests/Dockerfile, tests/claim/, tests/test.sh  the verifier's image: the same, with its own copy of the claim
  solution/                                      with --oracle: the claim's oracle proof (claims/KERNEL/oracle/)

and OUT/run-tva3-claim-KERNEL.sh, which runs the task once on the host: the credit meter (meter.py, with
--budget), then Harbor with the claim agent (harness/cc_agent.py) on --model, then the meter's report.

Both images start from the claim image tva3-claim:<tag> (claim/build.sh; by default the newest one), whose /app
holds the library (lean/), the docs, the worked example and the tool `tva-check`. Each image gets only the claim's
own files (claim_files.py), never a claim's oracle proof, which goes only to solution/. The agent writes only the
proof (`/app/claim/Proof.lean`, `/app/claim/Proof/`); Harbor hands only those files to the verifier, a machine
check (`tva-check verify`) whose reward is 1 or 0.
"""
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from claim_files import copy_claim  # noqa: E402
from lean_hash import lean_hash  # noqa: E402

EXAMPLE = "vec_add_f32"  # the claim image's worked example: not a task
CLAUDE_CODE = """RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm \\
    && rm -rf /var/lib/apt/lists/* \\
    && npm install -g @anthropic-ai/claude-code@2.1.316 && npm cache clean --force"""
CHECK = "/usr/local/bin/tva-check check /app/claim"
GRADE = "/usr/local/bin/tva-check grade /app/claim"
TIME = "/usr/local/bin/tva-check time"
VERIFIER_TIMEOUT = 3600


# The prover's network (environment/docker-compose.yaml, merged by Harbor with its own compose files).
COMPOSE = """# The prover reaches nothing but the credit meter. `main` (the prover's container) is only on `sealed`, an
# internal network with no route out: not the internet, not the host. Its one other member is `meter`
# (meter_relay.py), which is also on an ordinary network and relays one port, 8080, to the meter at
# TVA_METER_HOST:TVA_METER_PORT (set by the task's run script; unset, it relays nothing).
services:
  main:
    networks: [sealed]
    depends_on: [meter]
  meter:
    image: {image}
    command: ["python3", "-I", "/opt/meter_relay.py"]
    volumes: ["./meter_relay.py:/opt/meter_relay.py:ro"]
    environment:
      TARGET_HOST: "${{TVA_METER_HOST:-}}"
      TARGET_PORT: "${{TVA_METER_PORT:-}}"
      LISTEN_PORT: "8080"
    networks: [sealed, egress]
networks:
  sealed:
    internal: true
  egress: {{}}
"""


def write(path, text, mode=0o644):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, mode)


def claim_copy(src, dst):
    """The claim's own files (claim_files.py's allow list: never its GPU check or its oracle proof), with Claim.lean's
    sha256 in claim.json."""
    return copy_claim(src, dst)


def hm(s):
    return f"{s // 3600} h" + (f" {s % 3600 // 60} min" if s % 3600 else "")


def newest_image():
    r = subprocess.run(["docker", "images", "tva3-claim", "--format", "{{.CreatedAt}}\t{{.Repository}}:{{.Tag}}"],
                       capture_output=True, text=True, check=True)
    rows = sorted(l.split("\t") for l in r.stdout.splitlines() if l.strip() and not l.endswith(":<none>"))
    if not rows:
        sys.exit("no tva3-claim image: build one with sandbox/claim/build.sh")
    return rows[-1][1]


def instruction(k, cj, readme, a_timeout):
    ns = cj["namespace"]
    return f"""# Prove a fixed claim about the Triton kernel `{k}`

{cj['title']}.

## What to do

1. Read `/app/claim/Claim.lean` and the section **Your claim** below.
2. Write a Lean 4 proof of `theorem {ns}.main : {ns}.claim.Holds` in `/app/claim/Proof.lean` (and, if you
   want, more modules under `/app/claim/Proof/`).
3. Run `{CHECK}` until it ends with `PASS: {ns}.claim.Holds is proved`.
4. Run `{GRADE}` (in the background: it rebuilds your proof from scratch) and see `reward 1`.

You have {hm(a_timeout)}. Grading is all or nothing: the reward is 1 if the check passes and 0 otherwise. You
may not change the claim. [/app/CLAUDE.md](/app/CLAUDE.md) explains how to work in this container (paths,
builds, subagents, planning); Claude Code loads it for you (you start in `/app`).
**Read [/process_safety.md](/process_safety.md) before you stop or kill any process**: a broad `kill` or
`pkill` can end your own session, and the task then fails.

## The commands

Run them exactly like this, from any directory:

```
{CHECK}   # the grader's five steps, reusing your build (takes minutes)
{GRADE}   # the grader's procedure, from a fresh build of your proof
{TIME}               # the time used and left
```

`check` and `grade` take the workspace `/app/claim` as their one argument and check all of it: the fixed
`Claim.lean` and your proof, `Proof.lean` and every file under `Proof/`. To build while you work, run `lake`
from `/app/lean`: `timeout 1200 lake build Proof`, or one module with `lake build Proof.Lanes`.

You work as the user `prover`. Everything in this container is read-only to you except your proof
(`/app/claim/Proof.lean` and `/app/claim/Proof/`), the library's build folder (your builds add to it) and `/tmp`.

## What the theorem says

`Claim.lean` defines the model (the kernel transcribed into TVA's `tl.*` vocabulary), the instances it covers
(`Inst`, `Inst.ok`), the launch (`launch`), the initial memories covered (`Pre`) and the specification (`Post`),
and packs them as `def claim : TVA.Claim`. `TVA.Claim.Holds` (`/app/lean/TVA/Claim.lean`) is

```lean
∀ P, Device P → ∀ d, claim.ok d → ∀ init, claim.Pre d init →
  Correct ((claim.launch d).env P) (claim.kernel d) (claim.launch d) init (claim.Post d init)
```

`Correct`: the launch is valid and well behaved from `init` (no undefined behaviour, no race), and every way it
can end is a normal end whose final memory satisfies `Post`. `Device P` gives the device facts (accuracy of
`exp`, `tl.dot`, conversions, …) about the platform `P`. The proof kit `TVA.Proof` reduces `Correct` to one
proof per program (`Safe`, walked line by line with the rules of `TVA.Proof.Rules`), disjoint footprints, and
how the programs' results give `Post`: `correct_of_safe` (`/app/lean/TVA/Proof/Correct.lean`; its docstring
shows the skeleton of `main`). `/app/examples/vec_add_f32/Proof.lean` is a complete proof of another claim.

`Claim.lean` also proves `claim_satisfiable` and the `setup_*_ok` theorems (the claim covers the source's real
launches). They are done: `main` does not need them.

## Your claim

This is `/app/claim/README.md`. In it, "line N" is a line of `/app/claim/Claim.lean`, and "source line N" a
line of the kernel's source in `/app/claim/source/`. It may mention a GPU check (`check_gpu.py`): it was run
when the claim was written, and it is not in this container.

{readme}
## The proof file

```lean
-- /app/claim/Proof.lean
import Claim
import TVA.Proof

namespace {ns}

open TVA   -- the opens of Claim.lean do not carry over: open what you use

theorem main : claim.Holds := by
  ...

end {ns}
```

Your proof may use several files: `/app/claim/Proof/Lanes.lean` is the module `Proof.Lanes`, imported with
`import Proof.Lanes`. Proof files may import only `TVA`, `Mathlib`, `Claim` and `Proof` modules. A module that
`Proof.lean` does not import (directly or through another module) is not built, but it is still checked for the
forbidden constructs below.

## Grading

Only `/app/claim/Proof.lean` and `/app/claim/Proof/` are graded. The grader starts a fresh container from the
same image (no agent tools), with its own copy of the claim, copies those files into it, and runs `{CHECK}`,
the command you can run. Nothing else of yours reaches it: not your files elsewhere (`/tmp`, scratch modules),
not your build state. The grader's whole check must finish within {hm(VERIFIER_TIMEOUT)}: a proof whose build
takes much longer (heavy `decide` or `simp`, very large `maxHeartbeats`) scores 0. The reward is 1 when every
step passes, and 0 otherwise:

1. `Claim.lean` is the claim, unchanged (its sha256).
2. The proof files pass the hygiene list. It is matched on the **whole text, comments and strings included**,
   so do not write these words even in a comment:
   - proof escapes: `sorry`, `admit`, `native_decide`, `ofReduceBool`, `ofReduceNat`, `axiom`;
   - kernel or compiler bypasses: `implemented_by`, `extern`, `unsafe`, `skipKernelTC`, `addDecl`, `modifyEnv`,
     `setEnv`, the word `Environment`;
   - meta code: `#eval`, `#exit`, `run_cmd`, `run_tac`, `elab`, `macro`, `syntax`, `initialize`, `simproc`,
     `delab`, ..., and meta types (`IO`, `CoreM`, `MetaM`, `TacticM`, `Lean.Elab`, `Lean.Meta`, ...);
   - attributes that install handlers (`@[tactic ...]`, `@[norm_num ...]`, `@[aesop ...]`, `@[init]`, ...);
   - `notation`, `infix`, `prefix`, `postfix`, `export`, `renaming`, `unif_hint`;
   - `local` or `scoped` instances, attributes or notation, `@[instance]`, `@[default_instance]`, `@[coe]`, and
     attribute erasure (`attribute [-simp] ...`);
   - `set_option` other than the resource limits `maxHeartbeats`, `maxRecDepth`, `synthInstance.maxHeartbeats`;
   - a string literal holding `{{`, and a line break other than a line feed.

   They import only `TVA`, `Mathlib`, `Claim` and `Proof` modules. `{CHECK}` names the file and line of each
   match.
3. `lake build Claim Proof` succeeds.
4. `theorem tva_check : TVA.Claim.Holds {ns}.claim := {ns}.main` elaborates, and it depends on no axioms other
   than Lean's `propext`, `Classical.choice` and `Quot.sound`.
5. `leanchecker` re-checks every declaration of `Claim` and `Proof`.

## Time

You have {hm(a_timeout)} from the start of the session; then the session ends, whatever state the proof is in. A
proof that does not pass `{CHECK}` scores 0, however close it is. So plan: get `main` building early, as a
skeleton that states the lemmas it needs, and fill the lemmas in afterwards (a placeholder proof is fine while
you work, but none may be left at the end). `{TIME}` shows the time used and left; the harness also tells you at
the start, at half time, at 75% and 90%, and with 30 and 10 minutes left.

**Check yourself exactly as the grader will**, before you finish: `{GRADE}` deletes your proof's build outputs
(the grader has none), builds `Proof.lean` and `Proof/` from scratch, runs the five steps within the grader's
{hm(VERIFIER_TIMEOUT)}, and ends with `reward 1` or `reward 0`. It takes as long as a fresh build of your proof:
run it with the Bash tool's `run_in_background` and read its output when it ends, when no subagent is building.

The harness does not let you end the session while `{CHECK}` fails; when it passes, you are done.
"""


RUN = """#!/bin/bash
# Run the task {task} once: the credit meter (budget ${budget}), Harbor with the claim agent on {model}, then the
# meter's report. Everything goes to {runs}/{task}/: meter.jsonl (the ledger), meter.log, harbor.log, jobs/.
# The endpoint's key stays on the host (the meter reads KEY_FILE); the containers get only the meter's client key.
# The prover's container reaches nothing but the meter: it is on a sealed network whose one other member, `meter`
# (environment/docker-compose.yaml, meter_relay.py), relays to the meter at $METER_HOST:$PORT.
#   KEY_FILE=<file with the endpoint's key> {out}/run-{task}.sh
set -euo pipefail
: "${{KEY_FILE:?set KEY_FILE to the file that holds the endpoint key}}"
UPSTREAM=${{UPSTREAM:-{upstream}}}
PRICE=${{PRICE:-{price}}}
PORT=${{PORT:-18800}}
METER_HOST=${{TVA_METER_HOST:-$(hostname -I | awk '{{print $1}}')}}   # the host's address, as the relay reaches it
HARBOR=${{HARBOR:-harbor}}
d={runs}/{task}
mkdir -p "$d"
python3 -I {sandbox}/meter.py serve --upstream "$UPSTREAM" --key-file "$KEY_FILE" --price "$PRICE" \\
  --ledger "$d/meter.jsonl" --budget {budget} --port "$PORT" > "$d/meter.log" 2>&1 < /dev/null &
meter=$!
trap 'kill $meter 2>/dev/null || true' EXIT
for i in $(seq 1 30); do [ -s "$d/meter.jsonl.client-key" ] && break; sleep 1; done
TVA_METER_HOST="$METER_HOST" TVA_METER_PORT="$PORT" ANTHROPIC_BASE_URL="http://meter:8080/prover" ANTHROPIC_API_KEY="$(cat "$d/meter.jsonl.client-key")" \\
  PYTHONPATH={sandbox}/harness "$HARBOR" run -p {out}/{task} -a cc_agent:ClaudeCodeClaim \\
  -m anthropic/{model} -e docker -o "$d/jobs" > "$d/harbor.log" 2>&1 || echo "harbor exit $?" >> "$d/harbor.log"
python3 -I {sandbox}/meter.py report "$d/meter.jsonl"
"""


def make(k, a, image):
    cd = os.path.join(TVA, "claims", k)
    task = f"tva3-claim-{k}"
    out = os.path.join(a.out, task)
    if os.path.exists(out):
        shutil.rmtree(out)
    cj = claim_copy(cd, os.path.join(out, "environment", "claim"))
    claim_copy(cd, os.path.join(out, "tests", "claim"))
    if "source" in cj and "source_sha256" in cj:
        src = os.path.join(cd, cj["source"])
        if hashlib.sha256(open(src, "rb").read()).hexdigest() != cj["source_sha256"]:
            sys.exit(f"{src} does not match claim.json's source_sha256")
    rp = os.path.join(cd, "README.md")
    readme = open(rp).read().strip() + "\n" if os.path.isfile(rp) else "(This claim has no README.md.)\n"

    write(os.path.join(out, "environment", "instruction.md"), instruction(k, cj, readme, a.agent_timeout))
    write(os.path.join(out, "instruction.md"), f"""Prove a fixed claim about the Triton kernel `{k}` in Lean 4.

The task is in `/instruction.md`: read it first, all of it, and follow it. Read it again whenever you are unsure
what is asked. The task succeeds when `{CHECK}` passes.
""")
    write(os.path.join(out, "task.toml"), f'''schema_version = "1.4"

[task]
name = "tva3/claim-{k}"
description = "Prove TVA's fixed claim about the Triton kernel {k} in Lean 4; a machine check grades the proof."
keywords = ["lean4", "triton", "formal-verification"]

[metadata]
kernel = "{k}"
image = "{image}"
claim_sha256 = "{cj['claim_sha256']}"
model = "{a.model}"
budget_usd = {a.budget}

[agent]
timeout_sec = {a.agent_timeout}
user = "prover"

[verifier]
timeout_sec = {VERIFIER_TIMEOUT}
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
    write(os.path.join(out, "environment", "docker-compose.yaml"), COMPOSE.format(image=image))
    shutil.copy(os.path.join(HERE, "meter_relay.py"), os.path.join(out, "environment", "meter_relay.py"))
    write(os.path.join(out, "environment", "Dockerfile"), f"""# The prover's image: the claim image, Claude Code, and the claim in /app/claim.
FROM {image}
{CLAUDE_CODE}
ENV CLAUDE_CODE_MAX_CONTEXT_TOKENS={a.context_window} CLAUDE_CODE_AUTO_COMPACT_WINDOW={a.context_window} \\
    TVA_AGENT_TIMEOUT_SEC={a.agent_timeout} TVA_VERIFIER_TIMEOUT_SEC={VERIFIER_TIMEOUT}
COPY claim/ /app/claim/
COPY instruction.md /instruction.md
# the claim and the task are root's and read-only to the prover (user `prover`); it may only add files to
# /app/claim (sticky: its Proof.lean and Proof/, never replacing or removing the claim's)
RUN tva-check prepare \\
    && chown -R root:root /app/claim /instruction.md && chmod -R a+rX,go-w /app/claim && chmod 444 /instruction.md \\
    && chmod 1777 /app/claim
WORKDIR /app
""")
    write(os.path.join(out, "tests", "Dockerfile"), f"""# The verifier's image: the claim image and its own copy of the claim; the proof arrives as artifacts.
FROM {image}
COPY claim/ /app/claim/
RUN tva-check prepare
COPY test.sh /tests/test.sh
""")
    write(os.path.join(out, "tests", "test.sh"), "#!/bin/bash\n# the machine check: reward 1 when the proof "
          "passes, 0 otherwise (/logs/verifier/reward.json)\nexec tva-check verify\n", 0o755)
    if a.oracle:
        S = os.path.join(out, "solution")
        os.makedirs(S)
        shutil.copy(os.path.join(out, "tests", "claim", "claim.json"), os.path.join(S, "claim.json"))
        subprocess.run([sys.executable, "-I", os.path.join(HERE, "claim", "oracle.py"), k, S], check=True)
        os.remove(os.path.join(S, "claim.json"))
        write(os.path.join(S, "solve.sh"), """#!/bin/bash
# The oracle: the claim's proof (claims/KERNEL/oracle/, written by claim/oracle.py).
set -euo pipefail
cp /solution/Proof.lean /app/claim/Proof.lean
if [ -d /solution/Proof ]; then cp -r /solution/Proof /app/claim/; fi
/usr/local/bin/tva-check check /app/claim
""", 0o755)
    write(os.path.join(a.out, f"run-{task}.sh"), RUN.format(
        task=task, budget=a.budget, model=a.model, out=shlex.quote(os.path.abspath(a.out)),
        runs=shlex.quote(os.path.abspath(a.runs)), sandbox=shlex.quote(HERE), upstream=a.upstream, price=a.price),
        0o755)
    print(f"wrote {out} (from {image})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kernels", nargs="+", metavar="KERNEL")
    ap.add_argument("--out", default="tasks", help="the task folder (outside TVA; default ./tasks)")
    ap.add_argument("--runs", default="runs", help="where run-*.sh puts each run (default ./runs)")
    ap.add_argument("--image", help="the claim image (default: the newest tva3-claim)")
    ap.add_argument("--oracle", action="store_true", help="add the claim's oracle proof (claims/KERNEL/oracle/) as the task's solution")
    ap.add_argument("--agent-timeout", type=int, default=8 * 3600, help="the agent's time limit, seconds (8 h)")
    ap.add_argument("--model", default="accounts/fireworks/models/deepseek-v4p1-flash-1017",
                    help="the model run-*.sh gives Harbor (-m anthropic/MODEL)")
    ap.add_argument("--budget", type=float, default=25, help="the meter's budget for one run, dollars (25)")
    ap.add_argument("--upstream", default="https://api.fireworks.ai/inference",
                    help="the Anthropic-compatible endpoint the meter forwards to")
    ap.add_argument("--price", default="0.30,0.006,1.20",
                    help="the model's dollars per million uncached input, cached input and output tokens")
    ap.add_argument("--context-window", type=int, default=640000,
                    help="Claude Code's context window, tokens; it compacts near it (640000)")
    ap.add_argument("--cpus", type=int, default=16)
    ap.add_argument("--memory-mb", type=int, default=65536)
    a = ap.parse_args()
    for d in (a.out, a.runs):
        r = os.path.realpath(d)
        if r == TVA or r.startswith(TVA + os.sep):
            sys.exit(f"{d} is inside TVA's tree: choose a folder outside it")
    image = a.image or newest_image()
    if not image.startswith(f"tva3-claim:{lean_hash(os.path.join(TVA, 'lean'))}-"):
        print(f"note: {image} was built from another version of lean/ than the one here now", file=sys.stderr)
    for k in a.kernels:
        if k == EXAMPLE and not a.oracle:
            sys.exit(f"{EXAMPLE} is the worked example in every claim image, with its proof: not a task "
                     "(use --oracle to test the pipeline on it)")
        if not os.path.isfile(os.path.join(TVA, "claims", k, "Claim.lean")):
            sys.exit(f"no claim {k} (claims/{k}/Claim.lean)")
        make(k, a, image)


if __name__ == "__main__":
    main()
