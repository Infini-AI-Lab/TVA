# TVA sandbox: an agent proves a fixed claim, in a container

A claim task gives an agent one of TVA's fixed claims (`claims/<kernel>/`) and asks for a Lean proof of
`theorem main : claim.Holds`. A machine check grades the proof (`tva-check`): reward 1 when it passes, 0
otherwise. The task runs in the Harbor framework: the agent (Claude Code) works in one container, and a
verifier checks the proof in another.

## Layout

| Path | What |
|---|---|
| `build_base.sh`, `base/Dockerfile` | the base image `tva3-base:<hash>`: the Lean toolchain, Mathlib and the TVA library, built |
| `lean_hash.py` | `<hash>`: a content hash of `lean/`'s source (lakefile, manifest, toolchain, `TVA.lean`, `TVA/**.lean`) |
| `claim/build.sh`, `claim/Dockerfile` | the claim image `tva3-claim:<hash>-<sub>`, on the base: harness, docs, example, tool |
| `claim/tva-check` | the one tool: `check`, `grade`, `verify`, `prepare`, `time`, and the prover's hooks |
| `claim/CLAUDE.md`, `claim/settings.json`, `claim/lean-fixer.md` | the prover's harness: how to work, the hooks, a subagent |
| `claim/process_safety.md` | what the prover must never kill |
| `claim/lake` | the build lock in front of the toolchain's `lake` (one build at a time; readers share) |
| `claim/oracle.py`, `claim/oracle/vec_add_f32/` | the worked example's proof |
| `make_task.py` | writes the Harbor task `tva3-claim-<kernel>/` and a script that runs it once |
| `harness/cc_agent.py` | the Harbor agent `ClaudeCodeClaim`: Claude Code, resumed after an error exit |
| `meter.py` | the credit meter: a proxy on the host that records each request's tokens and cost, stops at a budget, and keeps the endpoint's key out of the containers |

## The claim image

```
/app/claim/                  the task's claim (each task adds it); the agent writes Proof.lean and Proof/ here
/app/examples/vec_add_f32/   the worked example: a fixed claim with its complete proof (checked at build)
/app/lean/                   the TVA library with Mathlib, built (lakefile: TVA, and ClaimTask = /app/claim)
/app/docs/                   TVA's docs, as they were when the image was built
/app/CLAUDE.md, /app/.claude/  the harness; the agent starts in /app, so Claude Code loads both
/process_safety.md           what never to kill; each task adds /instruction.md (the task)
/usr/local/bin/tva-check     the tool; /usr/local/bin/lake, the build lock
```

`tva-check check /app/claim` checks the workspace (`Claim.lean`, `claim.json` with `claim_sha256` and
`namespace`, the prover's `Proof.lean` and `Proof/**.lean`):

1. `Claim.lean` is the claim (its sha256);
2. the proof files pass the hygiene list: no `sorry`/`admit`/`axiom`/`native_decide`, no `unsafe`,
   `implemented_by`, `extern` or other kernel bypass, no meta code (`macro`, `syntax`, `elab`, `#eval`, …),
   no `notation`, no `local`/`scoped` instance, no `set_option` but `maxHeartbeats`, `maxRecDepth`,
   `synthInstance.maxHeartbeats`; imports only `TVA`, `Mathlib`, `Claim`, `Proof` (matched on the whole text,
   comments included);
3. `lake build Claim Proof` succeeds;
4. `theorem tva_check : TVA.Claim.Holds <ns>.claim := <ns>.main` elaborates and `#print axioms` shows only
   `propext`, `Classical.choice`, `Quot.sound`;
5. `leanchecker` re-checks `Claim` and `Proof` from the compiled files.

`grade` deletes the proof's build outputs, builds afresh and checks within the grader's limit (1 h); `verify`
writes `/logs/verifier/reward.json`. The hooks: `SessionStart` starts the clock and, after a compaction, shows
`/tmp/plan.md` again; `PreToolUse` refuses changes to `Claim.lean`/`claim.json` and runs subagents in the
background; `PostToolUse` reminds of the time (50%, 75%, 90%, 30 and 10 min left); `Stop` runs the check and
holds the agent until it passes (or until eight stops in a row with the proof unchanged).

## Use

Harbor 0.25.0 (`pip install harbor==0.25.0`) and Docker are needed. Generated tasks and Harbor's job folders
go **outside** TVA's tree (`make_task.py` refuses a folder inside it): run from a work folder elsewhere.
`TVA` stands for TVA's root.

```bash
# 1. the images (the base takes about 10 minutes, 11 GB; nothing is downloaded but Ubuntu's packages,
#    Claude Code and its Node in the task image). build.sh builds the base first when it is missing, and prints
#    the claim image's tag. The library must build, and the example's proof must pass the check.
$TVA/sandbox/build_base.sh          # optional: tva3-base:<hash>
$TVA/sandbox/claim/build.sh         # tva3-claim:<hash>-<sub>

# 2. tasks (default: the newest tva3-claim image), and one run script per task
python3 -I $TVA/sandbox/make_task.py softmax_f32 matmul_f16 --out tasks --runs runs \
    [--agent-timeout 28800] [--model <model-id>] [--budget 25] [--context-window 640000] [--oracle]

# 3a. plumbing: the oracle (only where the sandbox has a proof: vec_add_f32) → reward 1 (about 3 minutes)
python3 -I $TVA/sandbox/make_task.py vec_add_f32 --out tasks --oracle
harbor run -p tasks/tva3-claim-vec_add_f32 -a oracle -e docker -o jobs

# 3b. an agent, through the meter (the run script does all of this; see below)
KEY_FILE=<file with the endpoint's key> tasks/run-tva3-claim-softmax_f32.sh
```

Options of `make_task.py`: `--out` (default `tasks`), `--runs` (default `runs`), `--image`, `--oracle`,
`--agent-timeout` (8 h), `--model` (default DeepSeek V4.1 Flash on Fireworks), `--budget` (the meter's cap for
one run, $25), `--upstream` and `--price` (the endpoint and its dollars per million uncached input, cached input
and output tokens), `--context-window` (640 000: the window Claude Code assumes and compacts at; keep it below
the model's context length, with a margin), `--cpus`, `--memory-mb`.

### Running with the meter

`run-tva3-claim-<k>.sh` starts the meter on the host, runs Harbor with the claim agent through it, and prints the
spend. By hand:

```bash
python3 -I $TVA/sandbox/meter.py serve --upstream https://api.fireworks.ai/inference \
    --key-file <file with the key> --price 0.30,0.006,1.20 --ledger runs/k/meter.jsonl --budget 25 --port 18800 &
K=$(cat runs/k/meter.jsonl.client-key); H=<the host's address, as the containers reach it>
ANTHROPIC_BASE_URL=http://$H:18800/prover ANTHROPIC_API_KEY=$K PYTHONPATH=$TVA/sandbox/harness \
    harbor run -p tasks/tva3-claim-<k> -a cc_agent:ClaudeCodeClaim -m anthropic/<model-id> -e docker -o runs/k/jobs
python3 -I $TVA/sandbox/meter.py report runs/k/meter.jsonl     # at any time: requests, tokens, cost
```

**Safety rules.**
- The endpoint's key (the Fireworks token) stays on the host: only the meter reads it (`--key-file`). Never put
  it in a container, a task, a URL, a log or a commit, and never print it. The containers get only the meter's
  own client key (`<ledger>.client-key`, mode 600), which works nowhere else, and the meter refuses requests
  from outside the private address ranges.
- Run every experiment through the meter, with a budget: at the budget it answers HTTP 402 and the agent stops.
  The ledger is the account of a run's spend (the provider's billing covers the whole account and lags;
  Claude Code prices its own use at Anthropic's rates).
- Docker's disk is shared: remove the images you build for a test (`docker rmi`, no `-f`), and never touch
  other users' images or containers.
- Launch long runs detached (`setsid nohup …`), so that they outlive the shell that started them.

`ClaudeCodeClaim` resumes the session (`claude --resume`, at most 3 times) when a run ends on an error rather
than with the agent's last turn, and logs each run's exit in `/logs/agent/cc-runs.txt`.

## Results

In Harbor's job folder (`runs/<task>/jobs/<job>/<trial>/`):
- `verifier/reward.json`: `{"reward": 1}` or `0`; `verifier/details.json`: each of the five checks, PASS or FAIL,
  with its detail;
- `agent/claude-code.txt`: the agent's transcript (stream JSON), `agent/cc-runs.txt`: each run's exit;
- `artifacts/app/claim/`: the graded proof (`Proof.lean`, `Proof/`).

`meter.py report runs/<task>/meter.jsonl` gives the run's requests, tokens and cost.
