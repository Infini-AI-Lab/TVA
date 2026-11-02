# TVA sandbox: a kernel proved and reviewed by agents, in containers

The sandbox runs TVA's prover/reviewer protocol (`docs/REVIEW_PROTOCOL.md`) on one kernel with agents
in place of people, as a task of the Harbor framework:

- the **prover** agent works in one container, with TVA and the kernel's skeleton;
- the **review service** runs in another: it holds the authority copy of TVA, gates each submission,
  runs the **reviewer** agents (each as a user of its own, in a copy of the tree), and records every
  step in the kernel's ledger with `tvr`;
- a **verifier** in a third container re-runs the gate on the exported record and checks the
  certificate.

The outcome is the kernel's record as TVA keeps it (claim, gate, reviews, responses, certificate), in
the trial's `artifacts/export/kernel/`. [PROTOCOL.md](PROTOCOL.md) says how a round goes and what each
side can see and do. [HARNESS.md](HARNESS.md) describes the agent harness (a manual, hooks, skills,
subagents and the helper `tva-kit`, installed at `/` in both containers), which helps models weaker
than the protocol was written for, and how to run the agents on public models.

## Layout

| Path | What |
|---|---|
| `base/Dockerfile`, `build_base.sh` | the base image `tva-sandbox-base:<TVA commit>`: Lean, Mathlib's cache and the TVA library built, from the files TVA ships at that commit |
| `service/server.py` | the review service (reviewer container): intake, gate, response, reviewer agents, recording, export |
| `service/reviewer-note.md` | what the sandbox adds to TVA's reviewer prompt |
| `service/tva-review` | the prover's client: `status`, `submit`, `wait`, `log`, `fetch` |
| `service/prepare_kernel.py` | resets the task's kernel to its skeleton in an image |
| `verifier/verify.py`, `verifier/test.sh` | the verifier |
| `harness/` | the agent harness (HARNESS.md): `tvakit.py`, `hook.py`, `settings.json`, the manuals, skills and subagents |
| `make_task.py` | writes the Harbor task `tva-prove-<kernel>[-scripted]/` |
| `meter.py` | the credit meter (HARNESS.md): a proxy on the host that records every agent request's tokens and cost, stops at a budget, and keeps the endpoint's key out of the containers |

TVA's gate reads every file of TVA's tree, so generated tasks and Harbor's job folders go outside it:
`make_task.py` writes to `tasks/` in the working directory and refuses a folder inside TVA, and
`harbor run` writes `jobs/` in the working directory. Run both from a work folder outside TVA. The
sandbox's own files name the containers' paths, so the gate's text scan skips its generic path tests
for this folder (library issue L135).

## Use

Harbor 0.25.0 (`pip install harbor==0.25.0`) and Docker are needed. From a work folder outside TVA, with
`TVA` standing for TVA's root:

```bash
# 1. the base image, from TVA's tree at the last commit that changed anything outside sandbox/ (about
#    10 minutes, 11 GB; no uncommitted changes outside sandbox/); a sandbox-only commit needs no new one
$TVA/sandbox/build_base.sh

# 2. a task, for a kernel of TVA's tree at that commit
python3 -I $TVA/sandbox/make_task.py vec_add_f32              # reviewers: Claude agents
python3 -I $TVA/sandbox/make_task.py vec_add_f32 --scripted   # reviewers: the scripted plumbing reviewer

# 3a. plumbing test: the oracle (TVA's certified proof) against the scripted reviewers
harbor run -p tasks/tva-prove-vec_add_f32-scripted -a oracle -e docker

# 3b. agents: one proves, Claude agents review (the key reaches only the reviewer container)
export TVA_REVIEWER_ANTHROPIC_API_KEY=...        # or TVA_REVIEWER_CLAUDE_CODE_OAUTH_TOKEN
export TVA_REVIEWER_MODEL=claude-opus-5-5        # optional
harbor run -p tasks/tva-prove-vec_add_f32 -a claude-code -m <model> -e docker

# 3c. public models (any Anthropic-compatible endpoint; HARNESS.md has the details, and how to track
#     the spend through meter.py)
export ANTHROPIC_BASE_URL=https://... ANTHROPIC_API_KEY=...                       # the prover's
export TVA_REVIEWER_ANTHROPIC_BASE_URL=https://... TVA_REVIEWER_ANTHROPIC_API_KEY=... TVA_REVIEWER_MODEL=<model-id>
harbor run -p tasks/tva-prove-vec_add_f32 -a claude-code -m anthropic/<model-id> -e docker
```

Options of `make_task.py`: `--out DIR` (default `tasks`), `--reviewers N` (1–4, default 2), `--model`,
`--agent-timeout` (default 8 h), `--max-reviewed` (default 6), `--max-rejected` (default 40), `--cpus`,
`--memory-mb`, `--context-window` (the context window Claude Code assumes and compacts at, default 128 000
tokens: set it below the model's context length, HARNESS.md) and `--no-autobuild` (no Lean build after
each of the prover's edits).

**Kernels.** Any kernel folder of TVA's tree at the base image's commit, with its source and
`kernel.json` (`tvr new` makes one). A certified kernel's task carries its certified proof as the oracle
solution (`-a oracle`); a kernel without a certificate has no oracle. Of the twelve certified kernels
(`kernels/README.md`), `vec_add_f32` is the smallest; `paged_attention_f16` (indirect access) and
`streamk_matmul_f16` (atomics) are among the hardest.

**Result.** Harbor's reward (`/logs/verifier/reward.json`) is 1 when the verifier's own gate passes and
the certificate holds; it also reports `rounds_reviewed`, `rejected`, `findings`, the reviewers' tokens
(`reviewer_input_tokens`, `reviewer_cached_tokens`, `reviewer_output_tokens`; and `reviewer_cost_usd` with
Anthropic's own endpoint only), `service_error` (a reviewer or the service failed, not the prover) and
`escalated`. With another endpoint, the spend is in the credit meter's ledger (HARNESS.md). The exported record,
with the job logs and the reviewer agents' transcripts, is in the trial's `artifacts/export/`. Its
`kernel/` folder is the kernel's folder as TVA keeps it: put in place of `kernels/<kernel>/` in TVA's
tree at the same commit, `tvr status` reads its certificate as it reads any other (a record made by
the scripted reviewers, who read nothing, is a plumbing test and nothing more).

Without a key, or with a bad one, the reviewer container stops at start (it runs one short agent call
first), and Harbor fails the trial before the prover starts.

## Testing the plumbing

- `harbor run … -a oracle` on a `--scripted` task: the oracle submits TVA's certified proof; the scripted
  reviewers accept it in round 1; reward 1.
- `TVA_REVIEWER_MODE=scripted-findings` (set it in a scripted task's `docker-compose.yaml`) runs the rest
  of the protocol: in round 1 each reviewer raises a blocking finding; the prover answers
  (`tvr start`, `response.json`, `tva-review submit`); in round 2 each reviewer's first filing is
  unfinished, the service sends it back, and its second filing (`filing-2.json`) settles the finding.
- `tva-review submit` of a skeleton or of a failing proof: rejected, nothing recorded, the gate's output
  returned.
- The verifier on a tampered export (a ledger line removed) or on a scripted record under an agent task:
  reward 0.
