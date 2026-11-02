# The agent harness

The prover and the reviewers are Claude Code agents, and they may run on public models (DeepSeek, GLM
and the like, through an Anthropic-compatible endpoint) that are weaker than the models TVA's protocol
was written for. Those models lose their way in long tasks, get exact formats wrong, guess Lean names,
edit files they must not touch, and stop early. The harness turns each of these into a tool, a check or
a reminder. It is Claude Code's own extension points (a manual, settings with hooks, skills, subagents)
plus one helper program, `tva-kit`, and it is installed at the root of each container.

## Where it is

In the prover's container and in the reviewers' container alike:

| Path | What |
|---|---|
| `/CLAUDE.md` | the role's operating manual (prover: the path in nine stages; reviewer: the procedure and its calibration) |
| `/.claude/settings.json` | the hooks and environment settings |
| `/.claude/skills/` | prover: `tva-model-map`, `tva-claim`, `tva-proof`, `tva-gate`, `tva-respond`; reviewer: `tva-review-method`, `tva-filing` |
| `/.claude/agents/` | subagents: `tva-scout` (both roles), `lean-fixer` (prover) |
| `/usr/local/bin/tva-kit`, `/usr/local/lib/tva-harness/` | the helper and the hook script (`tvakit.py`, `hook.py`) |
| `/etc/tva-sandbox/harness.json` | the role, kernel, examples, entry kernels and reviewer ids |

Agents start in `/`: the prover's image has `WORKDIR /`, and the review service starts each reviewer
with working directory `/`. So `/.claude/` is the project's settings directory, and Claude Code loads its
hooks, skills and subagents. Harbor's agent sets `CLAUDE_CONFIG_DIR` to its log folder, so the
project level is the only one the agent reads besides that.

Claude Code (2.1.316, tested) does not read a `CLAUDE.md` at the filesystem root, although it reads
one in any directory below it. So the `SessionStart` hook delivers `/CLAUDE.md`'s text at every start,
resume and compaction, with the role's brief and the tail of `~/scratch/progress.md`.

`make_task.py` renders all of this per task (kernel, Lean module, examples, number of reviewers) into
`environment/harness/` and installs it in both images. The sources are in `harness/`.

## The parts

**The manual.** One page per role. The prover's manual is a path in stages: read, classes, model, map,
claim, proofs, write-up, gate and submit, rounds. Each stage names its command and skill, and the
manual lists the worked examples to copy from: three certified kernels chosen for the task's kernel
(the table `EXAMPLES` in `make_task.py`), never the kernel itself. The reviewer's manual is a procedure in six steps, with calibration
rules: accept nothing unread, invent nothing, and do not move the goalposts.

**`tva-kit`.** Short commands that print little. They write the mechanical parts of the protocol and
check them the way `tvr` does (they import `tvr`'s own functions):

| Command | Role | What |
|---|---|---|
| `brief`, `stage` | prover | where you are: the stage checklist with what is done and what is next |
| `source [A-B]` | both | the source with line numbers and each line's class |
| `examples REGEX`, `lookup NAME`, `doc QUERY` | both | how certified kernels map a construct; a library or kernel declaration with its signature; the passages of TVA's documents about a topic |
| `classify [--write]` | prover | proposes `kernel.json`'s `source.ranges`, `source.host` and `source.unused` from the JIT functions reachable from the launched kernel |
| `map-skeleton`, `map-check` | prover | an entry for every code line (imports and decorators pre-filled as drops), then the gate's map check without a build |
| `build [MODULE]`, `gate` | prover | Lean errors with their source lines; the gate with a hint for each failing check |
| `findings`, `respond …` | prover | the open findings in full; `response.json` and `notes.md` written and linted |
| `review-brief`, `targets` | reviewer | items, assumptions, your findings, the source beside each model-map entry; the targets a finding may name |
| `filing …` | reviewer | `filing.json` and `report.md` written together, consistent and linted |

**Hooks** (`hook.py`):

| Event | What it does |
|---|---|
| `SessionStart` | the manual, the role's brief and the progress notes (at start, resume and compaction) |
| `PreToolUse` | refuses writes outside the role's folders (prover: the kernel folder and `~/scratch`; reviewer: its round folder and `~/scratch`), recording commands (`tvr respond`, `tvr file`, …), toolchain changes (`lake update`/`clean`, `lake exe cache`, `elan …`, deleting `.lake`), for reviewers `lake build`/`tvr check`/`tva-kit gate`, and searches of `/` with no path. Each refusal says why and what to do instead |
| `PostToolUse` | after the prover edits a kernel file: JSON validity, the map check for `model-map.json`, and a build of the edited Lean module whose errors come back at once (`TVA_HARNESS_AUTOBUILD`, on by default) |
| `Stop` | the prover cannot stop while a review is running, findings are open, or the proof is unfinished, unless the service says the task is over; a reviewer cannot stop before `tvr lint` says `regular`. After 8 (prover) or 6 (reviewer) identical refusals it lets the agent stop, so a stuck model ends instead of looping |

The hooks guard against mistakes; they are not the security boundary. That is the review service: it
holds the authority copy of TVA, takes only whitelisted files from a submission, gates them itself, and
records the ledger. The verifier re-checks the record.

**Skills.** One per stage that needs know-how: how to transcribe and map, write a claim, structure
proofs (with the lemma layout the examples use), pass the gate, and answer findings. For the reviewer:
the review method part by part, with the question to ask and the finding kind for each defect, and
filing. Claude Code shows each skill's description to the model, which loads the skill when the stage
comes up.

**Subagents.** `tva-scout` answers one question about the library, the documents or the examples in a
few lines with file references. `lean-fixer` makes one Lean file build. Both keep long searches and
error loops out of the main context.

**Settings.** Long Bash timeouts (10 minutes by default, 30 at most; builds are slow), a 40 000-character
output limit, and the context window (`make_task.py --context-window`, default 128 000):
`CLAUDE_CODE_MAX_CONTEXT_TOKENS`, the window Claude Code assumes for a model it does not know (it
assumes 200 000 otherwise, and ignores this for Claude models), and `CLAUDE_CODE_AUTO_COMPACT_WINDOW`,
the size at which it compacts (at most the model's window).

## Running with a public model

The prover is Harbor's `claude-code` agent pointed at the provider's Anthropic-compatible endpoint:

```bash
export ANTHROPIC_BASE_URL=https://...      # the provider's Anthropic-compatible endpoint
export ANTHROPIC_API_KEY=...               # or ANTHROPIC_AUTH_TOKEN, as the provider asks
harbor run -p tasks/tva-prove-<kernel> -a claude-code -m anthropic/<model-id> -e docker
```

The `anthropic/` prefix tells Harbor that the model speaks Anthropic's Messages API. With a base URL set,
Harbor points every model Claude Code uses (main, subagents, side requests) at `<model-id>`. Extra
settings go in with `--ae`, for example `--ae CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192` when the provider
caps output, or `--ae API_TIMEOUT_MS=600000`.

The reviewers get their own endpoint, key and model, set on the host and passed only to the reviewer
container:

```bash
export TVA_REVIEWER_ANTHROPIC_BASE_URL=https://...
export TVA_REVIEWER_ANTHROPIC_API_KEY=...  # or TVA_REVIEWER_ANTHROPIC_AUTH_TOKEN
export TVA_REVIEWER_MODEL=<model-id>
# optional: TVA_REVIEWER_API_TIMEOUT_MS, TVA_REVIEWER_CLAUDE_CODE_MAX_OUTPUT_TOKENS, TVA_REVIEWER_MAX_THINKING_TOKENS
```

With a base URL the service sets every model alias to `TVA_REVIEWER_MODEL` and the API timeout to 10
minutes. It makes one short call at start and stops if that fails, so a wrong endpoint, key or model
fails the trial before the prover starts.

Make the task with `--context-window` below the model's context length, with a margin: an endpoint
without Anthropic's token counting (`/v1/messages/count_tokens`; Fireworks answers 404) leaves Claude
Code to estimate how full its context is. For DeepSeek V4.1 Flash (1M tokens) on Fireworks we use
`--context-window 650000`. `--no-autobuild` turns off the build after each edit, for a machine
where builds are too slow for it.

### Tracking the spend

`meter.py` runs on the host between the agents and the endpoint. It records every request's tokens and
cost in a ledger, labelled by who made it, stops the agents at a budget (HTTP 402 once it is spent), and
keeps the endpoint's key on the host: the containers get only the meter's own client key, which works
nowhere else. With Fireworks, for example (prices in dollars per million uncached input, cached input
and output tokens, as the account's billing summary rates the model):

```bash
python3 -I $TVA/sandbox/meter.py serve --upstream https://api.fireworks.ai/inference \
    --key-file <file with the key> --price 0.30,0.006,1.20 --ledger run/meter.jsonl --budget 20 &
K=$(cat run/meter.jsonl.client-key); H=<the host's address, as the containers reach it>
export ANTHROPIC_BASE_URL=http://$H:18800/prover ANTHROPIC_API_KEY=$K
export TVA_REVIEWER_ANTHROPIC_BASE_URL=http://$H:18800/reviewer TVA_REVIEWER_ANTHROPIC_API_KEY=$K
export TVA_REVIEWER_MODEL=<model-id>
harbor run -p tasks/tva-prove-<kernel> -a claude-code -m anthropic/<model-id> -e docker
python3 -I $TVA/sandbox/meter.py report run/meter.jsonl      # at any time: requests, tokens, cost by label
```

The ledger is the account of a run's spend: the provider's own billing covers the whole account and
may lag by a day, and Claude Code prices its own use at Anthropic's rates (the reward reports a
reviewer cost only with Anthropic's endpoint; in the first Fireworks pilot Claude Code's estimate was
$15.28 for reviewers whose metered spend was $0.67).

## How it was tested

- `tva-kit` against TVA's twelve certified kernels, starting from each kernel's skeleton. `classify`
  proposes the certified kernel ranges for 10 of 12, and the certified host lines and unused functions
  for 6 of 12. The rest differ in host-side code of the big tutorial sources (autotuning configurations,
  tests, an activation helper called only under a meta-parameter), where the class is a judgment call, so the manual has the
  prover check the proposal before writing it. `map-skeleton` covers every code line of all 12.
- The real Claude Code CLI (2.1.316) in the prover's image, against a scripted stand-in for the
  Anthropic API, with the reviewers as Claude Code agents in the review service against the same
  stand-in. The manual, the skills, the subagents and the hooks load from `/`. A write into the library
  is refused with its reason. The prover's files build, the gate passes, and the submission goes
  through review. Both reviewers' filings are recorded, and the certificate is granted.
- The same through Harbor (`-a claude-code -m anthropic/<model>` with `ANTHROPIC_BASE_URL`), verifier
  included.

These tests show that the harness is wired correctly. They do not measure how much it helps a given
model. That takes real runs.
