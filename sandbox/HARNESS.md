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
| `/CLAUDE.md` | the role's operating manual (prover: the path in eleven stages; reviewer: the procedure and its calibration) |
| `/.claude/settings.json` | the hooks and environment settings |
| `/.claude/skills/` | prover: `tva-model-map`, `tva-claim`, `tva-design`, `tva-proof`, `tva-gate`, `tva-respond`; reviewer: `tva-review-method`, `tva-filing` |
| `/.claude/agents/` | subagents: `tva-scout` (both roles), `lean-fixer` and `claim-audit` (prover) |
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

**The manual.** One page per role. The prover's manual gives the time limit (the task's agent timeout,
also in `harness.json` as `time_limit_sec`; `tva-kit stage` shows the time used) and a path in stages:
read, classes, model, map, claim, design, audit, lemmas, write-up, gate and submit, rounds. Each stage names its command and skill, and the
manual lists the worked examples to copy from: three certified kernels chosen for the task's kernel
(the table `EXAMPLES` in `make_task.py`), never the kernel itself. The reviewer's manual is a procedure in six steps, with calibration
rules: accept nothing unread, invent nothing, and do not move the goalposts.

**`tva-kit`.** Short commands that print little. They write the mechanical parts of the protocol and
check them the way `tvr` does (they import `tvr`'s own functions):

| Command | Role | What |
|---|---|---|
| `brief`, `stage` | prover | where you are: the stage checklist with what is done and what is next, and each `sorry` with its lemma (flagged when it sits inside a proof of more than 150 lines) |
| `outline [KERNEL] [FILE]` | both | a kernel's proof design: its Lean files bottom-up, each with its docstring and declarations (`sorry`s marked for one's own kernel); with FILE, that file's statements |
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
| `SessionStart` | the manual, the role's brief and the progress notes (at start, resume and compaction); the prover's clock starts at its first start |
| `SubagentStart` | every subagent (the harness's or Claude Code's own `general-purpose`) gets the role's rules and, for the prover, the Lean advice and the time left: the manual reaches only the main agent |
| `PreToolUse` | refuses a Write or Edit outside the role's folders (prover: the kernel folder and `~/scratch`; reviewer: its round folder and `~/scratch`), recording commands (`tvr respond`, `tvr file`, …), toolchain changes (`lake update`/`clean`, `lake exe cache`, `elan …`, deleting `.lake`), for reviewers `lake build`/`tvr check`/`tva-kit gate`, searches of `/` with no path, and waiting loops that keep a core busy (`read -t N < /dev/zero`). The tool's path is exact; a command is read without its here-documents, and never for the files it writes. Each refusal says why and what to do instead. Every subagent runs in the foreground: the hook sets `run_in_background` to false and says so (Claude Code 2.1.316 may run a subagent in the background when the field is absent), and `SendMessage` is refused: Claude Code resumes a subagent in the background, so the refusal says to start a new one with the earlier report |
| `PostToolUse` | after the prover edits a kernel file: JSON validity, the map check for `model-map.json`, and a build of the edited Lean module whose errors come back at once (`TVA_HARNESS_AUTOBUILD`, on by default), with the whole output kept in `~/scratch/build/<Module>.log`. After the prover's Bash command: the fixed files it changed (the tree outside the kernel's folder, and the kernel's source), compared with a snapshot taken at the session's start, each change reported once. A kernel module that Lake reports took a minute or more (`TVA_HARNESS_SLOW_BUILD`) gets the advice to split the file: in the automatic build, in `tva-kit build`, and in a Bash command that runs `lake build`, from its output or from a file it sent the output to and wrote while it ran (Lake prints each module's time when its output is not a terminal). After a subagent's report, which Claude Code ends with an invitation to continue it with `SendMessage`, the agent is told to start a new one instead. Every hour (and at three quarters and seven eighths of the limit), each of the prover's agents is told the time used and left, and while nothing is submitted, what that means: submit by three quarters, narrow what does not converge |
| `Stop` | the prover cannot stop while a review is running, findings are open, or the proof is unfinished, unless the service says the task is over; a reviewer cannot stop before `tvr lint` says `regular`. After 8 (prover) or 6 (reviewer) identical refusals it lets the agent stop, so a stuck model ends instead of looping |

The hooks guard against mistakes; they are not the security boundary. That is the review service: it
holds the authority copy of TVA, takes only whitelisted files from a submission, gates them itself, and
records the ledger. The verifier re-checks the record.

**Skills.** One per stage that needs know-how: how to transcribe and map, write a claim, design the
proof, prove its lemmas, pass the gate, and answer findings. The design skill (`tva-design`) is the
process the certified kernels follow: read the closest example's design (`tva-kit outline`); state
every lemma in layers (facts; the numbers, with each loop's invariant, step lemma and end lemma on values
only; the program walk; the composition), with `sorry` only as whole proofs and the walk and `main`
proved from the statements; have the claim and the design audited; then prove the lemmas, the hardest
first. `make_task.py` fills it with every certified kernel's design but the task's own (each file
bottom-up, with its title) and with real step lemmas of those kernels (`DESIGN_LEMMAS`). For the reviewer:
the review method part by part, with the question to ask and the finding kind for each defect, and
filing. Claude Code shows each skill's description to the model, which loads the skill when the stage
comes up.

**Subagents.** `tva-scout` answers one question about the library, the documents or the examples in a
few lines with file references. `lean-fixer` proves one stated lemma (or makes one Lean file build).
`claim-audit` checks the prover's claim and design before the long proofs (integer wrap-around under
`Inst.ok`, the tolerances at the extremes, each invariant's room for one trip's error, every `sorry` a
stated lemma) and writes `~/scratch/audit.md`, which stage 7 waits for. They keep long searches and
error loops out of the main context. They run in the foreground: the agent that starts one gets its
report as the call's result, and several started in one message run at once. None is resumed: to
continue one's work, the agent starts a new subagent with the earlier report.

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
- Claude Code 2.1.316 against a scripted API that asks for a subagent in the background and then
  writes a `read < /dev/zero` loop, with the clock set to 4.6 of 8 hours: the subagent ran in the
  foreground (its report was the call's result), it got the `SubagentStart` brief with the time, the
  loop was refused, and the prover was told the time once. Claude Code 2.1.316 blocks a foreground
  `sleep`, which is why the pilot's prover wrote such loops.
- The same, with the prover also trying to resume its subagent with `SendMessage` and running
  `lake build` with the output sent to a file (split threshold 0 for the test): the subagent's report
  came back with the note to start a new subagent instead, `SendMessage` was refused with that note,
  and the build got the advice to split, from Lake's time in the file. The hook alone: a slow kernel
  module in the output or in a file written while the command ran gets the advice; a library module, a
  module under the threshold, a file older than the command and a redirection inside a here-document
  do not.
- The design tools in the paged-attention task's image: `tva-kit outline attn_fwd_f16` gives its 13
  files bottom-up, and `tva-kit outline attn_fwd_f16 Row` the statement of `row_step_masked`; the task's
  own kernel is in neither the design skill's designs nor its lemmas. With the second paged pilot's
  final files in the kernel's folder, `tva-kit stage` holds stage 6 at the one `sorry`, inside the
  507-line `safe_program` (the certified proof's `safe_program` is 298 lines with no numeric reasoning:
  it calls `row_step`).

- The pilots on Fireworks (DeepSeek V4.1 Flash 1017): the provers used `tva-kit` in the early stages,
  the `lean-fixer` subagent for long proofs, and almost no skills; the stop hook kept them working
  four times, the build after each edit reported about 60 times in one run. A first version of the
  write guard also read each Bash command for the files it writes: it refused 32 commands in two runs,
  none of them a real violation (here-documents read as shell, `awk 'NR>=608'` read as a redirection,
  `cd` within a command ignored). It now checks only what is exact, and reports a changed fixed file
  after the command instead.
- What the paged-attention pilot (8 hours, timed out one proof short of the gate) taught, and what the
  harness does about it since:
  - the prover started its subagents in the background and then waited for them, 7.1 of its 7.9
    hours, in loops of `read -t 10 < /dev/zero` (Claude Code blocks `sleep`), each keeping a core busy
    and noticing a finished subagent up to 8 minutes late: subagents now run in the foreground, and
    such loops are refused;
  - the hard proof grew in one 1 700-line file built automatically 438 times, in 25 s to 5 minutes
    each (7.6 hours of builds of that file across the agents): the build's message now says to split
    a file that takes a minute;
  - the agents built again 28 times right after a failed automatic build, only to read a goal the
    summary had cut: the whole output is now kept in `~/scratch/build/`;
  - it never knew its time limit, and was still splitting its last proof when the limit came: the
    prover now knows the limit, is told the time every hour, and is told to narrow by half-time;
  - Claude Code's own `general-purpose` subagents did most of the proof without the manual: every
    subagent now gets the rules and the Lean advice when it starts.
- The second paged-attention pilot, with the harness above, ended the same way (one proof short, no
  submission), but with no waiting loop, and its builds took 4.2 hours instead of 7.4. It showed two
  gaps, closed since:
  - the prover resumed a subagent with `SendMessage`, as Claude Code's report invites, which runs it in
    the background, then polled `ListAgents` 12 times in 12 minutes: `SendMessage` is now refused, and
    each report comes with the note to start a new subagent instead;
  - the agents ran `lake build` in 495 Bash commands (52 of them took over a minute), mostly with the
    output sent to a file, and the advice after an edit never saw those builds: it now comes from
    Lake's own time for each module, in such a command's output or the file it wrote, as well as in
    the automatic build.
- In the third paged-attention pilot, Claude Code ran the prover's first subagent in the background
  although its call did not ask for it (the field was absent; Claude Code tests `run_in_background !==
  false` in some sessions). The hook now sets the field on every call; the running pilot's containers got
  the fixed hook 35 minutes in.
- The second paged pilot that used the design stage (DeepSeek) was certified. Its review showed two gaps, closed
  since: the reviewers could not run Lean (the Mathlib cache unpacks its build files with mode 600,
  readable by `tva` alone; the base image now makes them readable to every user), and the certificate
  was granted while round 2's `DOC` findings, four README sentences stating the round-1 tolerance
  constants, were open (TVA's new blocking kind `DOC_WRONG`, library issue L136; the reviewers' and
  prover's skills say when to use it).
  In the softmax pilot both reviewers lost 10 minutes to an exact `Fraction` computation of
  `(1 + 2^-24)^(2^20)`; the reviewer's manual now says to compute tolerances in floating point, under
  `timeout`.

These tests show that the harness is wired correctly. They do not measure how much it helps a given
model. That takes real runs.
