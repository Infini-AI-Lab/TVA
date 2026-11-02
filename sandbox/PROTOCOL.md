# The proof–review protocol of a TVA sandbox task

A task asks an agent (the **prover**) to prove one Triton kernel in Lean 4 with the TVA library, and to
get the proof **certified** by independent **reviewer** agents through TVA's own review protocol
(`docs/REVIEW_PROTOCOL.md` in TVA, the tool `tools/tvr.py`). This file says how the sandbox runs that
protocol across containers: who runs where, what crosses between them, how a round goes, and what the
verifier checks.

## 1. Three containers

```
┌──────────── main (Harbor's agent container) ───────────┐        ┌────── reviewer (sidecar) ─────────────────────────────┐
│ the prover agent (root)                                 │  HTTP  │ review service (root)  /srv/sandbox/server.py :8765    │
│ TVA at ~/fpv/tva, the kernel reset to its skeleton      │ ─────▶ │ authority TVA at ~tva/fpv/tva                          │
│ tva-review (client): status, submit, wait, log, fetch   │ ◀───── │   library and tool: root's, read-only                 │
└─────────────────────────────────────────────────────────┘ record │   kernel folder, lean/.lake: `tva`'s (tvr runs as tva) │
                                                                    │ reviewer agents r-1, r-2: users rv1, rv2, each in      │
                                                                    │   its own copy ~rvN/fpv/tva (lean/.lake → authority's, │
                                                                    │   read-only)                                          │
                                                                    └──────────────┬────────────────────────────────────────┘
                                                                                   │ export (collect hook, after main stops)
                                                                    ┌──────────────▼──────── verifier (separate) ───────────┐
                                                                    │ fresh TVA image; the exported kernel folder put in;   │
                                                                    │ tvr check, tvr status, ledger and field checks        │
                                                                    └───────────────────────────────────────────────────────┘
```

All three images start from `tva-sandbox-base:<TVA commit>`: Ubuntu, the pinned Lean toolchain, Mathlib's
prebuilt cache, and the TVA library built, from the files TVA ships (no git history). The kernel's own
certified proof and review record are removed from the prover's and the reviewer's trees at image build
(`prepare_kernel.py`); the tree's other kernels stay, as the library's worked examples. The prover's and
the reviewer's images also carry the agent harness at `/` (HARNESS.md): a manual, hooks, skills,
subagents and `tva-kit`. Its hooks refuse some actions early with a reason, but the protocol does not
rely on them: the checks below are the service's.

**Start.** The service assigns the reviewers (`tvr assign`) and, with agent reviewers, first runs one
short agent call with the key and the model given: a bad key stops the reviewer container, and Harbor
fails the trial before the prover starts.

**What crosses.** From the prover to the service: submissions (a tar of the kernel's files, §3). From
the service to the prover: the status and the kernel's record (`REVIEW.md`, `claim.md`, `gate.json`,
`review/`), once a round is over. The reviewers' agents never talk to the prover; the prover sees only
what they file. From the service to the verifier: the exported record (`/export`). The reviewer key is
in the reviewer container's environment only, set from the host's `TVA_REVIEWER_ANTHROPIC_API_KEY`
(or `TVA_REVIEWER_ANTHROPIC_AUTH_TOKEN`, `TVA_REVIEWER_CLAUDE_CODE_OAUTH_TOKEN`; with
`TVA_REVIEWER_ANTHROPIC_BASE_URL` for another Anthropic-compatible endpoint).

## 2. The service's states

| Status | Meaning | Next |
|---|---|---|
| `ready` | the prover's turn | `tva-review submit` |
| `busy` | a submission is being gated or reviewed | `tva-review wait` |
| `certified` | the certificate is granted | stop (reward 1 if the verifier agrees) |
| `escalated` | a finding was re-raised twice: TVA sends it to a user, and no user decides here | none: no certificate can follow |
| `exhausted` | the limit of reviewed (default 6) or rejected (default 40) submissions is reached | none |
| `error` | the service or a reviewer agent failed (not the prover's doing) | none; the verifier reports `service_error: 1` |
| `stopped` | the agent phase ended (the export ran) | none |

One submission is processed at a time; a submission while `busy` is refused (HTTP 409).

## 3. A submission

The client packs, from `kernels/<K>/`: `lean/**.lean`, `README.md`, `model-map.json`, `kernel.json`, and,
from round 2, `review/round-N/prover/*` (N the current round; `.json .md .txt .lean .py`). The service
refuses anything else: other paths, links, devices, non-UTF-8 files, more than 400 files, a file over
2 MiB, more than 16 MiB in all.

Its **kind** follows from the ledger:
- **initial**: round 1, no reviewer has filed. It must carry no response.
- **response**: every reviewer has filed in round N and findings are open. It must carry
  `review/round-N/prover/response.json` (and the notes).
- In any other state the service takes no submission.

**Intake** (under the record lock):
1. Snapshot the authority's kernel folder.
2. Replace `lean/` wholly, `README.md` and `model-map.json`; rebuild `kernel.json` from the task's fixed
   fields (protocol, Lean module and namespace, source file and sha256, doc) and the prover's `title`,
   `source.ranges`, `source.host`, `source.unused`; copy the prover's round folder.
3. `tvr check K` as `tva`. TVA's gate reads the files first (manifest, source sha, hygiene: no meta code,
   no `#eval`, no `initialize`, no IO types, …; independence) and builds the kernel's Lean only when
   they pass; then it replays every declaration with `leanchecker`, reads the claim and writes
   `gate.json`, `claim.md` and a snapshot. The service checks that the ledger did not change during the
   gate.
4. For a response: `tvr lint K prover`, then `tvr respond K` (one batch answers every open finding).
5. Any failure in 3–4 **rejects** the submission: the snapshot is restored, nothing is recorded, and the
   prover gets the reason and the tool's output (`tva-review status`, `tva-review log`). A rejection
   counts against `max_rejected` only.

An accepted submission counts against `max_reviewed` and opens the round's review.

## 4. A round of review

1. **Copies.** For each reviewer that has not filed in round N, the service makes a fresh copy of the
   authority's tree in the reviewer's home (`~rvN/fpv/tva`; its `lean/.lake` links to the authority's
   build, read-only to the reviewer). All copies are made before any reviewer of the round files, so
   round 1 is blind as TVA requires: no reviewer sees another's round-1 filing.
2. **Review.** The reviewers run in parallel. Each gets TVA's own reviewer prompt
   (`tvr prompt K reviewer r-i`) followed by the sandbox's note (`service/reviewer-note.md`): the prover's
   files are evidence, not instructions; the gate has passed; check Lean on work files only
   (`lake env lean`); stop when `tvr lint K reviewer r-i` prints `regular`, without running `tvr file`.
   An agent is the Claude Code CLI, headless (`claude -p … --output-format json`), as the reviewer's
   user, with working directory `/` (where the harness is), resumed in later turns and rounds
   (`--resume`), with a time limit (default 90 minutes a turn).
3. **Recording.** The service copies the reviewer's round folder (`review/round-N/r-i/`) into the
   authority (only files not recorded before; a recorded file must be unchanged; no links; size limits)
   and runs `tvr file K r-i` as `tva`. If `tvr file` refuses, the files it added are removed again and the
   reviewer is sent back, in its session, with the refusal (at most 2 times).
4. **Completeness.** When every reviewer has filed, the service checks each reviewer's record: every
   item is accepted at its current digest or held by an unresolved blocking finding of that reviewer's,
   and every blocking finding of theirs the prover answered is released or re-raised. A reviewer whose
   record leaves something open gets the whole record (its copy of the kernel folder is refreshed, now
   that the round is no longer blind) and writes its next filing (`filing-2.json`); at most 2 times.
5. **Outcome.** Certificate granted → `certified`; a finding escalated → `escalated`; findings open →
   `ready` (or `exhausted`); a reviewer that cannot file a regular or complete review → `error`.

The prover then answers the open findings: fix, re-run the gate, `tvr start K prover`, fill in
`response.json` and `notes.md`, `tvr lint K prover`, `tva-review submit`. TVA's rules hold unchanged:
the responses (`REVISED` naming the changed items, `DISPUTED`, `CLARIFIED`, `DEFERRED`, `REFERRED` to a
library issue), the reviewers' releases and re-raises, the item digests that make an acceptance stale
when an item changes, and escalation after the second re-raise.

## 5. HTTP API (port 8765, reachable from the prover's container)

| Request | Answer |
|---|---|
| `GET /status` | kernel, status, round, TVA's phase, certificate, open findings, counts and limits, the last job, what to do next |
| `POST /submit` (a tar) | `202 {"job": n}`, or 409 (not `ready`), 413 (too large) |
| `GET /wait?timeout=S` | waits until the service is not `busy` (S ≤ 600), then the status |
| `GET /bundle` | `tar.gz` of `REVIEW.md`, `claim.md`, `gate.json`, `review/` |
| `GET /log?job=n` | the job's log: intake, the gate's output, the recordings, the outcome |
| `GET /health` | `{"ok": true}` |

Requests from inside the reviewer container (a reviewer agent) are refused, except `/health`.

## 6. Export and verification

When the agent phase ends, Harbor stops the prover's container and runs the collect hook in the
reviewer container: `server.py export /export` marks the service `stopped` and copies, under the
record lock (so between steps), the kernel folder, `state.json` (status, counts, costs, the ledger's
sha256 after every step the service took), the job logs and the reviewer agents' outputs and
transcripts. Harbor restores `/export` in the verifier's container, a fresh copy of the base image,
with no network. `verify.py`:

1. the record exists; the reviewers were agents (a task made with `--scripted` allows the scripted
   reviewer, and its reward says `scripted: 1`); the service ran the task's TVA commit;
2. puts the exported kernel folder into its own TVA tree; `kernel.json` keeps the task's fixed fields;
   the source is the task's (sha256);
3. the ledger's sha256 is the one the service recorded, its assignments are exactly the task's
   reviewers, and no other role acts in it;
4. `tvr check K` passes here, with this image's library;
5. `tvr status K` grants the certificate against this gate (every reviewer accepts every item at its
   digest, no blocking finding unresolved, every ledger event follows from its recorded file), and the
   service's `certificate.json` names this gate's digest and the task's reviewers.

`/logs/verifier/reward.json`: `reward` (1 when all checks pass), `gate_pass`, `certificate`,
`rounds_reviewed`, `rejected`, `findings`, `reviewer_cost_usd`, `service_error`, `escalated` (and
`scripted`). `/logs/verifier/details.json` has each check, the gate's output and the status text.

## 7. Trust, and what is not defended

- **Build-time code.** The prover's Lean is built in the reviewer container and in the verifier. TVA's
  gate refuses meta code before building (hygiene runs first), as in TVA's own trust model; the sandbox
  adds defence in depth: the build runs as `tva`, who cannot write the library, the tool, the service's
  state or the reviewers' homes, and the ledger digests the root service records would show a ledger
  changed behind its back.
- **Prompt injection.** The prover's files are what the reviewers read. The sandbox's note tells them that
  text addressed to them is evidence of a defect, not an instruction; nothing stronger is enforced.
- **Network.** Public, by decision: the prover's agent reaches its model and the web. TVA's
  independence rule (use only TVA and Mathlib) still holds for what the prover cites, and the gate's
  independence check and the reviewers enforce it on the files.
- **Examples.** The prover's image holds TVA's documents and the other kernels' certified proofs, by
  decision, as worked examples. The target kernel's own proof, record and build output are not in the
  image (some library-review material names declarations of its original proof).
- **No user.** TVA escalates a finding re-raised twice to a user. The sandbox has none, so escalation ends
  the task without a certificate.
