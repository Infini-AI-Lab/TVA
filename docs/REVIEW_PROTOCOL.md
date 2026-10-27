# Prover/reviewer protocol (`tvr-2`)

Who writes what, who checks what, and what each of them trusts. The tool is `tools/tvr.py` (run
it as `python3 -I tools/tvr.py …` from the TVA root). The agents' prompts are in
[docs/prompts/](prompts/): each role's prompt states its scope, and `tvr prompt` fills it in.

## 1. Two layers, four roles

A kernel's claim rests on two layers. Each has an author and reviewers, and each layer's
reviewers check only that layer.

| | **Library** (`lean/TVA`, `docs/`, `evidence/`) | **Kernel** (`kernels/<kernel>/`) |
|---|---|---|
| Author | **library author**: what each Triton construct means (`TVA.Lang`, `TVA.Sem`), how to transcribe source into a model ([TRANSCRIPTION.md](TRANSCRIPTION.md)), the compiler rule and the device facts with their constants (`TVA.Rules`), lemmas | **kernel prover** ([PROVER_GUIDE.md](PROVER_GUIDE.md)): the model, the model map, the conditions, the specification, the kernel's own assumptions, the proof, the write-up |
| Reviewer | **library reviewer** ([LIBRARY_REVIEW.md](LIBRARY_REVIEW.md)): checks the library against Triton 3.8.0 and the target GPU, using the compiler's sources, PTX and runs on the GPU | **kernel reviewer** ([REVIEWER_GUIDE.md](REVIEWER_GUIDE.md)): checks the prover's choices against the kernel's source and its real use, **trusting the library** |

Two more roles:
- **The coordinator** assigns reviewers, keeps the ledgers, and triages library reports into
  library issues.
- **The user** decides escalated disputes and questions of scope.

### The trust boundary

Kernel review takes the library as given, and kernel reviewers do not need to understand how a
library fact is established. Take "on sm_90, `tl.math.exp2` on `float32` is within `E` of `2^x`".
Its lowering and its measurement are library matters, settled in library review with evidence in
`evidence/`. Everything in the library is trusted in kernel review:
- the meaning of every `tl.*` operation, of launches and of outcomes;
- the transcription rules;
- `Compiler.Faithful`;
- the device facts;
- the lemmas.

A kernel reviewer who doubts the library files a `LIBRARY_REPORT`. It does not block the kernel;
it goes to the library's issue list (§10). A library fix re-opens exactly the kernel items whose
meaning reaches the changed constants (§4).

## 2. Scopes

### Kernel prover

- **Writes:**
  - the model, following the transcription rules, one rule per source construct;
  - the model map;
  - `Inst`, `Inst.ok`, `Pre`, `Post`;
  - the `Assumptions` structure;
  - `main`, `consistent`;
  - the write-up.
- **Assumptions are of two kinds only** (§5):
  - `[rule]`: a library rule or device fact, cited as the library states it;
  - `[kernel]`: an unproved fact about the kernel's own values.
- **No device or compiler facts of the kernel's own.** The prover never states a device or
  compiler fact or picks accuracy constants. If a kernel needs a fact the library lacks (say, the
  accuracy of an fp16 `tl.dot`), the prover asks the library author for it (a library issue).
  The kernel waits until the library has it.
- **Transcription gaps.** If the source uses a construct the transcription rules do not cover,
  the prover asks the library for the rule.

### Kernel reviewer

- **Trusts the library** (§1). Does not read the compiler's sources or PTX, does not consult other
  projects' material, and runs nothing on a GPU.
- **Judges:**
  - **the model against the source**, by the transcription rules, and the model map;
  - **the conditions**: they cover the kernel's real use (the host code, the autotune configs),
    and they are what the write-up says;
  - **the specification**: it says what the kernel is for, and a tolerance is meaningful at the
    sizes the conditions allow;
  - **each `[rule]` assumption**: it is the library's fact for the operation the model uses
    (format, configuration, function);
  - **each `[kernel]` assumption**: it is true, needed, and clearly stated;
  - **the write-up**: it matches the Lean.
- **Not in scope:**
  - whether the kernel compiles. A launch Triton rejects before it runs is outside every claim
    (§6);
  - how Triton lowers an operation;
  - whether a device fact's constants are right;
  - whether a `tl.*` definition matches Triton.

  These go to the library as `LIBRARY_REPORT`s.

### Library author

- Maintains the semantics, the transcription rules, `Compiler.Faithful`, and one device-fact
  page per target: per operation, its lowering, its constants and the evidence.
- Writes the evidence in `evidence/<topic>/`: a `README.md` (the question, the method, Triton's
  version, the GPU, the date, the result), the script, and its output.
- Triages library reports into issues, and fixes them in batches.

### Library reviewer

- Checks the library against Triton 3.8.0's documentation and compiler (the pinned sources in
  `primitives/`) and against the target GPU.
- May read any source, compile, inspect PTX and run measurements.
- A device fact is accepted on TVA's own evidence. Another project's measurement can suggest
  where to look, but it is not evidence.

## 3. The kernel folder

Everything about one kernel is in one folder. Every file has one author, and `tvr` records a
reviewer's or the prover's file only from that author's place. `kernels/README.md` (generated)
lists the kernels and where each one stands.

```
kernels/<kernel>/
  README.md                  prover    the write-up: Specification, Conditions, Assumptions, Proof, Narrowings
  kernel.json                prover    manifest: title, Lean module and namespace, source file, sha256, line ranges
  source/<file>.py           prover    the kernel source, pinned by its sha256
  lean/<Module>.lean, lean/<Module>/*.lean
                             prover    the Lean bundle (a lake library of its own: lean/lakefile.toml)
  model-map.json             prover    each source line → model text, or a drop reason (§7)
  claim.md                   tvr       the sheet the reviewers judge (tvr check)
  gate.json                  tvr       the gate's checks and the item digests (tvr check)
  REVIEW.md                  tvr       the review, readable: status, items, findings, every round in full
  review/
    round-1/
      <reviewer>/filing.json reviewer  findings, releases, re-raises, acceptances (§9)
      <reviewer>/report.md   reviewer  the review in words: verdict, what was checked, what was not
      <reviewer>/work/       reviewer  scratch files behind the findings (Lean checks, arithmetic)
      prover/response.json   prover    one response per open finding (§9)
      prover/notes.md        prover    what changed, in words
    round-2/ …
    ledger.jsonl             tvr       every event, in order, append only (REVIEW.md is made from it)
    snapshots/*.json         tvr       each passing gate's item cones (for "what changed")
    certificate.json         tvr       present exactly while the certificate is granted
```

A kernel folder holds no device evidence: device facts are the library's.

- **Rounds.** Round `N` opens when the prover has answered round `N − 1` (round 1 at the
  start). Each reviewer files in round `N` (a second filing in the same round is `filing-2.json`
  and `report-2.md`). Then the prover answers every open finding once, which closes the round.
- **Recorded files are frozen.** The ledger keeps the sha256 of every filing, report, response
  and notes file it records, and `REVIEW.md` flags any recorded file that changed later.
- **A new kernel**: `tvr new <kernel> --module <Module> --namespace <NS> --source <file>`.

## 4. Claim items and digests

The claim splits into items. A reviewer accepts items, not files.

| Item | Roots | What the kernel reviewer judges |
|---|---|---|
| `model` | `model`, `launch` (and the source's sha256) | the transcription, the launch, the block shapes |
| `map` | `model-map.json` and the source | that each source line corresponds to its model text |
| `conditions` | `Inst`, `Inst.ok`, `Pre` | coverage of real use |
| `spec` | `Post`, `Spec` | what the outputs are; that a tolerance is meaningful |
| `assumption:<f>` | the statement of field `f` and its docstring | `[rule]`: the right library fact; `[kernel]`: true, needed, clear |

Each item's digest covers its *cone*: every `TVA.*` and kernel constant its definitions reach,
by elaborated term (definition bodies included; theorems by statement only), plus the roots'
docstrings. So:
- a proof edit, a comment or a reformatting changes no digest;
- a library fix re-opens exactly the items whose meaning reaches the changed constants;
- `REVIEW.md` names the constants that changed since a reviewer accepted.

The claim reader (`tools/claim_reader.lean.in`) computes the cones from the elaborated Lean
claim. It never reads the kernel's source: there is no translator, and the model is the
prover's transcription.

## 5. Assumptions

Each field of `Assumptions K P` has a docstring that starts with its kind:

| Kind | What it is | Its statement may mention | Who stands behind it |
|---|---|---|---|
| `[rule]` | a library rule or device fact (`TVA.Rules.*`), as the library states it | the library's constants, `K`, `P`, and the kernel's `Inst`, `Inst.ok`, `model`, `launch` — no other kernel constant | the library (library review) |
| `[kernel]` | an unproved fact about the kernel's own values, e.g. a property of an error constant | the kernel's constants; neither `K` nor `P` | the kernel reviewer |

The gate checks the kinds and these limits (§8). So a kernel cannot slip its own constants into
a library rule, and a `[kernel]` assumption cannot be about the device or the compiler.

## 6. What a claim covers

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance and input, every outcome
of a launch of the compiled kernel is a normal end whose memory satisfies `Post`. A launch that
Triton or the driver rejects before any program runs has no outcome. Examples are too much
shared memory, too many registers, or a `tl.dot` below its minimum size; Triton reports these on
the host. So a claim never promises that the kernel compiles, and nobody asks for compile logs
in kernel review (user, 2026-10-27).

## 7. The model map

`model-map.json` is `{"entries": [...]}`, one entry per source line or line range of the kernel:

```json
{"src": [153], "model": "let accumulator ← tl.dot_add cfg a b accumulator", "note": "…"}
{"src": [189], "drop": "config", "note": "FP8_OUTPUT = False on every covered instance"}
```

- `model` is a string or a list of strings. Each must occur, up to whitespace, in the kernel's
  Lean files.
- `drop` is one of these (each is a rule of [TRANSCRIPTION.md](TRANSCRIPTION.md)):
  - `decl`: a decorator or signature line, covered by the model's definition;
  - `no-op:debug_barrier`, `no-op:print`, `no-op:cache`, `no-op:range-kw`: the no-ops of the
    transcription rules;
  - `config`: a branch not taken in the covered configuration. The `note` names the condition;
  - `compile-check`: a compile-time assertion (`tl.static_assert`) that holds on every covered
    instance. The `note` names the condition of `Inst.ok` that makes it hold;
  - `dead`: a pure value no covered path uses. The `note` says why;
  - `comment`.
- The gate fails when a code line of the kernel's ranges is not covered, when model text is not
  found, or when a drop reason is not on the list.

## 8. The gate (`tvr check <kernel>`)

| Check | Passes when |
|---|---|
| `source` | the kernel source has the manifest's sha256 |
| `build` | `lake build <module>` succeeds |
| `hygiene` | the kernel's Lean files contain no `sorry`, `admit`, `native_decide`, `axiom`, `extern`, `implemented_by`, `unsafe` or `set_option debug.*` |
| `shape` | the claim reader elaborates: `main : ∀ K P, Assumptions K P → Spec K`, `consistent : Claim.Consistent Assumptions`, `Spec K = Claim.Meets Inst.ok Pre Post K` |
| `axioms` | `main` and `consistent` depend only on `propext`, `Classical.choice`, `Quot.sound` |
| `usage` | `main` uses every assumption, through its projection (no destructuring of `Assumptions`) |
| `namespace` | the kernel declares nothing in `TVA` and no instance of a `TVA` class |
| `kinds` | every assumption is `[rule]` or `[kernel]` and keeps to its kind's limits (§5) |
| `map` | §7 |
| `doc` | the write-up `README.md` has the sections Specification, Conditions, Assumptions, Proof |

The gate writes `gate.json` and `claim.md`:
1. the claim;
2. the source;
3. the conditions;
4. the specification;
5. the assumptions, each with its kind and the library fact it cites;
6. the model, with the rendered map;
7. the gate.

Declarations are quoted from their files by Lean's declaration ranges.

## 9. Kernel review

### The flow

1. **The prover** makes the gate pass and asks for review.
2. **The coordinator** assigns reviewers: `tvr assign <kernel> <reviewer> [--focus model|claim|all]`.
   They start each agent with its role's prompt (`tvr prompt <kernel> reviewer <id>`, §11) and
   may record notes (`tvr note <kernel> <text> [--files …]`).
3. **Round 1 (blind).** Each reviewer reads `claim.md`, the source, the Lean files, the write-up
   and the library pages the claim cites. They write `review/round-1/<reviewer>/filing.json`,
   `report.md` and any `work/`, and record the filing with `tvr file <kernel> <reviewer>`. Until
   every assigned reviewer has filed, `REVIEW.md` and `tvr status` hide round 1's findings
   (`--as <reviewer>` shows one's own).
4. **The prover answers every open finding in one batch** (the tool refuses a partial batch):
   fix, re-run the gate, write `review/round-N/prover/response.json` and `notes.md`, then run
   `tvr respond <kernel>`. This closes round `N`.
5. **Round N + 1.** Each reviewer re-reads what changed (`REVIEW.md` names the changed
   constants per item), then files in `review/round-<N+1>/<reviewer>/`: release settled
   findings, re-raise the others, raise new ones, accept items.
6. Steps 4 and 5 repeat until the certificate is granted or a finding is escalated.
7. **Escalation.** A finding re-raised twice goes to the user (`ESCALATED`). The coordinator
   writes a short memo with both positions (`tvr note`), and the user's decision is recorded
   with `tvr decide <kernel> <id> <text>`.

### Findings

| Kind | Target | Blocks | Meaning |
|---|---|---|---|
| `MODEL_MISMATCH` | `item:model`, `map:<line>`, `src:<line>`, `decl:<name>` | yes | the model is not the source under the transcription rules |
| `MAP_INCOMPLETE` | `item:map`, `map:<line>`, `src:<line>` | yes | a source line's map entry is missing, wrong, or drops what may not be dropped |
| `UNSUPPORTED` | `src:<line>`, `map:<line>` | yes | the source uses a construct the transcription rules do not cover; resolved in the library |
| `CONDITION_TOO_NARROW` | `item:conditions`, `decl:<name>` | yes | `Inst.ok`/`Pre` exclude real use |
| `CONDITION_WRONG` | `item:conditions`, `decl:<name>` | yes | unsatisfiable, or not what the write-up says |
| `SPEC_TOO_WEAK` | `item:spec`, `decl:<name>` | yes | `Post` does not say what the kernel is for, or a tolerance is meaningless at covered sizes |
| `SPEC_WRONG` | `item:spec`, `decl:<name>` | yes | `Post` says something other than the kernel's purpose |
| `RULE_MISCITED` | `item:assumption:<f>`, `[rule]` only | yes | not the library's fact for the operation the model uses |
| `ASSUMPTION_FALSE`, `ASSUMPTION_TOO_STRONG`, `ASSUMPTION_UNCLEAR` | `item:assumption:<f>`, `[kernel]` only | yes | the kernel's own assumption is false, stronger than needed, or unclear |
| `EVIDENCE_REQUEST` | `item:assumption:<f>`, `[kernel]` only | yes | a proof or an argument for the kernel's own assumption |
| `LIBRARY_REPORT` | `decl:<TVA name>`, `lib:<path>`, `src:<line>` | no | a doubt about the library; listed in `docs/library-review/REPORTS.md` for triage |
| `DOC`, `QUESTION` | any, or `doc:<path>` | no | wording, or a question |

- `decl:<name>` must be a constant in some item's cone. `src:`/`map:` lines must be in the
  manifest's ranges. `lib:`/`doc:` paths must exist.
- A finding is *about* some items:
  - an `item:` target names one;
  - `map:`/`src:` lines are about `model` and `map`;
  - a `decl:` target is about the items whose cone contains the constant;
  - any finding may list the items it `affects` (`"affects": ["item:model"]`).
- A reviewer cannot accept an item while one of their own blocking findings about it is
  unresolved.

**States**:
- a raised finding is `OPEN`;
- the prover's response makes it `ANSWERED`, or `REFERRED` if the response refers it to the
  library;
- a re-raise by a raiser makes it `OPEN` again;
- it is `CLOSED` once every raiser has released it, or by the user's decision;
- after the second re-raise it is `ESCALATED`.

`REFERRED` does not block.

**Responses**:

| Response | Meaning | What the tool checks |
|---|---|---|
| `REVISED` | the kernel changed | each item in `changed` has a different digest than when the finding was last raised |
| `DISPUTED` | an argument; nothing changed | |
| `CLARIFIED` | wording or the write-up only | |
| `DEFERRED` | kept as a stated narrowing or limitation | the reviewer agrees by releasing |
| `REFERRED` | the finding is about the library: it becomes library issue `L<n>`, named in the text | the issue exists in `ISSUES.md`; the finding stops blocking; the reviewer may re-raise it if it is about the kernel after all |

### Filings and responses

A reviewer's filing, `review/round-N/<reviewer>/filing.json` (recorded by
`tvr file <kernel> <reviewer>`):

```json
{"reviewer": "r-gemm",
 "findings": [{"kind": "SPEC_TOO_WEAK", "target": "item:spec", "summary": "…", "detail": "…",
               "evidence": "…", "blocking": true}],
 "release": [{"id": "F2", "note": "the revision settles it"}],
 "reraise": [{"id": "F3", "note": "the response does not address …"}],
 "accept": ["item:conditions", "item:assumption:dotTotal"],
 "notes": "non-blocking remarks"}
```

- `accept` may be `"all"`. Acceptance records each item's current digest, and it goes `STALE`
  when the digest changes.
- Only an `ANSWERED` or `REFERRED` finding can be re-raised.

The prover's responses, `review/round-N/prover/response.json` (recorded by `tvr respond <kernel>`):

```json
{"responses": [{"id": "F1", "response": "REVISED", "text": "…", "changed": ["item:spec"]}]}
```

The tool refuses three things:
- a stale gate (one older than a Lean file, the map or the manifest);
- a response before every reviewer has filed in the round;
- a batch that leaves an open finding unanswered.

### The certificate

The certificate is granted, and `review/certificate.json` written, when all of these hold:
- the gate passes;
- every assigned reviewer has accepted every item at its current digest;
- no blocking finding is unresolved.

It holds under the library as it stands. `REVIEW.md` lists the open library issues that touch the
kernel, and a library fix re-opens the affected items.

## 10. Library issues and library review

```
docs/library-review/
  ISSUES.md               coordinator   the library issues: L<n>, state, source, summary, resolution, constants touched
  REPORTS.md              tvr           every LIBRARY_REPORT and REFERRED finding from kernel review (generated)
  round-<N>/author/notes.md
                          author        a batch of fixes: what changed, why, the evidence
  round-<N>/<reviewer>/report.md
                          reviewer      the library reviewer's verdict on each issue in the batch
evidence/<topic>/         author        README.md, the script, its output
```

- **Sources of issues.** An issue comes from a kernel finding (`LIBRARY_REPORT`, or a response
  `REFERRED`), from library review, or from the author. The coordinator records it in
  `ISSUES.md` with its source, for example `grouped_gemm_f16#F2`.
- **States.** An issue is `OPEN`, `FIXED` (in a batch, with the constants it changed), `ACCEPTED`
  (a library reviewer accepted the fix), or `DECLINED` (with the reason, and the user's decision
  where the scope changes).
- **Fixes.** The author fixes issues in batches (one round each) with evidence, and a library
  reviewer who did not write the fix checks the batch.
- **Effect on kernels.** Each kernel item's digest covers the library constants it reaches, so
  a fix re-opens exactly the affected kernel items. Their reviewers re-accept them in the next
  kernel round.

## 11. Prompts

Each agent starts from its role's prompt in [docs/prompts/](prompts/):

| Prompt | Role | Fills in |
|---|---|---|
| [kernel-prover.md](prompts/kernel-prover.md) | kernel prover | the kernel, the round, what to answer |
| [kernel-reviewer.md](prompts/kernel-reviewer.md) | kernel reviewer | the kernel, the reviewer id, the round |
| [library-author.md](prompts/library-author.md) | library author | the issues of the batch |
| [library-reviewer.md](prompts/library-reviewer.md) | library reviewer | the batch (round), the reviewer id |

`tvr prompt <kernel> prover|reviewer [<id>]` prints a kernel prompt with the kernel's name, the
current round and the reviewer's id filled in. The coordinator adds nothing about the content of
the claim: what the agent reads is in the kernel's folder and the library, and an agent handed
the coordinator's view of the claim reviews that view instead of the claim.

## 12. Why it is built this way

- **Scope.** Kernel reviewers trust a separately reviewed library, so device and compiler
  knowledge is checked once, in library review, not again for every kernel.
- **Per-item digests computed in Lean.** An acceptance survives proof edits and changes to other
  items, and a library change re-opens only what it reaches.
- **One ledger and one folder per kernel.** Findings have explicit states, the tool enforces
  the transitions, and `REVIEW.md` renders the whole record beside the kernel's files.
- **Checked by Lean and by the gate.** The claim's shape is checked by Lean, not by text
  matching. The model map is checked for coverage, and the assumption kinds for ownership.
- **Fixed prompts.** Every agent of a role starts from the same prompt, which carries the scope.
