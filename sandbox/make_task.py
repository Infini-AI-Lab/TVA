#!/usr/bin/env python3
"""Generate the Harbor task in which agents prove and review one TVA kernel (PROTOCOL.md).

  sandbox/make_task.py KERNEL [--out DIR] [--tva DIR] [--scripted] [--reviewers N] [--model M]
               [--agent-timeout S] [--max-reviewed N] [--max-rejected N] [--cpus N] [--memory-mb N]
               [--context-window N] [--no-autobuild]

writes OUT/tva-prove-KERNEL/ (OUT/tva-prove-KERNEL-scripted/ with --scripted, whose reviewers are the
scripted plumbing reviewer that accepts everything unread, and whose verifier allows it). OUT is
`tasks` in the working directory by default, and must be outside TVA: the gate reads every file of
TVA's tree. The kernel is the one in TVA's tree at its commit (`--tva`, by default the tree that holds
this file). A certified kernel's task carries its certified proof as the oracle solution; a kernel
without a certificate has none. The task's images start from tva-sandbox-base:<commit>, which
build_base.sh builds from TVA's tree at the last commit that changed anything outside sandbox/.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVICE = os.path.join(HERE, "service")
VERIFIER = os.path.join(HERE, "verifier")
HARNESS = os.path.join(HERE, "harness")

# The certified kernels each kernel's prover is pointed to as worked examples, closest first.
EXAMPLES = {
    "vec_add_f32": ["softmax_f32", "attn_bwd_pre_f16", "paged_attention_f16"],
    "softmax_f32": ["vec_add_f32", "attn_bwd_pre_f16", "paged_attention_f16"],
    "attn_bwd_pre_f16": ["vec_add_f32", "softmax_f32", "attn_bwd_f16"],
    "matmul_f16": ["matmul_e5m2", "grouped_matmul_tma_f16", "vec_add_f32"],
    "matmul_e5m2": ["matmul_f16", "block_scaled_matmul_mxfp8", "vec_add_f32"],
    "grouped_matmul_tma_f16": ["matmul_f16", "block_scaled_matmul_mxfp8", "vec_add_f32"],
    "block_scaled_matmul_mxfp8": ["matmul_e5m2", "matmul_f16", "grouped_matmul_tma_f16"],
    "streamk_matmul_f16": ["matmul_f16", "vec_add_f32", "softmax_f32"],
    "paged_attention_f16": ["attn_fwd_f16", "softmax_f32", "vec_add_f32"],
    "attn_fwd_f16": ["attn_fwd_e5m2", "paged_attention_f16", "softmax_f32"],
    "attn_fwd_e5m2": ["attn_fwd_f16", "matmul_e5m2", "softmax_f32"],
    "attn_bwd_f16": ["attn_fwd_f16", "attn_bwd_pre_f16", "matmul_f16"],
}


# Lemmas of certified kernels that show a proof's design at work (skill tva-design): (kernel, file, lemma, what)
DESIGN_LEMMAS = [
    ("attn_fwd_f16", "Row", "row_step_masked", "one block of an online softmax keeps the row's state "
     "(running maximum, rescaled sum, accumulator): one hypothesis per model line of the loop body"),
    ("attn_fwd_f16", "Inner", "safe_loop_masked", "the loop's walk, which calls `row_step_masked` at each trip"),
    ("paged_attention_f16", "Facts", "row_step", "one page of an online softmax over a block table"),
    ("attn_fwd_e5m2", "Row", "row_step_masked", "the same with `float8` values"),
    ("attn_bwd_f16", "Step", "kv_step", "one step of the backward pass's loop for `dk` and `dv`"),
    ("matmul_f16", "Basic", "acc_step", "one step of a `tl.dot` accumulation along K"),
    ("matmul_f16", "Program", "safe_program", "the K loop's walk, with `acc_step` at each trip"),
    ("matmul_e5m2", "Basic", "acc_step", "the same with `float8` chunks"),
    ("block_scaled_matmul_mxfp8", "Basic", "acc_step", "the same with block-scaled `tl.dot_scaled`"),
    ("softmax_f32", "Err", "row_bound", "a row's error without a loop state (max, exp, sum, division)"),
    ("softmax_f32", "Proof", "safe_row", "one trip's walk, ending in `row_bound`"),
    ("attn_bwd_pre_f16", "Err", "row_bound", "a row's error: an elementwise product and a sum"),
]


def lean_order(tva, e):
    """A kernel's Lean modules bottom-up (each after the ones it imports), each with its file's title."""
    root = os.path.join(tva, "kernels", e, "lean")
    files = {os.path.relpath(p, root)[:-5].replace(os.sep, "."): p
             for p in glob.glob(os.path.join(root, "**", "*.lean"), recursive=True)}
    imports = {m: [x for x in re.findall(r"^import\s+(\S+)", open(p).read(), re.M) if x in files] for m, p in files.items()}
    order = []

    def visit(m):
        if m not in order:
            for x in sorted(imports[m]):
                visit(x)
            order.append(m)
    for m in sorted(files):
        visit(m)
    titles = {}
    for m in order:
        t = re.search(r"/-!\s*#\s*(.*)", open(files[m]).read())
        titles[m] = re.sub(r"^`[^`]*`:\s*", "", t.group(1).strip()) if t else ""
    return [(m, titles[m]) for m in order]


def design_fields(tva, k, ex):
    """The skill tva-design's examples: the certified kernels' designs (the task's examples first) and the
    lemmas of DESIGN_LEMMAS that are in them; the task's own kernel is never among them."""
    have = [d for d in sorted(os.listdir(os.path.join(tva, "kernels")))
            if os.path.isfile(os.path.join(tva, "kernels", d, "review", "certificate.json")) and d != k]
    kernels = [e for e in ex if e in have] + [e for e in have if e not in ex]
    designs = []
    for e in kernels:
        mods = lean_order(tva, e)
        designs.append(f"- `{e}`: " + " → ".join(f"{m.split('.')[-1]}" + (f" ({t[:70]})" if t else "") for m, t in mods))
    lemmas = [f"- `{e}`, `{f}.lean`, `{n}`: {what} (`tva-kit lookup {n}`, or `tva-kit outline {e} {f}`)"
              for e, f, n, what in DESIGN_LEMMAS if e in have]
    return {"{{designs}}": "\n".join(designs) or "- (none)", "{{design_lemmas}}": "\n".join(lemmas) or "- (none)"}


def examples_for(tva, k):
    have = [d for d in sorted(os.listdir(os.path.join(tva, "kernels")))
            if os.path.isfile(os.path.join(tva, "kernels", d, "review", "certificate.json")) and d != k]
    pick = [e for e in EXAMPLES.get(k, []) if e in have]
    return (pick + [e for e in ("vec_add_f32", "softmax_f32", "matmul_f16") if e in have and e not in pick])[:3]


def entry_kernels(tva, k, man):
    """The launched kernel: the JIT functions of the kernel lines that no other one there calls."""
    sys.path.insert(0, os.path.join(tva, "tools"))
    import tvr
    text = open(os.path.join(tva, "kernels", k, man["source"]["file"])).read()
    lines, jit = text.split("\n"), tvr.jit_functions(text) or {}
    inr = {i for a, b in man["source"]["ranges"] for i in range(a, b + 1)}
    mine = [n for n, sp in jit.items() if all(set(range(a, b + 1)) <= inr for a, b in sp)]
    called = {g for f in mine for a, b in jit[f] for g in mine
              if g != f and re.search(rf"\b{re.escape(g)}\s*\(", "\n".join(lines[a - 1:b]))}
    return [f for f in mine if f not in called]


def harness_files(E, tva, k, man, cfg, setup, a):
    """The harness (HARNESS.md) in the build context: environment/harness/, installed at / in each image."""
    H = os.path.join(E, "harness")
    ex = examples_for(tva, k)
    ex_man = json.load(open(os.path.join(tva, "kernels", ex[0], "kernel.json"))) if ex else None
    fields = {
        "{{kernel}}": k, "{{module}}": man["lean"]["module"], "{{namespace}}": man["lean"]["namespace"],
        "{{module_path}}": man["lean"]["module"].replace(".", "/"),
        "{{examples}}": ", ".join(f"`kernels/{e}/`" for e in ex) or "none",
        "{{example}}": ex[0] if ex else "vec_add_f32",
        "{{example_file}}": (f"~/fpv/tva/kernels/{ex[0]}/lean/{ex_man['lean']['module'].replace('.', '/')}.lean"
                             if ex else "an example's Lean file"),
        "{{context_window}}": str(a.context_window), "{{autobuild}}": "0" if a.no_autobuild else "1",
        "{{reviewers}}": {1: "one reviewer agent", 2: "two reviewer agents", 3: "three reviewer agents",
                          4: "four reviewer agents"}[a.reviewers],
        "{{time_limit}}": (f"{a.agent_timeout // 3600} hours" if a.agent_timeout % 3600 == 0
                           else f"{a.agent_timeout // 60} minutes"),
        **design_fields(tva, k, ex),
    }

    def render(src, dst, extra=None):
        text = open(src).read()
        for key, v in {**fields, **(extra or {})}.items():
            text = text.replace(key, v)
        if "{{" in text:
            sys.exit(f"{src}: a field this generator does not fill: {re.findall(r'{{\w+}}', text)}")
        write(dst, text)

    for f in ("tvakit.py", "hook.py"):
        os.makedirs(os.path.join(H, "lib"), exist_ok=True)
        shutil.copyfile(os.path.join(HARNESS, f), os.path.join(H, "lib", f))
    write(os.path.join(H, "tva-kit"), "#!/bin/sh\nexec python3 -I /usr/local/lib/tva-harness/tvakit.py \"$@\"\n", 0o755)
    copy(os.path.join(HARNESS, "lake"), os.path.join(H, "lake"), 0o755)
    entry = entry_kernels(tva, k, man)
    roles = {
        "prover": ({"role": "prover", "kernel": k, "tva": "/home/tva/fpv/tva", "entry": entry, "examples": ex,
                    "service": "http://reviewer:8765", "time_limit_sec": a.agent_timeout},
                   {"{{setup}}": "`SETUP.md`, " if setup else "",
                    "{{setup_read}}": ", `SETUP.md` (the launch the kernel is checked for)" if setup else ""},
                   ["tva-scout.md", "lean-fixer.md", "claim-audit.md"]),
        "reviewer": ({"role": "reviewer", "kernel": k, "entry": entry, "examples": ex,
                      "reviewers": {f"rv{i + 1}": r for i, r in enumerate(cfg["reviewers"])}},
                     {"{{setup}}": ", `SETUP.md`" if setup else ""}, ["tva-scout.md"]),
    }
    for role, (conf, extra, agents) in roles.items():
        R = os.path.join(H, role)
        write(os.path.join(R, "harness.json"), json.dumps(conf, indent=2) + "\n")
        render(os.path.join(HARNESS, role, "CLAUDE.md"), os.path.join(R, "CLAUDE.md"), extra)
        render(os.path.join(HARNESS, "settings.json"), os.path.join(R, ".claude", "settings.json"), extra)
        for s in sorted(os.listdir(os.path.join(HARNESS, role, "skills"))):
            render(os.path.join(HARNESS, role, "skills", s, "SKILL.md"), os.path.join(R, ".claude", "skills", s, "SKILL.md"), extra)
        for g in agents:
            render(os.path.join(HARNESS, "agents", g), os.path.join(R, ".claude", "agents", g), extra)
    return ex, entry


# Claude Code, pinned, in both agent images: one RUN line, so that the prover's and the reviewer's images share
# the layer, and Harbor's claude-code agent finds `claude` in the prover's image instead of installing it
CLAUDE_CODE = """RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm \\
    && rm -rf /var/lib/apt/lists/* \\
    && npm install -g @anthropic-ai/claude-code@2.1.316 && npm cache clean --force"""


def harness_install(role):
    """Dockerfile lines that put a role's harness at the image's root (/CLAUDE.md, /.claude/) and its tools in
    /usr/local; the agent starts in /, so Claude Code loads it as the project's configuration."""
    return f"""COPY harness/lib/ /usr/local/lib/tva-harness/
COPY harness/tva-kit /usr/local/bin/tva-kit
COPY harness/{role}/harness.json /etc/tva-sandbox/harness.json
COPY harness/{role}/CLAUDE.md /CLAUDE.md
COPY harness/{role}/.claude/ /.claude/
RUN chmod -R a+rX /usr/local/lib/tva-harness /.claude /CLAUDE.md /etc/tva-sandbox && chmod 755 /usr/local/bin/tva-kit""" + (
        "" if role != "prover" else """
# One `lake` at a time on the shared build folder (harness/lake): the wrapper takes elan's proxy's place.
COPY harness/lake /tmp/tva-lake
RUN mv /opt/elan/bin/lake /opt/elan/bin/lake.elan && install -m 755 /tmp/tva-lake /opt/elan/bin/lake && rm /tmp/tva-lake \\
    && touch /tmp/tva-lake.turn /tmp/tva-lake.lock && chmod 666 /tmp/tva-lake.turn /tmp/tva-lake.lock""")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def write(path, text, mode=0o644):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, mode)


def copy(src, dst, mode=0o644):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    os.chmod(dst, mode)


def sub(text, old, new):
    if old not in text:
        sys.exit(f"the prover prompt changed: no {old[:60]!r}")
    return text.replace(old, new)


def instruction(tva, k, man, cfg, setup):
    t = open(os.path.join(tva, "docs", "prompts", "kernel-prover.md")).read()
    t = re.sub(r"\]\(\.\./([^)]+)\)", lambda m: f"](docs/{m.group(1)})", t)
    t = sub(t, "This is **round {{round}}**.", "You start at **round 1**; `tva-review status` gives the current round.")
    t = sub(t, "make the gate pass, and\n  stop. There is nothing to respond to; your final message asks for review.",
            "make the gate pass, and\n  submit it (`tva-review submit`). There is nothing to respond to yet.")
    t = sub(t, "`review/round-{{round}}/prover/`", "`review/round-N/prover/` (N the current round)")
    t = sub(t, "`review/round-{{round}}/*/report.md`", "`review/round-N/*/report.md`")
    t = sub(t, "5. Record: `python3 -I tools/tvr.py respond {{kernel}}`.",
            "5. Submit: `tva-review submit`, then `tva-review wait` until the round is over. The review service\n"
            "   records your response (`tvr respond`); never run `tvr respond` yourself.")
    t = t[:t.index("## Final message")].rstrip("\n")
    t = t.replace("{{kernel}}", k)
    if "{{" in t:
        sys.exit("the prover prompt has a field this generator does not fill")
    lean = man["lean"]
    setup_line = " `SETUP.md` (the launch the kernel is checked for, and the launches to cover)," if setup else ""
    service = f"""

## In this task: the review service

This task runs TVA's protocol between you and {len(cfg['reviewers'])} reviewer agents
({', '.join(f'`{r}`' for r in cfg['reviewers'])}) that work in another container, run by a review service.
You reach the service only with `tva-review` (`tva-review --help`).

- **Your tree.** TVA is at `~/fpv/tva`; the documents' paths hold. Your kernel's folder,
  `kernels/{k}/`, holds the source,{setup_line} and the skeletons of `kernel.json`,
  `model-map.json` and `README.md`; `lean/` is empty. In `kernel.json` you set the title and
  `source.ranges`, `source.host` and `source.unused`; the service keeps the rest (the protocol, the Lean
  module `{lean['module']}` and namespace `{lean['namespace']}`, the source file and its sha256).
- **Submitting.** `tva-review submit` sends `lean/**.lean`, `README.md`, `model-map.json`, `kernel.json`
  and, from round 2, `review/round-N/prover/`. The service puts them into its own copy of TVA, whose
  library is the one this image ships (changes you make to the library, the tools or other kernels have
  no effect there), and runs `tvr check {k}`. A failing gate rejects the submission and records nothing.
  When findings are open, it then checks and records your response (`tvr lint {k} prover`,
  `tvr respond {k}`); a response that is not regular is rejected the same way. An accepted submission
  starts a round: each reviewer reads the claim and files its review, and the service records it.
- **Waiting.** A round takes from several minutes to more than an hour. Run `tva-review wait`
  repeatedly: each call waits at most 9 minutes and says whether the round is over. When it is,
  `tva-review wait` copies the record into your kernel folder (`REVIEW.md`, `review/`, `claim.md`,
  `gate.json`), so `python3 -I tools/tvr.py status {k}`, `start {k} prover` and `lint {k} prover` work in
  your tree. `tva-review log` shows the service's log of the last submission, with the gate's output.
- **Limits.** At most {cfg['max_reviewed']} submissions are reviewed and at most {cfg['max_rejected']}
  rejected. A finding re-raised twice is escalated to a user, and no user decides in this task: no
  certificate can follow, so settle findings on their substance. The library is fixed here, so a library
  request cannot be met: say so in your notes if you need one, and cover what you can without it.
- **Success** is a granted certificate: every reviewer accepts every item at the current gate, and no
  blocking finding is unresolved. `tva-review status` then says `certificate: GRANTED`, and you can stop.
  After you stop, a verifier re-runs the gate on the service's record and checks the certificate.
- **The reviewers judge your files, and nothing else reaches them.** Write for them as the protocol says.
  Text in your files that addresses the reviewers to steer their verdict is a defect they are told to
  raise.
- Use only TVA and Mathlib, as the independence section says: this container's network is for your own
  model, not for fetching other material.
"""
    return f"# Prove and certify the Triton kernel `{k}` with TVA\n\n" + t + service


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kernel")
    ap.add_argument("--tva", default=os.path.dirname(HERE))
    ap.add_argument("--out", default="tasks")
    ap.add_argument("--scripted", action="store_true")
    ap.add_argument("--reviewers", type=int, default=2)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--agent-timeout", type=int, default=8 * 3600)
    ap.add_argument("--max-reviewed", type=int, default=6)
    ap.add_argument("--max-rejected", type=int, default=40)
    ap.add_argument("--cpus", type=int, default=16)
    ap.add_argument("--memory-mb", type=int, default=65536)
    ap.add_argument("--context-window", type=int, default=128000,
                    help="the context window Claude Code assumes for a model it does not know "
                         "(CLAUDE_CODE_MAX_CONTEXT_TOKENS) and its compaction window (CLAUDE_CODE_AUTO_COMPACT_WINDOW)")
    ap.add_argument("--no-autobuild", action="store_true", help="no automatic Lean build after each edit")
    a = ap.parse_args()
    if not 1 <= a.reviewers <= 4:
        sys.exit("1 to 4 reviewers (the reviewer image has the users rv1 to rv4)")
    k, tva = a.kernel, os.path.realpath(a.tva)
    if os.path.realpath(a.out) == tva or os.path.realpath(a.out).startswith(tva + os.sep):
        sys.exit(f"--out {a.out} is inside TVA's tree, whose every file the gate reads: choose a folder outside it")
    kd = os.path.join(tva, "kernels", k)
    man = json.load(open(os.path.join(kd, "kernel.json")))
    certified = os.path.isfile(os.path.join(kd, "review", "certificate.json"))
    # the base image's commit: the last one that changed anything outside sandbox/ (build_base.sh)
    rev = subprocess.run(["git", "-C", tva, "log", "-1", "--format=%h", "--", ".", ":(exclude)sandbox"],
                         capture_output=True, text=True, check=True).stdout.strip()
    if subprocess.run(["git", "-C", tva, "status", "--porcelain", "--", f"kernels/{k}"], capture_output=True,
                      text=True, check=True).stdout.strip():
        sys.exit(f"kernels/{k} has uncommitted changes: the images hold TVA's tree at its commit, so commit them first")
    base = f"tva-sandbox-base:{rev}"
    name = f"tva-prove-{k}" + ("-scripted" if a.scripted else "")
    out = os.path.join(a.out, name)
    if os.path.exists(out):
        shutil.rmtree(out)
    reviewers = [f"r-{i + 1}" for i in range(a.reviewers)]
    src = man["source"]
    cfg = {"kernel": k, "module": man["lean"]["module"], "namespace": man["lean"]["namespace"],
           "source": src["file"], "source_sha256": src["sha256"], "reviewers": reviewers,
           "mode": "scripted" if a.scripted else "claude", "model": a.model,
           "max_reviewed": a.max_reviewed, "max_rejected": a.max_rejected, "gate_timeout_sec": 3600,
           "reviewer_timeout_sec": 5400, "file_retries": 2, "refiles": 2, "tva_commit": rev,
           "allow_scripted": a.scripted}
    if sha(os.path.join(kd, src["file"])) != src["sha256"]:
        sys.exit(f"kernels/{k}/{src['file']} does not match kernel.json's sha256")
    setup = os.path.isfile(os.path.join(kd, "SETUP.md"))
    cfg_text = json.dumps(cfg, indent=2) + "\n"
    E = os.path.join(out, "environment")

    write(os.path.join(out, "instruction.md"), instruction(tva, k, man, cfg, setup))
    mode_line = "scripted reviewers (plumbing test: they accept everything unread)" if a.scripted else \
        f"{a.reviewers} Claude Code reviewer agents ({a.model})"
    write(os.path.join(out, "task.toml"), f'''schema_version = "1.4"

[task]
name = "tva/{name.removeprefix('tva-')}"
description = "Prove the Triton kernel {k} in Lean 4 with the TVA library, and get the proof certified by {mode_line} through TVA's review protocol."
keywords = ["lean4", "triton", "formal-verification", "proof-review"]

[metadata]
kernel = "{k}"
tva_commit = "{rev}"
reviewers = {a.reviewers}
reviewer_mode = "{cfg['mode']}"
reviewer_model = "{a.model if not a.scripted else ''}"

[agent]
timeout_sec = {a.agent_timeout}

[verifier]
timeout_sec = 3600
environment_mode = "separate"
network_mode = "no-network"

# The review service's record, exported once the prover's container has stopped.
[[verifier.collect]]
command = "python3 -I /srv/sandbox/server.py export /export"
service = "reviewer"
timeout_sec = 1800

[environment]
build_timeout_sec = 3600
cpus = {a.cpus}
memory_mb = {a.memory_mb}

[[artifacts]]
source = "/export"
service = "reviewer"
''')

    # the prover's container (Harbor's `main`) and the reviewer's, each with its role's harness at /
    write(os.path.join(E, "task.json"), cfg_text)
    copy(os.path.join(SERVICE, "prepare_kernel.py"), os.path.join(E, "prepare_kernel.py"))
    copy(os.path.join(SERVICE, "tva-review"), os.path.join(E, "tva-review"), 0o755)
    ex, entry = harness_files(E, tva, k, man, cfg, setup, a)
    write(os.path.join(E, "Dockerfile"), f"""# The prover's container: Claude Code, TVA with the kernel reset to its skeleton, the review service's client,
# and the prover's harness at the root (/CLAUDE.md, /.claude/: Claude Code started in / loads it).
FROM {base}
{CLAUDE_CODE}
COPY task.json prepare_kernel.py /etc/tva-sandbox/
COPY tva-review /usr/local/bin/tva-review
RUN chmod 755 /usr/local/bin/tva-review \\
    && python3 -I /etc/tva-sandbox/prepare_kernel.py /etc/tva-sandbox/task.json \\
    && ln -s /home/tva/fpv /root/fpv && ln -s /opt/elan /root/.elan && mkdir -p /root/scratch \\
    && ln -s /root/scratch /home/tva/scratch
{harness_install("prover")}
ENV TVA_REVIEW_URL=http://reviewer:8765
WORKDIR /
""")
    for f in ("server.py", "reviewer-note.md"):
        copy(os.path.join(SERVICE, f), os.path.join(E, "reviewer", f))
    write(os.path.join(E, "reviewer", "Dockerfile"), f"""# The reviewer's container: the review service (root), the authority copy of TVA (the library root's,
# the kernel's folder `tva`'s), and a user per reviewer agent (rv1 to rv4).
FROM {base}
{CLAUDE_CODE}
RUN for i in 1 2 3 4; do \\
        useradd -m -u $((2000 + i)) -s /bin/bash rv$i && ln -s /opt/elan /home/rv$i/.elan && chmod 700 /home/rv$i \\
        && install -d -o rv$i -g rv$i /home/rv$i/scratch; \\
    done
{harness_install("reviewer")}
COPY task.json prepare_kernel.py /srv/sandbox/
COPY reviewer/server.py reviewer/reviewer-note.md /srv/sandbox/
RUN python3 -I /srv/sandbox/prepare_kernel.py /srv/sandbox/task.json \\
    && cd /home/tva/fpv/tva \\
    && find . -path ./lean/.lake -prune -o -exec chown -h root:root {{}} + \\
    && chown tva:tva kernels kernels/README.md docs/library-review/REPORTS.md \\
    && chown -R tva:tva kernels/{k} && chmod 700 kernels/{k} \\
    && chown root:root /home/tva /home/tva/fpv && chmod 755 /home/tva /home/tva/fpv \\
    && install -d -o tva -g tva /home/tva/.cache \\
    && chmod 755 /srv/sandbox && chmod 644 /srv/sandbox/*
EXPOSE 8765
ENTRYPOINT ["tini", "--"]
CMD ["python3", "-I", "/srv/sandbox/server.py", "serve"]
""")
    write(os.path.join(E, "docker-compose.yaml"), f"""# The reviewer sidecar. Its key comes from the environment that runs harbor, never from the task:
#   TVA_REVIEWER_ANTHROPIC_API_KEY or TVA_REVIEWER_ANTHROPIC_AUTH_TOKEN (or TVA_REVIEWER_CLAUDE_CODE_OAUTH_TOKEN),
#   and optionally TVA_REVIEWER_MODEL, TVA_REVIEWER_ANTHROPIC_BASE_URL (any Anthropic-compatible endpoint),
#   TVA_REVIEWER_API_TIMEOUT_MS, TVA_REVIEWER_CLAUDE_CODE_MAX_OUTPUT_TOKENS, TVA_REVIEWER_MAX_THINKING_TOKENS.
services:
  main:
    depends_on:
      reviewer:
        condition: service_healthy
  reviewer:
    build:
      context: .
      dockerfile: reviewer/Dockerfile
    pull_policy: build
    environment:
      TVA_REVIEWER_MODE: {cfg['mode']}
      TVA_REVIEWER_MODEL: ${{TVA_REVIEWER_MODEL:-{a.model}}}
      ANTHROPIC_API_KEY: ${{TVA_REVIEWER_ANTHROPIC_API_KEY:-}}
      ANTHROPIC_AUTH_TOKEN: ${{TVA_REVIEWER_ANTHROPIC_AUTH_TOKEN:-}}
      CLAUDE_CODE_OAUTH_TOKEN: ${{TVA_REVIEWER_CLAUDE_CODE_OAUTH_TOKEN:-}}
      ANTHROPIC_BASE_URL: ${{TVA_REVIEWER_ANTHROPIC_BASE_URL:-}}
      API_TIMEOUT_MS: ${{TVA_REVIEWER_API_TIMEOUT_MS:-}}
      CLAUDE_CODE_MAX_OUTPUT_TOKENS: ${{TVA_REVIEWER_CLAUDE_CODE_MAX_OUTPUT_TOKENS:-}}
      MAX_THINKING_TOKENS: ${{TVA_REVIEWER_MAX_THINKING_TOKENS:-}}
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://127.0.0.1:8765/health"]
      interval: 5s
      timeout: 10s
      retries: 60
      start_period: 60s
""")

    # the verifier's container
    T = os.path.join(out, "tests")
    write(os.path.join(T, "task.json"), cfg_text)
    copy(os.path.join(VERIFIER, "verify.py"), os.path.join(T, "verify.py"))
    copy(os.path.join(VERIFIER, "test.sh"), os.path.join(T, "test.sh"), 0o755)
    write(os.path.join(T, "Dockerfile"), f"""# The verifier's container: the same TVA image; it reads only the review service's export (/export).
FROM {base}
COPY test.sh verify.py task.json /tests/
RUN chmod 755 /tests/test.sh && chmod 644 /tests/verify.py /tests/task.json
""")

    # the oracle, for a certified kernel: TVA's own certified proof of it, submitted as is
    if certified:
        oracle(out, kd, k)
    print(f"wrote {out} (images from {base}; {mode_line}; examples {', '.join(ex)}; "
          f"launched kernel {', '.join(entry)}; " + ("oracle: the certified proof)" if certified else "no oracle: no certificate)"))


def oracle(out, kd, k):
    S = os.path.join(out, "solution")
    for dirpath, dirs, files in os.walk(os.path.join(kd, "lean")):
        for f in files:
            p = os.path.join(dirpath, f)
            q = os.path.join(S, "kernel", os.path.relpath(p, kd))
            os.makedirs(os.path.dirname(q), exist_ok=True)
            shutil.copyfile(p, q)
    for f in ("README.md", "model-map.json", "kernel.json"):
        shutil.copyfile(os.path.join(kd, f), os.path.join(S, "kernel", f))
    write(os.path.join(S, "solve.sh"), f"""#!/bin/bash
# The oracle: TVA's own certified proof of {k}, put in place and submitted as is.
set -euo pipefail
KD=/home/tva/fpv/tva/kernels/{k}
cp -r /solution/kernel/lean/. "$KD/lean/"
cp /solution/kernel/README.md /solution/kernel/model-map.json /solution/kernel/kernel.json "$KD/"
cd /home/tva/fpv/tva
python3 -I tools/tvr.py check {k} | tail -1
tva-review submit
while :; do
  set +e; tva-review wait 540; rc=$?; set -e
  [ "$rc" -eq 0 ] && break
  [ "$rc" -eq 3 ] || exit "$rc"
done
""", 0o755)


if __name__ == "__main__":
    main()
