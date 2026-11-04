#!/usr/bin/env python3
"""tva-kit: the sandbox harness's helpers for TVA's prover and reviewers (HARNESS.md).

Every command is short to type and prints little: the mechanical parts of the protocol (line classes,
the model map's skeleton, a filing and its report, a response and its notes) are written for you, and
checked the way `tvr` checks them.

Both roles
  tva-kit brief                 where you are and what to do next
  tva-kit source [A-B]          the kernel's source with line numbers and each line's class
  tva-kit examples REGEX [--in source|model|note|lean] [-n N]
                                how the certified kernels map source lines that match REGEX
  tva-kit lookup NAME [-n N]    a Lean declaration of the library or of a certified kernel
  tva-kit doc QUERY [-n N]      the sections of TVA's documents that mention QUERY
  tva-kit outline [KERNEL] [FILE]
                                a kernel's proof design: its Lean files bottom-up, each with its docstring and
                                declarations (`sorry`s marked for your own); with FILE, that file's statements
Prover
  tva-kit stage                 the checklist of the proof, with what is done and what is next, and each `sorry`
  tva-kit clock ["NOTE"]        the time used and left; with NOTE, the line `[time used] NOTE` added to
                                ~/scratch/progress.md, and how long it was since the last note
  tva-kit classify [--write]    propose kernel.json's line classes (source.ranges, source.host)
  tva-kit map-skeleton          add a model-map entry for every code line of the ranges not yet mapped
  tva-kit map-check             the gate's map check, without building
  tva-kit build [MODULE]        build the kernel's Lean; the errors with their source lines
  tva-kit try start|brief|check|status|accept|stop
                                several subagents at one lemma at once, each with its own approach; the first
                                proof Lean accepts wins (`tva-kit try` for help)
  tva-kit gate                  run the gate (tvr check); each failing check with what to do
  tva-kit findings [--all]      the open findings, in full
  tva-kit respond init|set|change|request|show|lint
                                write response.json and notes.md (`tva-kit respond` for help)
Reviewer
  tva-kit review-brief [--map A-B]
                                items, assumptions, your findings, and the source beside the model map
  tva-kit targets               the targets a finding may name, by kind
  tva-kit filing init|add|accept|release|reraise|checked|not-checked|remove|show|lint
                                write filing.json and report.md (`tva-kit filing` for help)
"""
import ast
import contextlib
import fcntl
import glob
import json
import os
import re
import subprocess
import sys
import time

CONFIG_PATH = os.environ.get("TVA_HARNESS_CONFIG", "/etc/tva-sandbox/harness.json")
CONFIG = json.load(open(CONFIG_PATH)) if os.path.exists(CONFIG_PATH) else {}
ROLE = CONFIG.get("role", "prover")
K = os.environ.get("TVA_KERNEL") or CONFIG.get("kernel", "")


def tva_root():
    if os.environ.get("TVA_ROOT"):
        return os.environ["TVA_ROOT"]
    if ROLE == "reviewer":
        return os.path.join(os.path.expanduser("~"), "fpv", "tva")
    return CONFIG.get("tva", "/home/tva/fpv/tva")


ROOT = tva_root()
KD = os.path.join(ROOT, "kernels", K)
SCRATCH = os.path.join(os.path.expanduser("~"), "scratch")  # the agent's own working files
sys.path.insert(0, os.path.join(ROOT, "tools"))
try:
    import tvr  # the tree's own tool: the gate's parsers and checks, so that what this says is what it checks
except Exception:  # noqa: BLE001  (a hook must still run without it)
    tvr = None


def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


def reviewer_id():
    import pwd
    user = pwd.getpwuid(os.getuid()).pw_name
    return CONFIG.get("reviewers", {}).get(user, os.environ.get("TVA_REVIEWER", ""))


def load(path, default=None):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return default


def manifest():
    return load(os.path.join(KD, "kernel.json"), {})


def source_text():
    man = manifest()
    return open(os.path.join(KD, man["source"]["file"])).read()


def lean_files():
    return sorted(glob.glob(os.path.join(KD, "lean", "**", "*.lean"), recursive=True))


def lean_text():
    return "\n".join(open(p).read() for p in lean_files())


def gate():
    return load(os.path.join(KD, "gate.json"), {"items": {}, "ok": False, "digest": "", "kinds": {}})


def state():
    return tvr.state(K)


def short(text, n=160):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 1] + "…"


CLOCK = os.path.join(SCRATCH, ".clock.json")  # {"start": …}, written at the prover's first session start


def clock():
    """(seconds used, seconds allowed) of the prover's time, or None: the limit is the task's agent timeout,
    counted from the first session's start."""
    limit, start = CONFIG.get("time_limit_sec"), (load(CLOCK, {}) or {}).get("start")
    return (time.time() - start, limit) if limit and start else None


def hm(s):
    m = int(s) // 60
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


def clock_line():
    c = clock()
    return f"time: {hm(c[0])} used of {hm(c[1])} ({hm(max(0, c[1] - c[0]))} left)" if c else ""


def parse_lines(spec):
    """'A-B' or 'A' → (A, B)."""
    m = re.fullmatch(r"(\d+)(?:-(\d+))?", spec or "")
    if not m:
        die(f"a line range is A-B or A, not {spec!r}")
    return int(m.group(1)), int(m.group(2) or m.group(1))


def runs(lines):
    out = []
    for i in sorted(lines):
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def run_tvr(*args, timeout=3600):
    p = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "tvr.py"), *args], cwd=ROOT,
                       capture_output=True, text=True, timeout=timeout, errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


# ---------------------------------------------------------------- the source and its classes

def line_classes():
    """{line: 'K'|'H'|'U'} from kernel.json, and the code lines (by the gate's own parser)."""
    man = manifest()
    s = man.get("source", {})
    cls = {}
    for a, b in s.get("host", []):
        for i in range(a, b + 1):
            cls[i] = "H"
    for a, b in s.get("ranges", []):
        for i in range(a, b + 1):
            cls[i] = "K"
    text = source_text()
    for name, a, b in (tvr.source_functions(text) or []):
        if name and name in s.get("unused", []):
            for i in range(a, b + 1):
                cls.setdefault(i, "U")
    code = tvr.code_lines(text)[0] or set()
    return cls, code


def map_cover():
    """{line: 'M'|'D'|'T'} from model-map.json: modelled, dropped, or a TODO entry."""
    mm = load(os.path.join(KD, "model-map.json"), {"entries": []})
    out = {}
    for e in mm.get("entries", []):
        try:
            lines = tvr.parse_lines(e.get("src", []))
        except (ValueError, TypeError):
            continue
        todo = "TODO" in json.dumps(e)
        for i in lines:
            out[i] = "T" if todo else ("D" if "drop" in e else "M")
    return out


def cmd_source(argv):
    text = source_text().split("\n")
    a, b = parse_lines(argv[0]) if argv else (1, len(text))
    cls, code = line_classes()
    cov = map_cover()
    print("class: K kernel lines, H host, U unused, . unclassed code, blank = no code;  map: M modelled, D dropped, "
          "T TODO entry, ! code line of the ranges with no entry")
    for i in range(max(1, a), min(len(text), b) + 1):
        c = cls.get(i, "." if i in code else " ")
        m = cov.get(i, "!" if c == "K" and i in code else " ")
        print(f"{i:4d} {c}{m} {text[i - 1]}")


IMPORT_TL = re.compile(r"^\s*(import\s+triton\s*(#.*)?$|import\s+triton\.language\s+as\s+tl\s*(#.*)?$)")


def propose_classes():
    """The proposal: the @triton.jit functions the launched kernel reaches (`entry` in the harness config, or
    every one), and the imports of `triton` and `triton.language as tl`, are kernel lines; a module-level
    function whose name nothing else mentions (and an unreached JIT function) is unused, its header a host
    line; every other function and module-level code line is a host line."""
    text = source_text()
    lines = text.split("\n")
    code = tvr.code_lines(text)[0] or set()
    jit = tvr.jit_functions(text) or {}
    funcs = tvr.source_functions(text) or []
    top = {n.name: (min([d.lineno for d in n.decorator_list] + [n.lineno]), n.end_lineno)
           for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    entry = [e for e in CONFIG.get("entry", []) if e in jit] or list(jit)
    reach, todo = set(), list(entry)
    while todo:
        f = todo.pop()
        if f in reach:
            continue
        reach.add(f)
        body = "\n".join(l for a, b in jit.get(f, []) for l in lines[a - 1:b])
        todo += [g for g in jit if g not in reach and re.search(rf"\b{re.escape(g)}\s*\(", body)]

    def mentioned_elsewhere(name, a, b):
        rx = re.compile(rf"\b{re.escape(name)}\b")
        return any(rx.search(l) for j, l in enumerate(lines, 1) if not a <= j <= b)
    unused = sorted([g for g in jit if g not in reach] +
                    [n for n, (a, b) in top.items() if n not in jit and not mentioned_elsewhere(n, a, b)])
    kern, host, notes = set(), set(), []
    for name, a, b in funcs:
        if name in reach and [a, b] in jit.get(name, []):
            kern |= set(range(a, b + 1))
    spans_unused = {i for n in unused if n in top for i in range(top[n][0], top[n][1] + 1)}
    for name, a, b in funcs:
        span = set(range(a, b + 1))
        if not span <= kern and not span <= spans_unused:
            host |= span - kern - spans_unused
    host |= tvr.def_headers(text, set(unused)) or set()
    for i in tvr.module_lines(text, unused) or []:
        if i in kern or i in host:
            continue
        if IMPORT_TL.match(lines[i - 1]):
            kern.add(i)
        else:
            host.add(i)
            if re.search(r"\btl\b|triton\.language|tl\.constexpr|\bjit\b", lines[i - 1]):
                notes.append(f"line {i} is classed as host but mentions Triton: check that the launch does not read "
                             f"what it binds (`{short(lines[i - 1], 80)}`)")
    if any(re.search(r"@triton\.(autotune|heuristics)", l) for l in lines):
        notes.append("the source has @triton.autotune or @triton.heuristics: outside the supported subset "
                     "(PROVER_GUIDE §1); say so in your notes")

    def ranges(own, other):
        out = []
        for a, b in runs(own):
            if out and all((j not in code) and (j not in other) for j in range(out[-1][1] + 1, a)):
                out[-1][1] = b
            else:
                out.append([a, b])
        return out
    return ranges(kern, host), ranges(host, kern), unused, notes


def cmd_classify(argv):
    rng, host, unused, notes = propose_classes()
    print(f"proposed source.ranges (kernel lines): {json.dumps(rng)}")
    print(f"proposed source.host   (host lines):   {json.dumps(host)}")
    print(f"proposed source.unused (nothing runs): {json.dumps(unused)}")
    for n in notes:
        print(f"note: {n}")
    print("Check them: the kernel lines hold the launched kernel and every JIT function it calls on the covered "
          "launches; a function nothing runs is unused (its header stays a host line). The reviewers judge the "
          "classes (PROVER_GUIDE §1, 'Class every function').")
    if "--write" in argv:
        p = os.path.join(KD, "kernel.json")
        man = load(p)
        man["source"]["ranges"], man["source"]["host"] = rng, host
        man["source"].pop("unused", None)
        if unused:
            man["source"]["unused"] = unused
        tvr.dump_json(p, man)
        print(f"wrote source.ranges, source.host and source.unused into kernels/{K}/kernel.json")
    else:
        print("(run `tva-kit classify --write` to write them into kernel.json)")


# ---------------------------------------------------------------- the model map

def statement_spans(text, kern):
    """The spans a model-map entry naturally covers, for the kernel lines: each import, decorator and `def`
    header, each simple statement, and each compound statement's header (`for …:`, `if …:`, `else:`)."""
    tree = ast.parse(text)
    lines = text.split("\n")
    spans = []

    def header(node):
        first = node.body[0].lineno if node.body else node.end_lineno
        return (node.lineno, max(node.lineno, first - 1))

    def block(stmts):
        for s in stmts:
            if isinstance(s, (ast.For, ast.While, ast.If, ast.With, ast.AsyncFor, ast.AsyncWith)):
                spans.append(header(s))
                block(s.body)
                orelse = getattr(s, "orelse", [])
                if orelse:
                    if isinstance(s, ast.If) and len(orelse) == 1 and isinstance(orelse[0], ast.If) and \
                            re.match(r"\s*elif\b", lines[orelse[0].lineno - 1]):
                        block(orelse)
                    else:
                        for j in range(s.body[-1].end_lineno + 1, orelse[0].lineno):
                            if re.match(r"\s*else\s*:", lines[j - 1]):
                                spans.append((j, j))
                        block(orelse)
            elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for d in s.decorator_list:
                    spans.append((d.lineno, d.end_lineno))
                body = s.body
                if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                        and isinstance(body[0].value.value, str):
                    body = body[1:]
                first = (body[0].lineno if body else s.end_lineno + 1)
                spans.append((s.lineno, max(s.lineno, first - 1)))
                block(body)
            else:
                spans.append((s.lineno, s.end_lineno))
    block(tree.body)
    code = tvr.code_lines(text)[0] or set()
    out = []
    for a, b in spans:  # trimmed to their code lines (a comment between a header and its body is no code)
        c = [i for i in range(a, b + 1) if i in code]
        if c and any(i in kern for i in c):
            out.append((c[0], c[-1]))
    return out


def skeleton_entry(text_lines, a, b):
    src = text_lines[a - 1].strip()
    entry = {"src": [a] if a == b else [f"{a}-{b}"]}
    if a == b and IMPORT_TL.match(text_lines[a - 1]):
        what = "`tl` to `triton.language`" if "language" in src else "`triton`, Triton's own module"
        entry.update(drop="import", note=f"binds {what} (SEMANTICS.md, the supported subset, Names); no other statement "
                     "of the module binds it; it adds no step to the program")
    elif a == b and src == "@triton.jit":
        entry.update(drop="decl", note="a bare `@triton.jit` on a top-level `def` (SEMANTICS.md, the supported subset: "
                     "the kernel), so its `debug` is unset (`Launch.debug = false`); the model is the `Kernel` it decorates")
    else:
        entry.update(model="TODO", note=f"TODO: the model text of: {short(src, 120)}")
    return entry


def cmd_map_skeleton(argv):
    text = source_text()
    tl = text.split("\n")
    man = manifest()
    kern = {i for a, b in man["source"].get("ranges", []) for i in range(a, b + 1)}
    if not kern:
        die("kernel.json has no source.ranges yet: run `tva-kit classify --write` first")
    code = tvr.code_lines(text)[0] or set()
    p = os.path.join(KD, "model-map.json")
    mm = load(p, {"entries": []})
    have = set()
    for e in mm.get("entries", []):
        try:
            have |= set(tvr.parse_lines(e["src"]))
        except (KeyError, ValueError, TypeError):
            pass
    added = []
    for a, b in statement_spans(text, kern):
        span = {i for i in range(a, b + 1) if i in code and i in kern}
        if span and not (span & have):
            added.append(skeleton_entry(tl, a, b))
            have |= set(range(a, b + 1))
    for i in sorted(code & kern - have):
        added.append(skeleton_entry(tl, i, i))
    if not added:
        print("every code line of the ranges already has an entry")
        return
    os.makedirs(SCRATCH, exist_ok=True)
    if mm.get("entries"):
        json.dump(mm, open(os.path.join(SCRATCH, f"model-map.before-skeleton.{int(time.time())}.json"), "w"), indent=2)
    mm["entries"] = sorted(mm.get("entries", []) + added,
                           key=lambda e: min(tvr.parse_lines(e["src"])) if e.get("src") else 0)
    tvr.dump_json(p, mm)
    todo = sum(1 for e in added if e.get("model") == "TODO")
    print(f"added {len(added)} entries to kernels/{K}/model-map.json ({todo} to fill: `\"model\": \"TODO\"`); "
          f"{len(mm['entries']) - len(added)} kept.")
    print("Fill each TODO: `model` is text that occurs in your Lean (up to whitespace), or replace `model` with "
          "`drop` (a reason) when the line changes nothing; `note` says why. `tva-kit map-check` checks them.")


def cmd_map_check(argv):
    man = manifest()
    mm = load(os.path.join(KD, "model-map.json"))
    if mm is None:
        die("model-map.json is not JSON: fix its syntax first")
    probs, missing = tvr.map_problems(man, mm, source_text(), lean_text())
    todo = [e["src"] for e in mm.get("entries", []) if "TODO" in json.dumps(e)]
    for x in probs:
        print(f"- {x}")
    if missing:
        print(f"- code lines of the ranges with no entry: {tvr._line_list(missing)} (`tva-kit map-skeleton` adds them)")
    if todo:
        print(f"- entries still TODO: {', '.join(str(t) for t in todo[:20])}{' …' if len(todo) > 20 else ''}")
    if not (probs or missing or todo):
        print("map: regular (every code line of the ranges covered, every model text found in the Lean)")
        return 0
    return 1


# ---------------------------------------------------------------- examples, lookup, documents

def certified():
    out = []
    for kd in sorted(glob.glob(os.path.join(ROOT, "kernels", "*"))):
        k = os.path.basename(kd)
        if k != K and os.path.exists(os.path.join(kd, "review", "certificate.json")):
            out.append(k)
    return out


def cmd_examples(argv):
    where, n, pat = "source", 8, None
    i = 0
    while i < len(argv):
        if argv[i] == "--in":
            where, i = argv[i + 1], i + 2
        elif argv[i] == "-n":
            n, i = int(argv[i + 1]), i + 2
        else:
            pat, i = argv[i], i + 1
    if not pat:
        die("tva-kit examples REGEX [--in source|model|note|lean] [-n N]")
    rx = re.compile(pat)
    shown = 0
    for k in certified():
        kd = os.path.join(ROOT, "kernels", k)
        if where == "lean":
            for p in sorted(glob.glob(os.path.join(kd, "lean", "**", "*.lean"), recursive=True)):
                ls = open(p).read().split("\n")
                for j, l in enumerate(ls):
                    if rx.search(l) and shown < n:
                        shown += 1
                        ctx = "\n".join(f"      {ls[x]}" for x in range(max(0, j - 2), min(len(ls), j + 3)))
                        print(f"{os.path.relpath(p, ROOT)}:{j + 1}\n{ctx}")
            continue
        man = load(os.path.join(kd, "kernel.json"), {})
        mm = load(os.path.join(kd, "model-map.json"), {"entries": []})
        src = open(os.path.join(kd, man["source"]["file"])).read().split("\n")
        for e in mm.get("entries", []):
            try:
                ls = tvr.parse_lines(e["src"])
            except (KeyError, ValueError, TypeError):
                continue
            stext = " ".join(src[x - 1].strip() for x in ls if 0 < x <= len(src))
            model = e.get("model")
            mtext = " / ".join(model) if isinstance(model, list) else (model or "")
            hay = {"source": stext, "model": mtext, "note": e.get("note", "")}.get(where, stext)
            if rx.search(hay) and shown < n:
                shown += 1
                print(f"[{k} lines {e['src']}]\n  source: {short(stext, 220)}")
                print(f"  model:  {short(mtext, 300)}" if model else f"  DROP ({e.get('drop')})")
                if e.get("note"):
                    print(f"  note:   {short(e['note'], 300)}")
    if not shown:
        print(f"no certified kernel has a match for {pat!r} in {where}")
    elif shown >= n:
        print(f"(first {n}; -n N for more; the full files are kernels/<kernel>/model-map.json and lean/)")


DECL = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|noncomputable|partial|nonrec|unsafe)\s+)*"
                  r"(theorem|lemma|def|abbrev|structure|inductive|class|instance|axiom|opaque)\s+([^\s(:{\[]+)")


def decls():
    files = sorted(glob.glob(os.path.join(ROOT, "lean", "TVA", "**", "*.lean"), recursive=True))
    files += [p for k in certified() for p in sorted(glob.glob(os.path.join(ROOT, "kernels", k, "lean", "**", "*.lean"),
                                                                recursive=True))]
    for p in files:
        ls = open(p).read().split("\n")
        ns, depth = [], 0
        for j, l in enumerate(ls):
            # skip block comments and docstrings (`/- … -/`, nested), which can hold Lean examples
            if depth > 0:
                depth = max(0, depth + l.count("/-") - l.count("-/"))
                continue
            code = re.sub(r"--.*", "", l)
            if code.count("/-") > code.count("-/"):
                depth = code.count("/-") - code.count("-/")
                if not DECL.match(l):
                    continue
            m = re.match(r"^namespace\s+(\S+)", l)
            if m:
                ns.append(m.group(1))
                continue
            m = re.match(r"^end\s+(\S+)", l)
            if m and ns and ns[-1] == m.group(1):
                ns.pop()
                continue
            m = DECL.match(l)
            if m:
                name = m.group(2)
                full = name[len("_root_."):] if name.startswith("_root_.") else ".".join(ns + [name])
                yield full, p, j, ls, m.group(1)


def cmd_lookup(argv):
    n = 5
    if "-n" in argv:
        n = int(argv[argv.index("-n") + 1])
        argv = argv[:argv.index("-n")] + argv[argv.index("-n") + 2:]
    if not argv:
        die("tva-kit lookup NAME [-n N]")
    q = argv[0]
    exact, near = [], []
    for full, p, j, ls, kind in decls():
        if full == q or full.endswith("." + q) or full.split(".")[-1] == q:
            exact.append((full, p, j, ls, kind))
        elif q.lower() in full.lower():
            near.append((full, p, j, ls, kind))
    hits = (exact or near)[:n]
    if not hits:
        print(f"no declaration named like {q!r} in lean/TVA or the certified kernels (Mathlib is not searched: "
              f"try `#check @{q}` in a scratch file with `lake env lean`)")
        return 1
    for full, p, j, ls, kind in hits:
        start = j
        while start > 0 and (ls[start - 1].strip().startswith(("/--", "-/")) or ls[start - 1].startswith("  ") and
                             "/--" in "\n".join(ls[max(0, start - 8):start])):
            start -= 1
            if ls[start].strip().startswith("/--"):
                break
        end = j
        while end + 1 < len(ls) and end - j < 10 and ls[end + 1].strip() and not DECL.match(ls[end + 1]) \
                and ":=" not in ls[end] and not ls[end].rstrip().endswith(" where"):
            end += 1
        print(f"{kind} {full}   ({os.path.relpath(p, ROOT)}:{j + 1})")
        for x in range(max(start, j - 6), end + 1):
            print("    " + ls[x])
    if len(exact or near) > n:
        print(f"({len(exact or near) - n} more; -n N)")
    return 0


def doc_sections(path):
    secs, cur = [], None
    for l in open(path).read().split("\n"):
        if re.match(r"^#{1,4} ", l):
            cur = [l, []]
            secs.append(cur)
        elif cur:
            cur[1].append(l)
    return secs


def cmd_doc(argv):
    n = 3
    if "-n" in argv:
        n = int(argv[argv.index("-n") + 1])
        argv = argv[:argv.index("-n")] + argv[argv.index("-n") + 2:]
    if not argv:
        die("tva-kit doc QUERY [-n N]")
    q = " ".join(argv).lower()
    files = [os.path.join(ROOT, "docs", f) for f in ("PROVER_GUIDE.md", "REVIEWER_GUIDE.md", "TRANSCRIPTION.md",
                                                     "SEMANTICS.md", "REVIEW_PROTOCOL.md", "exclusions.md", "LIBRARY.md")]
    files += sorted(glob.glob(os.path.join(ROOT, "docs", "rules", "*.md")))
    scored = []
    for f in files:
        if not os.path.exists(f):
            continue
        for h, body in doc_sections(f):
            # the section's blocks: a bullet with its indented lines, a table row, or a paragraph
            blocks, cur = [], []
            for l in body:
                if not l.strip() or re.match(r"^(- |\d+\. |\| )", l):
                    if cur:
                        blocks.append(cur)
                    cur = [l] if l.strip() else []
                else:
                    cur.append(l)
            if cur:
                blocks.append(cur)
            for blk in blocks:
                text = "\n".join(blk).lower()
                lead = re.match(r"^\s*(?:- |\d+\. )?\*\*(.+?)\*\*", blk[0])
                score = (30 if lead and q in lead.group(1).lower() else 0) + (8 if q in h.lower() else 0) + \
                    text.count(q) * 2
                if q in text or q in h.lower():
                    scored.append((score, f, h, blk))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        print(f"nothing in docs/ mentions {q!r}")
        return 1
    for score, f, h, blk in scored[:n]:
        print(f"== {os.path.relpath(f, ROOT)} § {h.lstrip('# ')}")
        print("\n".join(blk[:30]) + ("\n…" if len(blk) > 30 else ""))
    others = sorted({f"{os.path.relpath(f, ROOT)} § {h.lstrip('# ')}" for _, f, h, _ in scored[n:n + 30]})[:8]
    if others:
        print("also in: " + "; ".join(others))
    return 0


# ---------------------------------------------------------------- build and gate

def module_of(path):
    rel = os.path.relpath(os.path.realpath(path), os.path.realpath(os.path.join(KD, "lean")))
    return rel[:-5].replace(os.sep, ".") if rel.endswith(".lean") and not rel.startswith("..") else None


ERR = re.compile(r"^(error|warning): (?:\S*?/)?([^/\s:]+\.lean):(\d+):(\d+): (.*)$")


def summarize_build(out, limit=8):
    """The errors of a `lake build` output, each with its source line, and the count of the kernel's
    warnings and `sorry`s (Lake replays the library's own warnings at every build)."""
    lines = out.split("\n")
    errs, warns, sorries, cur = [], 0, 0, None
    for l in lines:
        m = ERR.match(l)
        if m:
            cur = None
            if m.group(1) == "warning":
                if f"kernels/{K}/" in l:
                    warns += 1
                    sorries += "declaration uses 'sorry'" in l
                continue
            cur = {"file": m.group(2), "line": int(m.group(3)), "col": int(m.group(4)), "msg": [m.group(5)]}
            errs.append(cur)
        elif cur is not None and l.strip() and not re.match(r"^(✖|⚠|✔|info:|Build|Some required|trace:)", l.strip()):
            if len(cur["msg"]) < 14:
                cur["msg"].append(l)
        elif l.startswith(("✖", "Some required")):
            cur = None
    files = {os.path.basename(p): p for p in lean_files()}
    out_lines = []
    for e in errs[:limit]:
        src = ""
        p = files.get(e["file"])
        if p:
            ls = open(p).read().split("\n")
            if 0 < e["line"] <= len(ls):
                src = f"\n    > {e['line']}: {ls[e['line'] - 1].rstrip()}"
        out_lines.append(f"- {e['file']}:{e['line']}:{e['col']}: " + "\n    ".join(m.rstrip() for m in e["msg"]) + src)
    other = [l for l in lines if l.startswith("error:") and not ERR.match(l)]
    return errs, warns, sorries, out_lines + [f"- {short(l, 300)}" for l in other[:4]]


BUILD_LOGS = os.path.join(SCRATCH, "build")


def save_build_log(module, text):
    """Keep a build's whole output (every error with its goal) where the agent can read it, so that it need
    not build again to see a goal; return the path as the agent writes it."""
    with contextlib.suppress(OSError):
        os.makedirs(BUILD_LOGS, exist_ok=True)
        open(os.path.join(BUILD_LOGS, f"{module}.log"), "w").write(text)
    return f"~/scratch/build/{module}.log"


SLOW_BUILD = float(os.environ.get("TVA_HARNESS_SLOW_BUILD", "60"))  # seconds: a module this slow should be split
# Lake's line for a module's job; when its output is not a terminal it ends with the job's time ("115s", "2.6s", "870ms")
LAKE_JOB = re.compile(r"\[\d+/\d+\] (?:Built|Building) ([^\s()]+) \((?:(\d+)ms|(\d+(?:\.\d+)?)s)\)")


def slow_modules(text):
    """The kernel's modules that Lake's output `text` reports took SLOW_BUILD seconds or more, slowest first."""
    root = manifest().get("lean", {}).get("module")
    found = {}
    for mod, ms, s in LAKE_JOB.findall(text):
        secs = int(ms) / 1000 if ms else float(s)
        if root and (mod == root or mod.startswith(root + ".")) and secs >= SLOW_BUILD:
            found[mod] = max(found.get(mod, 0), secs)
    return sorted(found.items(), key=lambda x: -x[1])


def split_advice(slow):
    """What to do about the slow modules `slow_modules` found ("" when there are none)."""
    if not slow:
        return ""
    return ("Lake reports " + ", ".join(f"{m} took {s:.0f} s" for m, s in slow[:3]) + " to build: Lean re-checks a "
            "whole file at every build. Split the file before going on: move the lemmas that are done into a module "
            f"of their own (a new file in kernels/{K}/lean/, imported by this one), and state the piece you are "
            "working on as its own lemma in a small file, so that a try takes seconds.")


LAST_BUILD = {"seconds": 0.0}  # how long the last `build` ran, waiting for another agent's build aside


@contextlib.contextmanager
def lake_lock(exclusive=True):
    """The prover's image runs every `lake` under these locks (harness/lake): a build holds the build folder
    alone, `lake env` shares it with other readers, and a waiting build stops new readers (the turnstile)."""
    with open("/tmp/tva-lake.turn", "a") as turn, open("/tmp/tva-lake.lock", "a") as lock:
        fcntl.flock(turn, fcntl.LOCK_EX)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        finally:
            fcntl.flock(turn, fcntl.LOCK_UN)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def build(module=None, timeout=1800):
    module = module or manifest().get("lean", {}).get("module")
    os.makedirs(os.path.join(ROOT, "lean", ".lake"), exist_ok=True)
    with lake_lock():
        t0 = time.time()
        try:
            p = subprocess.run(["lake", "build", module], cwd=os.path.join(ROOT, "lean"), capture_output=True,
                               text=True, timeout=timeout, errors="replace", env={**os.environ, "TVA_LAKE_LOCKED": "1"})
            return p.returncode, (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired:
            return 124, f"error: lake build {module} timed out after {timeout} s"
        finally:
            LAST_BUILD["seconds"] = time.time() - t0


def cmd_build(argv):
    mod = argv[0] if argv else None
    if mod and mod.endswith(".lean"):
        mod = module_of(mod)
    if not lean_files():
        man = manifest()
        die(f"no Lean file yet: create lean/{man['lean']['module'].replace('.', '/')}.lean in your kernel folder "
            "(start from the closest example's file)")
    mod = mod or manifest()["lean"]["module"]
    t0 = time.time()
    rc, out = build(mod)
    errs, warns, sorries, lines = summarize_build(out)
    head = f"lake build {mod}: " + ("OK" if rc == 0 else f"{len(errs)} error(s)")
    print(head + f" ({time.time() - t0:.0f} s; {warns} warning(s), {sorries} `sorry`); "
          f"the whole output: {save_build_log(mod, out)}")
    print("\n".join(lines))
    if len(errs) > 8:
        print(f"({len(errs) - 8} more errors: fix the first ones first, they often cause the rest)")
    if rc == 0 and sorries:
        print("the gate refuses `sorry`: replace each before `tva-kit gate`")
    advice = split_advice(slow_modules(out))
    if advice:
        print(advice)
    return rc


GATE_HINTS = {
    "manifest": "kernel.json: lean.module/namespace are the task's (do not change them); source.ranges and source.host "
                "are lists of [first, last] line pairs. `tva-kit classify` proposes them.",
    "source": "the source file is fixed: never edit kernels/<kernel>/source/.",
    "hygiene": "no `sorry`, `admit`, `#eval`, `#print`, `run_cmd`, `notation`, `macro`, `instance`, `set_option` "
               "(beyond resource limits), `open … renaming`, `axiom`, `unsafe`, `initialize`, string literals with `{`. "
               "Remove the construct the line names. PROVER_GUIDE §1, last points.",
    "independence": "import only `TVA`, `Mathlib.*` and your own kernel's modules; write no path or name of another project.",
    "build": "Lean errors: run `tva-kit build` for each error with its source line; fix the first error first.",
    "replay": "leanchecker re-checks your declarations; this fails only when the build did or with meta tricks.",
    "shape": "the theorems must be exactly `main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K`, "
             "`consistent : Claim.Consistent Assumptions`, `satisfiable : Claim.Satisfiable Inst.ok Pre`, with "
             "`Spec K := Claim.Meets Inst.ok Pre Post K` and `structure Assumptions (K : Claim.Behaviour Inst) (P : Platform)`. "
             "Copy the statements of a certified example (`tva-kit lookup main`).",
    "axioms": "only propext, Classical.choice, Quot.sound: no `sorry`, no `axiom`, no `native_decide`.",
    "usage": "every field of `Assumptions` must be used by `main`: remove the unused ones.",
    "namespace": "declare everything inside your kernel's namespace; no `instance`; no name that also names a library "
                 "constant (rename it, e.g. `myAdd` instead of `add`); no binder named `tl`, `Tensor`, ….",
    "kinds": "each Assumptions field is [rule] (`Rules.Compiler.Faithful P (model d) (launch d) (K d)` under `∀ d, d.ok →`, "
             "or a fact docs/rules/num.md lists, applied to P, e.g. `Rules.Num.ExpF32 P`) or [kernel] (a closed fact "
             "about your own constants, without K or P). Give each a docstring starting `[rule]` or `[kernel]`.",
    "map": "run `tva-kit map-check`; `tva-kit classify --write` and `tva-kit map-skeleton` fix most of it.",
    "doc": "README.md needs the sections Specification, Conditions, Assumptions, Proof (and Narrowings), each "
           "non-empty, in words matching the Lean.",
}


def cmd_gate(argv):
    t0 = time.time()
    rc, out = run_tvr("check", K)
    fails = []
    for l in out.split("\n"):
        m = re.match(r"^(PASS|FAIL)\s+(\S+)\s+(.*)$", l)
        if m:
            print(f"{m.group(1)}  {m.group(2):12s} {short(m.group(3), 140 if m.group(1) == 'PASS' else 600)}")
            if m.group(1) == "FAIL":
                fails.append(m.group(2))
        elif l.startswith("gate "):
            print(l)
    for f in fails:
        if f in GATE_HINTS:
            print(f"  → {f}: {GATE_HINTS[f]}")
    print(f"({time.time() - t0:.0f} s)" + ("" if rc else "  next: `tva-review submit` (prover) when you are ready"))
    if not any(l.startswith("gate ") for l in out.split("\n")):
        print(short(out, 2000))
    for l in readme_stale_lines():
        print(l)
    return rc


def readme_stale_lines():
    """The warning lines for constants README states other than the Lean defines (stale_constants)."""
    p = os.path.join(KD, "README.md")
    if not os.path.exists(p):
        return []
    st = stale_constants(open(p).read(), [open(f).read() for f in lean_files()])
    if not st:
        return []
    return [f"README.md may state constants the Lean no longer has (reviewers file DOC_WRONG, which blocks; a "
            f"history of earlier values is fine, say it is one):"] + [
        f"  README.md:{n}: `{name} = {v}`; the Lean: `{name} := {lean}`" for n, name, v, lean in st[:8]]


# ---------------------------------------------------------------- the proof's design

DECL0 = re.compile(r"^(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|noncomputable|partial|nonrec|unsafe)\s+)*"
                   r"(theorem|lemma|def|abbrev|structure|inductive|class|instance|axiom|opaque)\s+([^\s(:{\[]+)")
# where a declaration ends: the next one, or a command at the start of a line
BOUND = re.compile(r"^(?:namespace|end|section|open|variable|universe|attribute|set_option|example|#|@\[)|" +
                   DECL0.pattern[1:])
LONG_PROOF = 150  # lines: a `sorry` inside a longer proof is a lemma that is not stated yet
AUDIT = os.path.join(SCRATCH, "audit.md")  # the claim-audit subagent's report


def strip_comments(text):
    """`text` with its comments (`--` to the end of the line, `/- … -/` nested, docstrings) blanked out, so
    that every character keeps its offset and every line its number."""
    out, i, depth, n = [], 0, 0, len(text)
    while i < n:
        if text.startswith("/-", i):
            depth, i = depth + 1, i + 2
            out.append("  ")
        elif depth and text.startswith("-/", i):
            depth, i = depth - 1, i + 2
            out.append("  ")
        elif depth:
            out.append("\n" if text[i] == "\n" else " ")
            i += 1
        elif text.startswith("--", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def lean_decls(path):
    """The declarations that start a line of a Lean file, [(kind, name, first, last)] (0-based lines,
    inclusive), and the file's lines with comments blanked."""
    ls = strip_comments(open(path).read()).split("\n")
    out = []
    for i, l in enumerate(ls):
        m = DECL0.match(l)
        if m:
            j = i + 1
            while j < len(ls) and not BOUND.match(ls[j]):
                j += 1
            while j > i + 1 and not ls[j - 1].strip():
                j -= 1
            out.append((m.group(1), m.group(2), i, j - 1))
    return out, ls


def sorry_sites(files=None):
    """Each `sorry` of the kernel's Lean (the gate refuses the word anywhere, comments too): (file, line, the
    declaration it is in, that declaration's length in lines, how it sits: "whole" when it is the whole proof,
    "inside" a longer proof, or "comment")."""
    out = []
    for p in files if files is not None else lean_files():
        raw = open(p).read()
        ds, ls = lean_decls(p)
        code = "\n".join(ls)
        for m in re.finditer(r"\bsorry\b", raw):
            line = raw.count("\n", 0, m.start())
            d = next((d for d in reversed(ds) if d[2] <= line <= d[3]), None)
            name, n = (d[1], d[3] - d[2] + 1) if d else ("?", 0)
            if code[m.start():m.end()] != "sorry":
                how = "comment"
            else:
                body = "\n".join(ls[d[2]:d[3] + 1]) if d else ""
                how = "whole" if re.fullmatch(r"\s*(by\s+)?sorry\s*", body.split(":=", 1)[-1]) else "inside"
            out.append((p, line + 1, name, n, how))
    return out


def sorry_line(s, base):
    p, line, name, n, how = s
    where = f"{os.path.relpath(p, base)}:{line}" + (f" in {name}" if name != "?" else "")
    if how == "comment":
        return f"{where}: in a comment (the gate refuses the word anywhere)"
    if how == "whole":
        return f"{where} (its whole proof)"
    return f"{where}, inside its {n}-line proof" + (": state what is left as its own lemma (layer 2 if it is about "
                                                     "values) and use it here" if n > LONG_PROOF else "")


def module_doc(raw):
    m = re.search(r"/-!(.*?)-/", raw, re.S)
    return m.group(1).strip() if m else ""


def signature(raw_ls, code_ls, kind, a, b, limit=45):
    """What a declaration (lines a..b) states: a theorem up to the end of its statement (`:=`), a definition
    whole (an invariant is its body), at most `limit` lines."""
    end = a
    if kind in ("theorem", "lemma"):
        while end < b and ":=" not in code_ls[end]:
            end += 1
    else:
        end = b
    cut = min(end, a + limit - 1)
    return raw_ls[a:cut + 1] + (["    …"] if cut < end else [])


def cmd_outline(argv):
    k = argv[0] if argv else K
    lean_root = os.path.join(ROOT, "kernels", k, "lean")
    files = sorted(glob.glob(os.path.join(lean_root, "**", "*.lean"), recursive=True))
    if not files:
        die(f"no Lean files in kernels/{k}/lean/ (`tva-kit outline KERNEL [FILE]`; the examples: "
            f"{', '.join(CONFIG.get('examples', [])) or 'see kernels/'})")
    mods = {os.path.relpath(p, lean_root)[:-5].replace(os.sep, "."): p for p in files}
    imports = {m: [x for x in re.findall(r"^import\s+(\S+)", open(p).read(), re.M) if x in mods] for m, p in mods.items()}
    order, seen = [], set()

    def visit(m):
        if m not in seen:
            seen.add(m)
            for x in sorted(imports[m]):
                visit(x)
            order.append(m)
    for m in sorted(mods):
        visit(m)
    sorries = {}
    if k == K:
        for s in sorry_sites(files):
            sorries.setdefault((s[0], s[2]), []).append(s[4])
    if len(argv) > 1:  # one file's statements
        q = argv[1][:-5] if argv[1].endswith(".lean") else argv[1]
        hit = [m for m in order if m == q or m.endswith("." + q)]
        if not hit:
            die(f"no file {argv[1]!r} in kernels/{k}/lean/: {', '.join(order)}")
        p = mods[hit[0]]
        raw_ls = open(p).read().split("\n")
        ds, code_ls = lean_decls(p)
        print(f"== kernels/{k}/lean/{os.path.relpath(p, lean_root)}: the statements ({len(ds)} declarations)")
        for kind, name, a, b in ds:
            print("\n".join(signature(raw_ls, code_ls, kind, a, b)) +
                  (f"\n    -- sorry ({', '.join(sorries[(p, name)])})" if (p, name) in sorries else ""))
        return 0
    total = sum(open(p).read().count("\n") for p in files)
    print(f"kernels/{k}/lean: {len(files)} files, {total} lines, bottom-up (each file after the files it imports)")
    for m in order:
        p = mods[m]
        raw = open(p).read()
        ds, _ = lean_decls(p)
        print(f"\n== {os.path.relpath(p, lean_root)} ({raw.count(chr(10))} lines" +
              (f"; imports {', '.join(imports[m])}" if imports[m] else "") + ")")
        doc = module_doc(raw).split("\n")
        print("\n".join("   " + l for l in doc[:24]) + ("\n   …" if len(doc) > 24 else ""))
        for kinds, label in ((("theorem", "lemma"), "theorems"), (("def", "abbrev", "structure", "inductive", "class"),
                                                                      "definitions")):
            marked = [f"{n} (sorry: {', '.join(sorries[(p, n)])})" for kd, n, a, b in ds if kd in kinds and (p, n) in sorries]
            names = marked + [n for kd, n, a, b in ds if kd in kinds and (p, n) not in sorries]
            if names:
                print(f"   {label}: " + ", ".join(names[:40]) + (f" … ({len(names) - 40} more)" if len(names) > 40 else ""))
    print(f"\n`tva-kit outline {k} FILE` prints one file's statements; `tva-kit lookup NAME` one declaration in full.")
    return 0


# ---------------------------------------------------------------- the prover's checklist

def service_status():
    url = os.environ.get("TVA_REVIEW_URL", CONFIG.get("service", "http://reviewer:8765"))
    import urllib.request
    try:
        with urllib.request.urlopen(url + "/status", timeout=10) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001
        return None


LEAN_CONST = re.compile(r"^(?:@\[[^\]]*\]\s*)?(?:noncomputable\s+)?(?:abbrev|def)\s+([A-Za-z_][\w.']*)((?:\s+\(?\w+(?:\s*:\s*[^)]*)?\)?)*)\s*:\s*[^:=\n]+:=\s*(\S.*)$")
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
ASCRIBE = re.compile(r"\(\s*([^():]+?)\s*:\s*[ℤℕℚℝ]\s*\)")


def norm_expr(e):
    """An expression as README and Lean write it, compared as text: no spaces, no type ascriptions, `2^(-10)`
    as `2^-10`, `·` and `*` alike."""
    e = ASCRIBE.sub(r"\1", e)
    e = re.sub(r"\s+", "", e).replace("·", "*").replace("−", "-")
    return re.sub(r"\^\((-?\d+)\)", r"^\1", e)


def eval_expr(e):
    """The value of a numeric expression (digits, + - * / ^ and parentheses), or None."""
    from fractions import Fraction
    if not re.fullmatch(r"[\d+\-*/^().]+", e) or "**" in e:
        return None
    try:
        v = eval(re.sub(r"(\d+(?:\.\d+)?)", r"Fraction('\1')", e.replace("^", "**")), {"__builtins__": {}},
                 {"Fraction": Fraction})
        return Fraction(v)
    except Exception:  # noqa: BLE001
        return None


def stale_constants(readme, lean_texts):
    """[(README line, name, README's value, the Lean's)]: each `NAME = value` (in backticks) that README states
    for a one-line `def NAME … := value` of the Lean, where the two differ (a constant changed in the Lean
    and not in the write-up is a DOC_WRONG finding)."""
    from fractions import Fraction
    defs = {}
    for t in lean_texts:
        code = strip_comments(t).split("\n")
        for i, l in enumerate(code):
            m = LEAN_CONST.match(l)
            if m and not (i + 1 < len(code) and code[i + 1].startswith((" ", "\t")) and code[i + 1].strip()):
                nargs = sum(len(g[0].split(":")[0].split()) if g[0] else 1
                            for g in re.findall(r"[({\[]([^)}\]]*)[)}\]]|(\w+)", m.group(2)))
                defs[m.group(1).split(".")[-1]] = (m.group(1), nargs, norm_expr(m.group(3)), m.group(3).strip())
    out = []
    for n, line in enumerate(readme.split("\n"), 1):
        for span in re.findall(r"`([^`]+)`", line):
            for m in re.finditer(r"(?<![\w.])([A-Za-z_][\w']*)((?:\s+[a-z]\w*)*)\s*=\s*([^=,;≤≥<>]+?)\s*(?=$|,|;|≤|≥|<|>|\bwhere\b)", span):
                name, args, val = m.group(1), m.group(2).split(), norm_expr(m.group(3))
                if name not in defs or len(args) != defs[name][1]:
                    continue
                lean, shown = defs[name][2], defs[name][3]
                stated = " ".join([name, *args])
                a, b = eval_expr(val), eval_expr(lean)
                if a is not None and b is not None:
                    dec = re.search(r"\.(\d+)$", val)  # a decimal README rounds to its digits
                    if abs(a - b) > (Fraction(51, 100) * Fraction(1, 10 ** len(dec.group(1))) if dec else 0):
                        out.append((n, stated, m.group(3).strip(), shown))
                elif a is None and b is None and val != lean and NUM.sub("N", val) == NUM.sub("N", lean):
                    # the same expression with other numbers (`relQ d + 2^-9` for `relQ d + 2^-10`)
                    out.append((n, stated, m.group(3).strip(), shown))
    return out


def stages():
    """[(name, done, detail)] for the prover's proof, from the files."""
    out = []
    man = manifest()
    s = man.get("source", {})
    probs, missing = ([], [])
    try:
        mm = load(os.path.join(KD, "model-map.json"), {"entries": []})
        probs, missing = tvr.map_problems(man, mm, source_text(), lean_text())
    except Exception as e:  # noqa: BLE001
        mm, probs = {"entries": []}, [str(e)]
    cls_probs = [p for p in probs if "lies neither" in p or "leave out" in p or "outside every function" in p]
    out.append(("2 classes", bool(s.get("ranges")) and not cls_probs,
                "kernel.json source.ranges/host: `tva-kit classify --write`" if not s.get("ranges") else
                (short(cls_probs[0], 200) if cls_probs else "set")))
    files = lean_files()
    text = lean_text()
    out.append(("3 model", bool(files) and "def model" in text and "Kernel" in text,
                "write the model (`def <kernel> … : Kernel where body ctx := do …`, `def model`, `def launch`) in "
                f"lean/{man.get('lean', {}).get('module', '?')}.lean, from a certified example" if not files else
                f"{len(files)} Lean file(s)"))
    todo = sum(1 for e in mm.get("entries", []) if "TODO" in json.dumps(e))
    out.append(("4 map", bool(mm.get("entries")) and not missing and not todo and not probs,
                "`tva-kit map-skeleton`, then fill the TODOs" if not mm.get("entries") else
                f"{len(mm['entries'])} entries, {todo} TODO, {len(missing)} lines uncovered, {len(probs)} problem(s)"))
    out.append(("5 claim", all(x in text for x in ("structure Inst", "Inst.ok", "def Pre", "def Post", "def Spec",
                                                   "structure Assumptions")),
                "Inst, Inst.ok, Pre, Post, Spec, Assumptions"))
    tops = [x for x in ("main", "consistent", "satisfiable") if f"theorem {x}" in text]
    sites = sorry_sites(files)
    in_main = [x for x in sites if x[2] == "main" and x[4] != "comment"]
    deep = [x for x in sites if x[4] == "inside" and x[3] > LONG_PROOF]
    out.append(("6 design", len(tops) == 3 and not in_main and not deep,
                f"state every lemma (`sorry` as whole proofs only) and prove main, consistent, satisfiable from them; "
                f"written: {', '.join(tops) or 'none'}" if len(tops) < 3 else
                "`main` has a `sorry`: prove it from the statements of the lemmas below it" if in_main else
                f"a `sorry` inside a long proof: {sorry_line(deep[0], KD)}" if deep else
                f"the statements compose; {len(sites)} `sorry` left"))
    out.append(("7 audit", os.path.exists(AUDIT),
                "~/scratch/audit.md written: fix what it reports before the long proofs" if os.path.exists(AUDIT) else
                "run the claim-audit subagent on the claim and the design (it writes ~/scratch/audit.md)"))
    out.append(("8 lemmas", len(tops) == 3 and not sites,
                f"{len(sites)} `sorry` left in {len({(x[0], x[2]) for x in sites})} declaration(s): `tva-kit stage` "
                "lists them" if sites else "all proved" if len(tops) == 3 else "main, consistent, satisfiable not all written"))
    readme = open(os.path.join(KD, "README.md")).read() if os.path.exists(os.path.join(KD, "README.md")) else ""
    empty = [h for h, b in tvr.md_sections(readme)[1] if not "".join(b).strip()]
    oldc = stale_constants(readme, [open(f).read() for f in files]) if readme else []
    out.append(("9 write-up", not empty, (f"README.md sections still empty: {', '.join(empty)}" if empty else "filled") +
                (f"; {len(oldc)} constant(s) README may state stale (first: README.md:{oldc[0][0]} `{oldc[0][1]} = "
                 f"{oldc[0][2]}`; the Lean: `{oldc[0][3]}`): `tva-kit gate` lists them" if oldc else "")))
    g = gate()
    stale = tvr.gate_staleness(K, g) if g.get("digest") else ["never run"]
    out.append(("10 gate", bool(g.get("ok")) and not stale,
                f"PASS {g['digest'][:16]}" if g.get("ok") and not stale else
                ("FAIL" if g.get("digest") and not g.get("ok") else "not run on the current files") + ": `tva-kit gate`"))
    return out


SIDE_STAGES = ("4 map", "9 write-up")  # due before the gate; once the claim is written, proof work need not wait for them


def next_stage(st):
    """(the stage to work on, the side stages still open): the first unfinished stage, except that the map and
    the write-up do not hold up the proof once the claim is written (a prover told "next: 4 map" for hours
    went on proving without the design stage's advice)."""
    todo = [name for name, done, _ in st if not done]
    claim = any(name == "5 claim" and done for name, done, _ in st)
    side = [n for n in todo if n in SIDE_STAGES and claim]
    main = [n for n in todo if n not in side]
    if main == ["10 gate"]:  # the gate cannot pass without them
        main, side = side + main, []
    return (main[0] if main else side[0] if side else None), [n for n in side if main]


def cmd_stage(argv):
    st = stages()
    for name, done, detail in st:
        print(f"[{'x' if done else ' '}] {name:11s} {detail}")
    nxt, side = next_stage(st)
    sites = sorry_sites()
    if sites:
        print(f"`sorry` left ({len(sites)}):\n" + "\n".join("  " + sorry_line(s, KD) for s in sites[:10]) +
              (f"\n  … and {len(sites) - 10} more" if len(sites) > 10 else ""))
    svc = service_status()
    if svc:
        print(f"service: {svc['status']}, round {svc.get('round')}, certificate "
              f"{'GRANTED' if svc.get('certificate') else 'not yet'}; next: {svc.get('next')}")
    if clock_line():
        print(clock_line())
    if sites and nxt in ("8 lemmas", "9 write-up", "10 gate"):
        print("a lemma that resists: `tva-kit try start LEMMA` runs several subagents at it at once, each with its "
              "own approach (best of N)")
    deep = [x for x in sites if x[4] == "inside" and x[3] > LONG_PROOF]
    if deep and nxt != "6 design":
        print(f"design: {len(deep)} `sorry` inside a proof over {LONG_PROOF} lines (first: {sorry_line(deep[0], KD)}); "
              "state what is left as its own lemma first (/.claude/skills/tva-design/SKILL.md)")
    if nxt:
        skill = STAGE_SKILLS.get(nxt)
        print(f"next stage: {nxt}" + (f"; how to do it: read /.claude/skills/{skill}/SKILL.md" if skill else "") +
              (f" (also before the gate: {', '.join(side)})" if side else ""))
    return 0


PROGRESS = os.path.join(SCRATCH, "progress.md")
TIME_LOG = os.path.join(SCRATCH, ".time-log.json")  # [[seconds used, note], …], written by `tva-kit clock NOTE`


def cmd_clock(argv):
    """The time used and left (a model's own sense of it can be hours off); with a note, the note recorded
    with the time used, in ~/scratch/progress.md and the time log."""
    c = clock()
    if not c:
        die("no clock: it starts at the prover's first session")
    used, limit = c
    note = " ".join(argv).strip()
    log = load(TIME_LOG, []) or []
    print(clock_line() + (f"; submit by {hm(limit * 0.75)} ({hm(limit * 0.75 - used)} from now)"
                          if used < limit * 0.75 else "; three quarters are gone: submit now"))
    if log:
        print(f"last note, {hm(used - log[-1][0])} ago at {hm(log[-1][0])}: {short(log[-1][1], 200)}")
    if note:
        os.makedirs(SCRATCH, exist_ok=True)
        log.append([round(used), note])
        with open(TIME_LOG, "w") as f:
            json.dump(log, f)
        with open(PROGRESS, "a") as f:
            f.write(f"- [{hm(used)}] {note}\n")
        print(f"recorded in ~/scratch/progress.md: [{hm(used)}] {short(note, 200)}")
    return 0


STAGE_SKILLS = {"3 model": "tva-model-map", "4 map": "tva-model-map", "5 claim": "tva-claim", "6 design": "tva-design",
                "7 audit": "tva-design", "8 lemmas": "tva-proof", "10 gate": "tva-gate"}


def cmd_brief(argv):
    if ROLE == "reviewer":
        return cmd_review_brief(["--short"])
    man = manifest()
    print(f"prover of `{K}`: {man.get('title', K)}\nLean module {man['lean']['module']}, namespace "
          f"{man['lean']['namespace']}; your kernel folder: ~/fpv/tva/kernels/{K}/")
    text = source_text()
    jit = tvr.jit_functions(text) or {}
    print("source: " + man["source"]["file"] + f" ({text.count(chr(10))} lines); @triton.jit: " +
          ", ".join(f"{n} (lines {a}-{b})" for n, spans in jit.items() for a, b in spans))
    if os.path.exists(os.path.join(KD, "SETUP.md")):
        print(f"SETUP.md: the launch to cover (read it): ~/fpv/tva/kernels/{K}/SETUP.md")
    ex = CONFIG.get("examples", [])
    if ex:
        print("closest certified examples (read their lean/ and model-map.json): " +
              ", ".join(f"~/fpv/tva/kernels/{e}/" for e in ex) +
              f"; the design of a proof, file by file: `tva-kit outline {ex[0]}`")
    cmd_stage([])
    return 0


# ---------------------------------------------------------------- findings and responses (prover)

def finding_lines(i, f):
    out = [f"{i} [{f['state']}{'' if f['blocking'] else ', non-blocking'}] {f['kind']} on {f['target']} "
           f"(by {', '.join(sorted(f['raisers']))}, round {f['round']})",
           f"  summary: {f.get('summary', '')}"]
    for key in ("detail", "evidence"):
        if f.get(key):
            out.append(f"  {key}: {f[key]}")
    if f.get("affects"):
        out.append(f"  affects: {', '.join(f['affects'])}")
    for h in f.get("history", [])[1:]:
        out.append(f"  {h['type']} by {h['actor']} (round {h.get('round')}): "
                   f"{short(h.get('note') or h.get('text') or h.get('response') or h.get('decision') or '', 300)}")
    return out


def cmd_findings(argv):
    S = state()
    show_all = "--all" in argv
    ids = sorted(S["findings"], key=lambda x: int(x[1:]))
    sel = [i for i in ids if show_all or S["findings"][i]["state"] == "OPEN"]
    if not sel:
        print("no open finding" + ("" if show_all else " (--all shows every finding)"))
        return 0
    for i in sel:
        print("\n".join(finding_lines(i, S["findings"][i])))
    N = S["round"]
    reports = sorted(glob.glob(os.path.join(KD, "review", f"round-{N}", "*", "report*.md")))
    if reports:
        print("the reviewers' reports: " + ", ".join(os.path.relpath(p, ROOT) for p in reports))
    return 0


def respond_paths():
    N, rp, np_ = tvr.kernel_round_files(K, "prover")
    return N, rp, np_, os.path.join(SCRATCH, f"respond-round-{N}.json")


def write_notes(N, resp, side):
    g = gate()
    gate_line = f"PASS, digest `{g['digest'][:16]}`" if g.get("ok") else "TODO (the gate has not passed: `tva-kit gate`)"
    ch = side.get("changes") or []
    rq = side.get("requests") or []
    txt = (f"# Prover's notes on {K}, round {N}\n\n## Changes\n\n" +
           ("".join(f"- {c}\n" for c in ch) or "None.\n") + "\n## Responses\n\n" +
           "".join(f"- **{r['id']}** {r['response']} — {r['text']}\n" for r in resp["responses"]) +
           "\n## Library requests\n\n" + ("".join(f"- {q}\n" for q in rq) or "None.\n") +
           f"\n## Gate\n\n{gate_line}\n")
    open(tvr.kernel_round_files(K, "prover")[2], "w").write(txt)


RESPOND_HELP = """tva-kit respond: answer the open findings (one batch answers all of them)
  tva-kit respond init                       write response.json and notes.md for the current round
  tva-kit respond set ID RESPONSE --text T [--changed item:NAME ...] [--issue L<n>]
      RESPONSE is REVISED (you changed the claim: name the changed items), DISPUTED (you argue it is wrong),
      CLARIFIED (wording only), DEFERRED (a narrowing you keep and state), REFERRED (about the library: --issue)
  tva-kit respond change TEXT                add a bullet to the notes' Changes
  tva-kit respond request TEXT               add a library request
  tva-kit respond show | lint                show them; check them as `tvr respond` will
The notes' Gate section is filled from the last passing gate: run `tva-kit gate` after your last change, then
`tva-kit respond show` (it rewrites the notes), `tva-kit respond lint`, and `tva-review submit`."""


def cmd_respond(argv):
    if not argv:
        print(RESPOND_HELP)
        return 0
    sub, rest = argv[0], argv[1:]
    N, rp, np_, sidep = respond_paths()
    if sub == "init":
        rc, out = run_tvr("start", K, "prover")
        print(out.strip())
        resp = load(rp)
        if resp is None:
            return 1
        side = load(sidep, {"changes": [], "requests": []})
        os.makedirs(SCRATCH, exist_ok=True)
        json.dump(side, open(sidep, "w"))
        write_notes(N, resp, side)
        print(f"open findings to answer: {', '.join(r['id'] for r in resp['responses'])} (`tva-kit findings`)")
        return 0
    resp = load(rp)
    if resp is None:
        die("no response.json for this round: run `tva-kit respond init`")
    side = load(sidep, {"changes": [], "requests": []})
    if sub == "set":
        if len(rest) < 2:
            die(RESPOND_HELP)
        fid, kind = rest[0], rest[1].upper()
        opts = {"--text": None, "--changed": [], "--issue": None}
        cur = None
        for a in rest[2:]:
            if a in opts:
                cur = a
            elif cur == "--changed":
                opts["--changed"].append(a if a.startswith("item:") else f"item:{a}")
            elif cur:
                opts[cur] = (opts[cur] + " " + a) if opts[cur] else a
        if kind not in tvr.RESPONSES:
            die(f"RESPONSE is one of {', '.join(sorted(tvr.RESPONSES))}")
        if not opts["--text"]:
            die("--text says what you did or why")
        ent = next((r for r in resp["responses"] if r["id"] == fid), None)
        if ent is None:
            die(f"{fid} is not an open finding of this round ({', '.join(r['id'] for r in resp['responses'])})")
        ent.clear()
        ent.update(id=fid, response=kind, text=opts["--text"])
        if kind == "REVISED":
            if not opts["--changed"]:
                die("REVISED names the items you changed: --changed item:spec item:model …")
            f = state()["findings"][fid]
            base = f.get("raise_items", f["items"])
            items = gate()["items"]
            same = [c for c in opts["--changed"] if items.get(c[5:]) == base.get(c[5:])]
            if same:
                print(f"warning: {', '.join(same)} did not change since the finding (by the last gate): run `tva-kit gate` "
                      "after your change, or name the items that did change")
            ent["changed"] = opts["--changed"]
        if kind == "REFERRED":
            if not opts["--issue"]:
                die("REFERRED names the library issue: --issue L<n> (docs/library-review/ISSUES.md)")
            ent["issue"] = opts["--issue"]
        json.dump(resp, open(rp, "w"), indent=2)
    elif sub in ("change", "request"):
        if not rest:
            die(f"tva-kit respond {sub} TEXT")
        side["changes" if sub == "change" else "requests"].append(" ".join(rest))
        os.makedirs(SCRATCH, exist_ok=True)
        json.dump(side, open(sidep, "w"))
    elif sub == "lint":
        write_notes(N, resp, side)
        rc, out = run_tvr("lint", K, "prover")
        print(out.strip())
        if rc == 0:
            print("next: `tva-review submit`")
        return rc
    elif sub != "show":
        die(RESPOND_HELP)
    write_notes(N, resp, side)
    for r in resp["responses"]:
        print(f"{r['id']}: {r['response']} — {short(r['text'], 200)}" + (f" (changed {', '.join(r['changed'])})" if r.get("changed") else ""))
    left = [r["id"] for r in resp["responses"] if r["response"] == "TODO"]
    if left:
        print(f"still to answer: {', '.join(left)}")
    return 0


# ---------------------------------------------------------------- the reviewer

def cmd_review_brief(argv):
    who = reviewer_id()
    S, g, man = state(), gate(), manifest()
    N = S["round"]
    print(f"reviewer {who} of `{K}`, round {N}; your folder: ~/fpv/tva/kernels/{K}/review/round-{N}/{who}/; "
          f"gate {g.get('digest', '')[:16]} {'PASS' if g.get('ok') else 'FAIL'}")
    acc = S["accepts"].get(who, {})
    print("items (what you accept, each at its digest):")
    for it, dg in sorted(g.get("items", {}).items()):
        mark = "accepted" if acc.get(it) == dg else ("STALE (changed since you accepted)" if it in acc else "-")
        kind = g.get("kinds", {}).get(it.split(":", 1)[1], "") if it.startswith("assumption:") else ""
        print(f"  item:{it:30s} {dg[:12]}  {mark}" + (f"  [{kind}]" if kind else ""))
    mine = {i: f for i, f in S["findings"].items() if who in f["raisers"]}
    if mine:
        print("your findings:")
        for i in sorted(mine, key=lambda x: int(x[1:])):
            f = mine[i]
            todo = " → release or re-raise it" if f["state"] == "ANSWERED" and who not in f["released"] else ""
            print(f"  {i} [{f['state']}] {f['kind']} {f['target']}: {short(f['summary'], 100)}{todo}")
            last = [h for h in f.get("history", []) if h["type"] == "respond"]
            if last:
                print(f"     prover: {last[-1]['response']} — {short(last[-1].get('text', ''), 260)}")
    if "--short" in argv:
        print("next: read claim.md and the Lean in full, then `tva-kit review-brief` (the map), `tva-kit targets`, "
              "`tva-kit filing init`")
        return 0
    rng = None
    if "--map" in argv:
        rng = parse_lines(argv[argv.index("--map") + 1])
    src = source_text().split("\n")
    mm = load(os.path.join(KD, "model-map.json"), {"entries": []})
    print("the source beside the model map (judge each line by SEMANTICS.md):")
    for e in mm.get("entries", []):
        try:
            ls = tvr.parse_lines(e["src"])
        except (KeyError, ValueError, TypeError):
            continue
        if rng and not any(rng[0] <= x <= rng[1] for x in ls):
            continue
        stext = " ⏎ ".join(src[x - 1].strip() for x in ls if 0 < x <= len(src))
        model = e.get("model")
        right = ("MODEL " + short(" / ".join(model) if isinstance(model, list) else model, 200)) if model else \
            f"DROP ({e.get('drop')}): {short(e.get('note', ''), 200)}"
        print(f"  L{e['src']}: {short(stext, 150)}\n      {right}")
    print(f"the Lean to read in full: {', '.join(os.path.relpath(p, ROOT) for p in lean_files())}")
    return 0


def valid_targets():
    S, g, man = state(), gate(), manifest()
    items = sorted(g.get("items", {}))
    decls_ = set()
    for v in tvr.snapshot(K, g.get("digest", "")).get("items", {}).values():
        for n in v.get("cone", {}):
            if n.startswith(man["lean"]["namespace"] + "."):
                decls_.add(n)
    lines = man["source"].get("ranges", []) + man["source"].get("host", [])
    return items, sorted(decls_), lines


def cmd_targets(argv):
    items, decls_, lines = valid_targets()
    g = gate()
    print("items: " + ", ".join(f"item:{i}" for i in items))
    print("source lines (src:N / map:N, N inside): " + ", ".join(f"{a}-{b}" for a, b in lines))
    print(f"declarations (decl:NAME), {len(decls_)}: " + ", ".join(decls_[:60]) + (" …" if len(decls_) > 60 else ""))
    print("documents: lib:<path> or doc:<path> of an existing file (e.g. doc:docs/SEMANTICS.md)")
    print("kinds (blocking?, allowed target prefixes, assumption kind required):")
    for kind, (blk, pref, need) in tvr.KINDS.items():
        print(f"  {kind:22s} {'blocking' if blk else 'non-blocking':12s} {', '.join(pref)}" + (f"  [{need}] only" if need else ""))
    kinds = g.get("kinds", {})
    if kinds:
        print("assumptions: " + ", ".join(f"item:assumption:{n} [{k}]" for n, k in sorted(kinds.items())))
    return 0


FILING_HELP = """tva-kit filing: write your filing.json and report.md together, regular by construction
  tva-kit filing init                          create them for this round (tvr start)
  tva-kit filing add --kind KIND --target T --summary S --detail D [--evidence E] [--affects item:X ...]
                                               raise a finding (`tva-kit targets` lists kinds and targets)
  tva-kit filing accept all | item:X ...       accept items (only those no blocking finding of yours holds)
  tva-kit filing release F<n> --note N         release a finding of yours the prover settled
  tva-kit filing reraise F<n> --note N         re-raise a finding of yours the prover did not settle
  tva-kit filing checked TEXT                  add a bullet to the report's Checked (at least one)
  tva-kit filing not-checked TEXT              add a bullet to the report's Not checked
  tva-kit filing remove finding|release|reraise|accept INDEX|ITEM
  tva-kit filing show | lint                   show; check as `tvr file` will (stop when it says regular)"""


def filing_paths():
    who = reviewer_id()
    N, fp, rp = tvr.kernel_round_files(K, "reviewer", who)
    return who, N, fp, rp


def report_lists(rp):
    """The bullets of the report's Checked and Not checked sections, as the reviewer left them."""
    if not os.path.exists(rp):
        return [], []
    secs = dict(tvr.md_sections(open(rp).read())[1])
    def bl(sec, empty):
        return [b for b in tvr.bullets(secs.get(sec, [])) if "TODO" not in b and b != empty]
    return bl("Checked", ""), bl("Not checked", "Nothing.")


def write_report(who, N, filing, rp, checked, not_checked):
    items = sorted(gate().get("items", {}))
    acc = filing.get("accept", [])
    accepted = set(items) if acc == "all" else {a[5:] for a in acc}
    blocking = [f for f in filing.get("findings", []) if tvr.KINDS.get(f.get("kind"), (False,))[0]]
    verdict = "accept" if accepted == set(items) and not blocking and not filing.get("reraise") else \
        ("reject" if not accepted else "accept-partly")
    bul = [f"- **{f['kind']}** `{f['target']}` — {f['summary']}" + (f" {f['detail']}" if f.get("detail") else "")
           for f in filing.get("findings", [])]
    bul += [f"- **release {r['id']}** — {r['note']}" for r in filing.get("release", [])]
    bul += [f"- **re-raise {r['id']}** — {r['note']}" for r in filing.get("reraise", [])]
    if acc == "all":
        acc_txt = "All items.\n"
    else:
        acc_txt = "".join(f"- `{a}`\n" for a in acc) or "None.\n"
    txt = (f"# Review of {K}, round {N}, by {who}\n\n## Verdict\n\n{verdict}\n\n## Findings\n\n" +
           ("\n".join(bul) + "\n" if bul else "None.\n") + f"\n## Accepted\n\n{acc_txt}\n## Checked\n\n" +
           ("".join(f"- {c}\n" for c in checked) or "- TODO (what you checked: `tva-kit filing checked TEXT`)\n") +
           "\n## Not checked\n\n" + ("".join(f"- {c}\n" for c in not_checked) or "Nothing.\n"))
    open(rp, "w").write(txt)
    return verdict


def held_items(who, filing, items):
    """The items an unresolved blocking finding of this reviewer holds (old ones and this filing's)."""
    S = state()
    released = {r["id"] for r in filing.get("release", [])}
    held = set()
    for i, f in S["findings"].items():
        if who in f["raisers"] and f["blocking"] and f["state"] not in ("CLOSED", "REFERRED") and i not in released:
            if f["state"] != "ANSWERED" or i in {r["id"] for r in filing.get("reraise", [])}:
                held |= tvr.affected(K, f, items)
    for f in filing.get("findings", []):
        if tvr.KINDS.get(f.get("kind"), (False,))[0]:
            t = f.get("target", "")
            if t.startswith("item:"):
                held.add(t[5:])
            elif t.startswith(("map:", "src:")):
                held |= {"model", "map"}
            elif f.get("affects"):
                held |= {a[5:] for a in f["affects"]}
            else:
                held |= set(items)
    return held


def cmd_filing(argv):
    if not argv:
        print(FILING_HELP)
        return 0
    sub, rest = argv[0], argv[1:]
    who, N, fp, rp = filing_paths()
    if sub == "init":
        rc, out = run_tvr("start", K, "reviewer", who)
        print(out.strip())
        if rc:
            return rc
        filing = load(fp)
        write_report(who, N, filing, rp, *report_lists(rp))
        print("next: judge the claim; record with `tva-kit filing add/accept/release/reraise/checked`")
        return 0
    filing = load(fp)
    if filing is None:
        die("no filing for this round yet: run `tva-kit filing init` (if filing.json exists but is not JSON, fix it)")
    checked, not_checked = report_lists(rp)
    g = gate()
    items = g.get("items", {})

    def opts(names, multi=()):
        o, cur = {n: ([] if n in multi else None) for n in names}, None
        for a in rest:
            if a in o:
                cur = a
            elif cur in multi:
                o[cur].append(a)
            elif cur:
                o[cur] = (o[cur] + " " + a) if o[cur] else a
        return o

    if sub == "add":
        o = opts(["--kind", "--target", "--summary", "--detail", "--evidence", "--affects"], multi=("--affects",))
        kind, target = (o["--kind"] or "").upper(), (o["--target"] or "").strip()
        if kind not in tvr.KINDS:
            die(f"--kind is one of {', '.join(tvr.KINDS)}")
        blk, pref, need = tvr.KINDS[kind]
        if not any(target.startswith(p) for p in pref):
            die(f"{kind} takes a target starting {' or '.join(pref)} (`tva-kit targets`)")
        if not tvr.valid_target(K, target, g, manifest()):
            die(f"no target {target!r}: `tva-kit targets` lists the items, lines and declarations")
        if need and g.get("kinds", {}).get(target.split(":", 2)[2], "") != need:
            die(f"{kind} applies to [{need}] assumptions; {target} is [{g.get('kinds', {}).get(target.split(':', 2)[2], '?')}]"
                + (" (a library fact: use LIBRARY_REPORT, or RULE_MISCITED if it is the wrong fact)" if need == "kernel" else ""))
        if not o["--summary"] or len(o["--summary"]) > 200:
            die("--summary is one line of at most 200 characters")
        if blk and not o["--detail"]:
            die("a blocking finding explains itself: --detail says what is wrong, where, and what would resolve it")
        f = {"kind": kind, "target": target, "summary": o["--summary"]}
        for key in ("detail", "evidence"):
            if o[f"--{key}"]:
                f[key] = o[f"--{key}"]
        if o["--affects"]:
            f["affects"] = [a if a.startswith("item:") else f"item:{a}" for a in o["--affects"]]
            bad = [a for a in f["affects"] if a[5:] not in items]
            if bad:
                die(f"--affects names no item: {', '.join(bad)}")
        filing.setdefault("findings", []).append(f)
        if filing.get("accept") == "all" and blk:
            filing["accept"] = sorted(f"item:{i}" for i in items if i not in held_items(who, filing, items))
            print("note: `accept all` narrowed to the items this blocking finding does not hold")
    elif sub == "accept":
        if not rest:
            die("tva-kit filing accept all | item:X ...")
        want = list(items) if rest == ["all"] else [a[5:] if a.startswith("item:") else a for a in rest]
        bad = [a for a in want if a not in items]
        if bad:
            die(f"no item {', '.join(bad)} (items: {', '.join(sorted(items))})")
        held = held_items(who, filing, items)
        blocked = sorted(a for a in want if a in held)
        cur = set(items) if filing.get("accept") == "all" else {a[5:] for a in filing.get("accept", [])}
        new = (cur | set(want)) - held
        if blocked:
            print(f"not accepted: {', '.join(blocked)} (a blocking finding of yours holds them; release it first, or "
                  "keep it and leave them unaccepted)")
        filing["accept"] = "all" if new == set(items) else sorted(f"item:{a}" for a in new)
    elif sub in ("release", "reraise"):
        o = opts(["--note"])
        fid = next((a for a in rest if re.fullmatch(r"F\d+", a)), None)
        if not fid or not o["--note"]:
            die(f"tva-kit filing {sub} F<n> --note TEXT")
        f = state()["findings"].get(fid)
        if not f or who not in f["raisers"]:
            die(f"{fid} is not a finding of yours")
        if sub == "reraise" and f["state"] not in ("ANSWERED", "REFERRED"):
            die(f"{fid} is {f['state']}: only an answered or referred finding can be re-raised")
        other = "reraise" if sub == "release" else "release"
        filing[other] = [r for r in filing.get(other, []) if r["id"] != fid]
        filing[sub] = [r for r in filing.get(sub, []) if r["id"] != fid] + [{"id": fid, "note": o["--note"]}]
    elif sub in ("checked", "not-checked"):
        if not rest:
            die(f"tva-kit filing {sub} TEXT")
        (checked if sub == "checked" else not_checked).append(" ".join(rest))
    elif sub == "remove":
        if len(rest) < 2:
            die("tva-kit filing remove finding INDEX | release F<n> | reraise F<n> | accept item:X")
        what, x = rest[0], rest[1]
        if what == "finding":
            filing["findings"].pop(int(x))
        elif what in ("release", "reraise"):
            filing[what] = [r for r in filing.get(what, []) if r["id"] != x]
        elif what == "accept":
            cur = set(items) if filing.get("accept") == "all" else {a[5:] for a in filing.get("accept", [])}
            filing["accept"] = sorted(f"item:{a}" for a in cur - {x[5:] if x.startswith("item:") else x})
    elif sub == "lint":
        write_report(who, N, filing, rp, checked, not_checked)
        rc, out = run_tvr("lint", K, "reviewer", who)
        print(out.strip())
        if rc == 0:
            print("regular: finish with your final message (the service records your filing; do not run `tvr file`)")
        return rc
    elif sub != "show":
        die(FILING_HELP)
    json.dump(filing, open(fp, "w"), indent=2)
    open(fp, "a").write("\n")
    verdict = write_report(who, N, filing, rp, checked, not_checked)
    print(f"{os.path.relpath(fp, ROOT)}: verdict {verdict}; {len(filing.get('findings', []))} finding(s), "
          f"{len(filing.get('release', []))} release(s), {len(filing.get('reraise', []))} re-raise(s); accepted: "
          f"{'all' if filing.get('accept') == 'all' else ', '.join(a[5:] for a in filing.get('accept', [])) or 'none'}; "
          f"Checked {len(checked)} bullet(s)")
    for i, f in enumerate(filing.get("findings", [])):
        print(f"  [{i}] {f['kind']} {f['target']}: {short(f['summary'], 100)}")
    S = state()
    owed = [i for i, f in S["findings"].items() if who in f["raisers"] and f["state"] == "ANSWERED"
            and who not in f["released"] and i not in {r["id"] for r in filing.get("release", []) + filing.get("reraise", [])}]
    if owed:
        print(f"still owed: release or re-raise {', '.join(sorted(owed, key=lambda x: int(x[1:])))}")
    left = sorted(set(items) - (set(items) if filing.get("accept") == "all" else {a[5:] for a in filing.get("accept", [])})
                  - held_items(who, filing, items))
    if left:
        print(f"items neither accepted nor held by a blocking finding of yours: {', '.join(left)}")
    if not checked:
        print("the report's Checked needs at least one bullet: `tva-kit filing checked TEXT`")
    return 0


# ---------------------------------------------------------------- best-of-N: attempts at one lemma (prover)

TRIES = os.path.join(SCRATCH, "tries")
TRY_AGENTS = os.path.join(TRIES, ".agents.json")  # {agent id: [lemma, attempt]}: which attempt a subagent works on
TRY_MARK = "-- ==== tva-kit try: the text above this line is fixed; below it, helper lemmas and the proof of `{}` ===="
TRY_END = "\nNat.zero : "  # how the sentinel `#check`'s answer starts: the guard's type ends before it
STRATEGIES = {
    "example": "Follow the closest certified example. Find the example lemma that plays this lemma's role "
               "(`tva-kit outline <example>`, `tva-kit examples REGEX`, `tva-kit lookup NAME`) and transcribe its "
               "proof step by step, adapting names, constants and hypotheses.",
    "decompose": "Decompose first. Before any tactic work, write two to five helper lemmas above the target, each "
                 "one fact the proof needs (on plain values where you can), with `sorry` proofs; prove the target "
                 "from them; then prove the helpers, the easiest first. Even if you run out of time, stated helpers "
                 "with the target proved from them are progress the prover can take.",
    "forward": "Work forward from the hypotheses. Unfold the definitions in the goal (`simp only [defs]`, "
               "`unfold`), collect the facts you need as `have`s with explicit bounds, and close with "
               "`linarith`, `nlinarith`, `gcongr`, `positivity`, `omega` or a `calc` chain.",
    "backward": "Work backward from the goal. `refine`, `apply` or `constructor` with a lemma whose conclusion "
                "matches, leaving `?_` holes; read each remaining goal (`trace_state`, or the check's log) and close "
                "the easiest first, turning a hard one into a helper lemma.",
    "generalize": "Generalize. If the goal is about one loop trip, one index or a fixed constant, state a more "
                  "general helper (for every n, by induction on the trip count, or for any bound), prove it, and "
                  "specialize it to the target.",
}
TRY_ORDER = ["example", "decompose", "forward", "backward", "generalize"]


def decls_of(text):
    """lean_decls on a text instead of a file."""
    ls = strip_comments(text).split("\n")
    out = []
    for i, l in enumerate(ls):
        m = DECL0.match(l)
        if m:
            j = i + 1
            while j < len(ls) and not BOUND.match(ls[j]):
                j += 1
            while j > i + 1 and not ls[j - 1].strip():
                j -= 1
            out.append((m.group(1), m.group(2), i, j - 1))
    return out, ls


def region_start(raw, code, i):
    """The first line of declaration i's block: its docstring and attribute lines directly above it."""
    while i > 0 and raw[i - 1].strip() and (not code[i - 1].strip() or code[i - 1].lstrip().startswith("@[")):
        i -= 1
    return i


def namespaces_at(code, n):
    """The namespaces open before line n (comments blanked)."""
    ns = []
    for l in code[:n]:
        m = re.match(r"^namespace\s+(\S+)", l)
        if m:
            ns.append(m.group(1))
        elif re.match(r"^section\b", l):
            ns.append("")
        elif re.match(r"^end\b", l) and ns:
            ns.pop()
    return [x for x in ns if x]


def find_lemma(name):
    """(file, raw lines, first line of its block, its last line, kind, full name) of the kernel declaration `name`."""
    hits = []
    for p in lean_files():
        raw = open(p).read().split("\n")
        ds, code = decls_of("\n".join(raw))
        for kind, n, a, b in ds:
            if n == name or n.endswith("." + name):
                full = n[len("_root_."):] if n.startswith("_root_.") else ".".join(namespaces_at(code, a) + [n])
                hits.append((p, raw, region_start(raw, code, a), b, kind, full, bool(re.match(r"^(@\[[^\]]*\]\s*)?private\b", code[a]))))
    if not hits:
        die(f"no declaration `{name}` in the kernel's Lean (`tva-kit stage` lists each `sorry` with its lemma)")
    if len(hits) > 1:
        die(f"`{name}` names {len(hits)} declarations: " + ", ".join(f"{h[5]} ({os.path.relpath(h[0], KD)})" for h in hits)
            + "; give the name as the file writes it, qualified enough to be unique")
    if hits[0][6]:
        die(f"`{name}` is private: make it non-private first (the attempts check it by its full name)")
    return hits[0][:6]


def try_dir(name):
    return os.path.join(TRIES, re.sub(r"[^\w.'-]", "_", name))


def try_task(name):
    t = load(os.path.join(try_dir(name), "task.json"))
    if not t:
        die(f"no attempts at `{name}`: start them with `tva-kit try start {name}`")
    return t


def try_over(t):
    """Why the attempts of task `t` are over ("" while they run): a winner, the prover's accept or stop, the deadline."""
    d = try_dir(t["name"])
    w = load(os.path.join(d, "WINNER"))
    if t.get("done"):
        return f"the prover has {t['done']}"
    if w:
        return f"attempt {w['attempt']} proved `{t['name']}`"
    if time.time() > t["deadline"]:
        return "the attempts' time is over"
    return ""


def lean_check(path, timeout=900):
    """`lake env lean path` from TVA's lean/ (a reader: it runs beside other readers, never during a build)."""
    with lake_lock(exclusive=False):
        try:
            p = subprocess.run(["lake", "env", "lean", path], cwd=os.path.join(ROOT, "lean"), capture_output=True,
                               text=True, timeout=timeout, errors="replace", env={**os.environ, "TVA_LAKE_LOCKED": "1"})
            return p.returncode, (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired:
            return 124, f"{path}:1:0: error: lean timed out after {timeout} s"


MSG = re.compile(r"^(\S+?\.lean):(\d+):(\d+): (error|warning|info): ?(.*)$")


def lean_messages(out):
    """[(line, kind, text)] of `lean`'s output, each message with the lines that follow it (Lean prints a
    `#check`'s answer with no location: the guard's comes last, after every message, and is cut off by guard_split)."""
    msgs = []
    for l in out.split("\n"):
        m = MSG.match(l)
        if m:
            msgs.append([int(m.group(2)), m.group(4), m.group(5)])
        elif msgs:
            msgs[-1][2] += "\n" + l
    return [tuple(m) for m in msgs]


def guard_split(out, full):
    """(Lean's messages before the guard's answer, the type the guard's `#check` printed for `full` or None).
    Lean prints a `#check`'s answer with no location, after every message: the last answer for `full` (with
    `@` before the name when the lemma has implicit binders), up to the sentinel's."""
    out = "\n" + out
    i = max(out.rfind("\n" + full + " : "), out.rfind("\n@" + full + " : "))
    j = out.find(TRY_END, i + 1)
    if i < 0 or j < i:
        return out[1:], None
    return out[1:i + 1], " ".join(out[out.index(" : ", i) + 3:j].split())


def try_run(t, k, text):
    """Check attempt k's text: (state, details, the messages of the region). The state is one of solved
    (no error, no `sorry`, the statement unchanged), partial (no error, the statement unchanged, `sorry` left),
    failing (errors), changed (the statement is not the original's), refused (the fixed text or a forbidden
    construct)."""
    d = os.path.join(try_dir(t["name"]), f"a{k}")
    mark = TRY_MARK.format(t["name"])
    if mark not in text:
        return "refused", [f"the marker line is gone: restore it (the original file is {try_dir(t['name'])}/prefix.lean "
                           "followed by the marker and the lemma)"], []
    head, region = text.split(mark, 1)
    if head != t["prefix"]:
        return "refused", [f"the text above the marker was changed: it is fixed (the original is "
                           f"{try_dir(t['name'])}/prefix.lean); helpers go below the marker"], []
    bad = []
    for what, pat in tvr.HYGIENE:
        for m in re.finditer(pat, region, re.M):
            if re.fullmatch(r"sorry|admit", m.group(0).strip()):
                continue
            bad.append(f"line {head.count(chr(10)) + region.count(chr(10), 0, m.start()) + 1}: {what} ({m.group(0).strip()[:24]})")
    for i, l in enumerate(strip_comments(region).split("\n")):
        if re.match(r"^\s*(namespace|end|section|import)\b", l):
            bad.append(f"line {head.count(chr(10)) + i + 1}: `{l.split()[0]}`: the attempt stays in the file's namespace")
    if not any(n == t["decl"] for _, n, _, _ in decls_of(region)[0]):
        bad.append(f"no declaration `{t['decl']}` below the marker: keep the lemma, with its statement unchanged")
    if bad:
        return "refused", bad, []
    chk = os.path.join(d, ".check.lean")
    open(chk, "w").write(text.rstrip("\n") + f"\n\nset_option pp.all true in\n#check @_root_.{t['full']}\n"
                         f"#check @_root_.Nat.zero\n")
    rc, out = lean_check(chk)
    out = out.replace(chk, os.path.join(d, "Try.lean"))
    open(os.path.join(d, "check.log"), "w").write(out)
    body, ty = guard_split(out, t["full"])
    msgs = lean_messages(body)
    first, end = head.count("\n") + 1, text.rstrip("\n").count("\n") + 1
    errs = [m for m in msgs if m[1] == "error"]
    if any(m[0] < first for m in errs):
        return "refused", [f"line {m[0]}: {short(m[2], 200)}" for m in errs if m[0] < first][:4] + [
            "errors in the fixed part: an import is not built (ask the prover to build the module)"], msgs
    if [m for m in errs if m[0] <= end]:
        return "failing", [f"line {m[0]}: {m[2].rstrip()}" for m in errs if m[0] <= end], msgs
    if t["type"] is not None and ty != t["type"]:
        return "changed", ["the lemma's statement (its type, as Lean prints it with pp.all) is not the original's: "
                           "restore the statement exactly; change only the proof"], msgs
    sorries = [m for m in msgs if m[1] == "warning" and "sorry" in m[2] and m[0] >= first] or (
        [(0, "warning", "sorry")] if re.search(r"\b(sorry|admit)\b", strip_comments(region)) else [])
    if sorries:
        return "partial", [f"line {m[0]}: {short(m[2], 120)}" for m in sorries], msgs
    return "solved", [], msgs


def try_save_status(t, k, state, details):
    d = os.path.join(try_dir(t["name"]), f"a{k}")
    region = open(os.path.join(d, "Try.lean")).read().split(TRY_MARK.format(t["name"]), 1)[-1]
    st = {"state": state, "details": details[:12], "time": time.time(), "lines": region.count("\n"),
          "errors": len(details) if state == "failing" else 0, "sorries": len(details) if state == "partial" else 0}
    json.dump(st, open(os.path.join(d, "status.json"), "w"), indent=1)
    if state == "solved":
        with contextlib.suppress(FileExistsError):
            fd = os.open(os.path.join(try_dir(t["name"]), "WINNER"), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            os.write(fd, json.dumps({"attempt": k, "time": time.time()}).encode())
            os.close(fd)
    return st


def try_brief(t, k, strategy, others, hint, seeded):
    d = os.path.join(try_dir(t["name"]), f"a{k}")
    left = max(0, t["deadline"] - time.time())
    hint = f"\n{hint}\n" if hint else ""
    return f"""# Attempt {k} of {t['n']} at `{t['name']}`

You are one of {t['n']} attempts, started at once, at the same lemma of the TVA kernel `{K}`. Each attempt has
its own approach; the first whose proof Lean accepts wins, and the others are stopped. Work only in your
attempt's folder: `{d}/`.

- The lemma: `{t['full']}` in `{os.path.relpath(t['file'], KD)}` (lines {t['first'] + 1}-{t['last'] + 1}).
- Your file: `{d}/Try.lean`: the kernel file up to the lemma, a marker line, then the lemma{' (seeded from an earlier attempt)' if seeded else ''}.
  The text above the marker is fixed. Below it: write the proof, and any helper lemmas you need above
  the lemma (new names, in the same namespace). Never change the lemma's statement (the check compares its
  type as Lean prints it). No `axiom`, `native_decide`, `set_option` beyond `maxHeartbeats`, or the like.
- Check: `tva-kit try check {t['name']} {k}`: Lean on your file (a minute or less for most files; it runs
  beside the other attempts), the errors with their lines, the whole output in `{d}/check.log`. Run it
  first, before you change anything, to see where you start.
- Look things up: `tva-kit lookup NAME`, `tva-kit outline`, `tva-kit examples REGEX`; for Mathlib, `#check` in
  `{d}/T.lean` (`import TVA`) with `cd ~/fpv/tva/lean && lake env lean {d}/T.lean`.
- Time: about {hm(left)} for the attempts (until the prover's {hm(t['deadline'] - t['clock0'])} mark at the latest).
  When the check says `solved`, stop and report. When the harness tells you that another attempt won or
  the time is over, stop at once and report.

## Your approach: {strategy}

{STRATEGIES[strategy]}
{hint}
The other attempts take: {', '.join(others) or 'none'}. Keep to your own approach: the point of several
attempts is that they try different routes.

## Report (at most 10 lines)

The check's last state (solved, partial with the `sorry`s left, or failing with the first error), the
helper lemmas you stated, and what you would try next.
"""


def try_start(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="tva-kit try start")
    ap.add_argument("name")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--minutes", type=int, default=45)
    ap.add_argument("--strategies", default="")
    ap.add_argument("--hint", action="append", default=[], help="added to every attempt's brief (once per line)")
    ap.add_argument("--from", dest="seed", type=int, default=0, help="start every attempt from the earlier attempt N's text")
    a = ap.parse_args(argv)
    if not 2 <= a.n <= 5:
        die("--n: 2 to 5 attempts (each runs Lean beside the others)")
    strategies = [s.strip() for s in a.strategies.split(",") if s.strip()] or TRY_ORDER[:a.n]
    bad = [s for s in strategies if s not in STRATEGIES]
    if bad or len(strategies) != a.n:
        die(f"--strategies: {a.n} of {', '.join(TRY_ORDER)} (comma-separated)" + (f"; unknown: {', '.join(bad)}" if bad else ""))
    p, raw, first, last, kind, full = find_lemma(a.name)
    d = try_dir(a.name)
    seed = None
    old = load(os.path.join(d, "task.json"))
    if old:
        if a.seed:
            sp = os.path.join(d, f"a{a.seed}", "Try.lean")
            if not os.path.exists(sp):
                die(f"--from {a.seed}: no attempt {a.seed} at `{a.name}`")
            seed = open(sp).read().split(TRY_MARK.format(old["name"]), 1)[-1]
        if not old.get("done") and not try_over(old):
            die(f"attempts at `{a.name}` are running (`tva-kit try status {a.name}`); `tva-kit try stop {a.name}` first")
        n = 1
        while os.path.exists(f"{d}.{n}"):
            n += 1
        os.rename(d, f"{d}.{n}")
    elif a.seed:
        die(f"--from {a.seed}: no earlier attempts at `{a.name}`")
    c = clock()
    used = c[0] if c else 0
    limit = c[1] if c else used + 3600 * 8
    minutes = max(10, min(a.minutes, int((limit - used) / 60) - 20))
    module = module_of(p)
    print(f"building {module}'s imports (the attempts import them) …", flush=True)
    build(module)  # its imports must be built; the module's own errors do not matter here
    prefix = "\n".join(raw[:first]) + "\n"
    region = seed if seed is not None else "\n" + "\n".join(raw[first:last + 1]) + "\n"
    decl = next(n for _, n, _, _ in decls_of("\n".join(raw[first:last + 1]))[0])
    t = {"name": a.name, "decl": decl, "full": full, "file": p, "module": module, "first": first, "last": last,
         "prefix": prefix, "region": "\n".join(raw[first:last + 1]), "n": a.n, "strategies": strategies,
         "deadline": time.time() + 60 * minutes, "clock0": time.time() - used, "started": time.time(), "type": None}
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "prefix.lean"), "w").write(prefix)
    text = prefix + TRY_MARK.format(a.name) + "\n" + "\n".join(raw[first:last + 1]) + "\n"
    os.makedirs(os.path.join(d, "a1"), exist_ok=True)
    open(os.path.join(d, "a1", "Try.lean"), "w").write(text)
    st, det, _ = try_run(t, 1, text)
    t["type"] = guard_split(open(os.path.join(d, "a1", "check.log")).read(), full)[1]
    if st == "refused" or not t["type"]:
        die(f"cannot start: Lean on the file up to `{a.name}` ({d}/a1/check.log):\n  " + "\n  ".join(det[:6] or
            ["the lemma's statement does not elaborate"]))
    json.dump(t, open(os.path.join(d, "task.json"), "w"), indent=1)
    base = st
    for k in range(1, a.n + 1):
        ad = os.path.join(d, f"a{k}")
        os.makedirs(ad, exist_ok=True)
        open(os.path.join(ad, "Try.lean"), "w").write(prefix + TRY_MARK.format(a.name) + "\n" + region.lstrip("\n"))
        others = [s for j, s in enumerate(strategies) if j != k - 1]
        open(os.path.join(ad, "BRIEF.md"), "w").write(
            try_brief(t, k, strategies[k - 1], others, "\n".join(a.hint), seed is not None))
    print(f"{a.n} attempts at `{full}` ({os.path.relpath(p, KD)}:{first + 1}-{last + 1}; as it stands: {base}), "
          f"{minutes} min, in {d}/a1 … a{a.n}.\nNow start them all in ONE message (they run at once; you get the "
          f"reports when the last one ends, and the first to prove the lemma stops the others):")
    for k in range(1, a.n + 1):
        print(f"  Agent(subagent_type=\"proof-attempt\", description=\"attempt {k} at {a.name}\", "
              f"prompt=\"Attempt {k} at {a.name}: run `tva-kit try brief {a.name} {k}` and follow it. <add what you "
              f"know: the goal, the facts it needs, the example lemma>\")")
    print(f"Then `tva-kit try status {a.name}` and `tva-kit try accept {a.name}` (the winner), or "
          f"`tva-kit try accept {a.name} K --partial` to take attempt K's helpers with their `sorry`s, or a new "
          f"round from the best one: `tva-kit try start {a.name} --from K`.")
    return 0


def try_lines(t):
    w = load(os.path.join(try_dir(t["name"]), "WINNER"))
    out = []
    for k in range(1, t["n"] + 1):
        st = load(os.path.join(try_dir(t["name"]), f"a{k}", "status.json")) or {"state": "not checked", "details": []}
        det = "; ".join(st.get("details") or [])[:160]
        out.append(f"  a{k} {t['strategies'][k - 1]:10s} {st['state']:11s}" + (" WINNER" if w and w["attempt"] == k else "")
                   + (f"  {det}" if det else ""))
    return out


def try_accept(t, k, partial):
    d = os.path.join(try_dir(t["name"]), f"a{k}")
    text = open(os.path.join(d, "Try.lean")).read()
    st, det, _ = try_run(t, k, text)
    try_save_status(t, k, st, det)
    if st != "solved" and not (partial and st == "partial"):
        die(f"attempt {k} is {st}" + (": " + "; ".join(det[:3]) if det else "") +
            ("" if st != "partial" else f" (`--partial` takes it with its `sorry`s)"))
    region = text.split(TRY_MARK.format(t["name"]), 1)[1].strip("\n")
    if st == "partial":
        tmp = os.path.join(d, ".region.lean")
        open(tmp, "w").write(region + "\n")
        inside = [s for s in sorry_sites([tmp]) if s[4] != "whole"]
        if inside:
            die(f"attempt {k}: a `sorry` that is not a lemma's whole proof ({', '.join(f'{s[2]}' for s in inside[:3])}); "
                "take only stated lemmas")
    raw = open(t["file"]).read()
    ls = raw.split("\n")
    ds, code = decls_of(raw)
    hit = [(region_start(ls, code, a), b) for _, n, a, b in ds if n == t["decl"]]
    if len(hit) != 1 or "\n".join(ls[hit[0][0]:hit[0][1] + 1]) != t["region"]:
        die(f"`{t['name']}` changed in {os.path.relpath(t['file'], KD)} since the attempts started: merge by hand from "
            f"{d}/Try.lean (below the marker)")
    a, b = hit[0]
    new = "\n".join(ls[:a] + region.split("\n") + ls[b + 1:])
    open(t["file"], "w").write(new)
    rc, out = build(t["module"])
    log = save_build_log(t["module"], out)
    errs, warns, sorries, lines = summarize_build(out)
    if rc != 0:
        open(t["file"], "w").write(raw)
        die(f"attempt {k} merged into {os.path.relpath(t['file'], KD)}, but `lake build {t['module']}` failed "
            f"({len(errs)} error(s); {log}), so the file is back as it was:\n" + "\n".join(lines[:8]) +
            "\n(a helper's name may clash with a declaration below the lemma: rename it in the attempt, then accept again)")
    t["done"] = f"accepted attempt {k}"
    json.dump(t, open(os.path.join(try_dir(t["name"]), "task.json"), "w"), indent=1)
    left = [x for x in sorry_sites([t["file"]]) if x[4] != "comment"]
    print(f"attempt {k} ({t['strategies'][k - 1]}) merged into {os.path.relpath(t['file'], KD)}: `lake build "
          f"{t['module']}` OK; {len(left)} `sorry` left in the file" + (": " + ", ".join(x[2] for x in left[:5]) if left else "")
          + ". `tva-kit stage` shows what is left.")
    return 0


TRY_HELP = """tva-kit try: several subagents at one lemma at once, each with its own approach (best of N)

  tva-kit try start NAME [--n 3] [--minutes 45] [--strategies a,b,c] [--hint TEXT] [--from K]
        copy the kernel file up to lemma NAME into ~/scratch/tries/NAME/a1 … aN and print the subagent calls
        to start in one message; approaches: example, decompose, forward, backward, generalize
  tva-kit try brief NAME K      attempt K's brief (the attempt's first command)
  tva-kit try check NAME K      Lean on attempt K: solved, partial, failing, changed or refused
  tva-kit try status [NAME]     every attempt's last state, and the winner
  tva-kit try accept NAME [K] [--partial]
        merge attempt K (default: the winner) into the kernel file and build it; --partial takes stated
        helper lemmas with `sorry` proofs
  tva-kit try stop NAME         end the attempts (the harness stops the subagents at their next step)
"""


def cmd_try(argv):
    if ROLE == "reviewer":
        die("tva-kit try is the prover's")
    if not argv or argv[0] not in ("start", "brief", "check", "status", "accept", "stop"):
        print(TRY_HELP)
        return 0 if not argv else 1
    sub, rest = argv[0], argv[1:]
    if sub == "start":
        return try_start(rest)
    if sub == "status" and not rest:
        names = sorted(n for n in os.listdir(TRIES) if os.path.exists(os.path.join(TRIES, n, "task.json"))) \
            if os.path.isdir(TRIES) else []
        if not names:
            print("no attempts yet (`tva-kit try start NAME`)")
        for n in names:
            t = load(os.path.join(TRIES, n, "task.json"))
            print(f"{t['name']}: " + (try_over(t) or f"running, {hm(max(0, t['deadline'] - time.time()))} left"))
        return 0
    if not rest:
        die(TRY_HELP)
    t = try_task(rest[0])
    if sub == "status":
        print(f"{t['name']} ({t['full']}): " + (try_over(t) or f"running, {hm(max(0, t['deadline'] - time.time()))} left"))
        print("\n".join(try_lines(t)))
        return 0
    if sub == "stop":
        t["done"] = t.get("done") or "stopped the attempts"
        json.dump(t, open(os.path.join(try_dir(t["name"]), "task.json"), "w"), indent=1)
        print(f"attempts at `{t['name']}` stopped")
        return 0
    if sub == "accept":
        ks = [x for x in rest[1:] if x.isdigit()]
        w = load(os.path.join(try_dir(t["name"]), "WINNER"))
        if not ks and not w:
            die(f"no attempt has proved `{t['name']}`: give the attempt (`tva-kit try accept {t['name']} K --partial`)")
        return try_accept(t, int(ks[0]) if ks else w["attempt"], "--partial" in rest)
    if len(rest) < 2 or not rest[1].isdigit() or not 1 <= int(rest[1]) <= t["n"]:
        die(f"tva-kit try {sub} {t['name']} K, with K from 1 to {t['n']}")
    k = int(rest[1])
    d = os.path.join(try_dir(t["name"]), f"a{k}")
    if sub == "brief":
        print(open(os.path.join(d, "BRIEF.md")).read())
        over = try_over(t)
        if over:
            print(f"NOTE: {over}: the attempts are over; report and stop.")
        return 0
    t0 = time.time()
    st, det, _ = try_run(t, k, open(os.path.join(d, "Try.lean")).read())
    try_save_status(t, k, st, det)
    w = load(os.path.join(try_dir(t["name"]), "WINNER"))
    print(f"attempt {k}: {st} ({time.time() - t0:.0f} s; the whole output: {d}/check.log)")
    for x in det[:8]:
        print("  " + short(x, 400) if st != "failing" else "  " + x[:1500])
    if st == "solved":
        print("you proved it: stop and report." if w and w["attempt"] == k else
              f"proved, but attempt {w['attempt']} was first: stop and report.")
    elif try_over(t):
        print(f"{try_over(t)}: stop now and report your state.")
    else:
        print(f"{hm(max(0, t['deadline'] - time.time()))} left for the attempts")
    return 0 if st == "solved" else 1


COMMANDS = {"brief": cmd_brief, "source": cmd_source, "examples": cmd_examples, "lookup": cmd_lookup, "doc": cmd_doc,
            "outline": cmd_outline, "stage": cmd_stage, "clock": cmd_clock, "classify": cmd_classify, "map-skeleton": cmd_map_skeleton, "map-check": cmd_map_check,
            "build": cmd_build, "gate": cmd_gate, "findings": cmd_findings, "respond": cmd_respond,
            "review-brief": cmd_review_brief, "targets": cmd_targets, "filing": cmd_filing,
            "try": cmd_try}


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help", "help") or argv[1] not in COMMANDS:
        print(__doc__)
        return 0 if len(argv) < 2 or argv[1] in ("-h", "--help", "help") else 1
    if tvr is None:
        die(f"cannot load tvr from {ROOT}/tools")
    if not K or not os.path.isdir(KD):
        die(f"no kernel folder {KD} (set in {CONFIG_PATH})")
    return COMMANDS[argv[1]](argv[2:]) or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
