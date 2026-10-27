#!/usr/bin/env python3
"""tvr: TVA's prover/reviewer protocol (docs/REVIEW_PROTOCOL.md).

Each kernel is one folder, kernels/<kernel>/ (the layout is section 1 of the protocol).

  tvr new     KERNEL --module M --namespace NS --source FILE
                                scaffold kernels/KERNEL (folder, manifest, lake library)
  tvr check   KERNEL            run the gate: build, read the Lean claim (tools/claim_reader.lean.in); write gate.json,
                                claim.md and a snapshot of the item cones
  tvr assign  KERNEL REVIEWER   assign a reviewer
  tvr file    KERNEL REVIEWER   record the reviewer's review/round-N/REVIEWER/filing.json
  tvr respond KERNEL            record the prover's review/round-N/prover/response.json
  tvr decide  KERNEL ID TEXT    record the user's decision on an escalated finding
  tvr note    KERNEL TEXT       record a coordinator's note
  tvr status  KERNEL [--as R]   print the status (REVIEW.md has the whole record)
  tvr prompt  KERNEL prover|reviewer [ID]
  tvr prompt  library author|reviewer ROUND [ID]
                                print a role's prompt (docs/prompts/) with its fields filled in
  tvr index                     regenerate kernels/README.md and docs/library-review/REPORTS.md

Run it as `python3 -I tools/tvr.py ...` from the TVA root. Standard library only.
"""
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KROOT = os.path.join(ROOT, "kernels")
LAKE = os.environ.get("TVR_LAKE", os.path.expanduser("~/.elan/bin/lake"))
STD_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
KINDS = {
    # kind: (blocking by default, allowed target prefixes, assumption kind it may target)
    "MODEL_MISMATCH": (True, ("item:model", "map:", "src:", "decl:"), None),
    "MAP_INCOMPLETE": (True, ("item:map", "map:", "src:"), None),
    "UNSUPPORTED": (True, ("src:", "map:"), None),
    "CONDITION_TOO_NARROW": (True, ("item:conditions", "decl:"), None),
    "CONDITION_WRONG": (True, ("item:conditions", "decl:"), None),
    "SPEC_TOO_WEAK": (True, ("item:spec", "decl:"), None),
    "SPEC_WRONG": (True, ("item:spec", "decl:"), None),
    "RULE_MISCITED": (True, ("item:assumption:",), "rule"),
    "ASSUMPTION_FALSE": (True, ("item:assumption:",), "kernel"),
    "ASSUMPTION_TOO_STRONG": (True, ("item:assumption:",), "kernel"),
    "ASSUMPTION_UNCLEAR": (True, ("item:assumption:",), "kernel"),
    "EVIDENCE_REQUEST": (True, ("item:assumption:",), "kernel"),
    "LIBRARY_REPORT": (False, ("decl:", "lib:", "src:"), None),
    "DOC": (False, ("item:", "decl:", "src:", "map:", "lib:", "doc:"), None),
    "QUESTION": (False, ("item:", "decl:", "src:", "map:", "lib:", "doc:"), None),
}
RESPONSES = {"REVISED", "DISPUTED", "CLARIFIED", "DEFERRED", "REFERRED"}
DROPS = {"no-op:debug_barrier", "no-op:print", "no-op:cache", "no-op:range-kw",
         "decl", "config", "compile-check", "dead", "comment"}
NOTE_REQUIRED = {"config", "compile-check", "dead"}
ESCALATE_AFTER = 2  # re-raises after which a finding goes to the user
DOC_SECTIONS = ("Specification", "Conditions", "Assumptions", "Proof")
RULE_ARGS = ("Inst", "Inst.ok", "model", "launch")  # the kernel constants a [rule] may mention
LREVIEW = os.path.join(ROOT, "docs", "library-review")


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def file_sha(path):
    with open(path, "rb") as f:
        return sha(f.read())


def die(msg):
    print(f"tvr: {msg}", file=sys.stderr)
    sys.exit(1)


def kpath(k, *parts):
    return os.path.join(KROOT, k, *parts)


def rel(path, base=ROOT):
    return os.path.relpath(path, base)


def load_json(path, default=None):
    if not os.path.exists(path):
        if default is not None:
            return default
        die(f"missing {rel(path)}")
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            die(f"{rel(path)} is not valid JSON: {e}")


def dump_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def manifest(k):
    m = load_json(kpath(k, "kernel.json"))
    m["name"] = k
    return m


def lean_file(k, man, mod):
    """The file of a module: the kernel's own modules live in kernels/K/lean, the library's in lean/."""
    parts = mod.split(".")
    if parts[0] == man["lean"]["module"].split(".")[0]:
        return kpath(k, "lean", *parts) + ".lean"
    return os.path.join(ROOT, "lean", *parts) + ".lean"


def kernel_lean_files(k):
    out = []
    for dirpath, _, names in os.walk(kpath(k, "lean")):
        out += [os.path.join(dirpath, n) for n in sorted(names) if n.endswith(".lean")]
    return sorted(out)


def parse_lines(spec):
    out = []
    for x in spec:
        if isinstance(x, int):
            out.append(x)
        else:
            a, b = str(x).split("-")
            out += list(range(int(a), int(b) + 1))
    return out


def kernels():
    if not os.path.isdir(KROOT):
        return []
    return sorted(d for d in os.listdir(KROOT) if os.path.exists(kpath(d, "kernel.json")))


# ---------------------------------------------------------------- ledger

def ledger(k):
    path = kpath(k, "review", "ledger.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def append(k, events):
    path = kpath(k, "review", "ledger.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    seq = len(ledger(k))
    with open(path, "a") as f:
        for e in events:
            seq += 1
            f.write(json.dumps({"seq": seq, "time": now(), **e}, sort_keys=True, ensure_ascii=False) + "\n")


def state(k):
    """Fold the ledger. Events written before rounds were recorded belong to round 1."""
    S = {"reviewers": {}, "findings": {}, "accepts": {}, "filed": {}, "responses": 0,
         "archive": {}, "events": ledger(k)}
    for e in S["events"]:
        t, rnd = e["type"], e.get("round", 1)
        if t == "assign":
            S["reviewers"][e["actor"]] = e.get("focus", "all")
        elif t == "file":
            S["filed"].setdefault(e["actor"], {}).setdefault(rnd, 0)
            S["filed"][e["actor"]][rnd] += 1
        elif t == "response":
            S["responses"] += 1
        elif t == "raise":
            S["findings"][e["id"]] = {**e, "round": rnd, "state": "OPEN", "raisers": {e["actor"]},
                                      "released": set(), "reraises": 0, "history": [e]}
        elif t in ("respond", "release", "reraise", "escalate", "decide"):
            f = S["findings"][e["id"]]
            f["history"].append(e)
            if t == "respond":
                f["state"] = "REFERRED" if e["response"] == "REFERRED" else "ANSWERED"
                f["last_response"] = e["response"]
            elif t == "release":
                f["released"].add(e["actor"])
                if f["released"] >= f["raisers"]:
                    f["state"] = "CLOSED"
            elif t == "reraise":
                f["reraises"] += 1
                f["released"].discard(e["actor"])
                f["state"] = "OPEN"
                f["raise_items"] = e.get("items", f["items"])
            elif t == "escalate":
                f["state"] = "ESCALATED"
            elif t == "decide":
                f["state"] = "CLOSED"
                f["decision"] = e["decision"]
        elif t == "accept":
            S["accepts"].setdefault(e["actor"], {}).update(e["items"])
        for p, h in e.get("files", {}).items():
            S["archive"][p] = h
    S["round"] = S["responses"] + 1
    return S


def snapshot(k, gate_digest):
    return load_json(kpath(k, "review", "snapshots", gate_digest[:16] + ".json"), {"items": {}})


def affected(k, f, items):
    """The items a finding is about: an acceptance of one of them waits for the finding."""
    t = f["target"]
    if t.startswith("item:"):
        return {t[5:]}
    if t.startswith(("map:", "src:")):
        return {"model", "map"}
    if f.get("affects"):
        return {a[5:] if a.startswith("item:") else a for a in f["affects"]}
    if t.startswith("decl:"):
        hit = {it for it, v in snapshot(k, f["gate"])["items"].items() if t[5:] in v.get("cone", {})}
        return hit or set(items)
    return set(items)


# ---------------------------------------------------------------- the gate

def check(k):
    man = manifest(k)
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    src = kpath(k, man["source"]["file"])
    src_text = open(src).read() if os.path.exists(src) else ""
    src_sha = sha(src_text)
    add("source", src_sha == man["source"]["sha256"],
        f"{man['source']['file']} sha256 {src_sha[:16]}" +
        ("" if src_sha == man["source"]["sha256"] else f" (the manifest says {man['source']['sha256'][:16]})"))
    lean_dir = os.path.join(ROOT, "lean")
    mod, ns = man["lean"]["module"], man["lean"]["namespace"]
    r = subprocess.run([LAKE, "build", mod], cwd=lean_dir, capture_output=True, text=True)
    errs = [l for l in r.stdout.splitlines() if l.startswith("error")]
    add("build", r.returncode == 0, f"lake build {mod}" + ("" if r.returncode == 0 else ": " + " | ".join(errs[:3])))
    bad = []
    pat = re.compile(r"\bsorry\b|\badmit\b|\bnative_decide\b|^\s*axiom\b|implemented_by|@\[extern|\bunsafe\b|set_option\s+debug\.")
    for p in kernel_lean_files(k):
        for i, line in enumerate(open(p), 1):
            if pat.search(line.split("--")[0]):
                bad.append(f"{rel(p, kpath(k))}:{i}")
    add("hygiene", not bad, "no sorry, admit, native_decide, axiom, extern or unsafe" if not bad else ", ".join(bad[:5]))
    tmpl = open(os.path.join(ROOT, "tools", "claim_reader.lean.in")).read()
    gen = tmpl.replace("@MODULE@", mod).replace("@NS@", ns).replace("@PREFIX@", mod.split(".")[0])
    gpath = os.path.join(lean_dir, ".lake", "tvr", k, "claim_reader.lean")
    write(gpath, gen)
    surf = None
    if r.returncode == 0:
        r2 = subprocess.run([LAKE, "env", "lean", gpath], cwd=lean_dir, capture_output=True, text=True)
        js = [l for l in r2.stdout.splitlines() if l.startswith("TVR_JSON ")]
        if r2.returncode == 0 and js:
            surf = json.loads(js[0][len("TVR_JSON "):])
        add("shape", surf is not None,
            "main : ∀ K P, Assumptions K P → Spec K; consistent : Claim.Consistent Assumptions; Spec K = Meets Inst.ok Pre Post K"
            if surf else "the claim does not have the protocol's shape: " + " | ".join(r2.stdout.splitlines()[:4]))
    else:
        add("shape", False, "not checked (the build failed)")
    items, consts = {}, {}
    if surf:
        consts = {c["name"]: c for c in surf["consts"] if c}
        add("axioms", all(set(v) <= STD_AXIOMS for v in surf["axioms"].values()),
            "; ".join(f"{a}: {', '.join(v)}" for a, v in surf["axioms"].items()))
        unused = [f for f, u in surf["usage"].items() if not u]
        add("usage", not unused and not surf["destructured"],
            "every assumption is used by main" if not unused and not surf["destructured"] else
            ("unused: " + ", ".join(unused) if unused else "") +
            (" (Assumptions destructured: use projections)" if surf["destructured"] else ""))
        tva_insts = [i for i in surf["instances"] if i["class"].startswith("TVA.")]
        add("namespace", not surf["shadow"] and not tva_insts,
            "the kernel declares nothing in TVA and no instance of a library class" if not surf["shadow"] and not tva_insts
            else "declares " + ", ".join(surf["shadow"][:5] + [i["name"] for i in tva_insts][:5]))
        bad_kinds = []
        for f in surf["fields"]:
            m = re.match(r"\s*\[(rule|kernel)\]", surf["field_docs"].get(f, ""))
            u = surf["field_uses"].get(f, {"consts": [], "K": False, "P": False})
            own = sorted(c for c in u["consts"] if c.startswith(ns + ".") and c[len(ns) + 1:] not in RULE_ARGS)
            rules = [c for c in u["consts"] if c.startswith("TVA.Rules.")]
            if not m:
                bad_kinds.append(f"{f}: not tagged [rule] or [kernel]")
            elif m.group(1) == "rule" and (own or not rules):
                bad_kinds.append(f"{f}: a [rule] must cite a TVA.Rules fact" + (" with no constant of the kernel's own (" +
                                 ", ".join(c[len(ns) + 1:] for c in own) + ")" if own else ""))
            elif m.group(1) == "kernel" and (u["K"] or u["P"] or rules):
                bad_kinds.append(f"{f}: a [kernel] assumption may not mention K, P or a TVA.Rules fact")
        add("kinds", not bad_kinds, "every assumption is a [rule] citing the library or a [kernel] fact about the kernel's own values"
            if not bad_kinds else "; ".join(bad_kinds))
        for it, v in surf["items"].items():
            lines = sorted(f"{n} {consts[n]['hash']}" for n in v["cone"] if n in consts)
            docs = [consts[n].get("doc") or "" for n in v["roots"] if n in consts]
            if it.startswith("assumption:"):
                docs.append(surf["field_docs"].get(it.split(":", 1)[1], ""))
            extra = [src_sha] if it == "model" else []
            items[it] = {"digest": sha("\n".join(lines + docs + extra + [surf["lean"]])), "roots": v["roots"],
                         "cone": sorted(v["cone"])}
    mm_path = kpath(k, "model-map.json")
    mm = load_json(mm_path, {"entries": []})
    claim_text = re.sub(r"\s+", " ", " ".join(open(p).read() for p in kernel_lean_files(k)))
    covered, problems = set(), []
    for i, e in enumerate(mm["entries"]):
        covered |= set(parse_lines(e["src"]))
        if "model" in e:
            for snip in ([e["model"]] if isinstance(e["model"], str) else e["model"]):
                if re.sub(r"\s+", " ", snip).strip() not in claim_text:
                    problems.append(f"entry {i} (src {e['src']}): model text not found: {snip[:50]!r}")
        elif e.get("drop") not in DROPS:
            problems.append(f"entry {i}: drop reason {e.get('drop')!r} is not allowed")
        elif e["drop"] in NOTE_REQUIRED and not e.get("note"):
            problems.append(f"entry {i}: a {e['drop']} drop needs a note (the condition that makes it hold)")
    src_lines = src_text.splitlines()
    missing, in_doc = [], False
    for a, b in man["source"]["ranges"]:
        for i in range(a, b + 1):
            t = src_lines[i - 1].strip() if i <= len(src_lines) else ""
            q = t.count('"""')
            if in_doc or q:
                if q % 2 == 1:
                    in_doc = not in_doc
                continue
            if t and not t.startswith("#") and i not in covered:
                missing.append(i)
    add("map", not missing and not problems,
        f"{len(mm['entries'])} entries cover every code line of {man['source']['ranges']}" if not missing and not problems
        else "; ".join(([f"uncovered source lines {missing[:12]}"] if missing else []) + problems[:4]))
    items["map"] = {"digest": sha(open(mm_path).read() + src_sha) if os.path.exists(mm_path) else "",
                    "roots": [], "cone": []}
    doc = kpath(k, man.get("doc", "README.md"))
    heads = re.findall(r"^## (\w+)", open(doc).read(), re.M) if os.path.exists(doc) else []
    lack = [s for s in DOC_SECTIONS if s not in heads]
    add("doc", os.path.exists(doc) and not lack, f"{man.get('doc', 'README.md')} has the sections {', '.join(DOC_SECTIONS)}"
        if not lack else f"{man.get('doc', 'README.md')} lacks: {', '.join(lack)}")
    ok = all(c["ok"] for c in checks)
    gate_digest = sha("\n".join(f"{i} {v['digest']}" for i, v in sorted(items.items())))
    kinds = {}
    for f in (surf["fields"] if surf else []):
        m = re.match(r"\s*\[(rule|kernel)\]", surf["field_docs"].get(f, ""))
        kinds[f] = m.group(1) if m else ""
    gate = {"protocol": "tvr-2", "kernel": k, "time": now(), "ok": ok, "digest": gate_digest,
            "checks": checks, "items": {i: v["digest"] for i, v in items.items()}, "kinds": kinds}
    dump_json(kpath(k, "gate.json"), gate)
    if ok and surf:
        dump_json(kpath(k, "review", "snapshots", gate_digest[:16] + ".json"),
                  {"items": {i: {"digest": v["digest"], "cone": {n: consts[n]["hash"] for n in v["cone"] if n in consts}}
                             for i, v in items.items()}})
    if surf:
        render_claim(k, man, gate, surf, consts, items, mm, src_lines)
    refresh(k)
    for c in checks:
        print(f"{'PASS' if c['ok'] else 'FAIL'}  {c['name']:<10} {c['detail']}")
    print(f"gate {'PASS' if ok else 'FAIL'}  digest {gate_digest[:16]}")
    return ok


# ---------------------------------------------------------------- claim.md

def visible(name):
    return not any(part.startswith("_") or part.startswith("match_") or part in (
        "mk", "rec", "casesOn", "recOn", "noConfusion", "noConfusionType", "brecOn", "below", "eq_def",
        "eq_1", "inj", "injEq", "sizeOf_spec", "ext", "ext_iff") for part in name.split("."))


def render_claim(k, man, gate, surf, consts, items, mm, src_lines):
    ns = man["lean"]["namespace"]
    out = []
    w = out.append
    w(f"# Claim: `{k}`\n")
    w(f"> Generated by `tvr check` at {gate['time']}; do not edit. Gate: **{'PASS' if gate['ok'] else 'FAIL'}**, "
      f"digest `{gate['digest'][:16]}`. How to review: [docs/REVIEW_PROTOCOL.md](../../docs/REVIEW_PROTOCOL.md) and "
      f"[docs/REVIEWER_GUIDE.md](../../docs/REVIEWER_GUIDE.md). The review so far: [REVIEW.md](REVIEW.md).\n")
    w(f"{man.get('title', k)}. The prover's write-up: [README.md](README.md).\n")
    w("## 1. The claim\n")
    w(f"```lean\ntheorem {ns}.main : {surf['main_type']}\ntheorem {ns}.consistent : TVA.Claim.Consistent {ns}.Assumptions\n```\n")
    w("`Spec K` is checked to be `Claim.Meets Inst.ok Pre Post K`: on every instance `Inst.ok` covers and every input "
      "`Pre` allows, every outcome of the device `K` is a normal end whose memory satisfies `Post`. Lean checked "
      "`main` and `consistent`; you judge the items below.\n")
    w("## 2. The source\n")
    s = man["source"]
    w(f"[`{s['file']}`]({s['file']}) (sha256 `{s['sha256'][:16]}…`), kernel lines {s['ranges']}" +
      (f", host lines {s['host']}" if s.get("host") else "") + ".\n")

    def section(title, item, intro):
        w(f"## {title}\n")
        w(f"Item `{item}`, digest `{items[item]['digest'][:16]}`. {intro}\n")
        decls = []
        for n in items[item]["cone"]:
            c = consts.get(n)
            if not c or not n.startswith(ns + ".") or not visible(n) or c["kind"] == "ctor":
                continue
            if not c.get("range") or not c.get("module"):
                continue
            p = lean_file(k, man, c["module"])
            if os.path.exists(p):
                decls.append((rel(p, kpath(k)), c["range"][0], c["range"][1], p))
        last = {}
        for f, a, b, p in sorted(set(decls), key=lambda x: (x[0], x[1], -x[2])):
            if f in last and a <= last[f]:
                continue  # inside a declaration already shown (a structure's field)
            last[f] = b
            text = "\n".join(open(p).read().splitlines()[a - 1:b])
            w(f"[`{f}:{a}`]({f}#L{a})\n```lean\n{text}\n```\n")
        lib = sorted(n for n in items[item]["cone"] if n.startswith("TVA.") and visible(n) and n in consts
                     and consts[n]["kind"] in ("def", "inductive", "opaque", "axiom"))
        if lib:
            w(f"<details><summary>Library definitions this item reaches ({len(lib)})</summary>\n")
            for n in lib:
                d = (consts[n].get("doc") or "").strip().split("\n")[0]
                w(f"- `{n}` ({consts[n]['module']}){': ' + d if d else ''}")
            w("\n</details>\n")

    section("3. Conditions", "conditions", "The instances (`Inst.ok`) and inputs (`Pre`) covered.")
    section("4. Specification", "spec", "What the outputs are (`Post`).")
    w("## 5. Assumptions\n")
    for f in surf["fields"]:
        it = f"assumption:{f}"
        w(f"### `{f}` — item `{it}`, digest `{items[it]['digest'][:16]}`, used by `main`: "
          f"{'yes' if surf['usage'].get(f) else '**no**'}\n")
        w(surf["field_docs"].get(f, "").strip() + "\n")
        w(f"```lean\n{surf['field_types'].get(f, '')}\n```\n")
        cited = sorted(c for c in surf["field_uses"].get(f, {}).get("consts", []) if c.startswith("TVA.Rules."))
        if cited:
            w("Library facts cited: " + ", ".join(f"`{c}`" for c in cited) + ". A kernel reviewer checks that they are "
              "the library's facts for the operation the model uses, not whether they are true.\n")
    section("6. Model", "model", "The kernel in the `tl.*` vocabulary and the launch.")
    w("### The model map (item `map`)\n")
    w(f"Digest `{items['map']['digest'][:16]}`. Every code line of the source is mapped to model text or dropped "
      "with a reason.\n")
    w("| Source | Model / reason |\n|---|---|")
    for e in mm["entries"]:
        ls = parse_lines(e["src"])
        st = " ⏎ ".join(src_lines[i - 1].strip() for i in ls if i <= len(src_lines)).replace("|", "\\|")
        if "model" in e:
            m = e["model"] if isinstance(e["model"], str) else " ⏎ ".join(e["model"])
            right = f"`{m}`".replace("|", "\\|") + (f" — {e['note']}" if e.get("note") else "")
        else:
            right = f"**{e['drop']}**" + (f": {e['note']}" if e.get("note") else "")
        w(f"| [{e['src']}]({s['file']}#L{ls[0]}): `{st}` | {right} |")
    w("")
    w("## 7. Gate\n")
    w("| Check | | Detail |\n|---|---|---|")
    for c in gate["checks"]:
        w(f"| {c['name']} | {'PASS' if c['ok'] else 'FAIL'} | {c['detail'].replace('|', '/')} |")
    w("\n| Item | Digest |\n|---|---|")
    for i, v in sorted(items.items()):
        w(f"| `{i}` | `{v['digest'][:16]}` |")
    write(kpath(k, "claim.md"), "\n".join(out) + "\n")


# ---------------------------------------------------------------- review commands

def current_gate(k):
    g = load_json(kpath(k, "gate.json"))
    if not g["ok"]:
        die("the gate fails; run `tvr check` after fixing it")
    newest = max(os.path.getmtime(p) for p in kernel_lean_files(k) + [kpath(k, "model-map.json"), kpath(k, "kernel.json")])
    gt = datetime.datetime.strptime(g["time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()
    if newest > gt + 1:
        die("the gate is stale (a kernel file changed after it ran); run `tvr check`")
    return g


def valid_target(k, t, gate, man):
    if t.startswith("item:"):
        return t[5:] in gate["items"]
    if t.startswith("decl:"):
        return any(t[5:] in v["cone"] for v in snapshot(k, gate["digest"])["items"].values())
    if t.startswith(("src:", "map:")):
        try:
            n = int(t.split(":", 1)[1])
        except ValueError:
            return False
        return any(a <= n <= b for a, b in man["source"]["ranges"] + man["source"].get("host", []))
    if t.startswith(("lib:", "doc:")):
        return os.path.exists(os.path.join(ROOT, t.split(":", 1)[1].split("#")[0]))
    return False


def round_file(k, rnd, actor, base, count):
    """review/round-N/ACTOR/BASE.json for an actor's first filing of the round, BASE-2.json for the second, …"""
    stem, ext = os.path.splitext(base)
    return kpath(k, "review", f"round-{rnd}", actor, base if count == 0 else f"{stem}-{count + 1}{ext}")


def archive_entry(*paths):
    return {rel(p): file_sha(p) for p in paths if os.path.exists(p)}


def cmd_assign(k, reviewer, focus):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", reviewer) or reviewer in ("prover", "user", "tvr", "coordinator"):
        die(f"bad reviewer id {reviewer!r} (lower case, digits, '-')")
    S = state(k)
    append(k, [{"type": "assign", "actor": reviewer, "focus": focus, "round": S["round"]}])
    refresh(k)
    print(f"assigned {reviewer} ({focus}) to {k}")


def cmd_file(k, who):
    S = state(k)
    N = S["round"]
    if who not in S["reviewers"]:
        die(f"{who} is not assigned to {k}")
    count = S["filed"].get(who, {}).get(N, 0)
    path = round_file(k, N, who, "filing.json", count)
    report = round_file(k, N, who, "report.md", count)
    filing = load_json(path)
    if filing.get("reviewer", who) != who:
        die(f"{rel(path)} names reviewer {filing['reviewer']}")
    gate = current_gate(k)
    man = manifest(k)
    findings = S["findings"]
    events = [{"type": "file", "actor": who, "round": N, "gate": gate["digest"], "path": rel(path),
               "report": rel(report) if os.path.exists(report) else "", "notes": filing.get("notes", ""),
               "files": archive_entry(path, report)}]
    nxt = 1 + max([int(i[1:]) for i in findings] or [0])
    for f in filing.get("findings", []):
        if f.get("kind") not in KINDS:
            die(f"unknown kind {f.get('kind')!r}; the kinds are {', '.join(KINDS)}")
        if not any(f["target"].startswith(p) for p in KINDS[f["kind"]][1]) or not valid_target(k, f["target"], gate, man):
            die(f"invalid target {f['target']!r} for {f['kind']} (allowed: {', '.join(KINDS[f['kind']][1])})")
        need = KINDS[f["kind"]][2]
        if need:
            have = gate.get("kinds", {}).get(f["target"].split(":", 2)[2], "")
            if have != need:
                die(f"{f['kind']} applies to [{need}] assumptions; {f['target']} is [{have}]" +
                    (" (a library fact: file LIBRARY_REPORT, or RULE_MISCITED if it is the wrong fact)" if have == "rule" else ""))
        for a in f.get("affects", []):
            if (a[5:] if a.startswith("item:") else a) not in gate["items"]:
                die(f"`affects` names no item: {a}")
        events.append({"type": "raise", "actor": who, "round": N, "id": f"F{nxt}", "kind": f["kind"],
                       "target": f["target"], "affects": f.get("affects", []),
                       "blocking": f.get("blocking", KINDS[f["kind"]][0]), "summary": f["summary"],
                       "detail": f.get("detail", ""), "evidence": f.get("evidence", ""),
                       "gate": gate["digest"], "items": gate["items"]})
        nxt += 1
    released = set()
    for r in filing.get("release", []):
        if r["id"] not in findings or who not in findings[r["id"]]["raisers"]:
            die(f"{who} cannot release {r['id']} (not raised by {who})")
        events.append({"type": "release", "actor": who, "round": N, "id": r["id"], "note": r.get("note", "")})
        released.add(r["id"])
    reraised = set()
    for r in filing.get("reraise", []):
        f = findings.get(r["id"])
        if not f or who not in f["raisers"]:
            die(f"{who} cannot re-raise {r['id']} (not raised by {who})")
        if f["state"] not in ("ANSWERED", "REFERRED"):
            die(f"{r['id']} is {f['state']}; only an answered or referred finding can be re-raised")
        events.append({"type": "reraise", "actor": who, "round": N, "id": r["id"], "note": r.get("note", ""),
                       "items": gate["items"]})
        reraised.add(r["id"])
        if f["reraises"] + 1 >= ESCALATE_AFTER:
            events.append({"type": "escalate", "actor": "tvr", "round": N, "id": r["id"],
                           "memo": f"re-raised {f['reraises'] + 1} times; the user decides"})
    acc = filing.get("accept", [])
    if acc:
        names = list(gate["items"]) if acc == "all" else [a[5:] if a.startswith("item:") else a for a in acc]
        own = [f for i, f in findings.items() if who in f["raisers"] and f["blocking"]
               and ((f["state"] not in ("CLOSED", "REFERRED") and i not in released) or i in reraised)]
        own += [e for e in events if e["type"] == "raise" and e["blocking"]]
        for n in names:
            if n not in gate["items"]:
                die(f"no item {n}")
            block = sorted({f["id"] for f in own if n in affected(k, f, gate["items"])})
            if block:
                die(f"{who} cannot accept {n} while their blocking finding(s) on it are open: {', '.join(block)}")
        events.append({"type": "accept", "actor": who, "round": N, "items": {n: gate["items"][n] for n in names},
                       "gate": gate["digest"]})
    append(k, events)
    refresh(k)
    print(f"recorded {rel(path)}: {len(events) - 1} event(s)")
    print(status_text(k, who))


def cmd_respond(k):
    S = state(k)
    N = S["round"]
    waiting = [r for r in S["reviewers"] if not S["filed"].get(r, {}).get(N)]
    if waiting:
        die(f"round {N} is waiting for {', '.join(waiting)}")
    path = kpath(k, "review", f"round-{N}", "prover", "response.json")
    notes = kpath(k, "review", f"round-{N}", "prover", "notes.md")
    resp = load_json(path)
    gate = current_gate(k)
    findings = S["findings"]
    events = [{"type": "response", "actor": "prover", "round": N, "gate": gate["digest"], "path": rel(path),
               "notes": rel(notes) if os.path.exists(notes) else "", "files": archive_entry(path, notes)}]
    answered = set()
    for r in resp["responses"]:
        f = findings.get(r["id"])
        if not f:
            die(f"no finding {r['id']}")
        if f["state"] != "OPEN":
            die(f"{r['id']} is {f['state']}; only open findings are answered")
        if r["response"] not in RESPONSES:
            die(f"unknown response {r['response']!r}; the responses are {', '.join(sorted(RESPONSES))}")
        if r["response"] == "REFERRED":
            ids = re.findall(r"\bL\d+\b", r["text"])
            known = issue_ids()
            if not ids or not all(i in known for i in ids):
                die(f"{r['id']}: REFERRED must name a library issue of docs/library-review/ISSUES.md (L<n>); "
                    f"named {ids or 'none'}")
        if r["response"] == "REVISED":
            base = f.get("raise_items", f["items"])
            ch = [c[5:] if c.startswith("item:") else c for c in r.get("changed", [])]
            if not ch:
                die(f"{r['id']}: REVISED must name the changed items")
            unknown = [c for c in ch if c not in gate["items"]]
            same = [c for c in ch if c in gate["items"] and gate["items"][c] == base.get(c)]
            if unknown or same:
                die(f"{r['id']}: REVISED, but " + "; ".join(
                    ([f"no such item: {', '.join(unknown)}"] if unknown else []) +
                    ([f"these items did not change since the finding: {', '.join(same)}"] if same else [])))
        events.append({"type": "respond", "actor": "prover", "round": N, "id": r["id"], "response": r["response"],
                       "text": r["text"], "changed": r.get("changed", []), "gate": gate["digest"]})
        answered.add(r["id"])
    left = sorted((i for i, f in findings.items() if f["state"] == "OPEN" and i not in answered), key=lambda x: int(x[1:]))
    if left:
        die(f"one batch answers every open finding (the batch policy); not answered: {', '.join(left)}")
    append(k, events)
    refresh(k)
    print(f"recorded {rel(path)}: {len(events) - 1} response(s); round {N + 1} is open for the reviewers")


def cmd_decide(k, fid, text):
    S = state(k)
    if S["findings"].get(fid, {}).get("state") != "ESCALATED":
        die(f"{fid} is not escalated")
    append(k, [{"type": "decide", "actor": "user", "round": S["round"], "id": fid, "decision": text}])
    refresh(k)


def cmd_note(k, text, files=()):
    S = state(k)
    append(k, [{"type": "note", "actor": "coordinator", "round": S["round"], "text": text,
                "files": archive_entry(*[os.path.join(ROOT, f) for f in files])}])
    refresh(k)


# ---------------------------------------------------------------- views

def cert_state(k, S, gate):
    blocking = sorted((i for i, f in S["findings"].items() if f["blocking"] and f["state"] not in ("CLOSED", "REFERRED")),
                      key=lambda x: int(x[1:]))
    full = bool(S["reviewers"]) and gate["ok"] and all(
        S["accepts"].get(r, {}).get(i) == d for r in S["reviewers"] for i, d in gate["items"].items())
    return full and not blocking, blocking, full


def blind_round1(S):
    return any(not S["filed"].get(r, {}).get(1) for r in S["reviewers"])


def status_text(k, who=None):
    gate = load_json(kpath(k, "gate.json"), {"items": {}, "ok": False, "digest": ""})
    S = state(k)
    L = []
    if not S["reviewers"]:
        return "No reviewer assigned yet.\n"
    revs = list(S["reviewers"])
    L.append("| Item | Digest | " + " | ".join(revs) + " |")
    L.append("|---|---|" + "---|" * len(revs))
    cur = snapshot(k, gate["digest"])["items"]
    for it, dg in sorted(gate["items"].items()):
        row = []
        for r in revs:
            a = S["accepts"].get(r, {}).get(it)
            if a is None:
                row.append("—")
            elif a == dg:
                row.append("accepted")
            else:
                old = {}
                for e in S["events"]:
                    if e["type"] == "accept" and e["actor"] == r and it in e["items"]:
                        old = snapshot(k, e["gate"])["items"].get(it, {}).get("cone", {})
                new = cur.get(it, {}).get("cone", {})
                diff = sorted(n for n in set(old) | set(new) if old.get(n) != new.get(n))
                row.append("STALE" + (" (changed: " + ", ".join(f"`{n}`" for n in diff[:5]) +
                                      (" …" if len(diff) > 5 else "") + ")" if diff else ""))
        L.append(f"| `{it}` | `{dg[:12]}` | " + " | ".join(row) + " |")
    L.append("")
    hide = blind_round1(S)
    shown = {i: f for i, f in S["findings"].items()
             if not (hide and f["round"] == 1 and (who is None or who not in f["raisers"]))}
    if hide:
        L.append("Round 1 is blind: findings are hidden until every assigned reviewer has filed.\n")
    if shown:
        L.append("| Id | State | Kind | Target | By | Round | Summary |\n|---|---|---|---|---|---|---|")
        for i in sorted(shown, key=lambda x: int(x[1:])):
            f = shown[i]
            st = f["state"] + ("" if f["blocking"] else " (non-blocking)")
            if f["state"] == "ANSWERED":
                st += f" — {f['last_response']}"
            L.append(f"| [{i}](#{i.lower()}) | {st} | {f['kind']} | `{f['target']}` | {', '.join(sorted(f['raisers']))} "
                     f"| {f['round']} | {f['summary'][:120].replace('|', '/')} |")
        L.append("")
    cert, blocking, full = cert_state(k, S, gate)
    L.append(f"**Certificate: {'GRANTED' if cert else 'not yet'}**" + ("" if cert else (
        f" — {len(blocking)} blocking finding(s) unresolved" + ("" if full else
        "; not every item is accepted by every reviewer at its current digest"))) + "\n")
    return "\n".join(L)


def phase(k, S):
    N = S["round"]
    waiting = [r for r in S["reviewers"] if not S["filed"].get(r, {}).get(N)]
    open_ = [i for i, f in S["findings"].items() if f["state"] == "OPEN"]
    if not S["reviewers"]:
        return "no reviewer assigned"
    if waiting:
        return f"round {N}: waiting for {', '.join(waiting)}"
    if open_:
        return f"round {N}: waiting for the prover ({len(open_)} open finding(s))"
    return f"round {N}: all reviewers have filed"


def render_review(k):
    man = manifest(k)
    gate = load_json(kpath(k, "gate.json"), {"items": {}, "ok": False, "digest": "", "time": ""})
    S = state(k)
    out = []
    w = out.append
    w(f"# Review: `{k}`\n")
    w(f"> Generated by `tvr` from [review/ledger.jsonl](review/ledger.jsonl) at {now()}; do not edit. "
      f"Protocol: [docs/REVIEW_PROTOCOL.md](../../docs/REVIEW_PROTOCOL.md).\n")
    w(f"{man.get('title', k)}. The claim sheet: [claim.md](claim.md); the prover's write-up: [README.md](README.md).\n")
    cert, blocking, _ = cert_state(k, S, gate)
    w("## Status\n")
    w(f"- Gate: **{'PASS' if gate['ok'] else 'FAIL'}**, digest `{gate['digest'][:16]}` ({gate.get('time', '')}).")
    w(f"- Phase: {phase(k, S)}.")
    w(f"- Reviewers: {', '.join(f'{r} ({fo})' for r, fo in S['reviewers'].items()) or 'none'}.")
    w(f"- Certificate: **{'GRANTED' if cert else 'not yet'}**.")
    bad = [p for p, h in S["archive"].items() if not os.path.exists(os.path.join(ROOT, p)) or file_sha(os.path.join(ROOT, p)) != h]
    w(f"- Recorded files: {'all unchanged since they were recorded' if not bad else '**changed after recording: ' + ', '.join(bad) + '**'}.\n")
    w(status_text(k))
    rows = issues_for(k)
    if rows:
        w("### Library issues raised from this kernel\n")
        w("Kernel review trusts the library; these are tracked in "
          "[docs/library-review/ISSUES.md](../../docs/library-review/ISSUES.md).\n")
        w("| Issue | State | Source | Summary |\n|---|---|---|---|")
        for c in rows:
            w(f"| {c[0].replace('(#', '(../../docs/library-review/ISSUES.md#')} | {c[1]} | {c[2]} | {c[3]} |")
        w("")
    hide = blind_round1(S)
    rounds = sorted({e.get("round", 1) for e in S["events"]})
    for rnd in rounds:
        w(f"## Round {rnd}\n")
        if rnd == 1 and hide:
            done = [r for r in S["reviewers"] if S["filed"].get(r, {}).get(1)]
            w(f"Blind: {len(done)} of {len(S['reviewers'])} reviewers have filed; their filings are shown once all have.\n")
            continue
        evs = [e for e in S["events"] if e.get("round", 1) == rnd]
        for e in evs:
            t = e["type"]
            if t == "assign":
                w(f"- `{e['time']}` {e['actor']} assigned (focus: {e.get('focus', 'all')}).")
            elif t == "note":
                w(f"- `{e['time']}` coordinator's note: {e['text']}" +
                  (" Files: " + ", ".join(f"[{os.path.basename(p)}](../../{p})" for p in e.get("files", {})) if e.get("files") else ""))
            elif t == "file":
                # filings recorded before the kernels/ layout carry no path: use the round folder's
                d = os.path.dirname(e.get("path") or f"kernels/{k}/review/round-{rnd}/{e['actor']}/filing.json")
                fp = e.get("path") or os.path.join(d, "filing.json")
                rep = e.get("report") or (os.path.join(d, "report.md") if not e.get("path") else "")
                links = []
                for label, p in (("filing", fp), ("report", rep)):
                    if p and os.path.exists(os.path.join(ROOT, p)):
                        links.append(f"[{label}]({rel(os.path.join(ROOT, p), kpath(k))})")
                work = os.path.join(ROOT, d, "work")
                if os.path.isdir(work):
                    links.append(f"[work]({rel(work, kpath(k))}/)")
                w(f"\n### {e['actor']} files ({', '.join(l for l in links if l)})\n")
                w(f"`{e['time']}`, at gate `{e['gate'][:16]}`." + (f" Notes: {e['notes']}" if e.get("notes") else "") + "\n")
            elif t == "raise":
                w(f"#### {e['id']}\n")
                w(f"**{e['kind']}** on `{e['target']}`" + (" (blocking)" if e["blocking"] else " (non-blocking)") +
                  (f", affects {', '.join(e['affects'])}" if e.get("affects") else "") + f": {e['summary']}\n")
                if e.get("detail"):
                    w(e["detail"] + "\n")
                if e.get("evidence"):
                    w(f"*Evidence:* {e['evidence']}\n")
            elif t == "release":
                w(f"- **released {e['id']}**: {e.get('note', '')}")
            elif t == "reraise":
                w(f"- **re-raised {e['id']}**: {e.get('note', '')}")
            elif t == "escalate":
                w(f"- **{e['id']} escalated to the user**: {e.get('memo', '')}")
            elif t == "accept":
                w(f"- **accepted** at gate `{e['gate'][:16]}`: " + ", ".join(f"`{i}`" for i in sorted(e["items"])))
            elif t == "response":
                links = [f"[response]({rel(os.path.join(ROOT, e['path']), kpath(k))})"]
                if e.get("notes"):
                    links.append(f"[notes]({rel(os.path.join(ROOT, e['notes']), kpath(k))})")
                w(f"\n### The prover responds ({', '.join(links)})\n")
                w(f"`{e['time']}`, at gate `{e['gate'][:16]}`.\n")
            elif t == "respond":
                w(f"- **{e['id']} {e['response']}**" + (f" (changed: {', '.join(e['changed'])})" if e.get("changed") else "") +
                  f": {e['text']}")
            elif t == "decide":
                w(f"- **the user decides {e['id']}**: {e['decision']}")
        w("")
    write(kpath(k, "REVIEW.md"), "\n".join(out) + "\n")
    if cert:
        dump_json(kpath(k, "review", "certificate.json"),
                  {"kernel": k, "gate": gate["digest"], "items": gate["items"], "reviewers": sorted(S["reviewers"]),
                   "time": now()})
    elif os.path.exists(kpath(k, "review", "certificate.json")):
        os.remove(kpath(k, "review", "certificate.json"))


def render_index():
    out = ["# Kernels\n",
           "> Generated by `tvr`; do not edit. One folder per kernel; the layout and the process are in "
           "[docs/REVIEW_PROTOCOL.md](../docs/REVIEW_PROTOCOL.md).\n",
           "| Kernel | What | Gate | Phase | Open blocking | Certificate |", "|---|---|---|---|---|---|"]
    for k in kernels():
        man = manifest(k)
        gate = load_json(kpath(k, "gate.json"), {"items": {}, "ok": False, "digest": ""})
        S = state(k)
        cert, blocking, _ = cert_state(k, S, gate)
        out.append(f"| [{k}]({k}/README.md) | {man.get('title', '')} | {'PASS' if gate['ok'] else 'FAIL'} | "
                   f"[{phase(k, S)}]({k}/REVIEW.md) | {len(blocking)} | {'GRANTED' if cert else '—'} |")
    write(os.path.join(KROOT, "README.md"), "\n".join(out) + "\n")


def issue_ids():
    p = os.path.join(LREVIEW, "ISSUES.md")
    return set(re.findall(r"^\| \[(L\d+)\]", open(p).read(), re.M)) if os.path.exists(p) else set()


def issues_for(k):
    """The rows of ISSUES.md that name one of this kernel's findings."""
    p = os.path.join(LREVIEW, "ISSUES.md")
    if not os.path.exists(p):
        return []
    rows, last = [], None
    for l in open(p).read().splitlines():
        if not l.startswith("| [L"):
            continue
        cells = [c.strip() for c in l.strip("|").split("|")]
        src = cells[2]
        kern = None
        for part in re.split(r",\s*", src):
            m = re.match(r"([a-z0-9_]+)#F\d+", part)
            if m:
                kern = m.group(1)
            if kern == k and re.search(r"#F\d+", part):
                rows.append(cells)
                break
    return rows


def render_reports():
    out = ["# Library reports from kernel review\n",
           "> Generated by `tvr`; do not edit. Every `LIBRARY_REPORT` a kernel reviewer filed, and every finding the "
           "prover referred to the library. The coordinator triages them into [ISSUES.md](ISSUES.md).\n",
           "| Finding | Kind | Target | State | Summary | Referred as |", "|---|---|---|---|---|---|"]
    for k in kernels():
        S = state(k)
        for i in sorted(S["findings"], key=lambda x: int(x[1:])):
            f = S["findings"][i]
            if f["kind"] != "LIBRARY_REPORT" and f["state"] != "REFERRED" and f.get("last_response") != "REFERRED":
                continue
            ref = ""
            for e in f["history"]:
                if e["type"] == "respond" and e["response"] == "REFERRED":
                    ref = ", ".join(sorted(set(re.findall(r"\bL\d+\b", e["text"]))))
            out.append(f"| [{k}#{i}](../../kernels/{k}/REVIEW.md#{i.lower()}) | {f['kind']} | `{f['target']}` | {f['state']} | "
                       f"{f['summary'][:120].replace('|', '/')} | {ref} |")
    write(os.path.join(LREVIEW, "REPORTS.md"), "\n".join(out) + "\n")


def cmd_prompt(argv):
    if argv[2] == "library":
        role, rnd = argv[3], argv[4]
        rid = argv[5] if len(argv) > 5 else "lr-1"
        name = {"author": "library-author.md", "reviewer": "library-reviewer.md"}.get(role)
        fields = {"round": rnd, "reviewer": rid}
    else:
        k, role = argv[2], argv[3]
        if not os.path.exists(kpath(k, "kernel.json")):
            die(f"no kernel {k}")
        name = {"prover": "kernel-prover.md", "reviewer": "kernel-reviewer.md"}.get(role)
        fields = {"kernel": k, "round": str(state(k)["round"]), "reviewer": argv[4] if len(argv) > 4 else ""}
        if role == "reviewer" and not fields["reviewer"]:
            die("tvr prompt KERNEL reviewer ID")
    if not name:
        die("roles: prover, reviewer (kernel); author, reviewer (library)")
    text = open(os.path.join(ROOT, "docs", "prompts", name)).read()
    for a, b in fields.items():
        text = text.replace("{{" + a + "}}", b)
    # links in the prompts are relative to docs/prompts; an agent reads them from the TVA root
    text = re.sub(r"\]\(\.\./([^)]+)\)", lambda m: f"](docs/{m.group(1)})", text)
    left = re.findall(r"\{\{\w+\}\}", text)
    if left:
        die(f"unfilled fields: {', '.join(sorted(set(left)))}")
    print(text)


def refresh(k):
    render_review(k)
    render_index()
    render_reports()


# ---------------------------------------------------------------- scaffold

README_TEMPLATE = """# {title}

The prover's write-up. The reviewers judge [claim.md](claim.md) (generated); the review is
[REVIEW.md](REVIEW.md) (generated).

Source: [`{src}`]({src}). Lean: [`lean/`](lean/).

## Specification

## Conditions

## Assumptions

## Proof

## Narrowings and modelling decisions
"""


def cmd_new(k, module, ns, source):
    if not re.fullmatch(r"[a-z0-9_]+", k):
        die("a kernel name is lower case, digits and '_'")
    if os.path.exists(kpath(k)):
        die(f"kernels/{k} exists")
    base = os.path.basename(source)
    for d in ("source", "lean", "review"):
        os.makedirs(kpath(k, d))
    shutil.copyfile(os.path.join(ROOT, source), kpath(k, "source", base))
    dump_json(kpath(k, "kernel.json"), {
        "protocol": "tvr-1", "title": k,
        "lean": {"module": module, "namespace": ns},
        "source": {"file": f"source/{base}", "sha256": file_sha(kpath(k, "source", base)), "ranges": [], "host": []},
        "doc": "README.md"})
    dump_json(kpath(k, "model-map.json"), {"entries": []})
    write(kpath(k, "README.md"), README_TEMPLATE.format(title=k, src=f"source/{base}"))
    add_lake_lib(k, module)
    render_index()
    print(f"created kernels/{k}; write lean/{module.replace('.', '/')}.lean, fill kernel.json (ranges) and model-map.json")


def add_lake_lib(k, module):
    p = os.path.join(ROOT, "lean", "lakefile.toml")
    s = open(p).read()
    if re.search(rf'^name = "{re.escape(k)}"$', s, re.M):
        return
    s = re.sub(r'^defaultTargets = \[(.*?)\]', lambda m: f'defaultTargets = [{m.group(1)}, "{k}"]', s, count=1, flags=re.M)
    s = s.rstrip("\n") + f'\n\n[[lean_lib]]\nname = "{k}"\nsrcDir = "../kernels/{k}/lean"\nroots = ["{module.split(".")[0]}"]\n'
    write(p, s)


def main(argv):
    if len(argv) >= 2 and argv[1] == "index":
        for k in kernels():
            render_review(k)
        render_index()
        render_reports()
        return 0
    if len(argv) >= 4 and argv[1] == "prompt":
        cmd_prompt(argv)
        return 0
    if len(argv) < 3:
        print(__doc__)
        return 1
    cmd, k = argv[1], argv[2]

    def opt(name):
        return argv[argv.index(name) + 1] if name in argv else None

    if cmd == "new":
        cmd_new(k, opt("--module"), opt("--namespace"), opt("--source"))
        return 0
    if not os.path.exists(kpath(k, "kernel.json")):
        die(f"no kernel {k} (kernels/{k}/kernel.json)")
    if cmd == "check":
        return 0 if check(k) else 1
    if cmd == "assign":
        cmd_assign(k, argv[3], opt("--focus") or "all")
    elif cmd == "file":
        cmd_file(k, argv[3])
    elif cmd == "respond":
        cmd_respond(k)
    elif cmd == "decide":
        cmd_decide(k, argv[3], " ".join(argv[4:]))
    elif cmd == "note":
        files = argv[argv.index("--files") + 1:] if "--files" in argv else []
        text = " ".join(argv[3:argv.index("--files")] if "--files" in argv else argv[3:])
        cmd_note(k, text, files)
    elif cmd == "status":
        S = state(k)
        print(phase(k, S))
        print(status_text(k, opt("--as")))
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
