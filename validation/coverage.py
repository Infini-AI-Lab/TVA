#!/usr/bin/env python3
"""Write validation/COVERAGE.md: one row per tl.* operation of lean/TVA/Triton.

  coverage.py [RESULTS.json]

For each operation: what exec does at it (read from its definition, and from the tl operations
it calls), which single-operation cases exercise it (ops/cases.py, through the op names of
ops/Ops.lean), and which mutants are run on it. With RESULTS.json (Ops.lean against a device),
the cases' status. Helpers (relations, solvers, bounds) are not operations and are left out.
Standard library only. The table is generated: edit this script, not COVERAGE.md."""
import json, os, re, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
TRITON = os.path.join(os.path.dirname(HERE), "lean", "TVA", "Triton")

def strip_comments(s):
    s = re.sub(r"/-.*?-/", lambda m: "\n" * m.group(0).count("\n"), s, flags=re.S)
    return re.sub(r"--[^\n]*", "", s)

# The defs of `namespace tl`, with their text up to the next top-level declaration.
defs = {}
for fn in sorted(os.listdir(TRITON)):
    if not fn.endswith(".lean"): continue
    raw = open(os.path.join(TRITON, fn)).read()
    src = strip_comments(raw).split("\n")
    ns, cur = [], None
    for i, line in enumerate(src):
        m = re.match(r"(namespace|end) (\S+)", line)
        if m:
            if m.group(1) == "namespace": ns.append(m.group(2))
            elif ns and ns[-1] == m.group(2): ns.pop()
            cur = None; continue
        m = re.match(r"(?:noncomputable |private )?def ([^\s(:{\[]+)", line)
        if m and ns and ns[-1] == "tl":
            cur = m.group(1).strip("«»")
            defs[cur] = {"file": fn, "line": i + 1, "text": line + "\n"}
            continue
        if re.match(r"(theorem|instance|lemma|abbrev|structure|inductive|variable|section|@\[|/-)", line):
            cur = None
        if cur: defs[cur]["text"] += line + "\n"

# Builders and bounds the operations are written with, not operations of Triton.
INTERNAL = {"upToNaN", "math", "sumBound", "sumWith", "sum1With", "sum0With", "argCombine",
            "tripCount", "addrs", "rmwSeq", "atomicWith", "cumsumWith", "enabledBy", "forN"}

def signature(text):
    """The text before `:=`, or before the first `|` of a definition by pattern matching."""
    head = re.split(r":=|\n\s*\|", text, maxsplit=1)[0]
    return " ".join(head.split())

def is_helper(name, text):
    sig = signature(text)
    return (name[0].isupper() or name.startswith("_root_") or "Ans" in name or name in INTERNAL
            or sig.endswith("Prop"))

RANK = ["pure", "effect", "certified", "platform", "assumed", "no solver"]
def own(text):
    body = text.split(":=", 1)[1] if ":=" in text else ""
    if re.search(r"\.none \.ret|Solve\.none|\(\.none\)", body): return "no solver"
    if re.search(r"\.assume \(\.|map \.assume|Solve\.assume", body): return "assumed"
    if ".plat " in body or ".plat (" in body: return "platform"
    if ".pick" in body: return "certified"
    if re.search(r"\.(load|store|rmw|loop) |atomicWith|rmwSeq", body): return "effect"
    return "pure"

ops = {n: d for n, d in defs.items() if not is_helper(n, d["text"])}
names = set(defs)
def callees(n):
    body = defs[n]["text"].split(":=", 1)[-1]
    # a bare name or `tl.name`, not a field or method (`x.div?`), not a `?` function, and not a
    # parameter of the definition (`cdiv (x div : …)`)
    sig = signature(defs[n]["text"])
    params = {v for grp in re.findall(r"[({\[]([^(){}\[\]:]+):", sig) for v in grp.split()}
    found = re.findall(r"(?:(?<=tl\.)|(?<![.\w]))([a-z_][A-Za-z0-9_]*)\b(?!\?)", body)
    return {m for m in found if m in names and m != n and m not in params}

memo = {}
def status(n, seen=()):
    if n in memo: return memo[n]
    s = own(defs[n]["text"])
    for c in callees(n):
        if c in seen: continue
        t = status(c, seen + (n,))
        if RANK.index(t) > RANK.index(s) and t != "effect": s = t
        elif t == "effect" and s == "pure": s = "effect"
    memo[n] = s
    return s

# Single-operation cases: Ops.lean's op names and the tl definitions they call.
opsrc = open(os.path.join(HERE, "ops", "Ops.lean")).read()
reg = re.search(r"def ops : List \(String × Op\) := \[(.*?)\]\n", opsrc, re.S).group(1)
harness = defaultdict(set)   # tl def -> harness op names
for name, body in re.findall(r'\("([a-z_0-9]+)", (.*?)\)(?=,\s*\(|\s*$)', reg, re.S):
    for t in re.findall(r"tl\.([a-z_0-9]+)", body) + (["convert"] if name == "convert" else []) + (["sum"] if name == "sum" else []):
        harness[t].add(name)
for fnm, tln in (("dotOp", "dot"), ("dotAddOp", "dot_add"), ("sumOp", "sum"), ("convertOp", "convert")):
    for name in re.findall(r'\("([a-z_0-9]+)", %s' % fnm, reg):
        harness[tln].add(name)
mutants = defaultdict(list)
for name, why in re.findall(r'\("([a-z_0-9]+)", "([^"]+)", ', opsrc.split("def mutants", 1)[1].split("/-! ## Running", 1)[0]):
    mutants[name].append(why)

sys.path.insert(0, os.path.join(HERE, "ops"))
import cases as C
case_ids = defaultdict(list)
for c in C.cases():
    case_ids[c["op"]].append((c["id"], "→".join(c["dtypes"])))

results = {}
if len(sys.argv) > 1:
    r = json.load(open(sys.argv[1]))
    for x in r["cases"]: results[x["id"]] = x["status"]

def cell_cases(n):
    hs = sorted(harness.get(n, ()))
    if not hs: return ""
    cs = [c for h in hs for c in case_ids.get(h, [])]
    ids = [i for i, _ in cs]
    s = f"{len(ids)}: " + ", ".join(sorted({d for _, d in cs}))
    if results:
        bad = [i for i in ids if results.get(i) not in ("match", None)]
        s += " — " + ("all match" if not bad else "**" + ", ".join(f"{i} {results[i]}" for i in bad) + "**")
    return s

NOTE = {"no solver": "exec gives up", "assumed": "another site whose value is assumed and logged",
        "platform": "`Prog.plat`: the device's value, assumed and logged", "certified": "computed, with proof",
        "effect": "memory effect", "pure": "Lean function, no choice"}
rows = sorted(ops, key=lambda n: (ops[n]["file"], ops[n]["line"]))
counts = defaultdict(int)
for n in rows: counts[status(n)] += 1
tested = sum(1 for n in rows if harness.get(n))
out = ["# Coverage of the `tl.*` operations", "",
       "Generated by `validation/coverage.py` from `lean/TVA/Triton`, `ops/Ops.lean` and `ops/cases.py`;",
       "do not edit. What validates each definition otherwise (evidence, source reading) is in",
       "[TRUST.md](TRUST.md); the choice sites are explained in [PICKS.md](PICKS.md).", "",
       "**exec**: what the executor does at the operation (the most demanding of its own sites and of",
       "the operations it calls): " + "; ".join(f"*{k}*: {NOTE[k]}" for k in RANK) + ".", "",
       "**single-op cases**: cases of `ops/` that run the definition (count: dtypes)" +
       (", with their status against the device." if results else "; no device run recorded yet."), "",
       f"{len(rows)} operations: " + ", ".join(f"{counts[k]} {k}" for k in RANK if counts[k]) +
       f". {tested} have single-op cases. Not listed (builders the operations are written with): " +
       ", ".join(f"`{n}`" for n in sorted(INTERNAL & set(defs))) + ".", "",
       "| operation | defined | exec | single-op cases | mutants |", "|---|---|---|---|---|"]
for n in rows:
    d = ops[n]
    out.append(f"| `tl.{n}` | `{d['file']}:{d['line']}` | {status(n)} | {cell_cases(n)} | "
               f"{'; '.join(mutants.get(n, []))} |")
open(os.path.join(HERE, "COVERAGE.md"), "w").write("\n".join(out) + "\n")
print(f"COVERAGE.md: {len(rows)} operations, {tested} with cases; " + ", ".join(f"{k} {counts[k]}" for k in RANK))
