#!/usr/bin/env python3
"""Derive a proof of a fixed claim (claims/<kernel>/) from TVA's certified proof of the same kernel.

  oracle.py KERNEL OUT_DIR [--tva DIR]

writes OUT_DIR/Proof.lean (and OUT_DIR/Proof/*.lean). This works only when the claim's definitions are the
certified bundle's: the claim then states what the bundle proved, and the bundle's proof part, moved to the
claim's namespace, proves it. Each bundle file that holds no model (no `structure Inst`) becomes a proof
module; a file that holds the model gives what follows its `section Proof` line. The bundle's `consistent`
and what follows it are left out (the claim has its own); its `main` becomes `main_spec`, and
`theorem main : Statement := main_spec` closes the proof. The build (`tva-claim check`) is the test.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# a top-level declaration and its name
DECL = r"^(?:@\[[^\]]*\]\s*)?(?:(?:noncomputable|private|protected)\s+)*" \
       r"(?:theorem|lemma|def|abbrev|structure|instance)\s+([^\s:({\[]+)"
START = re.compile(r"^(?:/--|/-!|@\[|noncomputable|private|protected|theorem|lemma|def|abbrev|structure|instance|"
                   r"open|section|end|variable|attribute)\b")


def drop_declared(body, names):
    """`body` without the top-level declarations (and their doc comments) whose names are in `names`:
    the claim declares them already."""
    chunks, cur = [], []
    for line in body.split("\n"):
        doc_only = cur and cur[0].startswith("/--") and not any(re.match(DECL, l) for l in cur)
        if START.match(line) and not doc_only:
            chunks.append(cur)
            cur = []
        cur.append(line)
    chunks.append(cur)
    keep = []
    for c in chunks:
        m = next((re.match(DECL, l) for l in c if re.match(DECL, l)), None)
        if not (m and m.group(1) in names):
            keep.append("\n".join(c))
    return "\n".join(keep)


def main():
    args = sys.argv[1:]
    tva = args[args.index("--tva") + 1] if "--tva" in args else os.path.dirname(os.path.dirname(HERE))
    k, out = args[0], args[1]
    kd = os.path.join(tva, "kernels", k)
    km = json.load(open(os.path.join(kd, "kernel.json")))
    cm = json.load(open(os.path.join(tva, "claims", k, "claim.json")))
    root, kns, cns = km["lean"]["module"], km["lean"]["namespace"], cm["namespace"]
    lean = os.path.join(kd, "lean")
    # the bundle's modules in import order (its root module imports them)
    order, seen = [], set()

    def visit(mod):
        if mod in seen:
            return
        seen.add(mod)
        p = os.path.join(lean, *mod.split(".")) + ".lean"
        for m in re.findall(r"^import\s+(\S+)", open(p).read(), re.M):
            if m.split(".")[0] == root.split(".")[0]:
                visit(m)
        order.append(mod)

    visit(root)
    claim_names = set(re.findall(DECL, open(os.path.join(tva, "claims", k, "Claim.lean")).read(), re.M))
    parts = []
    for mod in order:
        text = open(os.path.join(lean, *mod.split(".")) + ".lean").read()
        if "theorem " not in text:
            continue  # the root module only imports the others
        body = text.split(f"namespace {kns}\n", 1)[1].rsplit(f"end {kns}", 1)[0]
        if re.search(r"^structure Inst\b", text, re.M):
            if "\nsection Proof\n" not in body:
                continue  # the model alone: the claim has it
            opens = re.findall(r"^open .*$", body.split("\nsection Proof\n")[0], re.M)
            body = "\n".join(opens) + "\n" + body.split("\nsection Proof\n", 1)[1]
        body = drop_declared(body.split("\ntheorem consistent")[0].replace(kns, cns), claim_names)
        doc = re.match(r"(?:import .*\n)*\s*(/-![\s\S]*?-/)", text)
        parts.append((mod, (doc.group(1) + "\n\n" if doc else ""), body.strip("\n")))
    if not parts or "theorem main :" not in parts[-1][2]:
        sys.exit(f"{k}: no `theorem main` in the bundle's proof part")
    names = {mod: ("Proof" if i == len(parts) - 1 else "Proof." + mod.split(".")[-1]) for i, (mod, _, _) in
             enumerate(parts)}
    os.makedirs(out, exist_ok=True)
    prev = "Claim"
    for i, (mod, doc, body) in enumerate(parts):
        last = i == len(parts) - 1
        if last:
            body = body.replace("theorem main :", "theorem main_spec :", 1) + \
                "\n\n/-- The claim: `main_spec` states exactly `Statement`. -/\n" \
                "theorem main : Statement := main_spec"
        name = names[mod]
        p = os.path.join(out, *name.split(".")) + ".lean"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(f"import {prev}\n\n{doc}namespace {cns}\n\n{body}\n\nend {cns}\n")
        prev = name
    print(f"{k}: {', '.join(names[m] for m, _, _ in parts)} written to {out}")


if __name__ == "__main__":
    main()
