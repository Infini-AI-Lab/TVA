"""TVA library evidence (library issue L149, round 34): a static scan of every claim's source for the loops that the
OptimizeThreadLocality exclusion (docs/exclusions.md; docs/SEMANTICS.md, "Float `+` and `-`") names.

Run from the TVA root:  python3 -I evidence/thread-locality/claims_scan.py
Output: evidence/thread-locality/claims_scan.txt. Reads claims/*/source/*.py only (as text, with Python's ast).

For every function of a claim's source and every `for` loop over `range` or `tl.range` in it (not
`tl.static_range`), it lists each reduction (`tl.sum`, `tl.max`, `tl.min`, `tl.reduce`, and the method forms
`x.sum(...)` etc.) anywhere in the loop's body: under an `if`, in a `tl.static_range` loop, in a nested loop, or in a
helper of the same source that the body calls (followed transitively). For each it says:
- operand: `load` when the operand is a `tl.load` (or `.load` of a block pointer), possibly through names assigned in
  the function and through operations that can give back the load's elements unchanged for some constants (`.to`,
  `tl.trans`, `tl.reshape`, `tl.broadcast_to`, `tl.where`, `tl.maximum`/`tl.minimum`, unary `-`, `*`, `/`, `+`, `-`
  with an operand that is not itself computed from a load), or a helper of the source returning such a value;
  `computed` otherwise (a product of two loads, a `tl.dot`, a value carried by the loop, ...). The test is
  deliberately generous: `load` is a candidate to read by eye, not a finding;
- carried: whether its result flows, through assignments in the loop's body, into a variable the loop carries (one
  assigned in the body that has a value before the loop: a parameter or an earlier assignment);
- candidate: both. A candidate is in the exclusion when, besides, the reduction is along the last axis of a tile of
  rank 2 or more, which the output gives the source text to check.
"""
import ast
import glob
import os

RED = {"sum", "max", "min", "reduce"}
IDENT_FN = {"trans", "reshape", "broadcast_to", "where", "maximum", "minimum", "view", "permute", "to", "cast",
            "multiple_of", "max_contiguous", "max_constancy", "expand_dims"}
OUT = []


def out(s=""):
    print(s)
    OUT.append(s)


def callee(n):
    f = n.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def is_runtime_for(n):
    it = n.iter
    if not isinstance(it, ast.Call):
        return False
    f = it.func
    if isinstance(f, ast.Name):
        return f.id == "range"
    if isinstance(f, ast.Attribute):
        return f.attr == "range"
    return False


def assigned_names(stmts):
    names = set()
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                tg = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in tg:
                    for m in ast.walk(t):
                        if isinstance(m, ast.Name):
                            names.add(m.id)
    return names


class Fn:
    def __init__(self, node, path):
        self.node, self.path = node, path
        self.params = {a.arg for a in node.args.args + node.args.kwonlyargs}
        self.assigns = {}  # name -> list of value exprs
        for n in ast.walk(node):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        self.assigns.setdefault(t.id, []).append(n.value)
                    elif isinstance(t, ast.Tuple) and isinstance(n.value, ast.Tuple):
                        for a, b in zip(t.elts, n.value.elts):
                            if isinstance(a, ast.Name):
                                self.assigns.setdefault(a.id, []).append(b)
            elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
                self.assigns.setdefault(n.target.id, []).append(ast.BinOp(n.target, n.op, n.value))
        self.returns = [n.value for n in ast.walk(node) if isinstance(n, ast.Return) and n.value is not None]


def classify(e, fn, fns, seen):
    """'load' if e can be a load up to identities (generous), else 'computed'."""
    if isinstance(e, ast.Call):
        name = callee(e)
        if name == "load":
            return "load"
        if name in IDENT_FN:
            base = e.func.value if name in ("to", "cast", "view", "permute", "trans", "reshape", "expand_dims") \
                and isinstance(e.func, ast.Attribute) and not (isinstance(e.func.value, ast.Name)
                                                               and e.func.value.id == "tl") else None
            args = ([base] if base is not None else []) + list(e.args)
            return "load" if any(classify(a, fn, fns, seen) == "load" for a in args) else "computed"
        if name in fns and name not in seen:
            h = fns[name]
            return "load" if any(classify(r, h, fns, seen | {name}) == "load" for r in h.returns) else "computed"
        return "computed"
    if isinstance(e, ast.UnaryOp):
        return classify(e.operand, fn, fns, seen)
    if isinstance(e, ast.BinOp):
        a, b = classify(e.left, fn, fns, seen), classify(e.right, fn, fns, seen)
        # an identity needs one side a load and the other a constant (not itself from a load)
        return "load" if (a == "load") != (b == "load") else "computed"
    if isinstance(e, ast.Subscript):
        return classify(e.value, fn, fns, seen)
    if isinstance(e, ast.Name):
        key = ("name", fn.node.name, e.id)
        if key in seen:
            return "computed"
        vals = fn.assigns.get(e.id, [])
        return "load" if any(classify(v, fn, fns, seen | {key}) == "load" for v in vals) else "computed"
    return "computed"


def reductions(stmts, fns, seen):
    """(node, function name) for each reduction in stmts, following helpers of the source transitively."""
    found = []
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Call):
                name = callee(n)
                if name in RED and isinstance(n.func, ast.Attribute):
                    found.append(n)
                elif name in fns and name not in seen:
                    for m in reductions(fns[name].node.body, fns, seen | {name}):
                        found.append(m)
    return found


def operand(n):
    f = n.func
    if isinstance(f.value, ast.Name) and f.value.id == "tl":
        return n.args[0] if n.args else None
    return f.value  # method form x.sum(...)


def flows_to(loop, red, carried):
    """Whether red's value flows, through assignments in the loop body, into a carried variable."""
    tainted = set()
    stmts = [n for n in ast.walk(loop) if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign))]
    changed = True
    holds = {id(m) for m in ast.walk(red)}
    while changed:
        changed = False
        for s in stmts:
            val = s.value
            if val is None:
                continue
            ids = {id(m) for m in ast.walk(val)}
            names = {m.id for m in ast.walk(val) if isinstance(m, ast.Name)}
            if (holds & ids) or (names & tainted):
                tg = s.targets if isinstance(s, ast.Assign) else [s.target]
                for t in tg:
                    for m in ast.walk(t):
                        if isinstance(m, ast.Name) and m.id not in tainted:
                            tainted.add(m.id)
                            changed = True
            # a reduction inside a call to a helper whose result is assigned
    return bool(tainted & carried), tainted & carried


def helper_red_flow(loop, fns, carried):
    """Reductions reached through a helper call: the helper call's result is what flows."""
    res = []
    for n in ast.walk(loop):
        if isinstance(n, ast.Call) and callee(n) in fns:
            hs = reductions(fns[callee(n)].node.body, fns, {callee(n)})
            if hs:
                res.append((n, hs))
    return res


def scan(path):
    src = open(path).read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    fns = {n.name: Fn(n, path) for n in tree.body if isinstance(n, ast.FunctionDef)}
    rows = []
    for fname, fn in fns.items():
        for loop in [n for n in ast.walk(fn.node) if isinstance(n, ast.For) and is_runtime_for(n)]:
            before = fn.params | {k for k, vs in fn.assigns.items()
                                  if any(getattr(v, "lineno", 10 ** 9) < loop.lineno for v in vs)}
            carried = assigned_names(loop.body) & before
            direct = [m for m in ast.walk(loop) if isinstance(m, ast.Call) and callee(m) in RED
                      and isinstance(m.func, ast.Attribute)]
            for red in direct:
                op = operand(red)
                kind = classify(op, fn, fns, set()) if op is not None else "computed"
                fl, which = flows_to(loop, red, carried)
                rows.append((path, red.lineno, fname, loop.lineno, kind, fl, sorted(which),
                             ast.unparse(red)[:100], ""))
            for call, hs in helper_red_flow(loop, fns, carried):
                fl, which = flows_to(loop, call, carried)
                for red in hs:
                    op = operand(red)
                    h = next((fns[k] for k in fns if any(m is red for m in ast.walk(fns[k].node))), fn)
                    kind = classify(op, h, fns, set()) if op is not None else "computed"
                    rows.append((path, red.lineno, fname, loop.lineno, kind, fl, sorted(which),
                                 ast.unparse(red)[:100], f" via {callee(call)} (line {call.lineno})"))
    return rows


def main():
    paths = sorted(glob.glob("claims/*/source/*.py"))
    out(f"{len(paths)} source files of {len(set(p.split('/')[1] for p in paths))} claims")
    cands = 0
    seen = set()
    for p in paths:
        for row in scan(p):
            path, ln, fname, lln, kind, fl, which, text, via = row
            key = (path, ln, lln, via)
            if key in seen:
                continue
            seen.add(key)
            cand = kind == "load" and fl
            cands += cand
            out(f"{'CANDIDATE ' if cand else ''}{path}:{ln}: in {fname}, runtime for at line {lln}{via}: "
                f"operand {kind}; carried {fl}{' ' + ','.join(which) if which else ''}; {text}")
    out(f"{cands} candidates")
    with open(os.path.join(os.path.dirname(__file__), "claims_scan.txt"), "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
