"""lr-17a: every call of the in-scope tutorial functions, with its positional count and keywords.

Library review round 17, library issue L88 (the supported subset's "Calls" item). The author's scan
(evidence/subset-calls-round17) matches names; this one prints each distinct call shape (callee,
number of positional arguments, keyword names, and the source text of short arguments) so that a
reader can check each against what SEMANTICS.md states about the arguments ("with arguments of the
kinds they state (an argument they do not state is excluded)"). It also lists every `return`, `for`
iterator and statement after an `if` that returns (L89, L90).

No GPU. Run from TVA's root: python3 -I docs/library-review/round-17/lr-17a/work/call_scan.py
"""
import ast, collections, json, os

ROOT = os.getcwd()
TUT = os.path.join(ROOT, "tutorials")
EXCLUDED = {("05-layer-norm.py", "_layer_norm_bwd_dx_fused"), ("08-grouped-gemm.py", "grouped_matmul_kernel"),
            ("10-block-scaled-matmul.py", "block_scaled_matmul_kernel_cdna4")}


def callee(node):
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        base = f.value
        parts = [f.attr]
        while isinstance(base, ast.Attribute):
            parts.append(base.attr)
            base = base.value
        if isinstance(base, ast.Name) and base.id in ("tl", "triton"):
            parts.append(base.id)
            return ".".join(reversed(parts))
        return "<expr>." + f.attr
    return "<call>"


def returns_on_instance(stmt):
    """An `if` whose taken branch could return (conservatively: any branch has a return)."""
    if not isinstance(stmt, ast.If):
        return False
    def has_ret(body):
        return any(isinstance(s, ast.Return) or returns_on_instance(s) for s in body)
    return has_ret(stmt.body) or has_ret(stmt.orelse)


def main():
    inv = json.load(open(os.path.join(TUT, "inventory.json")))
    shapes = collections.Counter()
    where = collections.defaultdict(set)
    fors, rets, after_ret_if, static_rets = [], [], [], []
    nfun = 0
    for t in inv["tutorials"]:
        src = t["source"]
        if src.startswith("11-"):
            continue
        tree = ast.parse(open(os.path.join(TUT, src)).read())
        funs = {f["name"] for f in t["functions"] if (src, f["name"]) not in EXCLUDED}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in funs:
                nfun += 1
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        args = []
                        for a in sub.args:
                            s = ast.unparse(a)
                            args.append(s if len(s) <= 24 else "…")
                        kws = tuple(sorted(k.arg or "**" for k in sub.keywords))
                        key = (callee(sub), len(sub.args), kws)
                        shapes[key] += 1
                        where[key].add(f"{src}:{node.name}")
                    if isinstance(sub, ast.For):
                        fors.append(f"{src}:{sub.lineno}: for {ast.unparse(sub.target)} in {ast.unparse(sub.iter)}")
                    if isinstance(sub, ast.Return):
                        rets.append(f"{src}:{sub.lineno} in {node.name}: {ast.unparse(sub)}")
                    if isinstance(sub, ast.For) and "static_range" in ast.unparse(sub.iter):
                        for s2 in ast.walk(sub):
                            if isinstance(s2, ast.Return):
                                static_rets.append(f"{src}:{s2.lineno}")
                    for field in ("body", "orelse"):
                        body = getattr(sub, field, None)
                        if isinstance(body, list):
                            for i, st in enumerate(body[:-1]):
                                if returns_on_instance(st):
                                    after_ret_if.append(f"{src}:{body[i + 1].lineno} after the if at {st.lineno}")
    print(f"in-scope functions scanned: {nfun}")
    print("call shapes (callee, positional count, keywords): uses, functions")
    for key in sorted(shapes, key=lambda k: (k[0], k[1], k[2])):
        print(f"  {key[0]}  pos={key[1]}  kw={list(key[2])}: {shapes[key]} uses; {sorted(where[key])[:3]}")
    print("for statements:")
    for f in fors:
        print("  " + f)
    print("returns:")
    for r in rets:
        print("  " + r)
    print("statements after an if with a return in a branch:", after_ret_if or "none")
    print("returns inside a tl.static_range body:", static_rets or "none")


if __name__ == "__main__":
    main()
