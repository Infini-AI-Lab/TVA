"""lr-17b: an independent AST scan of the in-scope tutorial functions (library issue L88).

Lists every AST node kind and every call (with the keywords given) in the 23 in-scope functions of
`tutorials/inventory.json` (all but tutorial 05's `_layer_norm_bwd_dx_fused`, tutorial 08's
`grouped_matmul_kernel`, tutorial 10's `block_scaled_matmul_kernel_cdna4` and tutorial 11), the
decorators included (they run on the host), so that each can be matched by hand with SEMANTICS.md's
supported subset. A method call on an expression is listed as `<obj>.<name>`.

No GPU. Run from TVA's root: python3 -I docs/library-review/round-17/lr-17b/work/call_scan.py
"""
import ast, collections, json

EXC = {("05-layer-norm.py", "_layer_norm_bwd_dx_fused"), ("08-grouped-gemm.py", "grouped_matmul_kernel"),
       ("10-block-scaled-matmul.py", "block_scaled_matmul_kernel_cdna4")}


def main():
    inv = json.load(open("tutorials/inventory.json"))
    calls, nodes, kws = collections.Counter(), collections.Counter(), collections.defaultdict(set)
    nfun = 0
    for t in inv["tutorials"]:
        if t["source"].startswith("11-"):
            continue
        tree = ast.parse(open("tutorials/" + t["source"]).read())
        want = {f["name"] for f in t["functions"] if (t["source"], f["name"]) not in EXC}
        for fn in tree.body:
            if not (isinstance(fn, ast.FunctionDef) and fn.name in want):
                continue
            nfun += 1
            for sub in ast.walk(fn):
                nodes[type(sub).__name__] += 1
                if isinstance(sub, ast.Call):
                    name = ast.unparse(sub.func)
                    if not name.startswith("tl.") and "." in name:
                        name = "<obj>." + name.rsplit(".", 1)[1]
                    elif name.startswith("tl.") and "(" in name:
                        name = "<obj>." + name.rsplit(".", 1)[1]
                    calls[name] += 1
                    kws[name] |= {k.arg or "**" for k in sub.keywords}
                    if any(isinstance(a, ast.Starred) for a in sub.args):
                        calls["<starred argument>"] += 1
    print(f"functions scanned: {nfun}")
    print("node kinds: " + ", ".join(f"{k} {v}" for k, v in sorted(nodes.items())))
    for n, c in sorted(calls.items()):
        print(f"{n}: {c}" + (f"; keywords {sorted(kws[n])}" if kws[n] else ""))


if __name__ == "__main__":
    main()
