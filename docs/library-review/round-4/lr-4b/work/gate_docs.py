"""lr-4b, library review round 4 (L27): which item digests of `vec_add_f32` a change of a library
docstring re-opens, for docstrings the cone may not reach: a structure field (`TVA.Launch.blocks`),
a constructor of an inductive (`TVA.Step.defer`), and a module docstring (`TVA.Claim`'s). On a copy
of the tree, with the kernel updated as evidence/gate-digests/run.py does; the real tree is never
written. Run from the TVA root:
  python3 -I docs/library-review/round-4/lr-4b/work/gate_docs.py SCRATCH
"""
import importlib.util, os, shutil, sys
TVA = os.getcwd()
SCRATCH = sys.argv[1]
spec = importlib.util.spec_from_file_location("gd", os.path.join(TVA, "evidence", "gate-digests", "run.py"))
gd = importlib.util.module_from_spec(spec); spec.loader.exec_module(gd)


def main():
    root = os.path.join(SCRATCH, "tva-docs")
    if os.path.exists(root):
        shutil.rmtree(root)
    gd.copy_tree(root)
    rr = os.path.join(root, "docs", "library-review", "round-4")
    for d in os.listdir(rr):
        shutil.rmtree(os.path.join(rr, d, "work"), ignore_errors=True)
    gd.update_kernel(root)
    base = gd.gate(root)
    gd.show("A: base", base)
    launch = os.path.join(root, "lean", "TVA", "Sem", "Launch.lean")
    step = os.path.join(root, "lean", "TVA", "Sem", "Step.lean")
    claim = os.path.join(root, "lean", "TVA", "Claim.lean")
    saved = {p: open(p).read() for p in (launch, step, claim)}

    def restore():
        for p, s in saved.items():
            open(p, "w").write(s)
    gd.edit(launch, "/-- The shapes of the source's block values on this instance",
            "/-- (changed) The shapes of the source's block values on this instance")
    gd.show("D1: the docstring of the field TVA.Launch.blocks changes", gd.gate(root), base)
    restore()
    gd.edit(step, "/-- An atomic stays pending while the program takes a later step first.",
            "/-- (changed) An atomic stays pending while the program takes a later step first.")
    gd.show("D2: the docstring of the constructor TVA.Step.defer changes", gd.gate(root), base)
    restore()
    gd.edit(claim, "* `init` is the global memory when the launch starts, and the outcome is the launch's own",
            "* (changed) `init` is the global memory when the launch starts, and the outcome is the launch's own")
    gd.show("D3: the module docstring of TVA.Claim (the claim's scope bullet) changes", gd.gate(root), base)
    restore()


if __name__ == "__main__":
    main()
