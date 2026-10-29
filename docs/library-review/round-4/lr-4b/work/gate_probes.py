"""lr-4b, library review round 4 (L20, L27): the gate's `namespace` and `hygiene` checks and the
item digests, on a copy of the tree. Never writes the real tree: the copy is made in the folder
given as the first argument (the round's scratch directory), with `vec_add_f32` updated for the
round-3 library as evidence/gate-digests/run.py does.

Run from the TVA root:  python3 -I docs/library-review/round-4/lr-4b/work/gate_probes.py SCRATCH
"""
import importlib.util, json, os, re, shutil, subprocess, sys
TVA = os.getcwd()
SCRATCH = sys.argv[1]
spec = importlib.util.spec_from_file_location("gd", os.path.join(TVA, "evidence", "gate-digests", "run.py"))
gd = importlib.util.module_from_spec(spec); spec.loader.exec_module(gd)
assert gd.TVA == TVA

PROBES = r'''
-- P1: a definition named `Launch.mk`, as the model may write the library's constructor
def Launch.mk (grid : Fin 3 → Nat) (fusion debug : Bool) (numCtas : Nat) (pdl : Bool)
    (blocks : List Shape) : TVA.Launch :=
  { grid := grid, fusion := fusion, debug := !debug, numCtas := numCtas, pdl := pdl, blocks := blocks }

-- P2: an instance of a core class on core types (`/` on `Nat`, as in the grid formula)
instance instHDivNatK : HDiv Nat Nat Nat := ⟨fun a b => a / b + 1⟩

-- P3: a look-alike of `tl` (U+2113)
def tℓ.add {s : Shape} {f : TVA.Math.Float.Format} (x y : Tensor s (FloatV f)) :
    Prog (Tensor s (FloatV f)) := TVA.tl.addc x y

-- P4: a structure whose field `add` a binder named `tl` exposes as `tl.add`
structure Ops where
  add : {s : Shape} → {f : TVA.Math.Float.Format} → Tensor s (FloatV f) → Tensor s (FloatV f) →
    Prog (Tensor s (FloatV f))
def binderText (x y : Tensor [4] F32) (tl : Ops := ⟨TVA.tl.addc⟩) : Prog (Tensor [4] F32) :=
  tl.add x y

-- P5: a name of the library namespace `Block`, opened on two lines
section P5
open TVA
  Block
theorem lane_tile : True := trivial
end P5

-- P6: a core name in the kernel's namespace
def min (a b : Nat) : Nat := a + b

-- C1 (control): a library name through `TVA` (expected flagged)
abbrev F16 := FloatV TVA.Math.Float.Format.binary32
'''


def clean(t):
    return t.replace(SCRATCH, "<scratch>").replace(os.path.expanduser("~"), "<home>")


def main():
    root = os.path.join(SCRATCH, "tva")
    if os.path.exists(root):
        shutil.rmtree(root)
    gd.copy_tree(root)
    gd.update_kernel(root)
    # reviewers' scratch output of this round is not part of what is probed
    rr = os.path.join(root, "docs", "library-review", "round-4")
    for d in os.listdir(rr):
        shutil.rmtree(os.path.join(rr, d, "work"), ignore_errors=True)
    lean = os.path.join(root, "kernels", "vec_add_f32", "lean", "VecAdd.lean")
    s = open(lean).read()
    # the only single-line `open Block` becomes a two-line `open`, so tvr's `open` reader misses it
    assert s.count("open Block\n") == 1
    s = s.replace("open Block\n", "open TVA\n  Block\n")
    s = s.replace("end Kernels.VecAdd", PROBES + "\nend Kernels.VecAdd")
    open(lean, "w").write(s)
    r = subprocess.run([sys.executable, "-I", "tools/tvr.py", "check", "vec_add_f32"], cwd=root,
                       capture_output=True, text=True)
    print("## tvr check on the copy, kernel with the probes appended")
    for l in r.stdout.splitlines():
        print("  " + clean(l[:300]))
    # read the claim reader's own output for the full lists
    cr = os.path.join(root, "lean", ".lake", "tvr", "vec_add_f32", "claim_reader.lean")
    r2 = subprocess.run([os.path.expanduser("~/.elan/bin/lake"), "env", "lean", cr], cwd=os.path.join(root, "lean"),
                        capture_output=True, text=True)
    js = [l for l in r2.stdout.splitlines() if l.startswith("TVR_JSON ")]
    if not js:
        print("claim reader gave no JSON:", clean(r2.stdout[-400:]), clean(r2.stderr[-400:]))
        return
    surf = json.loads(js[0][9:])
    print("\n## the claim reader's `shadow` list")
    for x in surf["shadow"]:
        print("  " + x)
    print("## the claim reader's instances")
    for i in surf["instances"]:
        print("  ", i)
    flagged = " ".join(surf["shadow"]) + " " + " ".join(i["name"] for i in surf["instances"] if i["library"])
    for tag, name in [("P1", "Kernels.VecAdd.Launch.mk"), ("P2", "instHDivNatK"), ("P3", "tℓ.add"),
                      ("P4", "Kernels.VecAdd.Ops"), ("P5", "lane_tile"), ("P6", "Kernels.VecAdd.min"),
                      ("C1", "Kernels.VecAdd.F16")]:
        print(f"{tag} {name}: {'flagged' if name in flagged else 'NOT flagged'}")
    print("\n## the model's cone (kernel constants)")
    print("  ", [c for c in surf["items"]["model"]["cone"] if c.startswith("Kernels.")])


if __name__ == "__main__":
    main()
