"""Compare the library's promotion definitions (`TVA.Lang.Promote`) with the front end's table.

`table.py` records what Triton 3.8.0's front end makes of each operator on each pair of operands
(`table.tsv`). This script evaluates the library's definitions on the same operands, in Lean
(`BinOp.typing?`, `UnOp.typing?`, `whereTyping?`, `boolOp?`), and compares the two line by line:
the result's dtype and kind (block `t:`, scalar `s:`), a compile-time Python value (`py`), or a
rejection (`error`). Lines whose operands are all Python numbers are Python's own arithmetic,
which the library does not model; they are skipped and counted.

Run (from TVA's root, after `lake build TVA`):
  .venv/bin/python -I evidence/promotion/lean_compare.py
It writes the Lean file it evaluates to a temporary folder and runs `lake env lean` in `lean/`.
"""
import collections
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LAKE = os.path.join(os.path.expanduser("~"), ".elan", "bin", "lake")

LEAN = r'''
import TVA.Lang.Promote
open TVA

def dts : List DType := DType.all

def operands : List (String × Operand) :=
  dts.map (fun d => ("t:" ++ d.tname, Operand.tensor d true)) ++
  dts.map (fun d => ("s:" ++ d.tname, Operand.tensor d false)) ++
  ([0, 1, -1, 2147483647, 2147483648, -2147483649, 9223372036854775808, 18446744073709551615] : List Int).map
    (fun v => ("int:" ++ toString v, Operand.int v)) ++
  [("float", Operand.float (FloatV.ofPat 4609434218613702656)), ("bool", Operand.bool true)]

def binops : List (String × BinOp) :=
  [("+", .add), ("-", .sub), ("*", .mul), ("/", .div), ("//", .floordiv), ("%", .mod),
   ("&", .band), ("|", .bor), ("^", .bxor), ("<<", .shl), (">>", .shr),
   ("<", .lt), ("<=", .le), (">", .gt), (">=", .ge), ("==", .eq), ("!=", .ne),
   ("tl.minimum", .minimum), ("tl.maximum", .maximum), ("min", .minimum), ("max", .maximum)]

def unops : List (String × UnOp) := [("neg", .neg), ("invert", .invert), ("not", .not), ("pos", .pos)]

def spell : Option Typing → String
  | some t => (if t.block then "t:" else "s:") ++ t.res.tname
  | none => "error"

def spellB : Option BoolOpRes → String
  | some .python => "py"
  | some (.tensor d b) => (if b then "t:" else "s:") ++ d.tname
  | none => "error"

def main : IO Unit := do
  for (on, op) in binops do
    for (ln, l) in operands do
      for (rn, r) in operands do
        if !(l.isPy && r.isPy) then
          IO.println s!"{on}\t{ln}\t{rn}\t{spell (op.typing? l r)}"
  for (on, isAnd) in [("and", true), ("or", false)] do
    for (ln, l) in operands do
      for (rn, r) in operands do
        if !(l.isPy && r.isPy) then
          IO.println s!"{on}\t{ln}\t{rn}\t{spellB (boolOp? isAnd l r)}"
  for (on, op) in unops do
    for (ln, l) in operands do
      if !l.isPy then
        IO.println s!"{on}\t{ln}\t-\t{spell (op.typing? l)}"
  for (cn, c) in operands do
    for (ln, l) in operands do
      for (rn, r) in operands do
        IO.println s!"where:{cn}\t{ln}\t{rn}\t{spell (whereTyping? c l r)}"

#eval main
'''


def norm(result):
    if result.startswith("py:"):
        return "py"
    return result


def main():
    front = {}
    for line in open(os.path.join(HERE, "table.tsv")):
        cols = line.rstrip("\n").split("\t")
        front[tuple(cols[:3])] = norm(cols[3])
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "PromoteTable.lean")
        with open(src, "w") as f:
            f.write(LEAN)
        p = subprocess.run([LAKE, "env", "lean", src], cwd=os.path.join(ROOT, "lean"),
                           capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-2000:])
        sys.exit(2)
    lean = {}
    for line in p.stdout.splitlines():
        cols = line.split("\t")
        if len(cols) == 4:
            lean[tuple(cols[:3])] = cols[3]
    compared = differ = 0
    missing = 0
    per_op = collections.Counter()
    for key, want in front.items():
        if key not in lean:
            missing += 1
            continue
        compared += 1
        if lean[key] != want:
            differ += 1
            per_op[key[0]] += 1
            if differ <= 40:
                print("DIFFER", "\t".join(key), "| front end:", want, "| library:", lean[key])
    print(f"{len(front)} distinct front-end cases; {compared} compared with the library; {differ} differ; "
          f"{missing} skipped (all operands Python numbers: Python's own arithmetic)")
    if per_op:
        print("differences per operator:", dict(per_op))
    sys.exit(1 if differ else 0)


if __name__ == "__main__":
    main()
