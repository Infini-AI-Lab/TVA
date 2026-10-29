-- lr-7a: a keyword namespace (`end`, accepted by manifest_problems) is read as a plain name component.
import Lean
open Lean
def x : Name := `end
example : True := by exact end.trivial
