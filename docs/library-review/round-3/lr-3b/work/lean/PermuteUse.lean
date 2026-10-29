import TVA
/-! Library review round 3 (lr-3b), L17: `[permute]` says a model writes `x.permute dims` with a
literal list and the permutation proof is found by itself. Check that this elaborates for the
forms the rule names (rank 3 and 4, `tl.trans(x)` of rank 3 as `[0, 2, 1]`), that the result
shape is `dims.map (Shape.dim s)`, and that `[1, 0]` is `trans`. -/
open TVA

variable (x3 : Tensor [2, 3, 4] Nat) (x4 : Tensor [2, 3, 4, 5] Nat) (x2 : Tensor [3, 4] Nat)

example : Tensor [2, 4, 3] Nat := x3.permute [0, 2, 1]
example : Tensor [4, 3, 2] Nat := x3.permute [2, 1, 0]
example : Tensor [5, 2, 4, 3] Nat := x4.permute [3, 0, 2, 1]
example : x2.permute [1, 0] = x2.trans := Tensor.permute_trans x2

-- the element convention: output coordinate k is input coordinate dims[k] (torch.permute's)
example (o : Idx [4, 3, 2]) (i : Idx [2, 3, 4])
    (hi : ∀ k, k < 3 → i.coord ([2, 1, 0].getD k 0) = o.coord k) :
    (x3.permute [2, 1, 0]).get o = x3.get i :=
  Tensor.get_permute x3 [2, 1, 0] (by decide) o i hi

#print axioms Tensor.get_permute
