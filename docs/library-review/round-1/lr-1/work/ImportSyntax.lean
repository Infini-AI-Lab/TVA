module

public import Aesop

/-! lr-1: a `module` file whose import line starts with `public`. `tvr`'s import check matches
only lines that begin with `import`, so this line is not seen. -/

public theorem lr1_probe : True := by aesop
