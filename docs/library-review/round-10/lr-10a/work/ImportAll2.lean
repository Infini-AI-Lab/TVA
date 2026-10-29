-- lr-10a, round 10: `import all` on a library module from a `module` file (Lean refuses it: TVA is not a module).
module
import all TVA.Rules.Num
#check 1
