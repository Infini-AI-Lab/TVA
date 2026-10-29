import VecAdd

/-! r-vecadd2, round 2: which instance each operator of the model and the launch resolves to
(`pp.explicit`), so that no `+`, `*`, `<`, `/` or `do` means anything but the library's. -/

set_option pp.fullNames true
set_option pp.explicit true
set_option pp.notation false

#print Kernels.VecAdd.add_kernel
#print Kernels.VecAdd.launch
#print Kernels.VecAdd.model
