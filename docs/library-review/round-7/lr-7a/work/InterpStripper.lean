def s : String := s!"{'"'} -- " #eval 7
#eval s
def t : String := s!"{'"'} -- " theorem bad : 1 = 1 := by run_tac pure ()
