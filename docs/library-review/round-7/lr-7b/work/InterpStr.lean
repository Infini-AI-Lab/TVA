-- Library hygiene (L40): Lean's lexer reads an interpolated string with a nested string holding `--`.
def hidden : String := s!"{"--"}"  #eval s!"meta code ran: {1 + 1}"

def hidden2 : String := s!"a{"/-"}b"  #eval "second command ran"
