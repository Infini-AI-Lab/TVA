"""lr-19b: CPython PyFloat_Pack2 (used by Triton 3.8.0's launcher extractFP16 for a tl.float16 scalar argument) on NaN
doubles, in TVA's Python (3.13) and in CPython 3.14. struct.pack("<e") calls PyFloat_Pack2."""
import struct, sys
for b in (0x7ff8000000000000, 0xfff8000000000000, 0x7ff4000000000000, 0x7ff0000000000001, 0x7fffffffffffffff):
    d = struct.unpack("<d", struct.pack("<Q", b))[0]
    h = struct.unpack("<H", struct.pack("<e", d))[0]
    print(sys.version.split()[0], f"{b:016x} -> f16 {h:04x}")
