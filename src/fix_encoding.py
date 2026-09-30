"""Repara texto UTF-8 que foi lido como cp1252 e regravado (mojibake)."""
import sys
from pathlib import Path


def fix(text: str) -> str:
    out = bytearray()
    for ch in text:
        try:
            out += ch.encode("cp1252")
        except UnicodeEncodeError:
            out += ch.encode("latin-1") if ord(ch) < 256 else ch.encode("utf-8")
    return out.decode("utf-8")


for name in sys.argv[1:]:
    path = Path(name)
    repaired = fix(path.read_text(encoding="utf-8-sig"))
    path.write_text(repaired, encoding="utf-8")
    print(f"{name}: reparado")