#!/usr/bin/env python3
"""Audit generated interleaved transcripts for formatting and boilerplate."""

from __future__ import annotations

import re
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "out_interleaved"

BAD_BYTES = re.compile(r"(\u00e2\u0080\u0098|\u00e2\u0080\u0099|\u00e2\u0080\u0094|\u00c3.)")
FORBID = [
    r"Watch the recording",
    r"Original audio recordings",
    r"Interpretative Resources",
    r"Frequently used words",
    r"Book Open",
    r"Themefisher",
    r"Jim (writes|píše):",
    r"In this context",
]


def main() -> int:
    err = 0
    for path in sorted(OUT.glob("session_*_interleaved.txt")):
        with path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                if not re.match(r"^\d{1,3}\.\d{1,3}\t.+", line):
                    print(f"[format]\t{path.name}:{lineno}\t{line.strip()}")
                    err = 1
                if BAD_BYTES.search(line):
                    print(f"[mojibake]\t{path.name}:{lineno}\t{line.strip()}")
                    err = 1
                for pat in FORBID:
                    if re.search(pat, line, re.IGNORECASE):
                        print(f"[boiler]\t{path.name}:{lineno}\t{line.strip()}")
                        err = 1
    return err


if __name__ == "__main__":
    sys.exit(main())
