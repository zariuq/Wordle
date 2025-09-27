"""Demonstration tests for the Ra Contact interleaver."""

from __future__ import annotations

import textwrap
from pathlib import Path

from interleave.ra_interleave import parse_session_file


def test_parse_session(tmp_path):
    content = textwrap.dedent(
        """
        999.0 (EN) Ra introduction 1.1 Questioner What is the plan? 1.2 Questioner Please continue. 1.10 This should stay together.
        1.1 (CS) Tazatel: Jaký je plán?
        1.2 (CS) Tazatel: Prosím, pokračujte.
        """
    ).strip()

    session_file = tmp_path / "session_999_cs_en.txt"
    session_file.write_text(content, encoding="utf-8")

    session_number, buckets = parse_session_file(session_file)

    assert session_number == 999
    assert buckets["999.0"]["en"]
    assert buckets["1.1"]["en"]
    assert buckets["1.2"]["en"]
    assert buckets["1.1"]["cs"]
    assert buckets["1.2"]["cs"]
    assert "stay together" in buckets["1.10"]["en"]
