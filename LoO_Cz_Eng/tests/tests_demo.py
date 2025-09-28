"""Demonstration tests for the Ra Contact interleaver."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from LoO_Cz_Eng.interleave.ra_interleave import parse_session_file, sanitize_text


def test_parse_session(tmp_path):
    content = textwrap.dedent(
        """
        999.0 (EN) Ra introduction 1.1 Questioner What is the plan? 1.2 Questioner Please continue. 1.10 This should stay together.
        1.1 (CS) Tazatel: Jaký je plán?[1]
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
    assert "[" not in buckets["1.1"]["cs"]
    assert buckets["1.2"]["cs"]
    assert "stay together" in buckets["1.10"]["en"]


def test_embedded_language_slices(tmp_path):
    content = textwrap.dedent(
        """
        1.0 (EN) At this time we speak. 1.1 Questioner It seems members of the Confederation have a specific purpose. <sup>note</sup> In our vibration paradoxes resolve. 1.2 Questione< >1.1 Tazatel: Zdá se, že členové Konfederace mají specifický cíl. Je to pravda? 1.2 Tazatel: Ano, dává. Děkuji. 1.2 Questioner Yes, it does. Thank you. Ra We appreciate your vibration. Is there another query?
        1.1 (CS) Tazatel: Zdá se, že členové Konfederace mají specifický cíl.
        1.2 (CS) Tazatel: Ano, dává. Děkuji.
        """
    ).strip()

    session_file = tmp_path / "session_001_cs_en.txt"
    session_file.write_text(content, encoding="utf-8")

    _, buckets = parse_session_file(session_file)

    assert "Confederation" in buckets["1.1"]["en"]
    assert "Thank you" in buckets["1.2"]["en"]
    assert buckets["1.1"]["cs"].startswith("Tazatel")
    assert buckets["1.2"]["cs"].startswith("Tazatel")


def test_sanitize_text_handles_mojibake_and_boilerplate():
    raw = (
        "Jim writes: This should disappear.\n"
        "Watch the recording later.\n"
        "Questioner \u00e2\u0080\u0094 how are you? [12]\n"
        "We are well."
    )

    cleaned = sanitize_text(raw)

    assert cleaned.startswith("Questioner — how are you?")
    assert "Jim writes" not in cleaned
    assert "Watch the recording" not in cleaned
    assert "[12]" not in cleaned
