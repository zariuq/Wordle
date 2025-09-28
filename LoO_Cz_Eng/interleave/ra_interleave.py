"""CLI tool for interleaving Czech and English Ra Contact sessions."""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

HEADER_RE = re.compile(r"^\s*(\d{1,3})\s*\.\s*(\d{1,3})\s*\((CS|EN)\)\s*(.*)$", re.IGNORECASE)
IMPLICIT_TOKEN_RE = re.compile(r"(?<!\d)(\d{1,3})\s*[\.\u00A0\u00B7]\s*(\d{1,3})(?!\d)")

MOJIBAKE_MAP = {
    "\u00e2\u0080\u0099": "\u2019",
    "\u00e2\u0080\u0098": "\u2018",
    "\u00e2\u0080\u009c": "\u201c",
    "\u00e2\u0080\u009d": "\u201d",
    "\u00e2\u0080\u0094": "\u2014",
    "\u00e2\u0080\u0093": "\u2013",
    "\u00e2\u0080\u00a6": "\u2026",
    "\u00e2\u0080\u00a2": "\u2022",
    "\u00e2\u0080\u00a0": "\u2020",
    "\u00e2\u0080\u00a1": "\u2021",
    "\u00c3\u00af": "\u00ef",
    "\u00c3\u00a9": "\u00e9",
    "\u00c3\u00a8": "\u00e8",
    "\u00c2 ": " ",
}

BOILERPLATE_PATTERNS = [
    r"^\s*(Watch the recording|Original audio recordings)\b",
    r"^\s*(Interpretative Resources|Frequently used words|Glossary)\b",
    r"^\s*Book Open\b",
    r"^\s*MB Book Open\b",
    r"^\s*(Our History|Team)\b",
    r"^\s*(←|→)\s*P(?:ředchozí|revious)\s+relace\b",
    r"Themefisher",
    r"Zákon jednoty, Vojtech Schlesinger",
    r"Čeština na lawofone\.info",
    r"youtube|facebook|instagram",
    r"Printed books L/L Research",
    r"Online Store",
    r"Amazon Kindle",
    r"Buy the eBook",
    r"Audiobook",
    r"Listen on Audible",
    r"MOBI Version",
    r"EPUB Version",
    r"^\s*Research\.?$",
    r"^\s*In this context,",
    r"^\s*Jim (writes|píše):",
]

TAIL_PATTERNS = [
    r"In this context",
    r"Jim (writes|píše):",
    r"\bO\s+nás\b",
    r"Staráme se o český překlad",
    r"channelingové komunikace s Ra",
    r"jsme satelitní organizací",
    r"L/L Research",
    r"Research\.",
]

FOOTNOTE_TOKEN = re.compile(r"\[(\d{1,3})\]")
LEADING_PUNCT = re.compile(r"^([0-9]{1,3}\.[0-9]{1,3})\t[.\s]+")

CS_KEYWORDS = ["tazatel", "já jsem", "vážíme si", "nástroj"]
EN_KEYWORDS = ["questioner", "i am", "we appreciate", "instrument"]


def _fix_mojibake(text: str) -> str:
    for bad, good in MOJIBAKE_MAP.items():
        text = text.replace(bad, good)
    return text


def detect_language(text: str, default: str) -> str:
    """Heuristically determine the language of *text*."""

    lowered = text.lower()
    if re.search(r"[ěščřžýáíéóúůďťň]", lowered):
        return "cs"
    if any(keyword in lowered for keyword in CS_KEYWORDS):
        return "cs"
    if any(keyword in lowered for keyword in EN_KEYWORDS):
        return "en"
    return default


def sanitize_text(text: str) -> str:
    """Collapse whitespace, drop boilerplate, and strip simple markup."""

    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = _fix_mojibake(text)
    text = re.sub(r"<[^>]*>", " ", text)

    kept_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(
            re.search(pattern, line, re.IGNORECASE | re.MULTILINE)
            for pattern in BOILERPLATE_PATTERNS
        ):
            continue
        for pattern in TAIL_PATTERNS:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                line = line[: match.start()].strip()
        if not line:
            continue
        kept_lines.append(line)

    if not kept_lines:
        return ""

    cleaned = " ".join(kept_lines)
    cleaned = FOOTNOTE_TOKEN.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _normalise_for_tokenisation(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = _fix_mojibake(text)
    return re.sub(r"<[^>]*>", " ", text)


def _should_skip_token(text: str, start: int, end: int) -> bool:
    i = start - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    if i >= 0 and text[i] in "[(":
        return True

    j = end
    n = len(text)
    while j < n and text[j].isspace():
        j += 1
    return j < n and text[j] in "-–—]/.,;:"


def parse_session_file(path: os.PathLike[str] | str) -> Tuple[int, Dict[str, Dict[str, str]]]:
    """Parse a session file and return its number and text buckets."""

    path = Path(path)
    match = re.search(r"session_(\d{3})", path.name)
    if not match:
        raise ValueError(f"Cannot determine session number from filename: {path}")
    session_number = int(match.group(1))

    buckets: Dict[str, Dict[str, str]] = defaultdict(lambda: {"cs": "", "en": ""})

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to read session file {path}: {exc}") from exc

    blocks = []
    current_block = None

    for line in content.splitlines():
        header_match = HEADER_RE.match(line)
        if header_match:
            if current_block:
                blocks.append(current_block)
            sq = f"{int(header_match.group(1))}.{int(header_match.group(2))}"
            lang = "cs" if header_match.group(3).lower() == "cs" else "en"
            initial_text = header_match.group(4).strip()
            current_block = {
                "sq": sq,
                "lang": lang,
                "lines": [initial_text] if initial_text else [],
            }
        elif current_block is not None:
            current_block["lines"].append(line)
    if current_block:
        blocks.append(current_block)

    for block in blocks:
        base_sq = block["sq"]
        default_lang = block["lang"]
        text = "\n".join(block["lines"]).strip()
        if not text:
            continue

        text = _normalise_for_tokenisation(text)
        last_index = 0
        current_sq = base_sq
        for match in IMPLICIT_TOKEN_RE.finditer(text):
            if _should_skip_token(text, match.start(), match.end()):
                continue
            start = match.start()
            if start > last_index:
                slice_text = text[last_index:start]
                _store_slice(slice_text, current_sq, default_lang, buckets)
            current_sq = f"{int(match.group(1))}.{int(match.group(2))}"
            last_index = match.end()
        if last_index < len(text):
            slice_text = text[last_index:]
            _store_slice(slice_text, current_sq, default_lang, buckets)

    return session_number, buckets


def _store_slice(slice_text: str, sq: str, default_lang: str, buckets: Dict[str, Dict[str, str]]) -> None:
    cleaned = sanitize_text(slice_text)
    if not cleaned:
        return
    lang = detect_language(cleaned, default_lang)
    bucket = buckets[sq]
    if len(cleaned) > len(bucket[lang]):
        bucket[lang] = cleaned


def write_interleaved(buckets: Dict[str, Dict[str, str]], out_path: os.PathLike[str] | str) -> None:
    """Write interleaved text pairs to *out_path*."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def sort_key(sq: str) -> Tuple[int, int]:
        major, minor = sq.split(".")
        return int(major), int(minor)

    sorted_items = sorted(buckets.items(), key=lambda item: sort_key(item[0]))

    with out_path.open("w", encoding="utf-8") as handle:
        for sq, texts in sorted_items:
            cs_text = texts.get("cs", "").strip()
            en_text = texts.get("en", "").strip()
            if cs_text:
                line = f"{sq}\t{cs_text}"
                line = LEADING_PUNCT.sub(r"\1\t", line)
                handle.write(f"{line}\n")
            if en_text:
                line = f"{sq}\t{en_text}"
                line = LEADING_PUNCT.sub(r"\1\t", line)
                handle.write(f"{line}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interleave Czech and English Ra Contact sessions.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        help="Directory containing session_###_cs_en.txt files",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        help="Directory where interleaved files will be written",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    input_dir = args.input_dir or root / "data" / "LoO_sessions"
    output_dir = args.output_dir or root / "out_interleaved"

    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    session_files = sorted(
        (path for path in input_dir.iterdir() if path.is_file() and "session_" in path.name),
        key=lambda p: p.name,
    )

    if not session_files:
        parser.error(f"No session files found in {input_dir}")

    exit_code = 0
    for session_file in session_files:
        try:
            session_number, buckets = parse_session_file(session_file)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed to parse {session_file}: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        out_file = output_dir / f"session_{session_number:03d}_interleaved.txt"
        try:
            write_interleaved(buckets, out_file)
        except OSError as exc:  # pragma: no cover - defensive
            print(f"Failed to write {out_file}: {exc}", file=sys.stderr)
            exit_code = 1

    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
