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
IMPLICIT_TOKEN_RE = re.compile(r"(?<!\d)(\d{1,3})\s*[\.\u00B7]\s*(\d{1,3})(?!\d)")
FOOTER_KEYWORDS = [
    "rychlé odkazy",
    "sociální sítě",
    "o nás",
    "šablona webu",
    "nastavení cookies",
    "©",
    "watch video",
    "listen",
    "original audio recordings",
    "glossary",
    "unusual words",
    "entities",
    "places",
    "protection rituals",
    "resource series",
    "amazon kindle",
    "audiobook",
    "youtube",
    "facebook",
    "instagram",
    "read time",
]

CS_KEYWORDS = ["tazatel", "já jsem", "vážíme si", "nástroj"]
EN_KEYWORDS = ["questioner", "i am", "we appreciate", "instrument"]


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
    text = re.sub(r"<[^>]*>", " ", text)

    for keyword in FOOTER_KEYWORDS:
        text = re.sub(re.escape(keyword), " ", text, flags=re.IGNORECASE)

    lines = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        lines.append(stripped)
    cleaned = " ".join(lines)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _normalise_for_tokenisation(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    return re.sub(r"<[^>]*>", " ", text)


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
                handle.write(f"{sq}\t{cs_text}\n")
            if en_text:
                handle.write(f"{sq}\t{en_text}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interleave Czech and English Ra Contact sessions.")
    parser.add_argument("input_dir", type=Path, help="Directory containing session_###_cs_en.txt files")
    parser.add_argument("output_dir", type=Path, help="Directory where interleaved files will be written")
    args = parser.parse_args(argv)

    input_dir = args.input_dir
    output_dir = args.output_dir

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
