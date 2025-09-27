#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interleave Ra Contact per-session texts: CZ -> EN for each question key (e.g., 106.0, 106.1, ...).
Input: a directory of files like 'session_001_cs_en.txt' (your current outputs).
Output:
  - out_interleaved/session_001_interleaved.txt  (106.0 CS, 106.0 EN, 106.1 CS, 106.1 EN, ...)
  - ra_bilingual_interleaved.html                (chaptered HTML for Calibre -> EPUB)
Usage:
  python interleave_ra_sessions.py --in out_sessions --out out_interleaved --wrap 100
"""

import re
import html
import argparse
from pathlib import Path
from collections import defaultdict, OrderedDict
from textwrap import fill

HEADER_RE = re.compile(r'^\s*(\d{1,3}\.\d{1,3})\s*\((CS|EN)\)\s*$', re.IGNORECASE)

def parse_session_file(path: Path):
    """
    Parse one session file into an OrderedDict: key -> {'CS': text, 'EN': text}
    We accept headers like: "106.2 (CS)"  / "106.2 (EN)"
    Text is all following lines until the next header or EOF.
    """
    with path.open('r', encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()

    key_lang = None
    buckets = OrderedDict()  # '106.2' -> {'CS': "...", 'EN': "..."}
    buf = []

    def flush():
        nonlocal key_lang, buf
        if key_lang and buf:
            key, lang = key_lang
            text = "\n".join([ln.rstrip() for ln in buf]).strip()
            if text:
                buckets.setdefault(key, {})
                # keep the longer text if duplicates appear
                prev = buckets[key].get(lang, "")
                if len(text) > len(prev):
                    buckets[key][lang] = text
        buf.clear()

    # Skip any preamble lines until the first header
    i = 0
    while i < len(lines):
        m = HEADER_RE.match(lines[i])
        if m:
            break
        i += 1

    # Main scan
    while i < len(lines):
        m = HEADER_RE.match(lines[i])
        if m:
            flush()
            key = f"{int(m.group(1).split('.')[0])}.{int(m.group(1).split('.')[1])}"
            lang = m.group(2).upper()
            key_lang = (key, 'CS' if lang == 'CS' else 'EN')
        else:
            buf.append(lines[i])
        i += 1
    flush()

    return buckets

def wrap_one_line(text: str, width: int) -> str:
    """Make a single logical line with soft wraps at `width` (no giant paragraphs)."""
    if width and width > 0:
        return fill(" ".join(text.split()), width=width)
    return " ".join(text.split())

def write_interleaved_txt(session_no: int, data, out_dir: Path, wrap: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"session_{session_no:03d}_interleaved.txt"
    with p.open("w", encoding="utf-8") as f:
        f.write(f"Session {session_no}\n")
        f.write("=" * (8 + len(str(session_no))) + "\n\n")
        for key in sorted(data.keys(), key=lambda k: (int(k.split('.')[0]), int(k.split('.')[1]))):
            cs = data[key].get('CS', '').strip()
            en = data[key].get('EN', '').strip()
            # Interleave strictly: CS then EN (even if one is empty, we keep the blank line)
            f.write(f"{key} (CS)\n")
            f.write(wrap_one_line(cs, wrap) + "\n\n")
            f.write(f"{key} (EN)\n")
            f.write(wrap_one_line(en, wrap) + "\n\n")
    return p

def build_html(all_sessions, html_path: Path):
    css = """
    body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height: 1.55; }
    main { max-width: 820px; margin: 2rem auto; padding: 0 1rem; }
    .session { page-break-before: always; margin-bottom: 2rem; }
    h1.title { font-size: 1.9rem; border-bottom: 1px solid #ddd; padding-bottom: .4rem; }
    h2.session-title { font-size: 1.5rem; margin-top: 2rem; border-bottom: 1px dashed #ddd; padding-bottom: .25rem; }
    .qa { margin: 1rem 0 1.25rem; }
    .key { font-weight: 600; color: #444; margin-bottom: .2rem; }
    .ln { margin: .15rem 0; }
    .tag { display:inline-block; font-size:.82rem; font-weight:600; color:#666; margin-right:.4rem; }
    .cs { }
    .en { color: #233; }
    """
    with html_path.open("w", encoding="utf-8") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'>")
        f.write("<title>Ra Contact — CS & EN (interleaved by question)</title>")
        f.write(f"<style>{css}</style></head><body><main>")
        f.write("<h1 class='title'>Ra Contact — Czech & English (interleaved per question)</h1>")
        for sess in sorted(all_sessions.keys()):
            f.write(f"<section class='session' id='session-{sess}'>")
            f.write(f"<h2 class='session-title'>Session {sess:03d}</h2>")
            for key in sorted(all_sessions[sess].keys(), key=lambda k: (int(k.split('.')[0]), int(k.split('.')[1]))):
                cs = all_sessions[sess][key].get('CS', '').strip()
                en = all_sessions[sess][key].get('EN', '').strip()
                if not (cs or en):
                    continue
                f.write("<div class='qa'>")
                f.write(f"<div class='key'>#{html.escape(key)}</div>")
                f.write(f"<div class='ln cs'><span class='tag'>CS</span>{html.escape(cs)}</div>")
                f.write(f"<div class='ln en'><span class='tag'>EN</span>{html.escape(en)}</div>")
                f.write("</div>")
            f.write("</section>")
        f.write("</main></body></html>")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="out_sessions", help="Input directory with session_XXX_cs_en.txt")
    ap.add_argument("--out", dest="outdir", default="out_interleaved", help="Output directory for interleaved TXT")
    ap.add_argument("--html", dest="html", default="ra_bilingual_interleaved.html", help="Output HTML for Calibre")
    ap.add_argument("--wrap", dest="wrap", type=int, default=100, help="Soft wrap column for TXT (0 = no wrap)")
    args = ap.parse_args()

    indir = Path(args.indir); outdir = Path(args.outdir)
    files = sorted(indir.glob("session_*_cs_en.txt"))
    if not files:
        raise SystemExit(f"No files like 'session_XXX_cs_en.txt' found in {indir}")

    all_sessions = {}  # sess_no -> { key -> {'CS':..., 'EN':...} }
    for p in files:
        # extract session number
        m = re.search(r'session_(\d{3})_cs_en\.txt$', p.name)
        if not m:
            continue
        sess_no = int(m.group(1))
        buckets = parse_session_file(p)
        # Move into session dict
        sess_map = defaultdict(dict)
        for key, kv in buckets.items():
            sess_map[key].update(kv)
        all_sessions[sess_no] = sess_map
        # Per-session TXT (interleaved)
        write_interleaved_txt(sess_no, sess_map, outdir, args.wrap)

    # Global HTML
    build_html(all_sessions, Path(args.html))
    print(f"Done.\n Interleaved TXT dir: {outdir}\n HTML: {args.html}")

if __name__ == "__main__":
    main()

