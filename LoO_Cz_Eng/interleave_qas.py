#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interleave CZ/EN lines by question index (e.g., 57.2) from per-session files like:
  session_106_cs_en.txt
Expected headers inside file:
  106.0 (CS) ...text...
  106.0 (EN) ...text...
The script emits:
  out_interleaved/session_106_interleaved.txt   # one line per language per key
  out_wrapped/session_106_wrapped.txt           # soft-wrapped at --wrap columns
and builds a single HTML with chapters:
  ra_interleaved.html
"""
import re, sys, argparse, html
from pathlib import Path
from typing import Dict, Tuple, List

HEADER_RE = re.compile(r'^\s*(\d{1,3})\s*\.\s*(\d{1,3})\s*\((CS|EN)\)\s*(.*)$', re.IGNORECASE)

def parse_session_file(path: Path) -> Tuple[int, Dict[str, Dict[str, str]]]:
    """
    Return (session_num, map: key -> {'cs': text, 'en': text})
    """
    s_num = None
    buckets: Dict[str, Dict[str, str]] = {}
    current = None  # (key, lang)
    buf: List[str] = []

    def flush():
        nonlocal current, buf
        if current is None:
            return
        key, lang = current
        txt = " ".join(x.strip() for x in buf if x.strip())
        txt = re.sub(r'\s+', ' ', txt).strip()
        if key not in buckets:
            buckets[key] = {}
        buckets[key][lang] = txt
        buf.clear()

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = HEADER_RE.match(line)
            if m:
                # finalize previous block
                flush()
                s_num = s_num or int(m.group(1))
                key = f"{int(m.group(1))}.{int(m.group(2))}"
                lang = 'cs' if m.group(3).upper() == 'CS' else 'en'
                rest = m.group(4).strip()
                current = (key, lang)
                buf = [rest] if rest else []
            else:
                if current is not None:
                    buf.append(line.strip())
                else:
                    # ignore preamble/footer
                    pass
    flush()
    if s_num is None:
        # infer from filename
        mfn = re.search(r'(\d{1,3})', path.stem)
        s_num = int(mfn.group(1)) if mfn else -1
    return s_num, buckets

def write_interleaved_txt(s_num: int, buckets: Dict[str, Dict[str, str]], out_path: Path):
    keys = sorted(buckets.keys(), key=lambda k: (int(k.split('.')[0]), int(k.split('.')[1])))
    with out_path.open("w", encoding="utf-8") as f:
        for k in keys:
            cz = buckets[k].get('cs', '').strip()
            en = buckets[k].get('en', '').strip()
            # one line per language
            f.write(f"{k}\tCS\t{cz}\n")
            f.write(f"{k}\tEN\t{en}\n")

def write_wrapped_txt(s_num: int, buckets: Dict[str, Dict[str, str]], out_path: Path, width: int):
    keys = sorted(buckets.keys(), key=lambda k: (int(k.split('.')[0]), int(k.split('.')[1])))
    with out_path.open("w", encoding="utf-8") as f:
        for k in keys:
            cz = buckets[k].get('cs', '').strip()
            en = buckets[k].get('en', '').strip()
            f.write(f"{k} (CS)\n")
            if cz:
                f.write("\n".join(_soft_wrap(cz, width)) + "\n")
            f.write(f"{k} (EN)\n")
            if en:
                f.write("\n".join(_soft_wrap(en, width)) + "\n")
            f.write("\n")

def _soft_wrap(text: str, width: int) -> List[str]:
    import textwrap
    # Keep paragraphs; wrap each separately.
    lines = []
    for para in text.split("\n"):
        para = re.sub(r'\s+', ' ', para).strip()
        if not para:
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, width=width, break_long_words=False, break_on_hyphens=True))
    return lines

def build_html(all_sessions: Dict[int, Dict[str, Dict[str, str]]], html_path: Path, max_width_ch: int = 90):
    css = f"""
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; 
            line-height: 1.55; margin: 2rem auto; max-width: {max_width_ch}ch; padding: 0 1rem; }}
    h1.title {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
    .toc a {{ margin-right: .5rem; text-decoration: none; font-variant-numeric: tabular-nums; }}
    .session {{ page-break-before: always; }}
    h2.session-title {{ border-bottom: 1px solid #ddd; padding-bottom: .25rem; margin-top: 2rem; }}
    .qa {{ margin: .6rem 0; }}
    .key {{ font-weight: 600; color: #444; font-variant-numeric: tabular-nums; }}
    .line {{ white-space: normal; word-break: break-word; }}
    .cs {{ color: #1f4d3a; }}
    .en {{ color: #2a3b7b; }}
    code.keytag {{ background: #f3f3f3; padding: .05rem .35rem; border-radius: .25rem; }}
    .lang {{ font-size: .85em; font-weight: 700; letter-spacing: .02em; opacity: .8; }}
    """
    with html_path.open("w", encoding="utf-8") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'>")
        f.write("<title>Ra Contact — CS & EN (interleaved)</title>")
        f.write(f"<style>{css}</style></head><body>")
        f.write("<h1 class='title'>Ra Contact — Czech & English (interleaved by question)</h1>")
        # TOC
        f.write("<div class='toc'><strong>Sessions:</strong> ")
        for s in sorted(all_sessions.keys()):
            f.write(f"<a href='#S{s:03d}'>[{s:03d}]</a>")
        f.write("</div>")

        for s in sorted(all_sessions.keys()):
            buckets = all_sessions[s]
            keys = sorted(buckets.keys(), key=lambda k: (int(k.split('.')[0]), int(k.split('.')[1])))
            f.write(f"<div class='session' id='S{s:03d}'>")
            f.write(f"<h2 class='session-title'>Session {s:03d}</h2>")
            for k in keys:
                cz = buckets[k].get('cs', '').strip()
                en = buckets[k].get('en', '').strip()
                if not cz and not en:
                    continue
                f.write("<div class='qa'>")
                f.write(f"<div class='key'><code class='keytag'>#{html.escape(k)}</code></div>")
                f.write(f"<div class='line cs'><span class='lang'>CS</span> — {html.escape(cz)}</div>")
                f.write(f"<div class='line en'><span class='lang'>EN</span> — {html.escape(en)}</div>")
                f.write("</div>")
            f.write("</div>")
        f.write("</body></html>")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="out_sessions", help="Folder containing session_###_cs_en.txt files")
    ap.add_argument("--out_interleaved", default="out_interleaved", help="Folder for one-line interleaved files")
    ap.add_argument("--out_wrapped", default="out_wrapped", help="Folder for soft-wrapped files")
    ap.add_argument("--wrap", type=int, default=100, help="Wrap column for wrapped files")
    ap.add_argument("--html", default="ra_interleaved.html", help="Path to write the combined HTML")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_inter = Path(args.out_interleaved); out_inter.mkdir(parents=True, exist_ok=True)
    out_wrap = Path(args.out_wrapped); out_wrap.mkdir(parents=True, exist_ok=True)

    all_sessions: Dict[int, Dict[str, Dict[str, str]]] = {}
    count_files = 0

    for p in sorted(in_dir.glob("session_*_cs_en.txt")):
        s_num, buckets = parse_session_file(p)
        if s_num == -1 or not buckets:
            continue
        # store
        all_sessions[s_num] = buckets
        # emit per-session
        write_interleaved_txt(s_num, buckets, out_inter / f"session_{s_num:03d}_interleaved.txt")
        write_wrapped_txt(s_num, buckets, out_wrap / f"session_{s_num:03d}_wrapped.txt", width=args.wrap)
        count_files += 1

    if not all_sessions:
        print("No sessions parsed. Check --in_dir and file format.")
        sys.exit(1)

    # combined HTML
    build_html(all_sessions, Path(args.html), max_width_ch=max(70, min(110, args.wrap)))

    # small report
    total_keys = sum(len(v) for v in all_sessions.values())
    have_both = sum(1 for v in all_sessions.values() for k, d in v.items() if d.get('cs') and d.get('en'))
    print({"sessions": len(all_sessions), "total_keys": total_keys, "have_both_langs": have_both, "interleaved_dir": str(out_inter), "wrapped_dir": str(out_wrap), "html": args.html})

if __name__ == "__main__":
    main()
