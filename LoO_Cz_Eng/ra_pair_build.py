#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build interleaved CZ↔EN Ra Contact files from session URLs.
- Reads two CSVs (CZ + EN), extracts all http(s) links.
- Downloads each session page (polite rate limit), parses text.
- Extracts pairs keyed like "57.2" from line text using regex.
- Outputs per-session bilingual TXT (CZ then EN per item) and one HTML for Calibre.
"""

import re, os, time, json, html, argparse
from pathlib import Path
from collections import defaultdict, OrderedDict

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

IDX_RE = re.compile(r"\b(\d{1,3})\.(\d{1,3})\b")          # 57.2
SESSION_RE = re.compile(r"\b(S|Session|Sezení|Sedění)\s*(\d{1,3})\b", re.I)

def find_urls_in_csv(path: Path, domain_hint=None):
    """Return a sorted, de-duplicated list of URLs from any column in a CSV."""
    urls = set()
    # try multiple encodings
    for enc in ("utf-8", "utf-8-sig", "cp1250", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        raise RuntimeError(f"Failed to read {path}")
    for c in df.columns:
        s = df[c].astype(str)
        for val in s:
            v = val.strip()
            if v.startswith("http://") or v.startswith("https://"):
                if (domain_hint is None) or (domain_hint in v):
                    urls.add(v.rstrip("/"))
    return sorted(urls)

def fetch(url, session, tries=3, sleep=0.5):
    for i in range(tries):
        r = session.get(url, timeout=30)
        if r.status_code == 200:
            return r.text
        time.sleep(sleep * (i+1))
    raise RuntimeError(f"GET failed {url} status {r.status_code}")

def text_blocks(html_text):
    """Return visible text blocks in reading order (p/li/div stripped)."""
    soup = BeautifulSoup(html_text, "lxml")
    # Try to focus on the main article area if present
    main = soup.select_one("main") or soup.select_one("article") or soup
    nodes = main.find_all(["p", "li", "div", "span"])
    out = []
    for n in nodes:
        # skip nav/footers by class hints
        cls = " ".join(n.get("class", [])).lower()
        if any(k in cls for k in ["nav","menu","footer","header","cookie","share","social"]):
            continue
        t = n.get_text(" ", strip=True)
        if t:
            out.append(t)
    # de-dupe adjacent boilerplate
    dedup = []
    prev = None
    for t in out:
        if t != prev:
            dedup.append(t)
        prev = t
    return dedup

def parse_session_num(blocks):
    for t in blocks[:30]:
        m = SESSION_RE.search(t)
        if m:
            try:
                return int(m.group(2))
            except:
                pass
    # fallback: scan any integer trip that looks like a session near the top
    for t in blocks[:50]:
        ms = re.findall(r"\b(\d{1,3})\b", t)
        for num in ms:
            n = int(num)
            if 1 <= n <= 200:
                return n
    return None

def extract_pairs(blocks):
    """
    Key heuristic:
    - whenever we see "57.2" in a line, start a new key bucket.
    - accumulate text until the next key appears.
    """
    pairs = OrderedDict()
    current_key = None
    buf = []
    def flush():
        if current_key is not None:
            txt = " ".join(buf).strip()
            if txt:
                # keep the longest in case of duplicates
                if current_key not in pairs or len(txt) > len(pairs[current_key]):
                    pairs[current_key] = txt
        buf.clear()

    for t in blocks:
        keys = list(IDX_RE.finditer(t))
        if keys:
            # flush previous bucket
            flush()
            # If multiple keys in one line, split at first; others will be seen on subsequent lines typically
            m = keys[0]
            key = f"{int(m.group(1))}.{int(m.group(2))}"
            current_key = key
            rest = t[m.end():].strip()
            if rest:
                buf.append(rest)
        else:
            if current_key is not None:
                buf.append(t)
            else:
                # ignore preamble
                pass
    flush()
    return pairs

def harvest(urls, lang_tag, session):
    data = {}
    for u in tqdm(urls, desc=f"Fetch {lang_tag}", unit="pg"):
        try:
            h = fetch(u, session)
            blocks = text_blocks(h)
            sess = parse_session_num(blocks)
            qmap = extract_pairs(blocks)
            # If no explicit session detected, try infer from URL tail
            if sess is None:
                tail_nums = re.findall(r"(\d{1,3})", u)
                if tail_nums:
                    sess = int(tail_nums[-1])
            if sess is None:
                # stash under -1, still usable by key intersection later
                sess = -1
            data.setdefault(sess, {}).update(qmap)
            time.sleep(0.15)
        except Exception as e:
            print(f"[{lang_tag}] fail {u}: {e}")
    return data

def interleave_and_write(cz_data, en_data, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    all_keys = set()
    for s, qm in cz_data.items():
        for q in qm.keys():
            all_keys.add(q)
    for s, qm in en_data.items():
        for q in qm.keys():
            all_keys.add(q)
    # build session->list[(qnum, key)]
    sessions = defaultdict(list)
    for key in all_keys:
        if not IDX_RE.match(key):
            continue
        s,q = key.split(".")
        sessions[int(s)].append((int(q), key))
    for s in sessions:
        sessions[s].sort()
    # write per-session TXT
    written = []
    for s, rows in sorted(sessions.items()):
        p = out_dir / f"session_{s:03d}_cs_en.txt"
        with p.open("w", encoding="utf-8") as f:
            f.write(f"Session {s}\n")
            f.write("=" * (8 + len(str(s))) + "\n\n")
            for q, key in rows:
                cz = cz_data.get(s, {}).get(key, "").strip()
                en = en_data.get(s, {}).get(key, "").strip()
                # only write if at least one language present
                if not cz and not en:
                    continue
                f.write(f"{key} (CS)\n{cz}\n\n")
                f.write(f"{key} (EN)\n{en}\n\n")
        written.append(p)
    return sessions, written

def build_html(cz_data, en_data, sessions, html_path: Path):
    css = """
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height: 1.5; }
    .session { page-break-before: always; }
    h1.session-title { border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }
    .qa { margin: 1em 0; }
    .qa .key { font-weight: 600; color: #555; }
    .lang { font-size: 0.85em; font-weight: 600; text-transform: uppercase; color: #666; }
    """
    with html_path.open("w", encoding="utf-8") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'>")
        f.write("<title>Ra Contact — CS & EN (interleaved)</title>")
        f.write(f"<style>{css}</style></head><body>")
        f.write("<h1>Ra Contact — Czech & English (interleaved by question)</h1>")
        for s, rows in sorted(sessions.items()):
            f.write(f"<div class='session' id='session-{s}'>")
            f.write(f"<h1 class='session-title'>Session {s:03d}</h1>")
            for q, key in rows:
                cz = cz_data.get(s, {}).get(key, "").strip()
                en = en_data.get(s, {}).get(key, "").strip()
                if not cz and not en:
                    continue
                f.write("<div class='qa'>")
                f.write(f"<div class='key'>#{key}</div>")
                f.write(f"<div class='cs'><span class='lang'>CS</span>: " + html.escape(cz) + "</div>")
                f.write(f"<div class='en'><span class='lang'>EN</span>: " + html.escape(en) + "</div>")
                f.write("</div>")
            f.write("</div>")
        f.write("</body></html>")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cz_csv", default="www.zakonjednoty.cz_27th_Sept_2025.csv")
    ap.add_argument("--en_csv", default="www.llresearch.org_27th_Sept_2025.csv")
    ap.add_argument("--outdir", default="ra_bilingual_sessions")
    ap.add_argument("--html", default="ra_bilingual_interleaved.html")
    args = ap.parse_args()

    cz_urls = find_urls_in_csv(Path(args.cz_csv), domain_hint="zakonjednoty.cz")
    en_urls = find_urls_in_csv(Path(args.en_csv), domain_hint="llresearch.org")
    print(f"CZ URLs: {len(cz_urls)}  EN URLs: {len(en_urls)}")

    s = requests.Session()
    s.headers.update({"User-Agent":"Mozilla/5.0 (X11; Linux) AppleWebKit/537.36 (KHTML, like Gecko) Safari"})
    # polite spacing
    cz_data = harvest(cz_urls, "CS", s)
    en_data = harvest(en_urls, "EN", s)

    out_dir = Path(args.outdir); out_dir.mkdir(exist_ok=True, parents=True)
    sessions, written = interleave_and_write(cz_data, en_data, out_dir)
    html_path = Path(args.html)
    build_html(cz_data, en_data, sessions, html_path)

    summary = {
        "per_session_files": len(written),
        "html": str(html_path),
        "outdir": str(out_dir),
        "cz_sessions": len(cz_data),
        "en_sessions": len(en_data)
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

