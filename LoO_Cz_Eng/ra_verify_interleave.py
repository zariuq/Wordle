#!/usr/bin/env python3
import re, sys, json
from pathlib import Path

# Explicit headers like "106.2 (CS)" and "106.2 (EN)"
HDR = re.compile(r'^\s*(\d{1,3})\s*\.\s*(\d{1,3})\s*\((CS|EN)\)\s*(.*)$',
                 re.IGNORECASE | re.MULTILINE)
# Implicit tokens "S.Q" anywhere (careful: don't match 1.1 inside 1.10)
TOK = re.compile(r'(?<!\d)(\d{1,3})\s*\.\s*(\d{1,3})(?!\d)')

CZ_DIACR = set("áéěíóúůýžščřďťňÁÉĚÍÓÚŮÝŽŠČŘĎŤŇ")
FOOTER_CLUES = {
 "rychlé odkazy","sociální sítě","o nás","šablona webu","nastavení cookies","©",
 "watch video","listen","original audio recordings","glossary","unusual words",
 "entities","places","protection rituals","resource series","living the law of one",
 "amazon kindle","audiobook","mobi","epub","youtube","facebook","instagram","read time"
}

def looks_cs(s: str) -> bool:
    return "Tazatel" in s or "Já jsem" in s or any(ch in CZ_DIACR for ch in s)

def looks_en(s: str) -> bool:
    return re.search(r'\b(Questioner|I am Ra|Ra I am Ra|Yes,|No,|Could you|Would you)\b', s) is not None

def clean(s: str) -> str:
    lines = []
    for ln in s.splitlines():
        t = ln.strip()
        if not t: 
            continue
        low = t.lower()
        if any(k in low for k in FOOTER_CLUES): 
            continue
        lines.append(t)
    s2 = " ".join(lines)
    return re.sub(r'\s+', ' ', s2).strip()

def segment_block(sess:int, declared_lang:str, text:str, hdr_key:str):
    """Split a language block by implicit S.Q anywhere inside; return {key:{'cs'|'en': text}}"""
    out = {}
    matches = [m for m in TOK.finditer(text) if int(m.group(1)) == sess]
    if not matches:
        t = clean(text)
        if t:
            out.setdefault(hdr_key, {})[declared_lang.lower()] = t
        return out
    # leading chunk belongs to hdr_key
    if matches[0].start() > 0:
        t = clean(text[:matches[0].start()])
        if t:
            out.setdefault(hdr_key, {})[declared_lang.lower()] = t
    # each implicit token starts a new S.Q
    for i, m in enumerate(matches):
        k = f"{int(m.group(1))}.{int(m.group(2))}"
        seg = clean(text[m.end() : (matches[i+1].start() if i+1 < len(matches) else len(text))])
        if not seg: 
            continue
        # trust content if declared label is wrong
        lang = declared_lang.lower()
        if looks_cs(seg): lang = 'cs'
        elif looks_en(seg): lang = 'en'
        out.setdefault(k, {})
        if len(seg) > len(out[k].get(lang, "")):
            out[k][lang] = seg
    return out

def parse_session_file(path: Path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    buckets = {}  # "S.Q" -> {'cs': str, 'en': str}
    m_sess = re.search(r'(\d{1,3})', path.stem)
    guessed_sess = int(m_sess.group(1)) if m_sess else -1

    matches = list(HDR.finditer(raw))
    if not matches:
        return guessed_sess, {}

    for idx, h in enumerate(matches):
        s = int(h.group(1)); q = int(h.group(2))
        lang = h.group(3).upper()  # 'CS' or 'EN'
        hdr_key = f"{s}.{q}"
        block = h.group(4) + raw[h.end(): (matches[idx+1].start() if idx+1 < len(matches) else len(raw))]
        pieces = segment_block(s, lang, block, hdr_key)
        for k, d in pieces.items():
            buckets.setdefault(k, {})
            for L, txt in d.items():
                if len(txt) > len(buckets[k].get(L, "")):
                    buckets[k][L] = txt

    # normalize: ensure both langs exist (even if empty)
    for k in list(buckets.keys()):
        buckets[k].setdefault('cs', '')
        buckets[k].setdefault('en', '')
    return s, buckets

def interleave_and_write(sess:int, buckets:dict, out_tsv:Path):
    keys = sorted(buckets.keys(), key=lambda k: (int(k.split('.')[0]), int(k.split('.')[1])))
    with out_tsv.open("w", encoding="utf-8") as f:
        for k in keys:
            f.write(f"{k}\tCS\t{buckets[k]['cs']}\n")
            f.write(f"{k}\tEN\t{buckets[k]['en']}\n")

def audit(sess:int, buckets:dict):
    missing = [k for k,v in buckets.items() if not v['cs'] or not v['en']]
    return {
        "session": sess,
        "total_keys": len(buckets),
        "complete": len(missing)==0,
        "missing": [{"key":k, "cs": not buckets[k]['cs'], "en": not buckets[k]['en']} for k in sorted(missing, key=lambda x:(int(x.split('.')[0]),int(x.split('.')[1])))]
    }

def main():
    if len(sys.argv) < 3:
        print("Usage: python ra_verify_interleave.py <in_dir> <out_dir>", file=sys.stderr)
        sys.exit(2)
    in_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)

    report = []
    for p in sorted(in_dir.glob("session_*_cs_en.txt")):
        sess, buckets = parse_session_file(p)
        if sess == -1 or not buckets:
            report.append({"session": p.name, "error":"no-headers"})
            continue
        out_tsv = out_dir / f"session_{sess:03d}_interleaved.tsv"
        interleave_and_write(sess, buckets, out_tsv)
        report.append(audit(sess, buckets))

    print(json.dumps({"summary": report}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

