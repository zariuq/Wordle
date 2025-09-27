#!/usr/bin/env python3
import re, sys, html, textwrap
from pathlib import Path

EXPL = re.compile(r'^\s*(\d{1,3})\s*\.\s*(\d{1,3})\s*\((CS|EN)\)\s*(.*)$',
                  re.IGNORECASE | re.MULTILINE)
# prevent matching 1.1 inside 1.10
IMPL = re.compile(r'(?<!\d)(\d{1,3})\s*\.\s*(\d{1,3})(?!\d)')

CZ_CHARS = set("áéěíóúůýžščřďťňÁÉĚÍÓÚŮÝŽŠČŘĎŤŇ")

def detect_lang(s:str)->str:
    s2 = s.strip()
    if not s2: return ""
    if "Tazatel" in s2 or "Já jsem" in s2 or any(ch in CZ_CHARS for ch in s2):
        return "cs"
    if re.search(r'\b(Questioner|I am Ra|Ra I am Ra|Yes,|No,|Could you|Would you|That is correct)\b', s2):
        return "en"
    letters = sum(ch.isalpha() for ch in s2)
    ascii_letters = sum(('a'<=ch<='z') or ('A'<=ch<='Z') for ch in s2)
    return "en" if letters and ascii_letters/letters > 0.7 else "cs"

DROP = {
 "rychlé odkazy","sociální sítě","o nás","šablona webu","nastavení cookies","©",
 "watch video","listen","original audio recordings","glossary","unusual words",
 "entities","places","protection rituals","resource series","living the law of one",
 "a wanderer","tilting at windmills","quixotic quest","prophetic qualities",
 "how the ra contact came to be","methodology","printed books","amazon kindle",
 "audiobook","mb mobi","epub version","book open","team","youtube","facebook",
 "instagram","read time"
}

def clean(s:str)->str:
    lines = [ln.strip() for ln in s.splitlines()]
    keep = []
    for ln in lines:
        if not ln: continue
        low = ln.lower()
        if any(x in low for x in DROP): continue
        keep.append(ln)
    s2 = " ".join(keep)
    return re.sub(r"\s+", " ", s2).strip()

def segment_block(s_num:int, lang_label:str, text:str, header_key:str):
    out = {}
    matches = [m for m in IMPL.finditer(text) if int(m.group(1))==s_num]
    if not matches:
        t = clean(text)
        if t: out.setdefault(header_key,{})[lang_label]=t
        return out
    # leading ->  S.0
    if matches[0].start()>0:
        t = clean(text[:matches[0].start()])
        if t: out.setdefault(header_key,{})[lang_label]=t
    for i,m in enumerate(matches):
        key = f"{int(m.group(1))}.{int(m.group(2))}"
        seg = clean(text[m.end(): matches[i+1].start() if i+1<len(matches) else len(text)])
        if not seg: continue
        detected = detect_lang(seg)
        label = detected if detected and detected!=lang_label else lang_label
        d = out.setdefault(key,{})
        if len(seg) > len(d.get(label,"")): d[label]=seg
    return out

def parse_one(path:Path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    buckets = {}
    ex = list(EXPL.finditer(raw))
    for i,m in enumerate(ex):
        s,q = int(m.group(1)), int(m.group(2))
        lang = 'cs' if m.group(3).upper()=='CS' else 'en'
        header_key = f"{s}.{q}"
        text = m.group(4) + raw[m.end(): ex[i+1].start() if i+1<len(ex) else len(raw)]
        pieces = segment_block(s, lang, text, header_key)
        for k, dd in pieces.items():
            bk = buckets.setdefault(k,{})
            for L,txt in dd.items():
                if len(txt) > len(bk.get(L,"")): bk[L]=txt
    return buckets

def write_tsv(buckets, out_path:Path):
    keys = sorted(buckets, key=lambda k:(int(k.split('.')[0]),int(k.split('.')[1])))
    with out_path.open("w", encoding="utf-8") as f:
        for k in keys:
            f.write(f"{k}\tCS\t{buckets[k].get('cs','')}\n")
            f.write(f"{k}\tEN\t{buckets[k].get('en','')}\n")

if __name__=="__main__":
    in_dir = Path(sys.argv[1]); out_dir = Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(in_dir.glob("session_*_cs_en.txt")):
        m = re.search(r"(\d{1,3})", p.stem)
        s_num = int(m.group(1)) if m else 0
        buckets = parse_one(p)
        write_tsv(buckets, out_dir / f"session_{s_num:03d}_interleaved.tsv")

