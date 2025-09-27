#!/usr/bin/env python3
import re, sys
from pathlib import Path

HDR = re.compile(r'^\s*(\d{1,3})\s*\.\s*(\d{1,3})\s*\((CS|EN)\)\s*(.*)$',
                 re.IGNORECASE | re.MULTILINE)
# robust: do NOT match 1.1 inside 1.10
TOK = re.compile(r'(?<!\d)(\d{1,3})\s*\.\s*(\d{1,3})(?!\d)')

CZ_DIACR = set("áéěíóúůýžščřďťňÁÉĚÍÓÚŮÝŽŠČŘĎŤŇ")
FOOTER = {"rychlé odkazy","sociální sítě","o nás","šablona webu","nastavení cookies","©",
          "watch video","original audio recordings","glossary","unusual words",
          "entities","places","resource series","book open","amazon kindle","audiobook",
          "youtube","facebook","instagram","read time"}

def looks_cs(s): return "Tazatel" in s or "Já jsem" in s or any(ch in CZ_DIACR for ch in s)
def looks_en(s): return re.search(r'\b(Questioner|I am Ra|Ra I am Ra|Yes,|No,|Could you)\b', s) is not None
def clean(s):
    out=[]
    for ln in s.splitlines():
        t=ln.strip()
        if not t: continue
        if any(k in t.lower() for k in FOOTER): continue
        out.append(t)
    return re.sub(r"\s+"," "," ".join(out)).strip()

def segment(sess, declared_lang, text, hdr_key):
    out={}
    m=[x for x in TOK.finditer(text) if int(x.group(1))==sess]
    if not m:
        t=clean(text)
        if t: out.setdefault(hdr_key,{})[declared_lang]=t
        return out
    if m[0].start()>0:
        t=clean(text[:m[0].start()])
        if t: out.setdefault(hdr_key,{})[declared_lang]=t
    for i,x in enumerate(m):
        k=f"{int(x.group(1))}.{int(x.group(2))}"
        seg=clean(text[x.end(): (m[i+1].start() if i+1<len(m) else len(text))])
        if not seg: continue
        lang = declared_lang
        if looks_cs(seg): lang='cs'
        elif looks_en(seg): lang='en'
        if k not in out: out[k]={}
        if len(seg) > len(out[k].get(lang,"")): out[k][lang]=seg
    return out

def parse_one(path:Path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    buckets={}
    ex = list(HDR.finditer(raw))
    for i,h in enumerate(ex):
        s,q = int(h.group(1)), int(h.group(2))
        lang = 'cs' if h.group(3).upper()=='CS' else 'en'
        text = h.group(4) + raw[h.end(): ex[i+1].start() if i+1<len(ex) else len(raw)]
        pieces = segment(s, lang, text, f"{s}.{q}")
        for k,d in pieces.items():
            if k not in buckets: buckets[k]={}
            for L,txt in d.items():
                if len(txt) > len(buckets[k].get(L,"")): buckets[k][L]=txt
    return buckets

def write_tsv(buckets, outp:Path):
    keys = sorted(buckets, key=lambda k:(int(k.split('.')[0]),int(k.split('.')[1])))
    with outp.open("w",encoding="utf-8") as f:
        for k in keys:
            f.write(f"{k}\tCS\t{buckets[k].get('cs','')}\n")
            f.write(f"{k}\tEN\t{buckets[k].get('en','')}\n")

if __name__=="__main__":
    in_dir = Path(sys.argv[1]); out_dir = Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(in_dir.glob("session_*_cs_en.txt")):
        s = int(re.search(r'(\d{1,3})', p.stem).group(1))
        b = parse_one(p)
        write_tsv(b, out_dir / f"session_{s:03d}_interleaved.tsv")

