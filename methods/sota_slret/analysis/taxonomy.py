"""Heuristic, pre-declared error taxonomy for T2V/V2T top-1 misses (val). Categories are checked in order."""
import json, re, sys
from collections import Counter
GREET = re.compile(r"\b(evening|viewers|spectators|goodbye|good bye|bye|wish you|have a nice|good night|hello|welcome)\b", re.I)
INTRO = re.compile(r"weather forecast|forecast for tomorrow|and now the weather", re.I)
NUMW = re.compile(r"\d+|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|twelfth|twentieth|thirtieth|minus|degrees?)\b", re.I)
TOK = re.compile(r"[a-z0-9]+")
def cat(q, n):
    if q.strip().lower() == n.strip().lower(): return "identical_caption"
    if GREET.search(q) and GREET.search(n) and len(TOK.findall(q.lower())) <= 12: return "formulaic_greeting"
    if INTRO.search(q) and INTRO.search(n): return "forecast_intro_date"
    a, b = set(TOK.findall(q.lower())), set(TOK.findall(n.lower()))
    if NUMW.search(q + " " + n) and set(x.lower() for x in NUMW.findall(q)) != set(x.lower() for x in NUMW.findall(n)) and len(a & b) / max(1, len(a | b)) >= 0.3:
        return "number_or_entity_diff"
    if len(a & b) / max(1, len(a | b)) >= 0.3: return "partial_overlap_other"
    return "semantic_confusion_low_overlap"
rows = [json.loads(l) for l in open(sys.argv[1])]
for d in ("t2v", "v2t"):
    rs = [r for r in rows if r["dir"] == d]
    c = Counter(cat(r["text"], r["top1_text"]) for r in rs)
    near = Counter(cat(r["text"], r["top1_text"]) for r in rs if r["rank"] <= 5)
    print(d, len(rs), {k: f"{v} ({near[k]} in top5)" for k, v in c.most_common()})
