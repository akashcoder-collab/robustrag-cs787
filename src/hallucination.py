"""Hallucination proxy (report Section 5.1): share of questions where a method answers (no "unknown") but is wrong.
Usage: python src/hallucination.py results.json   (merges results_keyword*.json and drops invalid targets, as plots.py)"""
import json, os, sys, re, string
import numpy as np
def norm(s):
    s = "".join(c for c in s.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())
f = sys.argv[1] if len(sys.argv) > 1 else "results.json"
kwf = f.replace("results", "results_keyword", 1)
r = json.load(open(f))
bad = {x["q"] for x in r if any(norm(g) and (norm(g) in norm(x["target"]) or norm(x["target"]) in norm(g)) for g in x["gold"])}
r = [x for x in r if x["q"] not in bad]
kw = {(x["q"], x["m"]): x for x in json.load(open(kwf))}
r = [x for x in r if (x["q"], x["m"]) in kw]
for x in r:
    x["key1"], x["key1_ok"] = x["key"], x["key_ok"]
    x["key"], x["key_ok"] = kw[(x["q"], x["m"])]["key"], kw[(x["q"], x["m"])]["key_ok"]
abst = lambda o: "unknown" in o.lower() or not o.strip()
print(f"n = {len({x['q'] for x in r})}")
print("| Method | Abstains (m=0) | Wrong but answered (m=0) | Wrong but answered (m=1) |\n|---|---|---|---|")
for k, name in [("van", "Vanilla RAG"), ("key", "Keyword aggregation (final)"),
                ("key1", "Keyword aggregation v1 (fixed threshold 3)"), ("dec", "Decoding aggregation")]:
    pct = lambda m, fn: 100 * np.mean([fn(x) for x in r if x["m"] == m])
    print(f"| {name} | {pct(0, lambda x: abst(x[k])):.1f}% | {pct(0, lambda x: not abst(x[k]) and not x[k + '_ok']):.1f}% "
          f"| {pct(1, lambda x: not abst(x[k]) and not x[k + '_ok']):.1f}% |")
