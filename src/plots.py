import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
f = sys.argv[1] if len(sys.argv) > 1 else "results.json"
kwf = f.replace("results", "results_keyword", 1)
figf = f.replace("results", "fig_main", 1).replace(".json", ".png")
import re, string
def norm(s):
    s = "".join(c for c in s.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())
def bad_target(x):   # attacker target that is actually (or contains) a gold answer: attack cannot succeed, drop it
    t = norm(x["target"]); return any(norm(g) and (norm(g) in t or t in norm(g)) for g in x["gold"])
r = json.load(open(f))
drop = {x["q"] for x in r if bad_target(x)}
r = [x for x in r if x["q"] not in drop]; ms = sorted({x["m"] for x in r})
import os
if os.path.exists(kwf):   # paper-faithful keyword aggregation re-run; keep the first version as key1
    kw = {(x["q"], x["m"]): x for x in json.load(open(kwf))}
    r = [x for x in r if (x["q"], x["m"]) in kw]
    for x in r:
        y = kw[(x["q"], x["m"])]
        x["key1_ok"], x["key1_asr"], x["cert1"] = x["key_ok"], x["key_asr"], x["cert"]
        x["key_ok"], x["key_asr"], x["cert"], x["key"] = y["key_ok"], y["key_asr"], y["cert"], y["key"]
    print("merged paper-faithful keyword results; n =", len({x["q"] for x in r}))
print(f"dropped {len(drop)} questions with invalid targets; n = {len({x['q'] for x in r})}")
def curve(k): return [100 * np.mean([x[k] for x in r if x["m"] == m]) for m in ms]
names = {"van": "Vanilla RAG", "key": "RobustRAG keyword agg.", "dec": "RobustRAG decoding agg."}
col = {"van": "#d95f02", "key": "#1b9e77", "dec": "#7570b3"}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for k in names:
    ax[0].plot(ms, curve(k + "_ok"), "o-", c=col[k], label=names[k])
    ax[1].plot(ms, curve(k + "_asr"), "o-", c=col[k], label=names[k])
if "key1_ok" in r[0]:
    ax[0].plot(ms, curve("key1_ok"), "^:", c="#1b9e77", alpha=.6, label="Keyword agg. v1 (fixed threshold 3)")
    ax[1].plot(ms, curve("key1_asr"), "^:", c="#1b9e77", alpha=.6, label="Keyword agg. v1 (fixed threshold 3)")
ax[0].plot(ms, curve("cert"), "s--", c="gray", label="Keyword agg. heuristic robustness indicator (not a certificate)")
ax[0].set(xlabel="# poisoned passages (of top-10)", ylabel="Accuracy (%)", title="Accuracy under poisoning")
ax[1].set(xlabel="# poisoned passages (of top-10)", ylabel="Attack success rate (%)", title="Attack success rate")
for a in ax: a.grid(alpha=.3); a.set_xticks(ms)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
plt.tight_layout(rect=(0, 0.14, 1, 1)); plt.savefig(figf, dpi=200)
ks = ['van_ok', 'key_ok', 'dec_ok', 'cert', 'van_asr', 'key_asr', 'dec_asr'] + (['key1_ok', 'key1_asr'] if "key1_ok" in r[0] else [])
for m in ms: print(m, {k: round(np.mean([x[k] for x in r if x['m'] == m]) * 100, 1) for k in ks})
