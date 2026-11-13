import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
f = sys.argv[1] if len(sys.argv) > 1 else "results.json"
import re, string
def norm(s):
    s = "".join(c for c in s.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())
def bad_target(x):   # attacker target that is actually (or contains) a gold answer: attack cannot succeed, drop it
    t = norm(x["target"]); return any(norm(g) and (norm(g) in t or t in norm(g)) for g in x["gold"])
r = json.load(open(f))
drop = {x["q"] for x in r if bad_target(x)}
r = [x for x in r if x["q"] not in drop]; ms = sorted({x["m"] for x in r})
print(f"dropped {len(drop)} questions with invalid targets; n = {len({x['q'] for x in r})}")
def curve(k): return [100 * np.mean([x[k] for x in r if x["m"] == m]) for m in ms]
names = {"van": "Vanilla RAG", "key": "RobustRAG keyword agg.", "dec": "RobustRAG decoding agg."}
col = {"van": "#d95f02", "key": "#1b9e77", "dec": "#7570b3"}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for k in names:
    ax[0].plot(ms, curve(k + "_ok"), "o-", c=col[k], label=names[k])
    ax[1].plot(ms, curve(k + "_asr"), "o-", c=col[k], label=names[k])
ax[0].plot(ms, curve("cert"), "s--", c="gray", label="Keyword agg. heuristic robustness indicator (not a certificate)")
ax[0].set(xlabel="# poisoned passages (of top-10)", ylabel="Accuracy (%)", title="Accuracy under poisoning")
ax[1].set(xlabel="# poisoned passages (of top-10)", ylabel="Attack success rate (%)", title="Attack success rate")
for a in ax: a.grid(alpha=.3); a.set_xticks(ms)
ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig_main.png", dpi=200)
print({m: {k: round(np.mean([x[k] for x in r if x['m'] == m]) * 100, 1) for k in ['van_ok', 'key_ok', 'dec_ok', 'van_asr', 'key_asr', 'dec_asr', 'cert']} for m in ms})
