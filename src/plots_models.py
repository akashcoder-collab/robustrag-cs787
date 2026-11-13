"""fig_models.png: accuracy and attack success rate vs number of poisoned passages for the three models (one column each).
Qwen runs: per-question files (same merging/filtering as plots.py). Mistral-7B: aggregates of its complete run
(results_mistral7b_run1_summary.json; its per-question files were overwritten by an interrupted rerun)."""
import json, re, string, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def norm(s):
    s = "".join(c for c in s.lower() if c not in string.punctuation); return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())
def curves(tag):
    r = json.load(open(f"results{tag}.json"))
    bad = {x["q"] for x in r if any(norm(g) and (norm(g) in norm(x["target"]) or norm(x["target"]) in norm(g)) for g in x["gold"])}
    kw = {(x["q"], x["m"]): x for x in json.load(open(f"results_keyword{tag}.json"))}
    r = [x for x in r if x["q"] not in bad and (x["q"], x["m"]) in kw]
    for x in r:
        y = kw[(x["q"], x["m"])]; x["key1_ok"], x["key1_asr"] = x["key_ok"], x["key_asr"]; x["key_ok"], x["key_asr"] = y["key_ok"], y["key_asr"]
    ms = sorted({x["m"] for x in r})
    out = {k: [100 * np.mean([x[k] for x in r if x["m"] == m]) for m in ms] for k in
           [a + b for a in ("van", "key", "key1", "dec") for b in ("_ok", "_asr")]}
    return out, len({x["q"] for x in r})
cols = [("Qwen2.5-1.5B", *curves("")), ("Qwen2.5-7B", *curves("_qwen7b"))]
mi = json.load(open("results_mistral7b_run1_summary.json")); cols.append(("Mistral-7B", mi, mi["n"]))
names = {"van": "Vanilla RAG", "key": "Keyword agg. (paper threshold)", "key1": "Keyword agg. v1 (fixed threshold 3)", "dec": "Decoding agg."}
sty = {"van": ("#d95f02", "o-"), "key": ("#1b9e77", "o-"), "key1": ("#1b9e77", "^:"), "dec": ("#7570b3", "s-")}
fig, ax = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True, sharey="row")
ms = [0, 1, 2, 3]
for j, (title, c, n) in enumerate(cols):
    for k, (col, st) in sty.items():
        ax[0, j].plot(ms, c[k + "_ok"], st, c=col, ms=4, label=names[k])
        ax[1, j].plot(ms, c[k + "_asr"], st, c=col, ms=4, label=names[k])
    ax[0, j].set_title(f"{title} (n = {n})", fontsize=10); ax[1, j].set_xlabel("# poisoned passages (of top-10)")
    for i in (0, 1): ax[i, j].grid(alpha=.3); ax[i, j].set_xticks(ms); ax[i, j].set_ylim(-3, 100)
ax[0, 0].set_ylabel("Accuracy (%)"); ax[1, 0].set_ylabel("Attack success rate (%)")
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.savefig("fig_models.png", dpi=200)
