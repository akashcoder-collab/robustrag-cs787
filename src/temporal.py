"""Temporal conflicts + bounded poisoning: plain voting vs naive recency vs thresholded recency (ours).
Fully synthetic data (invented orgs/people so the LLM cannot rely on memory). Run AFTER robustrag.py (shares the GPU)."""
import re, json, random, sys
from collections import Counter, defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
import robustrag as R

FIRST = "James Maria Robert Linda Peter Susan Daniel Helen Victor Nora Oscar Irene Felix Clara Hugo Alma".split()
LAST = "Whitfield Okafor Lindqvist Moreau Castellano Brandt Nakamura Petrov Haldane Ferreira Voss Adeyemi Kowalski Dunmore Reyes Tanaka".split()
CITY = "Marlow Ashford Brenton Calder Dunmere Eastvale Fairhaven Glenross".split()
THING = "Hawks Wolves Comets Falcons Titans Rovers Pilots Rangers".split()
ROLE = [("head coach", "the {o}"), ("mayor", "the city of {c}"), ("chief executive", "{c} Dynamics"), ("principal", "{c} Academy")]
MON = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

def fmt_date(y, m, d, rng):
    s = rng.choice([0, 1, 2])
    return [f"{MON[m-1]} {d}, {y}", f"{d} {MON[m-1][:3]} {y}", f"{y}-{m:02d}-{d:02d}"][s]

def parse_date(text):
    """temporal claim normalization -> (y, m, d) tuple; handles 3 formats"""
    mm = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if mm: return tuple(int(x) for x in mm.groups())
    mm = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}), (\d{4})", text)
    if mm: return (int(mm.group(3)), MON.index(mm.group(1)) + 1, int(mm.group(2)))
    mm = re.search(r"(\d{1,2}) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{4})", text)
    if mm: return (int(mm.group(3)), [x[:3] for x in MON].index(mm.group(2)) + 1, int(mm.group(1)))
    return None

def build(n, seed=0):
    rng = random.Random(seed); out = []
    for i in range(n):
        city, thing = rng.choice(CITY), rng.choice(THING)
        role, tmpl = rng.choice(ROLE)
        subj = tmpl.format(o=f"{city} {thing}", c=city)
        names = rng.sample([f"{a} {b}" for a in FIRST for b in LAST], 3)
        old, new, fake = names
        q = f"Who is the current {role} of {subj}?"
        def stmt(name, y_lo, y_hi):
            d = fmt_date(rng.randint(y_lo, y_hi), rng.randint(1, 12), rng.randint(1, 28), rng)
            return f"As of {d}, {name} is the {role} of {subj}. Officials confirmed the appointment."
        n_old, n_new = rng.choice([(4, 3), (3, 4)])
        old_p = [stmt(old, 2019, 2021) for _ in range(n_old)]
        new_p = [stmt(new, 2023, 2024) for _ in range(n_new)]
        fake_p = [stmt(fake, 2025, 2025) for _ in range(3)]     # attack A: new fabricated name, newest date
        forge_p = [stmt(old, 2025, 2025) for _ in range(3)]     # attack B: stale (real) answer re-dated to newest date
        irr = [f"As of {fmt_date(2022, 5, 9, rng)}, {rng.choice(FIRST)} {rng.choice(LAST)} is the {r} of {rng.choice(CITY)} {rng.choice(THING)}." for r, _ in ROLE for _ in range(2)]
        out.append(dict(q=q, old=old, new=new, fake=fake, old_p=old_p, new_p=new_p, fake_p=fake_p, forge_p=forge_p, irr=irr))
    return out

def decide(claims, method, B):
    """claims: list of (answer, date); B = assumed adversary budget; returns chosen answer"""
    if not claims: return None
    cnt = Counter(a for a, _ in claims)
    if method == "vote":
        return cnt.most_common(1)[0][0]
    if method == "recency":                         # naive: latest-dated claim wins
        dated = [(d, a) for a, d in claims if d]
        return max(dated)[1] if dated else cnt.most_common(1)[0][0]
    elig = {a for a, c in cnt.items() if c >= B + 1}  # only answers with >= B+1 supporting passages
    if method == "thresh_max":                      # first version: newest date of ANY supporter (forgeable)
        dated = [(d, a) for a, d in claims if d and a in elig]
        return max(dated)[1] if dated else cnt.most_common(1)[0][0]
    # thresholded recency (final): an answer's date is the (B+1)-th newest date among its supporters,
    # so B forged dates cannot move it later than some clean supporter's date
    best = []
    for a in elig:
        ds = sorted((d for b, d in claims if b == a and d), reverse=True)
        if len(ds) > B: best.append((ds[B], a))
    return max(best)[1] if best else cnt.most_common(1)[0][0]

METHODS = ("vote", "recency", "thresh_max", "thresh")

def main(N=40, MS=(0, 1, 2, 3), BUDGET=2):
    data = build(N); rag = R.RAG(); res = []
    for it in data:
        for atk in ("fake", "forge"):
            for m in MS:
                ctx = (it[atk + "_p"][:m] + it["new_p"] + it["old_p"] + it["irr"])[:10]
                need = [c for c in set(ctx) if c not in rag.cache]
                if need:
                    outs = rag.gen([rag.prompt(it["q"], c) for c in need], 12)
                    for c, o in zip(need, outs): rag.cache[c] = o
                claims = []
                for c in ctx:
                    a = rag.cache[c].strip().rstrip(".")
                    if "unknown" in a.lower() or not a: continue
                    claims.append((R.norm(a), parse_date(c)))
                row = dict(atk=atk, m=m)
                target = it["fake"] if atk == "fake" else it["old"]   # attacker's desired answer
                for meth in METHODS:
                    ans = decide(claims, meth, BUDGET)
                    row[meth + "_ok"] = bool(ans) and R.norm(it["new"]) in ans
                    row[meth + "_asr"] = bool(ans) and R.norm(target) in ans
                res.append(row)
    json.dump(res, open("results_temporal.json", "w"), indent=1)
    ms = sorted({r["m"] for r in res})
    cv = lambda atk, k: [100 * np.mean([r[k] for r in res if r["m"] == m and r["atk"] == atk]) for m in ms]
    names = {"vote": "Majority vote", "recency": "Naive recency", "thresh_max": "Thresholded, newest supporter date (v1)",
             "thresh": f"Thresholded, (B+1)-th newest date (ours, B={BUDGET})"}
    col = {"vote": "#d95f02", "recency": "#7570b3", "thresh_max": "#e7298a", "thresh": "#1b9e77"}
    sty = {"vote": "o-", "recency": "s-", "thresh_max": "^--", "thresh": "o-"}
    titles = {"fake": "Attack A: fabricated answer, newest date", "forge": "Attack B: stale answer re-dated as newest"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for i, atk in enumerate(("fake", "forge")):
        for k in names:
            ax[i].plot(ms, cv(atk, k + "_ok"), sty[k], c=col[k], label=names[k], ms=5)
        ax[i].set(xlabel="# adversarial passages (of top-10)", title=titles[atk]); ax[i].grid(alpha=.3); ax[i].set_xticks(ms)
    ax[0].set_ylabel("Accuracy on current answer (%)"); ax[0].set_ylim(-5, 105)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False)
    plt.tight_layout(rect=(0, 0.13, 1, 1)); plt.savefig("fig_temporal.png", dpi=200)
    for atk in ("fake", "forge"):
        for m in ms:
            print(atk, m, {k: round(cv(atk, k)[ms.index(m)], 1) for k in [x + s for s in ("_ok", "_asr") for x in METHODS]})

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
