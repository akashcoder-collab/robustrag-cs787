"""Certified robustness of keyword aggregation: Algorithm 3 of Xiang et al. (2024), group size 1.
For each question and attacker budget k' in {1, 2}: take the top k-k' benign passages (an injection attacker can only
push out the bottom k'), get their isolated responses, and enumerate every keyword set the attacker could produce:
for each k'_eff in 0..k', mu' = min(alpha*(n+k'_eff), beta); W_A = {count >= mu'} always kept, W_C = {mu'-k'_eff <= count < mu'}
attacker-controlled; every subset W_C' gives a response LLM(i2, q, sorted(W_A | W_C')). Certified iff all responses are correct.
Same questions, evidence passages, prompts and keyword extraction as robustrag.py / keyword_rerun.py.
Soundness fix: Algorithm 3 only enumerates keywords seen in benign responses, which is correct only if mu' > k'_eff;
otherwise a keyword that only the attacker writes is retained, so we return "not certified" (see report Section 4).
Deviation: we cap |W_C| at 10 (paper: 15); above the cap the question counts as NOT certified (conservative, still sound).
Writes certify{RR_TAG}.json."""
import sys, json, time, itertools
from collections import Counter
import torch
sys.path.insert(0, "src")
import robustrag as R
import keyword_rerun as KW

CAP = 10

def certify(rag, q, gold, benign, kp):
    resps = rag.indiv(q, benign, [("cert", q, j) for j in range(len(benign))])
    C, n = Counter(), 0
    for r in resps:
        if "unknown" in r.lower() or not r.strip(): continue
        n += 1; C.update(KW.keyphrases(r))
    seen, wc_max = set(), 0
    for ke in range(kp + 1):
        mu = min(KW.ALPHA * (n + ke), KW.BETA)
        if mu <= ke: return False, wc_max, n   # attacker alone reaches the threshold: any injected keyword survives
        WA = {w for w, c in C.items() if c >= mu}
        WC = sorted(w for w, c in C.items() if mu - ke <= c < mu)
        wc_max = max(wc_max, len(WC))
        if len(WC) > CAP: return False, wc_max, n
        for r in range(len(WC) + 1):
            for sub in itertools.combinations(WC, r):
                W = frozenset(WA | set(sub))
                if W: seen.add(W)
                else:   # empty retained set: RobustRAG outputs "I don't know" -> not correct
                    return False, wc_max, n
    seen = sorted(seen, key=len)
    for i in range(0, len(seen), 64):   # early exit at the first wrong response
        outs = rag.gen([KW.final_prompt(rag, q, W) for W in seen[i:i + 64]], 20)
        if not all(R.contains(rag.ans(o), gold) for o in outs): return False, wc_max, n
    return True, wc_max, n

def main(N=50, KPS=(1, 2)):
    items, passages = R.load_data(6 * N)
    rag = R.RAG()
    pe = rag.emb.encode(passages, batch_size=128, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    items = [(x["q"], x["gold"]) for x in json.load(open("results.json")) if x["m"] == 0][:N]
    res, t0 = [], time.time()
    for qi, (q, gold) in enumerate(items):
        qe = rag.emb.encode([q], normalize_embeddings=True, convert_to_tensor=True)
        top = torch.topk(pe @ qe[0], R.K).indices.tolist()
        gi = next(i for i in top if any(R.norm(a) in R.norm(passages[i]) for a in gold))
        others = [passages[i] for i in top if i != gi]
        clean = ([passages[gi]] + R.make_support(rag, passages[gi]) + others)[:R.K]
        for kp in KPS:
            ok, wc, n = certify(rag, q, gold, clean[:R.K - kp], kp)
            res.append(dict(q=q, kp=kp, certified=ok, wc_max=wc, n=n))
        if qi % 10 == 9: print(f"{qi+1}/{len(items)} {time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(f"certify{R.TAG}.json", "w"), indent=1)
    for kp in KPS:
        r = [x for x in res if x["kp"] == kp]
        print(f"k'={kp}: certified {sum(x['certified'] for x in r)}/{len(r)}, |W_C|>{CAP}: {sum(x['wc_max'] > CAP for x in r)}")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
