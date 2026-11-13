# RobustRAG Recreation and Thresholded Recency for Temporal Conflicts under Bounded Poisoning

CS787, Group 8. Authors: Akash Tiwari (241090403) and Vaibhav Soni (241090419), IIT Kanpur ([@vbsoni-hpc](https://github.com/vbsoni-hpc)). Small-scale recreation of *Certifiably Robust RAG against Retrieval Corruption* (Xiang et al., 2024, arXiv:2405.15556), plus one extension.

## 1. Summary
1. **Recreation.** We implement vanilla RAG and RobustRAG's isolate-then-aggregate defense (keyword aggregation and decoding aggregation) and attack them with PoisonedRAG-style injected passages. Vanilla RAG collapses under poisoning: its attack success rate (ASR) goes from 2% to 85% with one poisoned passage out of ten, and to 91% with two. Both RobustRAG variants resist far better at one poisoned passage (keyword 11%, decoding 13% ASR). This matches the paper's empirical finding (vanilla above 90% ASR, RobustRAG at most 15%). Our first keyword-aggregation version (fixed threshold) lost most of its clean accuracy (15%); implementing the paper's adaptive threshold and final prompt raises it to 35%, at the cost of higher ASR once the attacker controls 2 or more passages. Clean accuracy is still below vanilla (78%), unlike the paper's near-zero drop.
2. **Extension: thresholded recency.** When the correct answer changes over time, majority voting returns the stale majority, and naive "newest date wins" is broken by one passage with a fake newer date. We require an answer to have at least B+1 supporting passages, and we date each answer by the (B+1)-th newest date among its supporters, so that B forged dates cannot make an answer look newer. Our first version used the newest supporter date, and a review found it breaks with one forged date on the stale answer. Against both attacks the final rule keeps 100% accuracy up to the assumed budget B=2 and fails at 3, as the argument predicts. The data are synthetic and the test is a controlled sanity check, not a benchmark.

## 2. Setup (what is and is not faithful to the paper)
| Item | Paper | Ours |
|---|---|---|
| LLM | Mistral-7B-Instruct, Llama2-7B-Chat, GPT-3.5-turbo | Qwen2.5-1.5B-Instruct (fp16, 4 GB laptop GPU) |
| Data | RealtimeQA(-MC), Natural Questions, Biography generation; 100 queries each (50 for Bio) | Open-domain SQuAD (validation paragraphs as corpus, about 2k passages); 50 questions, 46 after filtering |
| Retrieval | Google Search, top k=10 | all-MiniLM-L6-v2 dense retriever, top k=10 |
| Attack | PoisonedRAG and prompt injection; mainly k'=1 corrupted passage | PoisonedRAG black-box style: question text + fabricated passage for an LLM-generated wrong target; injected passages take the top ranks and push out the bottom benign passages (as in the paper's threat model); m in {0,1,2,3} |
| Metric | answer correct if reference answer appears in response | same |

## 3. Implementation and deviations from the paper
**Keyword aggregation (paper Algorithm 1).** The paper prompts each isolated passage with "answer the query ..., say 'I don't know' if no relevant information found", counts the n non-abstaining responses, extracts unique keyphrases from each response, keeps keywords with count >= mu = min(alpha·n, beta) (alpha=0.2, beta=3 for short QA), and asks the LLM to "answer the query using provided keywords", with keywords sorted alphabetically.

We ran two versions (`src/keyword_rerun.py` reruns only the keyword part, on exactly the same questions, evidence and poisoned passages):
- **v1 (first attempt):** fixed threshold mu = 3; single words only, with question words removed; the final prompt reused the per-passage prompt (which allows "unknown") and listed keywords by count. 74% of its outputs at m=0 were "unknown", hence 15% clean accuracy.
- **Final (paper-faithful):** mu = min(0.2·n, 3) with n the number of non-abstaining responses; keywords = all informative words plus keyphrases of consecutive informative words (a stop-word list replaces the paper's POS tagger, and we do not lemmatize); final prompt "Answer the query using the provided keywords" with keywords sorted alphabetically. Only 4% of outputs at m=0 are "unknown". On our data n averages 6.9, so mu is usually 1 or 2, much lower than v1's 3. The per-passage prompt says "unknown" where the paper says "I don't know".

Reproducibility note: the first version of `load_data` built each gold-answer list from a Python `set`, whose order changes between runs, so question selection was not reproducible. It is now sorted, and `keyword_rerun.py` reads the exact question list stored in `results.json`.

**Decoding aggregation (paper Algorithm 2).** The paper keeps passages with P("I don't know") < gamma = 0.99, sums their next-token probability vectors, and takes the top token if its margin over the runner-up exceeds eta (eta = 0 for short QA); otherwise it uses the no-retrieval token. **Ours:** we drop a passage if its most likely first token starts "unk" (stricter than gamma=0.99), average the probability vectors (same argmax as summing) and decode greedily; with eta = 0 the no-retrieval fallback only matters for exact ties, which we ignore.

**"Heuristic robustness indicator" (not a certificate).** The paper certifies robustness (Algorithm 3) from the benign top k-k' passages by enumerating every keyword set an attacker could produce. We did not implement this. Our grey curve counts a keyword-aggregation answer as robust if it is correct, mu > m, and a gold keyword has count >= mu + m in the attacked run. This is a necessary-style heuristic, not certified accuracy, and is not comparable to the paper's certifiable accuracy (26-36% on NQ).

**Other deviations, forced by compute:**
- Questions were kept only if the gold answer appears in the clean top-10, and yes/no questions were removed.
- **Synthetic redundant evidence.** SQuAD has one evidence paragraph per question, but isolate-then-aggregate needs several agreeing passages. For each question we add 4 LLM paraphrases of the gold paragraph, giving 5 evidence passages plus 5 other retrieved passages. This favours the defenses; absolute numbers are not comparable to the paper.
- 4 of the 50 generated attack targets were themselves correct answers (e.g. target "Encourage growth" for gold "encourage growth", target "XXI" containing gold "XX"). The attack cannot succeed on these, so we exclude them; all results use the remaining 46 questions.

## 4. Results: recreation (n = 46 questions, k = 10)
Figure: `fig_main.png`.

Accuracy:

| Poisoned passages m | Vanilla | Keyword (final) | Keyword v1 | Decoding | Heuristic indicator (keyword final) |
|---|---|---|---|---|---|
| 0 | 78.3% | 34.8% | 15.2% | 52.2% | 34.8% |
| 1 | 15.2% | 30.4% | 15.2% | 47.8% | 28.3% |
| 2 | 8.7% | 28.3% | 15.2% | 39.1% | 0.0% |
| 3 | 10.9% | 28.3% | 13.0% | 34.8% | 0.0% |

Attack success rate:

| Poisoned passages m | Vanilla | Keyword (final) | Keyword v1 | Decoding |
|---|---|---|---|---|
| 0 | 2.2% | 8.7% | 0.0% | 0.0% |
| 1 | 84.8% | 10.9% | 0.0% | 13.0% |
| 2 | 91.3% | 39.1% | 2.2% | 32.6% |
| 3 | 89.1% | 39.1% | 17.4% | 47.8% |

At m = 0 there is no attack, so a non-zero ASR is a measurement artefact: the attacker's target string happens to appear in the answer (e.g. short targets such as "ten" or "5"). Treat about 2-9% as the noise floor of this metric.

Reading:
- **Consistent with the paper:** vanilla RAG is highly vulnerable (85-91% ASR), and both RobustRAG variants keep ASR low at the paper's main setting of one corrupted passage (keyword 11%, decoding 13%; the paper reports at most 15%).
- **Not consistent with the paper:** the paper reports that RobustRAG barely reduces clean accuracy. Ours drops from 78% (vanilla) to 35% (keyword) and 52% (decoding). Likely causes: a 1.5B model rebuilding answers from keyword lists, no lemmatization or POS tagging, and our partly synthetic evidence.
- **Threshold trade-off (our own observation):** the paper's adaptive threshold mu = min(0.2·n, 3) is usually 1-2 on our data. That more than doubles clean accuracy compared with the fixed mu = 3 (v1), but once the attacker has m >= mu passages its keywords pass the filter, and ASR jumps to 39% at m = 2. The fixed threshold of v1 is far more robust (ASR at most 2% for m <= 2) but far less useful. This matches the paper's statement that robustness is only possible for small k' relative to the number of useful passages, and its focus on k' = 1.
- Decoding aggregation keeps the best accuracy under attack, but its ASR grows with m, as expected from averaging rather than thresholded voting.

## 5. Extension: temporal claim normalization + thresholded recency
**Problem.** Facts change ("who is the current head coach?"). Passages dated in the past were correct for their date, so majority voting can return a stale answer. Naive recency (newest date wins) fixes this but gives the attacker an easy win.

**Method.**
1. *Claim extraction and date parsing:* each isolated passage yields a claim (answer, date). The answer comes from the LLM; the date is parsed from the passage in three formats (`March 4, 2023`, `4 Mar 2023`, `2023-03-04`). The generator writes exactly these formats, so the parser's success here is by construction.
2. *Eligibility:* an answer is eligible only if at least B+1 passages support it (B = assumed adversary budget; B = 2).
3. *Robust dating:* each eligible answer is dated by the (B+1)-th newest date among its supporters; the newest eligible answer wins.

**Guarantee (sketch, not a full proof).** Assume claims are extracted correctly. An attacker with at most B passages (a) cannot make a fabricated answer eligible, since it needs B+1 supporters, and (b) cannot make an eligible answer look newer than its newest clean supporting date, since at most B of its dates are forged, so the (B+1)-th newest is a clean date. Hence the output is an answer backed by clean passages, chosen by clean dates. It does *not* guarantee the current answer is chosen: if the current answer has at most B clean supporters in the retrieved set (for example, very recent news), the rule falls back to an older answer. That case is not tested here.

**Why robust dating matters.** Our first version (v1) dated an answer by its newest supporter. An attacker can then re-date the stale answer (which already has enough clean supporters) with one fake 2025 passage, and v1 picks it. Attack B below tests exactly this.

**Data** (synthetic, 40 questions, invented organizations and people so the LLM cannot rely on memory). Each question has 4+3 or 3+4 old/new supporting passages (old: 2019-2021, new: 2023-2024), the rest irrelevant. The new-answer passages are placed above the old ones, so injected passages push out old or irrelevant passages, never new ones (favourable to all recency methods). Attack A adds m passages claiming a third, fabricated name dated 2025. Attack B adds m passages claiming the stale answer dated 2025. Figure: `fig_temporal.png` (lines that coincide hide each other).

Accuracy on the current answer:

| Attack | m | Majority vote | Naive recency | Thresholded v1 (newest date) | Thresholded, robust date (ours) |
|---|---|---|---|---|---|
| A: fabricated name | 0 | 45% | 100% | 100% | 100% |
| | 1 | 45% | 0% | 100% | 100% |
| | 2 | 45% | 0% | 100% | 100% |
| | 3 (> B) | 45% | 0% | 0% | 0% |
| B: stale answer re-dated | 0 | 45% | 100% | 100% | 100% |
| | 1 | 0% | 0% | 0% | 100% |
| | 2 | 0% | 0% | 0% | 100% |
| | 3 (> B) | 0% | 0% | 0% | 0% |

Reading: only the robust-date rule resists both attacks, and it breaks exactly when the attacker exceeds the budget (m = 3 = B+1). Majority vote picks the stale answer whenever the old answer has more passages (55% of questions here). The numbers are all-or-nothing because the passages are templated and the LLM extracts names essentially perfectly; this shows the decision rule behaves as argued, not how it performs on real data.

## 6. Limitations
- Small samples (46 and 40 questions), one seed, no confidence intervals.
- A 1.5B local LLM instead of the paper's 7B models.
- Synthetic redundant evidence in the recreation; fully synthetic, templated temporal data.
- No real certification; the grey curve is a heuristic.
- Keyword extraction approximates the paper's (stop-word list instead of POS tags, no lemmatization); clean accuracy remains well below vanilla.
- The main results come from one run whose question list is stored in `results.json`; rerunning `robustrag.py` after the determinism fix selects a different (but reproducible) set of questions.
- The temporal guarantee assumes correct claim extraction, and the setup never tests a current answer with few supporters.

## 7. Reproduce
```
pip install torch transformers datasets sentence-transformers matplotlib
python src/robustrag.py 50 results.json   # about 75 min on an RTX 3050 4GB
python src/keyword_rerun.py 50            # paper-faithful keyword aggregation, about 10 min
python src/plots.py results.json          # merges keyword results, drops invalid targets, writes fig_main.png
python src/temporal.py 40                 # fig_temporal.png, results_temporal.json
```
