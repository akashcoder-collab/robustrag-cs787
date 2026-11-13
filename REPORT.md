# RobustRAG: Generation Against Temporal Conflicts, Adversarial Poisoning and Hallucination

CS787, Group 8. Authors: Akash Tiwari (241090403) and Vaibhav Soni (241090419), IIT Kanpur ([@vbsoni-hpc](https://github.com/vbsoni-hpc)). Base paper: *Certifiably Robust RAG against Retrieval Corruption* (Xiang et al., 2024, arXiv:2405.15556). Code and results: [github.com/akashcoder-collab/robustrag-cs787](https://github.com/akashcoder-collab/robustrag-cs787).

## 1. Abstract
Retrieval-augmented generation (RAG) can be misled by its own context: an attacker can inject passages that push a wrong answer (**adversarial poisoning**), passages correct at different times can disagree so that an outdated majority wins (**temporal conflicts**), and the model can answer without support from the evidence (**hallucination**). We reimplement RobustRAG's isolate-then-aggregate defense (keyword and decoding aggregation) and its certification procedure, and evaluate them against PoisonedRAG-style attacks with three LLMs: Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct and the paper's main model, Mistral-7B-Instruct-v0.2.

At 7B our results agree with the paper's Natural Questions numbers: one poisoned passage out of ten hijacks vanilla RAG on 31-41% of questions (paper: 37%), and RobustRAG lowers this to 4-6% (paper: 7-12%). The 1.5B model is hijacked far more often (85%) and the defenses cost it much more clean accuracy, so model scale matters for both vulnerability and defense cost. Implementing the paper's keyword certification (Algorithm 3), we found that it over-certifies when few passages answer; with a fix, Qwen2.5-7B is certified on 35% of questions against one corrupted passage (paper: 26%), and we show that with the paper's parameters no question can be certified against two. For hallucination, isolation alone does not reduce wrong answers; abstaining does, at a large cost in correct answers. Finally, we propose **thresholded recency** for temporal conflicts: answers need at least B+1 supporting passages and are dated by their (B+1)-th newest supporter, which resists date forgery by up to B passages on a synthetic benchmark, identically for all three models.

## 2. Introduction and threat model
RAG retrieves the top-k passages for a query and conditions the LLM on them. PoisonedRAG (Zou et al., 2024) showed that injecting a few passages that contain the question and a fabricated answer makes the LLM repeat that answer. RobustRAG's key idea is **isolate-then-aggregate**: the LLM answers from each passage separately, and the k answers are combined by a rule that a few malicious answers cannot dominate. Because the rule is simple, its worst-case behaviour can be computed (certified).

**Threat model (as in the paper).** The attacker injects k' passages that rank above all benign passages and thereby push the bottom k' benign passages out of the top-k (k = 10). The attacker knows the defense and controls the injected text completely. We measure accuracy (the reference answer appears in the response) and attack success rate (ASR: the attacker's target appears and the reference does not), for k' = m in {0, 1, 2, 3}.

## 3. RobustRAG and our implementation
**Keyword aggregation (paper Algorithm 1).** Each passage is prompted separately ("answer ...; if the passage does not contain the answer, reply 'unknown'", the paper's "I don't know"). Of the n non-abstaining responses, unique keywords and keyphrases are counted; keywords with count >= mu = min(alpha·n, beta) (alpha = 0.2, beta = 3, the paper's short-QA setting) are kept and given, sorted alphabetically, to the LLM with "answer the query using the provided keywords". We approximate the paper's POS-based keyphrase extraction with a stop-word list (informative words plus runs of consecutive informative words) and do not lemmatize. We also keep our first version **v1** (fixed mu = 3, single words) as a reference point for the threshold trade-off.

**Decoding aggregation (paper Algorithm 2).** At each decoding step the next-token distributions of the isolated passages are combined and the top token is emitted. The paper drops passages with P("I don't know") >= gamma = 0.99; we drop a passage if its most likely first token starts "unk" (stricter), and average instead of sum (same argmax). With eta = 0 the paper's no-retrieval fallback only matters for exact ties, which we ignore.

**Certification (paper Algorithm 3, `src/certify.py`).** Because injected passages only push out the bottom k' benign ones, the top k-k' benign passages are fixed. From their responses (n non-abstaining, keyword counts C), for each number k'_eff in {0..k'} of non-abstaining malicious responses, mu' = min(alpha·(n+k'_eff), beta); keywords with count >= mu' are always kept (W_A), keywords with count in [mu'-k'_eff, mu') are under the attacker's control (W_C), and each subset of W_C yields one possible keyword set and one LLM answer. A question is certified if every possible answer is correct. Like the paper we give up when W_C is too large; our cap is 10 keywords (paper: 15), which is conservative.

**A soundness gap in Algorithm 3.** The algorithm only enumerates keywords that occur in benign responses. If mu' <= k'_eff, the attacker's responses alone reach the threshold, so a keyword no benign passage contains (for example the attacker's target) is also kept, and the enumeration misses it. The paper's illustration assumes this away (mu' = 2 > k'_eff = 1). We detected the gap empirically: before the fix, 2 questions at k' = 1 and 6 at k' = 2 were "certified" yet answered with the attacker's target in the real attack run (e.g. "Los Angeles" instead of "Fresno", with n = 5 benign answers and mu' = 1.4 at k'_eff = 2). We therefore declare a question not certified whenever mu' <= k'_eff for some k'_eff. After the fix, no certified question is answered wrongly under attack. A consequence is a simple necessary condition: certification against k' requires alpha·(n + k') > k', i.e. n > 4 non-abstaining benign answers for k' = 1 and n > 8 for k' = 2. With k = 10 only k - k' = 8 benign passages remain at k' = 2, so **with alpha = 0.2 no question can ever be certified against two corrupted passages**.

## 4. Experimental setup
| Item | Paper | Ours |
|---|---|---|
| LLMs | Mistral-7B-Instruct, Llama2-7B-Chat, GPT-3.5 | Qwen2.5-1.5B-Instruct (fp16); Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.2 (bf16) |
| Data | RealtimeQA, NQ (100 queries each), Biography | Open-domain SQuAD (about 2k validation paragraphs as corpus); the same 50 questions for every model, 46-49 after filtering |
| Retrieval | Google Search, top-10 | all-MiniLM-L6-v2 dense retriever, top-10 |
| Attack | PoisonedRAG, prompt injection; mainly k' = 1 | PoisonedRAG black-box style (question + fabricated passages for an LLM-written wrong target); m in {0..3} |
| Hardware | not a constraint | 4 GB RTX 3050 laptop GPU (1.5B); 96 GB RTX PRO 6000 (7B) |

**Why not the paper's exact setup.** Google Search results need a paid API and change over time, and RealtimeQA and the biography task depend on live search or paid GPT judges; we use a fixed, free, reproducible corpus. Llama2-7B-Chat is gated on Hugging Face and was not used. Our first runs had to fit a 4 GB GPU, hence the 1.5B model; the same code was later run at 7B.

**Deviations that favour or affect the defenses.** (1) Questions are kept only if the gold answer is in the clean top-10, and yes/no questions are removed. (2) SQuAD has one evidence paragraph per question, but isolate-then-aggregate needs several agreeing passages, so we add four LLM paraphrases of the gold paragraph (5 evidence passages + 5 other retrieved passages). This favours the defenses. (3) Questions whose attack target contains the gold answer (e.g. target "XXI" for gold "XX") cannot be attacked and are dropped: 4, 1 and 2 questions for the three models. (4) Our metric is substring match, as in the paper, so a response naming several candidates can count as correct.

**Adapting Mistral-7B.** With our original prompt Mistral never answered "unknown" (it writes "The passage does not provide ..."), appended explanations, and produced attack targets such as "Sacramento International Airport\\n\\nExplanation:", which can never appear in an answer, so the measured ASR would have been zero whatever the defense. For Mistral only, we use a stricter instruction ("reply with ONLY the answer ... If the passage does not contain the answer, reply exactly: unknown"), keep only the first line of each answer up to the first sentence end or parenthesis, and allow decoding aggregation 20 instead of 10 tokens, because Mistral splits numbers into digits and answers in sentences (with 10 tokens "1227" came out as "Genghis Khan died in 122"). Qwen2.5-7B uses exactly the 1.5B prompt and settings, so the 1.5B-to-7B comparison changes only model size. The complete Mistral run (n = 48) used the 10-token limit; a rerun with 20 tokens was stopped by a GPU hardware failure after 30 questions (28 after filtering) and is reported separately. The rerun overwrote the per-question outputs of the complete run, so for the n = 48 Mistral rows only the aggregate numbers are preserved (`results_mistral7b_run1_summary.json`); all per-question analyses of Mistral (Sections 5.3 and 5.4) use the 28-question rerun.

## 5. Results: adversarial poisoning
Figure: `fig_models.png` shows accuracy and ASR for all models; the table gives the numbers.

| Model | Method | m = 0 | m = 1 | m = 2 | m = 3 |
|---|---|---|---|---|---|
| Qwen2.5-1.5B (n = 46) | Vanilla | 78.3 / 2.2 | 15.2 / 84.8 | 8.7 / 91.3 | 10.9 / 89.1 |
| | Keyword | 34.8 / 8.7 | 30.4 / 10.9 | 28.3 / 39.1 | 28.3 / 39.1 |
| | Keyword v1 | 15.2 / 0.0 | 15.2 / 0.0 | 15.2 / 2.2 | 13.0 / 17.4 |
| | Decoding | 52.2 / 0.0 | 47.8 / 13.0 | 39.1 / 32.6 | 34.8 / 47.8 |
| Qwen2.5-7B (n = 49) | Vanilla | 75.5 / 0.0 | 51.0 / 40.8 | 40.8 / 46.9 | 34.7 / 61.2 |
| | Keyword | 59.2 / 2.0 | 53.1 / 6.1 | 42.9 / 26.5 | 42.9 / 28.6 |
| | Keyword v1 | 6.1 / 0.0 | 6.1 / 0.0 | 6.1 / 0.0 | 4.1 / 0.0 |
| | Decoding | 65.3 / 0.0 | 65.3 / 6.1 | 59.2 / 18.4 | 40.8 / 34.7 |
| Mistral-7B (n = 48) | Vanilla | 64.6 / 2.1 | 50.0 / 31.2 | 47.9 / 33.3 | 43.8 / 43.8 |
| | Keyword | 45.8 / 4.2 | 39.6 / 6.2 | 33.3 / 18.8 | 37.5 / 22.9 |
| | Keyword v1 | 16.7 / 6.2 | 16.7 / 6.2 | 18.8 / 6.2 | 14.6 / 8.3 |
| | Decoding (10 tok.) | 27.1 / 4.2 | 35.4 / 4.2 | 37.5 / 4.2 | 39.6 / 14.6 |
| Mistral-7B (n = 28) | Decoding (20 tok.) | 39.3 / 3.6 | 50.0 / 3.6 | 53.6 / 3.6 | 57.1 / 10.7 |

Each cell is accuracy / ASR in %. At m = 0 there is no attack, so a non-zero ASR means a short target (e.g. "five") happens to occur in the answer; about 2-9% is the noise floor of the metric. With 46-49 questions the 95% Wilson interval of a proportion is about ±13 points (e.g. Qwen2.5-7B vanilla ASR at m = 1: 40.8%, interval 28-55%), so differences of a few points are not meaningful.

### 5.1 Comparison with the paper (k = 10, one corrupted passage)
The paper's closest setting is Natural Questions with the poisoning attack (its Tables I and II, Mistral-7B).

| | Paper, NQ | Mistral-7B | Qwen2.5-7B | Qwen2.5-1.5B |
|---|---|---|---|---|
| Vanilla: clean acc. / ASR | 61.0 / 37.0 | 64.6 / 31.2 | 75.5 / 40.8 | 78.3 / 84.8 |
| Keyword: clean acc. | 61.0 | 45.8 | 59.2 | 34.8 |
| Keyword: acc. / ASR under attack | 64.0 / 12.0 | 39.6 / 6.2 | 53.1 / 6.1 | 30.4 / 10.9 |
| Decoding: acc. / ASR under attack | 63.0 / 7.0 | 35.4 / 4.2 (50.0 / 3.6 with 20 tok.) | 65.3 / 6.1 | 47.8 / 13.0 |
| Keyword: certified acc. | 26.0 | not run | 34.7 | not run |

- **Reproduced:** vanilla RAG is easily hijacked, and both RobustRAG variants keep ASR low at k' = 1. At 7B the ASR values are within a few points of the paper's, well inside our confidence intervals.
- **Partly reproduced:** the paper finds that RobustRAG costs almost no clean accuracy. At 1.5B it costs a lot (keyword -44, decoding -26 points); with Qwen2.5-7B the cost shrinks to -16 and -10 points. Mistral's keyword cost (-19) is similar; its decoding result is limited by the effect discussed in 5.3.
- **Certified accuracy** (Section 5.2) is in the paper's range once the soundness fix is applied.

### 5.2 Certified robustness (keyword aggregation, Qwen2.5-7B, n = 49)
| k' | Certified | Fails mu' > k'_eff (too few answers) | W_C too large (> 10) | Some possible answer wrong |
|---|---|---|---|---|
| 1 | 17 (34.7%) | 9 | 10 | 13 |
| 2 | 0 (0.0%) | 49 | 0 | 0 |

At k' = 1, 17 of the 29 questions that keyword aggregation answers correctly without attack are certified. The failures have three distinct causes of similar size: too few passages answer (Qwen2.5-7B abstains often: on average only 5.0 of the 9 benign passages give an answer), too many medium-count keywords to enumerate, or a keyword subset that leads the LLM to a wrong answer. At k' = 2 the necessary condition of Section 3 (n > 8) can never hold, so the result is 0% for any model; certifying more than one corrupted passage needs a larger k or a different alpha. Certification for Mistral-7B and the 1.5B model was started but stopped by the GPU failure; the 1.5B run reports only a heuristic indicator (correct, mu > m, and a gold keyword with count >= mu + m), which gave 28% at m = 1 but is not a certificate.

### 5.3 Analysis
**Model scale changes vulnerability more than defense.** One poisoned passage flips 30 of the 36 questions that 1.5B vanilla RAG answers correctly to the attacker's answer, but only 12 of 37 for Qwen2.5-7B and 6 of 21 for Mistral (on the 28-question rerun). For the defenses the number of such flips is 0-3 at every model size. Across the 46 questions shared by the two Qwen models, 39 are hijacked at 1.5B and 19 at 7B, and 17 of the 19 are also hijacked at 1.5B: the larger model resists the easy cases, but the same questions remain vulnerable. A plausible reading is that the larger model weighs the injected "the answer is X" passage against nine benign passages rather than copying the top-ranked text.

**The threshold trades clean accuracy for robustness.** With the paper's adaptive threshold, mu is usually 1-2 (the number of non-abstaining responses averages 6.9 at 1.5B and 5.1 for Qwen2.5-7B, so mu = 0.2·n rounds to small values). Then two or three poisoned passages are enough to pass the filter, and keyword ASR jumps from about 6-11% at m = 1 to 19-39% at m = 2 for all models. The fixed threshold of v1 (mu = 3) keeps ASR near zero up to m = 2 but leaves too few keywords, so it mostly abstains (61-86% "unknown" answers) and has 6-17% accuracy. This matches the paper's position that robustness is only achievable for k' small relative to the number of useful passages.

**Keyword aggregation fails on distractor majorities.** The most common reason why vanilla RAG is right and keyword aggregation wrong (22, 11 and 7 questions for the three models) is that several retrieved passages share a related but wrong entity. For "Who is largely considered the first United Methodist theologian?" eight of the ten isolated answers (Mistral-7B) say "John Wesley" and only one passage names Albert C. Outler; vanilla RAG picks the specific passage, while every isolate-then-aggregate variant votes for "John Wesley". The defense assumes that the majority of retrieved passages is benign *and relevant*; with dense retrieval on a small corpus the second part often fails. Other failures come from the keyword-to-answer step (e.g. "Testament to Victorian housing style." for "terraces").

**Decoding aggregation with Mistral prefers "Unknown".** Mistral's decoding aggregation has low clean accuracy and gains accuracy as poisoned passages are added. In the outputs, the averaged first-token distribution often ranks "Unknown" first: passages that contain the answer still put noticeable probability on "Unknown", and only passages whose *most likely* token is "unknown" are dropped. Confident poisoned passages dilute this mass, so more gold answers win at higher m, while ASR stays low (4% up to m = 2). Qwen models put little mass on "unknown" when the passage is relevant, so this does not occur for them. The paper's gamma = 0.99 rule would behave the same way, so this is a property of probability averaging with this model rather than of our implementation.

### 5.4 Hallucination: wrong answers instead of abstaining
We use a simple proxy: the share of questions where a method answers (does not say "unknown") and the answer is wrong. Each cell is abstain rate (m = 0) / wrong-but-answered at m = 0 / at m = 1, in %.

| Method | Qwen2.5-1.5B | Qwen2.5-7B | Mistral-7B (n = 28) |
|---|---|---|---|
| Vanilla RAG | 2.2 / 19.6 / 84.8 | 8.2 / 16.3 / 46.9 | 3.6 / 21.4 / 42.9 |
| Keyword aggregation | 4.3 / 60.9 / 65.2 | 6.1 / 34.7 / 44.9 | 3.6 / 42.9 / 53.6 |
| Keyword aggregation v1 | 71.7 / 13.0 / 13.0 | 85.7 / 8.2 / 10.2 | 60.7 / 21.4 / 21.4 |
| Decoding aggregation | 4.3 / 43.5 / 52.2 | 6.1 / 28.6 / 34.7 | 32.1 / 28.6 / 32.1 |

Without attack, isolate-then-aggregate does **not** reduce unsupported answers: the final RobustRAG variants rarely abstain and are less accurate than vanilla RAG, so they answer wrongly more often (e.g. 35% vs 16% for Qwen2.5-7B keyword vs vanilla). Under one poisoned passage they are better than vanilla at 1.5B and comparable or better at 7B, mostly because vanilla's wrong answers are the attacker's. The strict threshold (v1) has the fewest wrong answers in every model because it abstains on 61-86% of questions. In our data, abstaining is the mechanism that suppresses hallucination, and it costs most of the correct answers. The proxy is coarse: substring matching counts partially correct answers (e.g. "terraced housing" for "terraces") as wrong.

## 6. Extension: thresholded recency for temporal conflicts
**Problem.** Facts change ("who is the current head coach?"). Passages dated in the past were correct for their date, so majority voting can return a stale answer. "Newest date wins" fixes this, but one injected passage with a fake newer date then decides the answer.

**Method.** (1) *Claims:* each isolated passage yields (answer, date); the answer comes from the LLM and the date is parsed from the passage (`March 4, 2023`, `4 Mar 2023`, `2023-03-04`). (2) *Eligibility:* an answer needs at least B+1 supporting passages (B = assumed adversary budget, here 2). (3) *Robust dating:* each eligible answer is dated by the (B+1)-th newest date among its supporters, and the newest eligible answer wins.

**Guarantee (sketch).** If claims are extracted correctly, an attacker with at most B passages (a) cannot make a fabricated answer eligible, and (b) cannot make an eligible answer look newer than one of its clean supporters, because at most B of its dates are forged, so the (B+1)-th newest is clean. The output is therefore backed by clean passages and chosen by clean dates. It does not guarantee the *current* answer: if the current answer has at most B clean supporters (very recent news), the rule falls back to an older answer; this case is not tested. Our first version (v1) dated an answer by its newest supporter, which one forged date on the stale answer breaks (Attack B).

**Data and attacks.** 40 synthetic questions with invented people and organizations, so the LLM cannot answer from memory. Each has 4+3 or 3+4 passages for the old (2019-2021) and new (2023-2024) answer plus irrelevant passages. Attack A injects m passages for a fabricated name dated 2025; Attack B injects m passages re-dating the stale answer to 2025. Figure: `fig_temporal.png`.

| Attack | m | Majority vote | Naive recency | Thresholded v1 (newest date) | Thresholded, robust date (ours) |
|---|---|---|---|---|---|
| A: fabricated name | 0 | 45% | 100% | 100% | 100% |
| | 1-2 | 45% | 0% | 100% | 100% |
| | 3 (> B) | 45% | 0% | 0% | 0% |
| B: stale answer re-dated | 0 | 45% | 100% | 100% | 100% |
| | 1-2 | 0% | 0% | 0% | 100% |
| | 3 (> B) | 0% | 0% | 0% | 0% |

Only the robust-date rule resists both attacks, and it fails exactly when the attacker exceeds the budget (m = B+1). Majority vote returns the stale answer whenever it has more passages (55% of questions). **All three models give identical tables**: the passages are templated and every model extracts the names essentially perfectly. This confirms that the decision rule behaves as argued, independent of the LLM, but says nothing about real, noisy temporal data.

## 7. Limitations
- **Small samples:** 46-49 poisoning questions and 40 temporal questions, one seed; 95% intervals are about ±13 points.
- **Data:** SQuAD with synthetic paraphrased evidence instead of the paper's search-based NQ/RealtimeQA; fully synthetic temporal data. Absolute numbers are not directly comparable with the paper.
- **Incomplete runs:** a GPU hardware failure stopped the 20-token Mistral rerun at 28 questions and prevented certification for Mistral-7B and the 1.5B model. Llama2-7B-Chat (gated) was not run.
- **Model-specific prompting:** Mistral needed a stricter prompt and answer clean-up, so its numbers are not a pure model-size comparison.
- **Approximations:** keyword extraction without POS tags or lemmatization; certification only for keyword aggregation (not decoding, Algorithm 4), with a W_C cap of 10.
- **Hallucination** is measured with a coarse proxy; no hallucination-specific defense was built.

## 8. Conclusion
Our recreation reproduces RobustRAG's central empirical claim: isolating passages and aggregating the answers stops a single poisoned passage from taking over the answer. At 7B the attack success rates (vanilla 31-41%, RobustRAG 4-6%) are close to the paper's (37%, 7-12%), and the clean-accuracy cost of the defense, large at 1.5B, shrinks substantially. Implementing the certification procedure gave 35% certified accuracy against one corrupted passage (paper: 26%), but only after fixing a case in which Algorithm 3 as written over-certifies; we also showed that with the paper's alpha and k no question can be certified against two corrupted passages. Our analysis identifies distractor majorities in retrieval as the main reason RobustRAG loses clean accuracy, and shows that against hallucination isolation alone does not help while abstention does, at a high price. For temporal conflicts, dating each answer by its (B+1)-th newest supporter resists both fabricated and re-dated answers up to the assumed budget, independently of the LLM. Next steps are completing the interrupted runs, certifying decoding aggregation, evaluating on real time-stamped corpora, and adding an answer-verification step against hallucination.

## 9. Reproducibility
All code, per-question outputs and figures are in the repository. Main entry points: `src/robustrag.py` (vanilla, keyword v1 and paper-faithful keyword aggregation, decoding aggregation, attack), `src/certify.py`, `src/temporal.py`, `src/hallucination.py`, `src/plots_models.py`, `src/make_pdf.py`. Models are chosen with environment variables; the same 50 questions are reused via `results.json`:
```
export RR_QUESTIONS=results.json RR_DTYPE=bfloat16
# Qwen2.5-7B
export RR_LLM=Qwen/Qwen2.5-7B-Instruct RR_TAG=_qwen7b
python src/robustrag.py 50
python src/temporal.py 40
python src/certify.py 50
# Mistral-7B (same three scripts)
export RR_LLM=mistralai/Mistral-7B-Instruct-v0.2
export RR_TAG=_mistral7b RR_PROMPT=strict
python src/robustrag.py 50
# figures, tables, PDF
python src/plots_models.py
python src/hallucination.py results_qwen7b.json
python src/make_pdf.py
```
The 1.5B run (`results.json`, about 75 minutes on an RTX 3050) uses the defaults; a 7B run takes 4-10 minutes on an RTX PRO 6000.

## References
1. C. Xiang, T. Wu, Z. Zhong, D. Wagner, D. Chen, P. Mittal. Certifiably Robust RAG against Retrieval Corruption. arXiv:2405.15556, 2024.
2. W. Zou, R. Geng, B. Wang, J. Jia. PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models. arXiv:2402.07867, 2024.
3. P. Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.
4. P. Rajpurkar, J. Zhang, K. Lopyrev, P. Liang. SQuAD: 100,000+ Questions for Machine Comprehension of Text. EMNLP 2016.
5. Qwen Team. Qwen2.5 Technical Report. arXiv:2412.15115, 2024.
6. A. Q. Jiang et al. Mistral 7B. arXiv:2310.06825, 2023.
7. N. Reimers, I. Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP 2019.
