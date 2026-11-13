# RobustRAG Recreation and Thresholded Recency for Temporal Conflicts under Bounded Poisoning

CS787, Group 8. Authors: Akash Tiwari (241090403) and Vaibhav Soni (241090419), IIT Kanpur ([@vbsoni-hpc](https://github.com/vbsoni-hpc)). Small-scale recreation of *Certifiably Robust RAG against Retrieval Corruption* (Xiang et al., 2024, arXiv:2405.15556), plus one extension.

## 1. Summary
1. **Recreation.** We implement vanilla RAG and RobustRAG's isolate-then-aggregate defense (keyword aggregation and decoding aggregation) and attack them with PoisonedRAG-style injected passages. Vanilla RAG collapses under poisoning (attack success rate, ASR, 2% to 78% with one poisoned passage out of ten). Both RobustRAG variants resist far better: keyword aggregation keeps ASR at 0-2% up to 2 poisoned passages, decoding aggregation keeps accuracy at 38-52%.
2. **Extension: thresholded recency.** When the correct answer changes over time, plain voting picks the stale majority (57.5% accuracy), and naive "latest date wins" is broken by one fake newer-dated passage (100% ASR). Requiring a claim to be supported by at least m+1 passages before recency is applied keeps 100% accuracy and 0% ASR up to the assumed adversary budget, and fails beyond it, as expected.

## 2. Setup (what is and is not faithful to the paper)
| Item | Paper | Ours |
|---|---|---|
| LLM | Mistral-7B, Llama, GPT | Qwen2.5-1.5B-Instruct (fp16, 4 GB laptop GPU) |
| Data | NQ, RealtimeQA, Biogen | Open-domain SQuAD (validation paragraphs as corpus, about 2k passages), 50 questions |
| Retriever | Contriever | all-MiniLM-L6-v2 dense retriever, k=10 |
| Attack | PoisonedRAG | PoisonedRAG black-box style: question text + fabricated passage for an LLM-generated wrong target; poisoned passages ranked on top (worst case), m in {0,1,2,3} |
| Defense | keyword and decoding aggregation | same idea, simplified (below) |

Deviations, all forced by compute and time:
- **Questions** were kept only if the gold answer appears in the clean top-10, so clean accuracy is meaningful. Yes/no questions were removed.
- **Redundant evidence is synthetic.** SQuAD has one evidence paragraph per question, but isolate-then-aggregate needs several agreeing passages. For each question we add 4 LLM paraphrases of the gold paragraph, giving 5 evidence passages plus 5 other retrieved passages. This favours the defenses relative to real corpora; absolute numbers are not comparable to the paper.
- **Keyword aggregation:** each passage is answered in isolation; keywords are word types in the answer (stop-words and question words removed); keywords with count >= tau = ceil(0.3 k) = 3 are kept and the LLM writes the final answer from them.
- **Decoding aggregation:** the next-token distributions of the isolated passages are averaged, then greedy decoding. We drop passages that abstain ("unknown") at the first step; the paper's entropy-based weighting is not implemented.
- **Certification** is approximate: a keyword-aggregation answer counts as certified if it is correct, tau > m, and a gold keyword has count >= tau + m. It is not the paper's full certificate.
- Accuracy means the normalized gold span appears in the output. This is strict for keyword aggregation, which outputs short word lists.

## 3. Results: recreation (n = 50 questions, k = 10)
Figure: `fig_main.png`.

| Poisoned passages m | Acc. vanilla | Acc. keyword | Acc. decoding | Certified (keyword) | ASR vanilla | ASR keyword | ASR decoding |
|---|---|---|---|---|---|---|---|
| 0 | 76% | 14% | 52% | 14% | 2% | 0% | 0% |
| 1 | 22% | 14% | 48% | 12% | 78% | 0% | 12% |
| 2 | 16% | 14% | 42% | 10% | 84% | 2% | 30% |
| 3 | 18% | 12% | 38% | 0% | 82% | 16% | 44% |

Reading:
- Vanilla RAG is best when clean but loses about 54 points of accuracy to a single poisoned passage, and the attacker's answer appears in 78% of outputs.
- Keyword aggregation is the most robust against the attacker (ASR at most 2% for m <= 2, the regime where tau > m), matching the paper's guarantee, but its clean accuracy is low (14%) in our setup because multi-word SQuAD spans are rarely reproduced from single-word keywords by a 1.5B model. It trades utility for robustness.
- Decoding aggregation keeps the best accuracy under attack (48 -> 38%) but its ASR rises as m grows, consistent with averaging being weaker than thresholded voting.
- At m = 3 = tau the keyword guarantee no longer applies and ASR jumps to 16%, as predicted.

## 4. Extension: temporal claim normalization + thresholded recency
Problem. Facts change ("who is the current head coach?"). Passages dated in the past are correct for their date, so majority voting can return a stale answer. Naive recency (latest date wins) fixes this but hands the attacker an easy win: one passage with a fake newer date.

Method.
1. *Temporal claim normalization:* each isolated response becomes a claim (answer, date), with dates parsed from three formats (`March 4, 2023`, `4 Mar 2023`, `2023-03-04`) into (y, m, d).
2. *Evidence-aware conflict resolution:* a claim is eligible only if at least tau = B + 1 passages support it, where B is the assumed adversary budget (B = 2, tau = 3). The most recent eligible claim wins.
3. *Why it preserves a bounded-adversary guarantee (sketch, not a full proof):* an adversary controlling at most B passages cannot make any claim reach B + 1 supporters without a clean passage also supporting it. So the set of claims eligible for the recency step contains no purely fabricated claim, and the fake newer date is filtered out. This guarantee covers the selection step only, assuming claim extraction is correct.

Data (synthetic, 40 questions, invented organizations and people so the LLM cannot rely on memory). Each question has 4+3 or 3+4 old/new supporting passages (old: 2019-2021, new: 2023-2024), the rest irrelevant, and m adversarial passages claiming a third name dated 2025. Claim answers are extracted by Qwen2.5-1.5B per passage. Figure: `fig_temporal.png`.

| Fake newer-dated passages m | Plain voting acc. | Naive recency acc. | Thresholded recency acc. (ours) | ASR naive | ASR ours |
|---|---|---|---|---|---|
| 0 | 57.5% | 100% | 100% | 0% | 0% |
| 1 | 57.5% | 0% | 100% | 100% | 0% |
| 2 | 57.5% | 0% | 100% | 100% | 0% |
| 3 (> budget) | 57.5% | 0% | 0% | 100% | 100% |

Reading: thresholded recency is the only method that is both temporally correct and attack-resistant; it breaks exactly when the attacker exceeds the budget (m = 3 >= tau), which is the expected boundary.

## 5. Limitations (please read)
- Small samples (50 and 40 questions), one seed, no confidence intervals. Differences of a few points are noise.
- Small local LLM; the paper uses 7B+ models.
- Redundant evidence in the recreation and the whole temporal set are synthetic. The temporal results show the mechanism works under controlled conditions; they are not a benchmark result. The clean numbers are near-trivial because the templates are simple.
- Certification is approximate; decoding aggregation has no certificate here.
- The thresholded-recency argument is a sketch that assumes correct claim extraction and correct dates in clean passages.

## 6. Reproduce
```
pip install torch transformers datasets sentence-transformers matplotlib
python src/robustrag.py 50 results.json   # about 75 min on an RTX 3050 4GB
python src/plots.py results.json          # fig_main.png
python src/temporal.py 40                 # fig_temporal.png, results_temporal.json
```
