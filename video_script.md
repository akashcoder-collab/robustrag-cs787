# RobustRAG — 3-Minute Video Script

**Paper:** *Certifiably Robust RAG against Retrieval Corruption* (Xiang, Wu, Zhong, Wagner, Chen, Mittal)
**Speakers:** A and B  ·  **Length:** about 440 spoken words, roughly 3:00 at a normal pace

---

### [0:00 – 0:20] Hook: the threat — **Speaker A**
*Visual: Fig. 1 top row. The vanilla RAG answer flips to "Fuji!"*

> Retrieval corruption is the problem. If an attacker gets just **one** poisoned passage into the top-10 results, through SEO spam, PoisonedRAG or an indirect prompt injection, vanilla RAG can be hijacked. In the paper's experiments, prompt injection succeeds on **over 90%** of queries. Existing defenses are heuristic, so a stronger adaptive attack can break them.

### [0:20 – 0:40] Threat model and goal — **Speaker B**
*Visual: on-screen text "k′ of k passages corrupted, attacker knows everything"*

> The paper assumes a very strong attacker. They can inject **k′ passages with any content, at any rank**, and they know the defense, the model weights and the query. The goal is not "we survived some attacks". It is **τ-certifiable robustness**: a provable lower bound on response quality against every possible attack. One caveat: this only works if useful benign passages outnumber malicious ones. Without that, no defense can succeed.

### [0:40 – 1:00] Core idea: isolate then aggregate — **Speaker A**
*Visual: Fig. 1 bottom row. Passages split into groups, three isolated answers*

> RobustRAG's key idea is to **isolate, then aggregate**. It splits the passages into disjoint groups of size κ, gets a separate LLM answer from each group, and then combines them. A corrupted passage can now spoil only **its own group**. The hard part is combining free-form text securely, because "Mount Everest" and "Everest is the highest" are the same answer, and plain majority voting can't see that.

### [1:00 – 1:30] Method 1: secure keyword aggregation — **Speaker B**
*Visual: Fig. 2. Keyword sets → counts → filtered set → final answer*

> The first method is **keyword aggregation**. It extracts the unique keywords from each isolated response and counts each keyword **at most once per response**. That means the attacker can raise any count by at most k′. Keywords below a threshold, min(α·n, β), are dropped. The LLM then answers using **only the surviving keywords**, with no raw passages. "Everest" appears twice and survives. "Fuji" appears once and is filtered out.

### [1:30 – 1:55] Method 2: secure decoding aggregation — **Speaker A**
*Visual: Fig. 3. Per-group probability bars summed into one vector*

> Keywords lose detail in long outputs like biographies, so the second method works **token by token**. At each step, it sums the next-token probability vectors from all groups. Each probability is at most 1, so an attacker can shift any sum by at most k′. If the gap between the top two tokens is larger than a threshold η, it emits the top token. Otherwise it falls back to the token the model would predict **without retrieval**, which the attacker can't touch.

### [1:55 – 2:20] Certification — **Speaker B**
*Visual: Fig. 4 (keyword groups W_A / W_B / W_C), then Fig. 5 (decoding tree)*

> Because the attacker's influence is bounded, the set of possible outputs becomes **finite**, so it can be checked exhaustively. For keywords: high-count words are always kept, low-count words are never kept, and the paper enumerates every subset of the "medium" words. For decoding, it builds a **binary tree** in which each step either always takes the top-1 token, branches into top-1 or no-retrieval, or always takes no-retrieval. If none of those cases applies, certification fails. The worst-scoring leaf gives τ, and Theorems 1 and 2 prove that this τ is correct.

### [2:20 – 2:45] Results — **Speaker A**
*Visual: Table I and Table II highlights*

> They tested on RealtimeQA, Natural Questions and Biography generation, with Mistral-7B, Llama-2-7B and GPT-3.5, at **k = 10, k′ = 1**. On RealtimeQA, accuracy is **71% clean with 38% certified**. Vanilla RAG gets 69% clean and **0% certified**. Multiple-choice QA reaches **71% certified**. Against real attacks, success rates drop from **over 90% to about 10%**. The runtime cost is **1.2–3.7×**, compared with more than 100× for randomized smoothing.

### [2:45 – 3:00] Trade-offs, limits, takeaway — **Speaker B**
*Visual: Fig. 6 and Fig. 8 curves, then the title card*

> Group size κ trades robustness against utility. At κ = 3, the clean-accuracy drop shrinks from 7% to 0%. When 5 of the 10 passages are corrupted, robustness falls to zero, as the theory predicts. Open problems include defending the retrieval phase and handling multi-hop queries. **RobustRAG is the first certifiably robust RAG for free-form generation.** Thanks for watching!

---

**Tips for recording**
- If you run long, shorten the certification section (2:00 slot) to two sentences: "bounded influence makes the output set finite, so we enumerate it and take the worst score."
- Symbols shown on screen: κ = group size, k′ = corrupted passages, α/β = keyword thresholds, η = decoding gap threshold, τ = certified quality bound.
