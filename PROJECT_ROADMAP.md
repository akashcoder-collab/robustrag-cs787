# 🛡️ RobustRAG: Defending Retrieval-Augmented Generation Against Temporal Conflicts, Adversarial Poisoning, and Hallucination

**Course:** CS787 — Advanced NLP/Deep Learning  
**Professor:** Arnab Bhattacharya  
**Group Number:** 8 | **Team Size:** 2  
**Project Status:** ✅ Approved  

---

## 📌 Project Overview

Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by grounding them in external knowledge retrieved at inference time. However, RAG systems face three critical vulnerabilities:

1. **Temporal Conflicts** — Retrieved documents may contain outdated or contradictory time-sensitive facts
2. **Adversarial Poisoning** — Attackers can inject malicious documents into the knowledge base to manipulate outputs
3. **Hallucination** — The model may generate unfaithful content despite having access to correct retrieved passages

This project proposes **RobustRAG**, a unified defense framework that addresses all three threats through an integrated pipeline of temporal-aware retrieval, adversarial document filtering, and hallucination-aware generation.

---

## 🗺️ Project Roadmap

### Phase 1: Literature Survey & Problem Formulation (Week 1–2)

| Task | Description | Status |
|------|-------------|--------|
| Survey RAG fundamentals | Read Lewis et al. (2020), REALM, RETRO | ⬜ |
| Study temporal conflict papers | ConflictRAG, T-GRAG, DRAGged Into Conflict | ⬜ |
| Study adversarial poisoning papers | PoisonedRAG, ToxicRAG, RobustRAG (Xiang et al.) | ⬜ |
| Study hallucination mitigation | VOTE-RAG, Stable-RAG, Self-RAG, attribution methods | ⬜ |
| Identify base paper | **RobustRAG (arXiv:2405.15556)** — Certifiably Robust RAG | ⬜ |
| Write problem statement | Define scope, threat models, evaluation criteria | ⬜ |

**Deliverable:** Problem statement document + annotated bibliography

---

### Phase 2: System Architecture Design (Week 3–4)

#### Proposed Architecture: Three-Layer Defense Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              LAYER 1: TEMPORAL-AWARE RETRIEVAL                  │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ Timestamp-based  │  │ Recency Scoring  │  │ Temporal      │  │
│  │ Metadata Filter  │  │ & Decay Function │  │ Consistency   │  │
│  │                  │  │                  │  │ Checker       │  │
│  └─────────────────┘  └──────────────────┘  └───────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│           LAYER 2: ADVERSARIAL DOCUMENT FILTERING               │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ Isolate-then-   │  │ Poisoned Doc     │  │ Source         │  │
│  │ Aggregate       │  │ Detector (ML)    │  │ Credibility   │  │
│  │ (RobustRAG)     │  │                  │  │ Assessment    │  │
│  └─────────────────┘  └──────────────────┘  └───────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│           LAYER 3: HALLUCINATION-AWARE GENERATION               │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ Faithful         │  │ Self-Reflection  │  │ Attribution   │  │
│  │ Generation       │  │ Tokens           │  │ Verification  │  │
│  │ Constraints      │  │ (Self-RAG)       │  │               │  │
│  └─────────────────┘  └──────────────────┘  └───────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Final Robust Response                         │
└─────────────────────────────────────────────────────────────────┘
```

| Task | Description | Status |
|------|-------------|--------|
| Design overall pipeline | Three-layer defense architecture | ⬜ |
| Define Layer 1 | Temporal-aware retrieval with timestamp metadata | ⬜ |
| Define Layer 2 | Isolate-then-aggregate + ML-based poison detection | ⬜ |
| Define Layer 3 | Hallucination-aware generation with self-reflection | ⬜ |
| Write architecture document | Diagrams, data flow, module specifications | ⬜ |

**Deliverable:** Architecture design document with diagrams

---

### Phase 3: Dataset Preparation & Baseline Setup (Week 4–5)

| Task | Description | Status |
|------|-------------|--------|
| Select evaluation datasets | NQ, TriviaQA, HotpotQA, MS-MARCO | ⬜ |
| Create temporal conflict dataset | Augment existing QA datasets with time-varied answers | ⬜ |
| Create poisoned corpus | Use PoisonedRAG methodology to inject adversarial docs | ⬜ |
| Setup baseline RAG pipeline | Implement vanilla RAG with Contriever/DPR + LLM | ⬜ |
| Implement evaluation metrics | Accuracy, ASR (Attack Success Rate), Faithfulness | ⬜ |

**Datasets:**
| Dataset | Purpose | Size |
|---------|---------|------|
| Natural Questions (NQ) | Open-domain QA baseline | ~300K |
| TriviaQA | Knowledge-intensive QA | ~100K |
| HotpotQA | Multi-hop reasoning | ~113K |
| Custom Temporal | Time-sensitive QA pairs | ~5K (augmented) |

**Deliverable:** Processed datasets + baseline results

---

### Phase 4: Implementation (Week 5–8)

#### 4.1 — Temporal Conflict Defense Module

| Task | Description | Status |
|------|-------------|--------|
| Implement timestamp extraction | NER-based temporal entity extraction | ⬜ |
| Build recency scoring | Decay-weighted scoring for retrieved documents | ⬜ |
| Temporal conflict detector | Identify contradictions across time-stamped passages | ⬜ |
| Resolution strategy | Most-recent-first + consensus-based resolution | ⬜ |

**Key Idea:** Assign temporal metadata to retrieved documents, apply recency decay weighting, and use a conflict detector to identify when passages contradict each other on temporal grounds. Resolution favors the most authoritative recent source.

#### 4.2 — Adversarial Poisoning Defense Module

| Task | Description | Status |
|------|-------------|--------|
| Implement isolate-then-aggregate | Core RobustRAG mechanism from base paper | ⬜ |
| Keyword-based aggregation | Majority voting over keyword sets | ⬜ |
| Decoding-based aggregation | Token-level consensus during generation | ⬜ |
| ML-based poison classifier | Train lightweight classifier to flag suspicious docs | ⬜ |
| Source credibility scoring | Entropy-TOPSIS based credibility (from ConflictRAG) | ⬜ |

**Key Idea:** Partition retrieved passages into K disjoint groups, generate responses independently for each group, then aggregate using secure voting (keyword or token-level). Additionally, train a lightweight ML classifier to pre-filter likely poisoned documents.

#### 4.3 — Hallucination Mitigation Module

| Task | Description | Status |
|------|-------------|--------|
| Implement self-reflection mechanism | Inspired by Self-RAG reflection tokens | ⬜ |
| Attribution verification | Check if each generated claim maps to a source passage | ⬜ |
| Faithfulness scorer | NLI-based entailment checking | ⬜ |
| Ensemble voting | VOTE-RAG style multi-path generation consensus | ⬜ |

**Key Idea:** After generation, verify each claim against retrieved passages using NLI-based entailment. Use self-reflection tokens to assess retrieval quality and generation faithfulness. Apply ensemble voting to reduce hallucination via multi-path consensus.

**Deliverable:** Working implementation of all three modules

---

### Phase 5: Experiments & Evaluation (Week 8–10)

| Experiment | Description | Metric |
|------------|-------------|--------|
| Baseline comparison | Vanilla RAG vs. RobustRAG | Accuracy, F1 |
| Temporal robustness | Test with time-conflicting passages | Temporal Accuracy |
| Adversarial robustness | PoisonedRAG attack + defense | ASR ↓, Clean Accuracy |
| Hallucination reduction | Measure faithfulness of generated text | Faithfulness Score |
| Ablation study | Contribution of each defense layer | Per-module improvement |
| Efficiency analysis | Latency and compute overhead | Time (ms), FLOPs |

#### Evaluation Metrics

| Metric | Measures | Formula/Tool |
|--------|----------|--------------|
| **Exact Match (EM)** | Answer correctness | String match |
| **F1 Score** | Partial answer overlap | Token overlap |
| **Attack Success Rate (ASR)** | Adversarial vulnerability | % queries misled |
| **Faithfulness** | Grounding in evidence | NLI entailment |
| **CARS Score** | Conflict-aware RAG quality | ConflictRAG metric |
| **Latency** | Computational overhead | Wall-clock time |

**Deliverable:** Experimental results + analysis

---

### Phase 6: Report Writing & Presentation (Week 10–12)

| Task | Description | Status |
|------|-------------|--------|
| Write introduction & motivation | Why RAG robustness matters | ⬜ |
| Write related work section | Cover all three threat categories | ⬜ |
| Write methodology section | Three-layer defense architecture | ⬜ |
| Write experiments section | Results, tables, ablation | ⬜ |
| Write conclusion | Key findings + future work | ⬜ |
| Prepare presentation slides | 15-20 slides | ⬜ |
| Final review & submission | Polish + submit | ⬜ |

**Deliverable:** Final report (IEEE/ACL format) + presentation slides

---

## 📚 Key Papers (Downloaded)

All papers saved in: `c:\cs787\RobustRAG_project\papers\`

### 🔴 Base Paper (MUST READ FIRST)
| # | Paper | ArXiv ID | Key Contribution |
|---|-------|----------|-------------------|
| 1 | **Certifiably Robust RAG against Retrieval Corruption** (RobustRAG) | `2405.15556` | Isolate-then-aggregate defense with certifiable guarantees |

### 🟠 Adversarial Poisoning Attack Papers
| # | Paper | ArXiv ID | Key Contribution |
|---|-------|----------|-------------------|
| 2 | **PoisonedRAG: Knowledge Corruption Attacks to RAG** | `2402.07867` | Formulates RAG poisoning as optimization; 90% ASR with 5 docs |
| 3 | **ToxicRAG: Single-Shot Knowledge Poisoning Attacks** | `2609.11082` | Single-doc poisoning; temporal alteration attack category |

### 🟡 Temporal Conflict Papers
| # | Paper | ArXiv ID | Key Contribution |
|---|-------|----------|-------------------|
| 5 | **ConflictRAG: Detecting & Resolving Knowledge Conflicts** | `2504.01233` | Entropy-TOPSIS credibility; conflict detection framework |
| 8 | **(D)RAGged Into a Conflict: Taxonomy of Knowledge Conflicts** | `2406.13805` | Conflict taxonomy + CONFLICTS benchmark |

### 🟢 Hallucination & Defense Papers
| # | Paper | ArXiv ID | Key Contribution |
|---|-------|----------|-------------------|
| 6 | **Hallucination Mitigation for Retrieval-Augmented LLMs** | `2501.01904` | Comprehensive survey of hallucination in RAG |
| 7 | **RADAR: Dynamic RAG Defense Framework** | `2405.17067` | Balances robustness vs efficiency |
| 10 | **Self-RAG: Learning to Retrieve, Generate, Reflect** | `2310.11511` | Reflection tokens for faithful generation |

### 🔵 Foundational Papers
| # | Paper | ArXiv ID | Key Contribution |
|---|-------|----------|-------------------|
| 4 | **Comprehensive Survey of RAG** | `2410.12837` | Full RAG landscape survey |
| 9 | **RAG for Knowledge-Intensive NLP Tasks** (Lewis et al.) | `2005.11401` | Original RAG paper — foundational |

---

## 🧰 Technical Stack (Recommended)

| Component | Tool/Library |
|-----------|-------------|
| **LLM Backend** | Llama-3 / GPT-4 API / Mistral |
| **Retriever** | Contriever / DPR / ColBERT |
| **Vector DB** | FAISS / ChromaDB |
| **NLI Model** | DeBERTa-v3 (for faithfulness checking) |
| **Framework** | PyTorch + HuggingFace Transformers |
| **Experiment Tracking** | Weights & Biases / MLflow |
| **Language** | Python 3.10+ |

---

## ⏰ Timeline Summary

```
Week 1-2  ████████░░░░░░░░░░░░░░░░  Literature Survey & Problem Formulation
Week 3-4  ░░░░░░░░████████░░░░░░░░  Architecture Design + Dataset Prep
Week 5-8  ░░░░░░░░░░░░░░░░████████  Implementation (3 modules)
Week 8-10 ░░░░░░░░░░░░░░░░░░██████  Experiments & Evaluation
Week 10-12░░░░░░░░░░░░░░░░░░░░░░██  Report + Presentation
```

---

## 📂 Project Directory Structure

```
c:\cs787\RobustRAG_project\
├── papers/                          # Downloaded research papers (10 PDFs)
│   ├── 1_BASE_RobustRAG_...pdf      # ⭐ BASE PAPER
│   ├── 2_PoisonedRAG_...pdf
│   ├── 3_ToxicRAG_...pdf
│   ├── 4_RAG_Survey_...pdf
│   ├── 5_ConflictRAG_...pdf
│   ├── 6_RAG_Hallucination_...pdf
│   ├── 7_RADAR_...pdf
│   ├── 8_DRAGged_Into_Conflict_...pdf
│   ├── 9_RAG_Foundational_...pdf
│   └── 10_Self_RAG_...pdf
├── src/                             # Source code (to be created)
│   ├── retrieval/                   # Layer 1: Temporal-aware retrieval
│   ├── defense/                     # Layer 2: Adversarial filtering
│   ├── generation/                  # Layer 3: Hallucination-aware gen
│   └── pipeline.py                  # End-to-end pipeline
├── data/                            # Datasets (to be created)
├── experiments/                     # Experiment configs & results
├── report/                          # LaTeX report
└── PROJECT_ROADMAP.md               # This file
```

---

## 🎯 Key Contributions (Proposed)

1. **Unified Defense Framework** — First to integrate temporal conflict resolution, adversarial poisoning defense, AND hallucination mitigation into a single RAG pipeline
2. **Temporal-Aware Retrieval** — Novel timestamp-based filtering and recency decay scoring for handling time-sensitive knowledge
3. **Enhanced Isolate-then-Aggregate** — Extend RobustRAG's mechanism with ML-based pre-filtering to improve both robustness and efficiency
4. **Comprehensive Evaluation** — Systematic benchmarking across all three threat categories with ablation studies

---

## ⚠️ IMMEDIATE ACTION NEEDED

Professor has asked "Approved. Base paper?" — Reply with:
- **Base Paper:** *"Certifiably Robust RAG against Retrieval Corruption"* by Chong Xiang et al. (arXiv:2405.15556, 2024)
- This paper introduces the RobustRAG framework with isolate-then-aggregate defense and certifiable robustness guarantees
- **Deadline:** Reply by Tuesday, September 22nd, 2026
