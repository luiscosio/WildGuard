# WildGuard — PROJECT.md
**Scalable Oversight Harness for Dark-Pattern Monitoring in Real LLM Chat Logs (WildChat)**

> **One-sentence summary:** WildGuard is a scalable oversight system that uses **DarkBench (elicitation prompts)** to elicit and define dark-pattern behaviors, then applies **two independent detectors (LLM Judge + trained classifier)** to monitor **real-world conversations (WildChat)**, quantify prevalence, and measure the **benchmark-to-reality (ecological validity) gap**—plus a reliability study of **LLM-as-a-judge**.

**Primary Track:** Track 2 — Real-World Analysis / Monitoring  
**Secondary Track:** Track 1 — Measurement & Evaluation (ecological validity gap + judge reliability)

---

## 0) Core References (read this first)
### WildChat (Real-world logs)
- WildChat paper (May 2024): **1M conversations**, **2.5M+ turns**, multilingual, opt-in collection, released at wildchat.allen.ai.  
  https://arxiv.org/abs/2405.01470 :contentReference[oaicite:0]{index=0}  
- Dataset homepage: https://wildchat.allen.ai/ :contentReference[oaicite:1]{index=1}  
- Hugging Face dataset: https://huggingface.co/datasets/allenai/WildChat-1M :contentReference[oaicite:2]{index=2}  

### DarkBench (Elicitation set + taxonomy)
- DarkBench paper (Mar 2025): **660 prompts**, **6 categories** (brand bias, user retention, sycophancy, anthropomorphism, harmful generation, sneaking).  
  https://arxiv.org/abs/2503.10728 :contentReference[oaicite:3]{index=3}  
- DarkBench GitHub: https://github.com/smarter/DarkBench :contentReference[oaicite:4]{index=4}  
- DarkBench site: https://darkbench.ai/ :contentReference[oaicite:5]{index=5}  

### LLM-as-a-judge reliability (we will explicitly measure)
- “Can You Trust LLM Judgments? Reliability of LLM-as-a-Judge” (Dec 2024): shows limitations of single-shot judging; proposes reliability framework and emphasizes multi-sample importance.  
  https://arxiv.org/abs/2412.12509 :contentReference[oaicite:6]{index=6}  

---

## 1) What we’re building (not “just a classifier”)
### WildGuard = Oversight Harness
WildGuard triangulates three signals:

1) **DarkBench Elicitation (Controlled, Synthetic)**
   - Provides taxonomy + controlled elicitation prompts for dark patterns. :contentReference[oaicite:7]{index=7}  

2) **LLM Judge (Auditor, Higher precision, higher cost)**
   - Scores and labels samples using a strict rubric; used for audits and judge reliability measurement. :contentReference[oaicite:8]{index=8}  

3) **Classifier (Monitor, scalable, stable)**
   - Fast, cheap scanning of large WildChat slices; used in the final monitoring tool.

**Core contribution:** build and evaluate a **scalable oversight harness** that answers:
- What dark patterns are present **in the wild**?
- How does that differ from **benchmark predictions** (ecological validity gap)?
- How reliable is the LLM judge—and where does it disagree with the classifier? :contentReference[oaicite:9]{index=9}  

---

## 2) Why this wins (hackathon alignment)
- **Real-world monitoring:** WildChat is a large-scale real log dataset designed for studying real usage. :contentReference[oaicite:10]{index=10}  
- **Ecological validity:** explicitly measures the gap between controlled benchmark elicitation (DarkBench) and real deployment behavior (WildChat). :contentReference[oaicite:11]{index=11}  
- **Empirical + scalable:** run at 50k–200k+ turns; produce prevalence stats, trends, and context clusters.  
- **Judge reliability as a first-class result:** LLM-as-a-judge reliability is a known issue; we measure it and build a practical harness around it. :contentReference[oaicite:12]{index=12}  

---

## 3) Problem statement
Benchmarks can measure manipulation in controlled prompts, but we lack tooling that:
1) detects dark-pattern markers in real logs,
2) quantifies prevalence + contexts,
3) compares benchmark exposure to real deployment behavior (ecological validity gap),
4) provides an operational oversight harness (scalable + auditable).

---

## 4) Goals (Definition of Done)
### G1 — Monitoring Tool
Input: chat logs (WildChat format or similar)  
Output:
- flagged assistant turns
- category + confidence
- conversation-level summary + risk-over-time plot
- exportable report JSON/CSV

### G2 — Real-world prevalence analysis (WildChat)
Run on ≥50k–200k assistant turns:
- prevalence by category
- prevalence vs conversation turn number
- confidence distributions
- top contexts (clusters/domains)

### G3 — Benchmark vs reality gap report
Compare:
- DarkBench category distribution (elicited)
vs
- WildChat observed distribution (in-the-wild)
Output:
- JS divergence / KL divergence
- rank correlations
- “biggest mismatches” narrative

### G4 — Reliability and disagreement report
- LLM judge self-consistency (repeatability)  
- Judge vs classifier disagreement rate
- High-confidence disagreement audit + failure modes  
(Explicitly motivated by LLM judge reliability research.) :contentReference[oaicite:13]{index=13}  

---

## 5) Non-goals
- We do **not** claim intent; we detect **dark-pattern markers**.
- No private customer logs (WildChat only; synthetic demo logs if needed).
- No user profiling / de-anonymization.
- No publishing “best manipulation prompt recipes.”

---

## 6) Taxonomy (DarkBench-aligned)
We classify assistant turns into one of DarkBench’s 6 categories: :contentReference[oaicite:14]{index=14}  
- `brand_bias`
- `user_retention`
- `sycophancy`
- `anthropomorphism`
- `harmful_generation`
- `sneaking`

Optional (only if reliably labelable within time):
- `guilt_pressure`
- `refusal_erosion`

---

## 7) What we will measure (build-in from the start)
These are the “guaranteed story” measurements:

| Measurement | Why it matters | Likely story |
|---|---|---|
| **Category distribution: DarkBench vs WildChat** | ecological validity gap | “X is 3× more common in the wild than benchmarks predict” |
| **Confidence scores by source** (judge vs classifier; DarkBench vs WildChat) | detectability differences | “Wild patterns are harder to detect; confidence drops by Y%” |
| **Prevalence vs conversation turn number** | multi-turn emergence | “Markers spike after turn N (relationship depth effect)” |
| **Judge vs classifier disagreement rate** | oversight reliability | “Judge misses X% of retention markers classifiers catch (or vice versa)” |
| **Top contexts where patterns appear** | actionability | “Patterns cluster in customer support / companionship / shopping” |

---

## 8) System architecture (ASCII diagram)

```

```
                       ┌─────────────────────────────┐
                       │         DarkBench           │
                       │  660 elicitation prompts    │
                       │  6-category taxonomy        │
                       └──────────────┬──────────────┘
                                      │ (subset 200–300)
                                      v
                          ┌───────────────────────┐
                          │  ELICITATION RUNNER    │
                          │  Run prompts on models │
                          └──────────────┬────────┘
                                         │ outputs
                                         v
                            ┌─────────────────────┐
                            │ DarkBench Outputs   │
                            │ (prompt, output,    │
                            │  category, model)   │
                            └───────────┬─────────┘
                                        │
                                        │ (optional judge labeling)
                                        v
```

┌───────────────────────────────┐      ┌───────────────────────────────┐
│            WildChat            │      │          LLM JUDGE            │
│ 1M real conversations          │      │ rubric-based scoring          │
│ 2.5M+ assistant turns          │      │ category + confidence + notes │
└───────────────┬───────────────┘      └───────────────┬───────────────┘
│ sample turns (1k–5k)                  │
v                                       │
┌───────────────────────┐                         │
│  LABELING SET BUILDER  │◄────────────────────────┘
│ combine:               │
│ - DarkBench outputs    │
│ - WildChat sample      │
│ - LLM judge labels     │
│ - human audit subset   │
└──────────────┬────────┘
│ train/eval data
v
┌────────────────────────┐
│   CLASSIFIER TRAINER    │
│ (MiniLM / DistilRoBERTa)│
└──────────────┬─────────┘
│ trained model
v
┌───────────────────────────────┐
│   SCALABLE MONITOR (INFER)     │
│ run on 50k–200k WildChat turns │
└──────────────┬────────────────┘
│ detections + confidences
v
┌────────────────────────────┐
│    PREVALENCE ANALYTICS     │
│ categories, contexts,       │
│ turn-index trends,          │
│ confidence histograms       │
└──────────────┬─────────────┘
│
├─────────────┐
│             │
v             v
┌────────────────────────┐   ┌──────────────────────────┐
│  GAP REPORT (Benchmark │   │  RELIABILITY REPORT       │
│  vs Reality)           │   │  Judge vs classifier      │
│ DarkBench dist vs      │   │  Judge self-consistency   │
│ WildChat observed dist │   │  Disagreement analysis    │
└──────────────┬─────────┘   └──────────────┬───────────┘
│                            │
└──────────────┬─────────────┘
v
┌───────────────────┐
│   STREAMLIT APP    │
│ Log scan demo      │
│ Risk report export │
└───────────────────┘

```

---

## 9) Functional requirements
### FR1 — DarkBench elicitation runner
- Run a subset (200–300) of DarkBench prompts. :contentReference[oaicite:15]{index=15}  
- Support N models (2–5) and store outputs in `outputs/darkbench_outputs.jsonl`.

### FR2 — LLM judge module
- Rubric-based classification: outputs `category`, `confidence`, `explanation`, `spans` (optional).
- Must support:
  - temperature controls
  - repeatability test (same input, multiple runs)
  - prompt template sensitivity test  
  (motivated by judge reliability research). :contentReference[oaicite:16]{index=16}  

### FR3 — Labeling set builder
- Build `data/labeled/train.jsonl` + `eval.jsonl` using:
  - DarkBench outputs (known taxonomy)
  - WildChat sampled assistant turns labeled by judge
  - gold subset with human audit (100–200)

### FR4 — Classifier training pipeline
- Train a 6-way classifier:
  - baseline: embeddings + logistic regression
  - preferred: fine-tuned transformer (DistilRoBERTa / MiniLM)
- Track macro-F1 and precision@high-confidence.

### FR5 — WildChat ingestion + inference at scale
- Load WildChat and extract assistant turns. :contentReference[oaicite:17]{index=17}  
- Run classifier on ≥50k–200k turns.
- Run LLM judge on:
  - top-k high-risk flags
  - low-confidence uncertain cases
  - random QA sample

### FR6 — Analytics + reports
Generate:
- prevalence tables + plots
- confidence histograms
- turn-index emergence plots
- top contexts clustering
- gap report (DarkBench vs WildChat)
- reliability report (judge vs classifier + self-consistency)

### FR7 — Streamlit demo app
- Upload chat transcript JSONL
- Show flagged turns + categories + confidence
- Show risk over time
- Export detections + summary JSON

### FR8 — Jupyter notebooks for Phase 2 “Find the Story”
We include notebooks as core deliverables:
- `notebooks/01_exploration.ipynb` — raw prevalence, confidence, contexts, turn-index plots
- `notebooks/02_gap_and_reliability.ipynb` — confirmatory gap metrics + reliability + final plots

---

## 10) Data spec
### Inputs
- DarkBench prompts + taxonomy. :contentReference[oaicite:18]{index=18}  
- WildChat dataset (assistant turns). :contentReference[oaicite:19]{index=19}  

### Outputs (standardized)
- `outputs/darkbench_outputs.jsonl`
- `outputs/judge_labels.jsonl`
- `models/classifier/`
- `outputs/wildchat_detections.csv`
- `outputs/prevalence.json`
- `outputs/gap_report.json`
- `outputs/reliability_report.json`
- `figures/` (plots)
- `app/streamlit_app.py`

---

## 11) Evaluation metrics (judge-friendly)
### Classifier quality
- Macro F1 on gold set
- Precision@conf≥0.8
- Calibration bins (confidence vs accuracy)

### LLM judge reliability (explicitly measured)
- Self-consistency (repeat runs)  
- Prompt-template sensitivity  
- Disagreement rate with classifier  
(Aligned with reliability concerns raised in LLM-as-a-judge work.) :contentReference[oaicite:20]{index=20}  

### Monitoring utility
- Review load (flags per 1,000 turns)
- % high-risk flags confirmed by judge
- estimated cost savings vs judging everything

### Benchmark vs reality gap
- JS divergence (category distributions)
- Spearman rank correlation
- “largest mismatch categories” list

---

## 12) MVP scope (48 hours)
### Strong MVP (recommended)
- 3 categories (if tight time): `user_retention`, `sycophancy`, `sneaking`
- Judge-label 1,000 WildChat turns; human audit 200
- Train classifier
- Run inference on ≥50k turns
- Produce:
  - prevalence report
  - gap report
  - reliability report
  - Streamlit demo
  - 2 notebooks

### Full MVP (stretch)
- all 6 categories
- run on 200k+ turns
- multi-turn patterns / accumulation

---

## 13) 48-hour execution plan (Exploratory → Confirmatory)
### Phase 1 — Build + Run (Day 1)
- Get pipeline working end-to-end
- Run on 50k+ turns
- Generate basic prevalence + confidence tables

### Phase 2 — Find the Story (Day 1 night / Day 2 morning)
- Use notebooks to identify:
  - biggest gaps
  - strongest turn-index effects
  - top contexts
  - most interesting disagreement cases

### Phase 3 — Frame + Present (Day 2)
- Lead with the finding
- Infrastructure is the proof you discovered it responsibly

---

## 14) Risks & mitigations
**Risk:** LLM judge unreliability / prompt sensitivity  
**Mitigation:** explicitly measure repeatability + sensitivity and report it (core contribution). :contentReference[oaicite:21]{index=21}  

**Risk:** intent attribution critique  
**Mitigation:** call it “markers,” triage, and auditing support.

**Risk:** privacy / PII  
**Mitigation:** no user identifiers; redact example excerpts; publish aggregates only. WildChat is opt-in and released under AI2 ImpACT license. :contentReference[oaicite:22]{index=22}  

**Risk:** label noise  
**Mitigation:** small taxonomy, double-label 100 samples, report agreement.

---

## 15) Deliverables (submission-ready)
- GitHub repo with:
  - scripts for ingestion, elicitation, judging, training, inference
  - outputs + plots + gap/reliability reports
  - Streamlit demo app
  - 2 notebooks
  - report + limitations/dual-use appendix
- Optional: 3–5 minute demo video

---

## 16) Required appendix: Limitations & Dual-Use
Include:
- false positives/negatives
- edge cases (multilingual, sarcasm)
- dual-use risk: detection can reveal what works
- mitigation: publish aggregates + minimal redacted examples
- responsible disclosure guidance
- future work: multi-turn accumulation, model family comparisons, multilingual analysis

---

## 17) Recommended repo structure
```

wildguard/
README.md
PROJECT.md
data/
labeled/
annotations.csv
label_guide.md
train.jsonl
eval.jsonl
samples/
sample_wildchat.jsonl
sample_darkbench.jsonl
src/
run_darkbench.py
ingest_wildchat.py
sample_for_labeling.py
judge_label.py
train_classifier.py
infer_wildchat.py
evaluate.py
analytics.py
gap_report.py
reliability_report.py
utils.py
models/
classifier/
outputs/
darkbench_outputs.jsonl
judge_labels.jsonl
wildchat_detections.csv
prevalence.json
gap_report.json
reliability_report.json
figures/
notebooks/
01_exploration.ipynb
02_gap_and_reliability.ipynb
app/
streamlit_app.py
reports/
report.md
limitations_dual_use.md

```

---

## 18) Demo script (3–5 minutes)
1) Streamlit: upload a transcript → show flagged turns (categories + confidence)
2) Prevalence dashboard: category frequency + confidence histograms
3) Turn-index plot: “markers increase after turn N”
4) Gap report: DarkBench vs WildChat mismatch plot
5) Reliability report: judge repeatability + judge vs classifier disagreement
6) Close: “We built a scalable oversight harness for detecting manipulation markers in the wild.”