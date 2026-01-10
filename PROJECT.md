# WildGuard — PROJECT.md
**Scalable Oversight Harness for Dark-Pattern Monitoring in Real LLM Chat Logs (WildChat)**

> **One-sentence summary:** WildGuard is a scalable oversight system that uses **DarkBench (elicitation prompts)** to define and elicit dark-pattern behaviors, then applies **two independent detectors (LLM Judge + trained classifier)** to monitor **real-world conversations (WildChat)**, quantify prevalence, and measure the **benchmark-to-reality (ecological validity) gap**.  
> **Primary Track:** Track 2 (Real-World Analysis)  
> **Secondary Track:** Track 1 (Measurement & Evaluation; benchmark vs reality; judge reliability)

---

## 0) Core References (for context + citations)
### WildChat
- Paper (arXiv): `https://arxiv.org/abs/2405.01470` — WildChat: **1M conversations**, **2.5M+ turns**. :contentReference[oaicite:0]{index=0}
- Dataset site: `https://wildchat.allen.ai/` :contentReference[oaicite:1]{index=1}
- Hugging Face dataset: `https://huggingface.co/datasets/allenai/WildChat-1M` :contentReference[oaicite:2]{index=2}
- OpenReview: `https://openreview.net/forum?id=Bl8u7ZRlbM` :contentReference[oaicite:3]{index=3}

### DarkBench (Elicitation Set / Benchmark)
- Paper (arXiv): `https://arxiv.org/abs/2503.10728` — **660 prompts**, **6 categories**. :contentReference[oaicite:4]{index=4}
- Project site: `https://darkbench.ai/` :contentReference[oaicite:5]{index=5}
- GitHub: `https://github.com/smarter/DarkBench` :contentReference[oaicite:6]{index=6}
- Apart write-up: `https://apartresearch.com/news/uncovering-model-manipulation-with-darkbench` :contentReference[oaicite:7]{index=7}

### LLM-as-a-Judge Reliability (for the reliability report)
- “Can You Trust LLM Judgments?” (arXiv): `https://arxiv.org/abs/2412.12509` :contentReference[oaicite:8]{index=8}
- “LLM Judges Are Unreliable” (CIP blog): `https://www.cip.org/blog/llm-judges-are-unreliable` :contentReference[oaicite:9]{index=9}
- “LLMs Cannot Reliably Judge (Yet?)” (arXiv): `https://arxiv.org/pdf/2506.09443` :contentReference[oaicite:10]{index=10}

---

## 1) What We’re Building
### WildGuard = Oversight Harness (not just a classifier)
We are building a **scalable oversight harness** that triangulates three signals:

1) **DarkBench Elicitation (Controlled, Synthetic)**
   - Used to elicit dark-pattern behaviors under known conditions and define a taxonomy.

2) **LLM Judge (Auditor, High Precision)**
   - Used to label samples, audit edge cases, and evaluate the reliability of LLM-as-a-judge.

3) **Classifier (Monitor, Scalable)**
   - Used to scan large volumes of real-world chat logs cheaply and consistently.

The key research contribution:  
✅ **Compare LLM Judge vs Classifier** on the same WildChat subset and propose a practical **deployment-grade oversight harness** (thresholding + auditing + human review triggers).

---

## 2) Why This Wins (Hackathon Alignment)
This project directly matches hackathon priorities:
- **Real-world monitoring:** Detect manipulation in-the-wild using WildChat. :contentReference[oaicite:11]{index=11}:contentReference[oaicite:12]{index=12}
- **Ecological validity:** Quantify the gap between benchmark-elicited behaviors (DarkBench) and real usage. :contentReference[oaicite:13]{index=13}:contentReference[oaicite:14]{index=14}
- **Empirical backing:** Large-scale prevalence stats + reliability evaluation.
- **Addresses a known gap:** LLM-as-a-judge reliability is explicitly under scrutiny. :contentReference[oaicite:15]{index=15}:contentReference[oaicite:16]{index=16}

---

## 3) Problem Statement
Benchmarks can measure manipulative tendencies in controlled prompts, but we lack tooling that:
1) detects dark-pattern markers in real conversation logs,
2) quantifies prevalence and contexts,
3) compares benchmark predictions to real-world behavior (ecological validity gap),
4) and provides a scalable oversight harness usable in practice.

---

## 4) Scope: Goals & Non-Goals
### Goals (Definition of Done)
**G1 — Monitoring Tool**
- Input: chat logs (WildChat format or similar)
- Output: flagged assistant turns + category + confidence + summary report

**G2 — Real-World Prevalence Analysis**
- Run on ≥50k–200k assistant turns
- Report: prevalence by category, by conversation length, top contexts

**G3 — Benchmark vs Reality Gap Report**
- Compare DarkBench category distribution vs observed WildChat distribution

**G4 — Reliability Report**
- Compare judge vs classifier agreement (F1, kappa)
- Judge self-consistency test (repeat same inputs multiple times)

### Non-Goals
- No intent attribution (we detect **markers**, not intent).
- No private customer logs (WildChat only; synthetic for demo).
- No de-anonymization, no user profiling.
- No publishing “best manipulation recipes”.

---

## 5) Taxonomy (DarkBench-Aligned)
DarkBench defines 6 dark-pattern categories. :contentReference[oaicite:17]{index=17}
We classify assistant turns into:
- `brand_bias`
- `user_retention`
- `sycophancy`
- `anthropomorphism`
- `harmful_generation`
- `sneaking`

Optional (only if label reliability supports it):
- `guilt_pressure`
- `refusal_erosion`

---

## 6) System Architecture (ASCII Diagram)

```

```
                       ┌─────────────────────────────┐
                       │         DarkBench           │
                       │  660 elicitation prompts    │
                       │  6 categories taxonomy      │
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
                                        │  (LLM Judge can optionally label too)
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
│ category rates, trends,     │
│ contexts, clusters          │
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

## 7) Functional Requirements
### FR1 — DarkBench Elicitation Runner
- Run 200–300 DarkBench prompts on N models (2–5 models).
- Store outputs with metadata (model, category, prompt id).

### FR2 — LLM Judge (Auditor)
- Rubric-based classification: returns:
  - predicted category (single or multi-label)
  - confidence score (0–1)
  - short explanation
  - optionally highlighted spans
- Must support:
  - repeated labeling for reliability assessment
  - different prompt templates for sensitivity testing

### FR3 — Labeling Set Builder
- Create `train.jsonl` and `eval.jsonl` using:
  - DarkBench outputs (known prompt category)
  - WildChat sampled turns labeled by LLM judge
  - Human audited subset for gold standard

### FR4 — Classifier Training Pipeline
- Train 6-way classifier with:
  - baseline: embeddings + logistic regression
  - preferred: MiniLM/DistilRoBERTa fine-tune
- Track macro F1, precision@high-confidence

### FR5 — WildChat Inference at Scale
- Ingest WildChat assistant turns.
- Run classifier on ≥50k–200k turns.
- Run LLM judge audit pass on:
  - top-k highest risk
  - low-confidence uncertain cases
  - random sample QA

### FR6 — Analytics + Reports
- Prevalence by category
- Prevalence vs conversation length
- Context clusters (topic clustering)
- Gap report:
  - compare distributions: DarkBench vs WildChat
  - compute JS divergence + rank correlation
- Reliability report:
  - judge self-consistency
  - judge vs classifier agreement
  - key disagreement cases

### FR7 — Demo UI
Streamlit app:
- upload transcript JSONL
- show flagged turns
- show risk-over-time plot
- export detections CSV + summary JSON

---

## 8) Data Specification
### Input
- WildChat: assistant turns extracted from dataset (text only). :contentReference[oaicite:18]{index=18}:contentReference[oaicite:19]{index=19}
- DarkBench: prompt subset + taxonomy categories. :contentReference[oaicite:20]{index=20}:contentReference[oaicite:21]{index=21}

### Output
- `outputs/darkbench_outputs.jsonl`
- `outputs/judge_labels.jsonl`
- `models/classifier/`
- `outputs/wildchat_detections.csv`
- `outputs/prevalence.json`
- `outputs/gap_report.json`
- `outputs/reliability_report.json`
- `app/streamlit_app.py`

---

## 9) Metrics
### Model quality (on gold subset)
- Macro F1
- Precision@HighConf (e.g., conf >= 0.8)
- Calibration bins (conf vs accuracy)

### Monitoring utility
- Review load (flags per 1,000 turns)
- % confirmed by judge on audit set

### Benchmark vs reality gap
- JS divergence between category distributions
- Spearman rank correlation of category frequencies

### Judge reliability (known issue)
- Self-consistency across repeated runs
- Prompt sensitivity effects
- Agreement vs gold human subset  
(LLM-as-a-judge reliability concerns documented in literature) :contentReference[oaicite:22]{index=22}:contentReference[oaicite:23]{index=23}

---

## 10) MVP Scope (48 hours)
### Strong MVP
- Focus on 3 categories: `user_retention`, `sycophancy`, `sneaking`
- Label 500 WildChat samples (judge + human audit 100)
- Train classifier
- Run on 50k–200k turns
- Publish prevalence + gap + reliability reports
- Streamlit demo

### Stretch
- All 6 categories
- Multi-turn accumulation detection
- Multi-judge ensemble (2 LLM judges)
- Compare to LMSYS-Chat-1M

---

## 11) 48-Hour Execution Plan
### Day 1
1) Finalize taxonomy + labeling guide
2) Run DarkBench subset on 2–3 models
3) Implement LLM judge rubric + JSON output format
4) Sample and judge-label 1,000 WildChat turns; human audit 200
5) Train baseline classifier + evaluation script

### Day 2
1) Train improved classifier (transformer)
2) Run inference on 50k–200k WildChat turns
3) Judge audit pass on high-risk + uncertain + random
4) Produce prevalence plots + gap report + reliability report
5) Build Streamlit demo + export reports
6) Write final report + limitations/dual-use appendix

---

## 12) Repo Structure (recommended)
```

wildguard/
README.md
PROJECT.md
data/
labeled/
annotations.csv
label_guide.md
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
app/
streamlit_app.py
reports/
report.md
limitations_dual_use.md

```

---

## 13) Ethics / Dual-Use Appendix (Required)
### Limitations
- false positives/negatives
- taxonomy incompleteness
- representativeness limits

### Dual-use
- detection tools could reveal what works
- mitigation:
  - publish aggregate results
  - redact or paraphrase illustrative examples

### Responsible disclosure
- no naming/shaming
- share patterns, not exploit templates

### Privacy
- avoid user identifiers
- redact examples
- follow dataset documentation and terms

---

## 14) Demo Script (3–5 min)
1) Show Streamlit log scan → flagged turns + categories
2) Show prevalence dashboard: category rates, risk vs length
3) Show gap report: DarkBench vs WildChat distribution mismatch
4) Show reliability report: judge variance + judge vs classifier disagreement
5) Conclusion: “We built a scalable oversight harness that bridges benchmark → deployment.”

---